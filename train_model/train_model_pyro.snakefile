import os 
import pandas as pd 
import numpy as np 
raw_metadata_fn = '../../data/hg19/roadmap_metadata_consolidated_epig_summary.csv'
processed_raw_metadata_fn = '../../data/hg19/processed_raw_metadata_roadmap_epig_summary.csv'
hg19_chromSize_fn = '../../data/hg19.chrom-sizes'
train_folder = '../../pyro_model/old_generative/K562_hg19/'
ct_to_predict = 'E123' # K562
roadmap_S25_emission = '../../data/hg19/roadmap/emission/emissions_25_imputed12marks.txt'

rule all: 
	input:
		expand(os.path.join(train_folder, 'train_output', 'frac_{train_frac}', 'params', '{output_fn}'), train_frac = 0.1, output_fn = ['beta_state_transition.txt', 'posterior_pi.txt']),

rule train_model:
	input:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'ref_epig_segment_for_training.bed.gz'), # data of chromatin state maps in reference epigenomes
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'observed_mark_signals_for_training.bed.gz'), # data of chromatin mark signals from experiments
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'uniform_alpha.txt'),
		roadmap_S25_emission
	output:
		os.path.join(train_folder, 'train_output', 'frac_{train_frac}', 'params', 'beta_state_transition.txt'),
		os.path.join(train_folder, 'train_output', 'frac_{train_frac}', 'params', 'posterior_pi.txt'),
	params:
		output_folder =	os.path.join(train_folder, 'train_output', 'frac_{train_frac}', 'params'),
	shell:
		"""
		python train_model_using_pyro.py {input[0]} {input[1]} {input[2]} {input[3]} {params.output_folder}
		"""

rule get_one_chrom_basic_bed:
	# get basic bed: chrom, start, end, index(0-based). Each segment 200bp apart based on the length of the chromosome
	input:
		hg19_chromSize_fn,
	output:
		os.path.join(ct_folder, 'basic_bed', 'chr{chrom}.bed.gz'),
	params: 
		NUM_BP_PER_WINDOW = 200
	run: 
		this_chrom = 'chr' + wildcards.chrom
		chromSize_df = pd.read_csv(input[0], header = None, sep = '\t', index_col = None)
		chrom_len = chromSize_df[chromSize_df[0] == this_chrom].iloc[0,1]
		num_bins = int(np.floor(chrom_len / params.NUM_BP_PER_WINDOW))
		result_df = pd.DataFrame({'chrom' : np.repeat(this_chrom, num_bins)})
		result_df['start'] = np.arange(num_bins) * params.NUM_BP_PER_WINDOW
		result_df['end'] = result_df['start'] + params.NUM_BP_PER_WINDOW
		result_df['bin_index'] = np.arange(num_bins)
		result_df.to_csv(output[0], header = False, sep = '\t', index = False, compression = 'gzip')

