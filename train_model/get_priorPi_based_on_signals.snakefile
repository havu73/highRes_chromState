import os 
import pandas as pd 
import numpy as np 
raw_metadata_fn = '../../data/hg19/roadmap_metadata_consolidated_epig_summary.csv'
processed_raw_metadata_fn = '../../data/hg19/processed_raw_metadata_roadmap_epig_summary.csv'
roadmap_S25_segment_folder = '/gstore/project/gepiviz_data/vuh6/roadmap/S25_segment/'
roadmap_S25_signal_folder = '/gstore/project/gepiviz_data/vuh6/roadmap/marks_signal/'
roadmap_S25_emission = '/gstore/project/gepiviz_data/vuh6/roadmap/emission/emissions_25_imputed12marks.txt'
basic_bed_folder = '../../data/hg19/basic_bed/'
hg19_chrom_length = '../../data/hg19.chrom-sizes'
train_folder = '../../pyro_model/old_generative/'
ct_to_predict = 'E123' # K562
all_three_marks_ref_epig_fn = '../../data/hg19/DNase_K4me3_K27ac_ref_epig_id.csv'
three_mark_signal_dir = '../../data/hg19/K562/three_mark_model_training/'
CHROM_MARK_LIST = ['DNase', 'H3K4me3', 'H3K27ac']
CHROM_MARK_LOGPVAL_THRESHOLD = {'DNase': 2.0, 'H2A.Z': 2.0, 'H3K27ac': 2.0, 'H3K27me3': 1.825, 'H3K36me3': 2.0, 'H3K4me1': 2.0, 'H3K4me2': 2.0, 'H3K4me3': 2.0, 'H3K79me2': 2.0, 'H3K9ac': 2.0, 'H3K9me1': 2.0, 'H3K9me3': 1.23, 'H4K20me1': 2.0} 

def get_ct_list(processed_raw_metadata_fn):
	df = pd.read_csv(processed_raw_metadata_fn, header = 0, index_col = None, sep = ',')
	df = df.rename(columns = {'Epigenome ID (EID)': 'EID'})
	all_ct_list = list(df.EID)
	# all_ct_list.remove(ct_to_predict)
	return all_ct_list #list(df.EID)

def get_ct_list_with_3Marks(): # only get list of ct id where the mark DNase has been profiled
	df = pd.read_csv(all_three_marks_ref_epig_fn, header = None, index_col = None, squeeze = True)
	return list(df.values)

rule all:
	input:
		# expand(os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'indv_marks', 'genome_chr{chrom}_binary_{mark}.txt.gz'), ct = get_ct_list(processed_raw_metadata_fn), chrom = [21,22], mark = ['H3K4me3']),
		expand(os.path.join(train_folder, 'similarity_between_ref_epig', '{mark}_chr{chrom}', '{mark}_chr{chrom}_{index}.txt.gz'), index = ['jaccard', 'MI'], mark = 'H3K4me3', chrom = 22), 

rule download_roadmap_pval_bigwig:
	input:
		processed_raw_metadata_fn
	output:
		temp(os.path.join(roadmap_S25_signal_folder, '{ct}', 'downloaded_pval_bigwig', '{ct}-{mark}.pval.signal.bigwig'))
	params:
		output_folder = os.path.join(roadmap_S25_signal_folder, '{ct}', 'downloaded_pval_bigwig')
	shell:
		"""
		link_prefix=https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/pval/
		link="${{link_prefix}}/{wildcards.ct}-{wildcards.mark}.pval.signal.bigwig"
		wget $link -P {params.output_folder}
		"""

rule bigwigAverageOverBed:
	input:
		os.path.join(basic_bed_folder, 'chr{chrom}.bed.gz'),
		os.path.join(roadmap_S25_signal_folder, '{ct}', 'downloaded_pval_bigwig', '{ct}-{mark}.pval.signal.bigwig')
	output:
		temp(os.path.join(roadmap_S25_signal_folder, '{ct}', 'downloaded_pval_bigwig', '{mark}_chr{chrom}.tab')), 
		# 6 columns: name (same as the fourth column in get_one_chrom_basic_bed, size: 200bp, covered: mostly 200bp, sum, mean0: mean when non-covered bases are 0, mean: average of only covered bases)
	shell: 
		"""
		../../program_source/ucsc_tools/bigWigAverageOverBed {input[1]} {input[0]} {output}
 		"""

rule binarize_multiple_bigwigAverageOverBed_output: # this rule will 
	input:
		os.path.join(roadmap_S25_signal_folder, '{ct}', 'downloaded_pval_bigwig', '{mark}_chr{chrom}.tab'), # from bigwigAverageOverBed
	output:
		os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'indv_marks', 'genome_chr{chrom}_binary_{mark}.txt.gz')
	run:	
		MEAN0_COLUMN_INDEX = 4
		all_region_df = pd.DataFrame()
		fn = input[0]
		mark = fn.split('/')[-1].split('_')[0]
		mark_threshold = CHROM_MARK_LOGPVAL_THRESHOLD[mark]
		df = pd.read_csv(fn, header = None, sep = '\t', index_col = None)
		all_region_df[mark] = df[MEAN0_COLUMN_INDEX].apply(lambda x: 1 if x >= mark_threshold else 0)
		all_region_df.to_csv(output[0], header = True, index = False, sep = '\t', compression = 'gzip')


rule get_jaccard_MI_across_ct:
	input:
		expand(os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'indv_marks', 'genome_chr{{chrom}}_binary_{{mark}}.txt.gz'), ct = get_ct_list(processed_raw_metadata_fn))
	output:
		expand(os.path.join(train_folder, 'similarity_between_ref_epig', '{{mark}}_chr{{chrom}}', '{{mark}}_chr{{chrom}}_{index}.txt.gz'), index = ['jaccard', 'MI'])
	params:
		output_prefix = os.path.join(train_folder, 'similarity_between_ref_epig', '{mark}_chr{chrom}', '{mark}_chr{chrom}'),
	shell:
		"""
		python calcualte_jaccard_index_all_ref_ct.py {roadmap_S25_signal_folder} {params.output_prefix} {wildcards.chrom} {wildcards.mark} 
		"""