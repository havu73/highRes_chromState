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
		# get raw data segmentaiton from online
		#get sample data
		expand(os.path.join(train_folder, 'train_segmentation_data', 'maxBin_{max_bin}_chr{chrom}', 'ref_epig_segment_for_training.bed.gz'), max_bin = 5000, chrom = 22) , # data of chromatin state maps in reference epigenomes
		expand(os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', 'observed_mark_signals_for_training.bed.gz'), max_bin = 5000, chrom = 22, ct = ['E123']), # data of chromatin mark signals from experiments
		# expand(os.path.join(train_folder, '{ct}_hg19', 'train_data', 'frac_{train_frac}', 'uniform_alpha.txt'), train_frac = 0.1)
		# data of model parameters after training
		# expand(os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params', '{output_fn}'), ct = ['E123'], max_bin = 5000, chrom = 22, output_fn = ['beta_state_transition.txt', 'posterior_pi.txt']),
		# expand(os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'posteror_state_prob', 'posterior_state_prob.txt.gz'), ct = ['E123'], max_bin = 5000, chrom = 22)



rule sample_genome_for_training: # sample 200bp window on the genome for training the model on pyro, resutling in one file chrom, start, end --> coordinates that we randomly chose to be in the training data
	input:	
		hg19_chrom_length,
	output:
		os.path.join(train_folder, 'train_segmentation_data', 'frac_{train_frac}_chr{chrom}', 'sample_genome_sorted.bed.gz')
	params:
		unsorted_fn = os.path.join(train_folder, 'train_segmentation_data', 'frac_{train_frac}_chr{chrom}', 'sample_genome.bed.gz'),
		output_no_gz = os.path.join(train_folder, 'train_segmentation_data', 'frac_{train_frac}_chr{chrom}', 'sample_genome_sorted.bed')
	shell:
		"""
		python sample_genome_for_training.py {input[0]} {wildcards.train_frac} {params.unsorted_fn} {wildcards.chrom}
		zcat {params.unsorted_fn} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule get_training_ref_segmentation: # this rule will produce the matrix of chromatin state maps in all 127 reference epigenomes. When later used in the training function, we will specify list of rerefence epigenome id used for training through the alpha vector.
	input:
		'/gstore/home/vuh6/pyro_model/old_generative/K562_hg19/train_data/maxBin_5000_chr22/maxBin_5000_chr22_equal_state_rep.bed.gz', # from sample_one_refEpig_by_state.py
		expand(os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_segments_sorted.bed.gz'), ct = get_ct_list(processed_raw_metadata_fn)), # from download_roadmap_S25_segment
	output:
		os.path.join(train_folder, 'train_segmentation_data', 'maxBin_{max_bin}_chr{chrom}', 'ref_epig_segment_for_training.bed.gz')
	params:
		ref_epig_fn_list_str = ' '.join(list(map(lambda x: os.path.join(roadmap_S25_segment_folder, '{}'.format(x), '{}_25_segments_sorted.bed.gz'.format(x)), get_ct_list(processed_raw_metadata_fn)))), # a little horrifying, excuse coder!
		output_no_gz = os.path.join(train_folder, 'train_segmentation_data', 'maxBin_{max_bin}_chr{chrom}', 'ref_epig_segment_for_training.bed'),
		header = 'chrom\tstart\tend\t' + '\t'.join(get_ct_list(processed_raw_metadata_fn)), # header of the output file
	shell:
		"""
		rm -f {params.output_no_gz}
		command="zcat {input[0]}"
		for fn in {params.ref_epig_fn_list_str}
		do
			command="$command | bedtools map -a - -b $fn -c 4 -o collapse"
		done
		command="$command >> {params.output_no_gz}"
		echo "{params.header}"
		echo -e "{params.header}" > {params.output_no_gz} # write header
		eval $command  # map the reference epig for all the reference epigenome
		gzip -f {params.output_no_gz}
		"""

# now get the data of chromatin mark signals for training
rule get_chrom_mark_signals_for_training:
	input:
		os.path.join(train_folder, 'K562_hg19/train_data/maxBin_5000_chr22/maxBin_5000_chr22_equal_state_rep.bed.gz' ), # from what I called outside right now
		os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'genome_chr{chrom}_binary.txt.gz')
	output:
		(os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', 'observed_mark_signals_for_training.bed.gz'))
	params:
		binarized_signal_folder = os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals')
	shell:
		"""
		python get_chrom_mark_signals_for_training.py {params.binarized_signal_folder} {input[0]} {output[0]}
		"""

rule get_uniform_alpha_for_training:
	input:
		processed_raw_metadata_fn
	output:
		os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', 'uniform_alpha.txt')
	run:
		ct_list = get_ct_list(processed_raw_metadata_fn)
		ct_list.remove(wildcards.ct)
		alpha = np.array([1.5] * len(ct_list))
		alpha = pd.Series(alpha)
		alpha.index = ct_list
		alpha.to_csv(output[0], header = None, index = True, sep = '\t')

rule get_alpha_based_on_jaccard:
	input:
		(os.path.join(train_folder, 'similarity_between_ref_epig', 'H3K4me3_chr22', 'H3K4me3_chr22_{index}.txt.gz'))
	output:
		os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', '{index}_based_alpha.txt')
	run:
		sim_df = pd.read_csv(input[0], header = 0, index_col = 0, sep = '\t')
		sim_one_ct = sim_df.loc[wildcards.ct] # get the similarity vecto associated with this ct
		sim_one_ct.drop(wildcards.ct, inplace = True) # within the similarity vector, drop the ct so we only get data from the remaining ct
		sim_one_ct = sim_one_ct * (1/min(sim_one_ct)) # make the smallest entry become 1
		sim_one_ct.to_csv(output[0], header = None, index = True, sep = '\t')

rule train_model:
	input:
		os.path.join(train_folder, 'train_segmentation_data', 'maxBin_{max_bin}_chr{chrom}', 'ref_epig_segment_for_training.bed.gz'), # data of chromatin state maps in reference epigenomes
		os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', 'observed_mark_signals_for_training.bed.gz'), # data of chromatin mark signals from experiments
		os.path.join(train_folder, '{ct}_hg19', 'train_data', 'maxBin_{max_bin}_chr{chrom}', 'MI_based_alpha.txt'),
		roadmap_S25_emission
	output:
		os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params', 'beta_state_transition.txt'),
		os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params', 'posterior_pi.txt'),
	params:
		output_folder =	os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params'),
		NUM_TRAIN_ITERATIONS = 100,
		NUM_BINS_SAMPLE_PER_ITER = 5000
	shell:
		"""
		python train_model_using_pyro.py {input[0]} {input[1]} {input[2]} {input[3]} {params.output_folder} {params.NUM_TRAIN_ITERATIONS} {params.NUM_BINS_SAMPLE_PER_ITER}
		"""

rule calculate_posterior_state_probabilities:
	input:
		os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params', 'beta_state_transition.txt'), # from rule train_model
		os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params', 'posterior_pi.txt'), # from rule train_model
		os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'genome_chr{chrom}_binary.txt.gz'), # from rule binarize_multiple_bigwigAverageOverBed_output
	output:
		os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'posteror_state_prob', 'posterior_state_prob.txt.gz')
	params:
		params_folder = os.path.join(train_folder, '{ct}_hg19', 'train_output', 'maxBin_{max_bin}_chr{chrom}', 'params'),
		obs_chromMark_signal_fn = os.path.join(roadmap_S25_signal_folder, '{ct}', 'binarized_signals', 'genome_chr{chrom}_binary.txt.gz'), # from rule binarize_multiple_bigwigAverageOverBed_output
		chrom_basic_bed = os.path.join(basic_bed_folder, 'chr{chrom}.bed.gz'),
	shell:
		"""
		python calculate_gw_state_posterior_prob.py {params.params_folder} {roadmap_S25_emission} {roadmap_S25_segment_folder} {params.obs_chromMark_signal_fn} {output[0]} {params.chrom_basic_bed}
		"""