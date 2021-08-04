import os 
import pandas as pd 
import numpy as np 
raw_metadata_fn = '../../data/hg19/roadmap_metadata_consolidated_epig_summary.csv'
processed_raw_metadata_fn = '../../data/hg19/processed_raw_metadata_roadmap_epig_summary.csv'
roadmap_S25_segment_folder = '../../data/hg19/roadmap/S25_segment/'
hg19_chrom_length = '../../data/hg19.chrom-sizes'
train_folder = '../../pyro_model/K562_hg19/'
ct_to_predict = 'E123' # K562
three_mark_signal_dir = '../../data/hg19/K562/three_mark_model_training/'

def get_ct_list(processed_raw_metadata_fn):
	df = pd.read_csv(processed_raw_metadata_fn, header = 0, index_col = None, sep = '\t')
	all_ct_list = list(df.EID)
	all_ct_list.remove(ct_to_predict)
	return all_ct_list #list(df.EID)

rule all:
	input:
		# get raw data segmentaiton from online
		processed_raw_metadata_fn,
		expand(os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_segments_sorted.bed.gz'), ct = get_ct_list(processed_raw_metadata_fn)),
		#get sample data
		expand(os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'ref_epig_segment_for_training.bed.gz'), train_frac = 0.1) , # data of chromatin state maps in reference epigenomes
		expand(os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'observed_mark_signals_for_training.bed.gz'), train_frac = 0.1), # data of chromatin mark signals from experiments
		expand(os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'uniform_alpha.txt'), train_frac = 0.1)

rule process_raw_metadata: # get clean metadata
	input:
		raw_metadata_fn,
	output:
		processed_raw_metadata_fn
	shell:
		"""
		cat {input} | awk -F',' '{{print $2,$4,$5,$6,$17}}' | awk  'BEGIN{{OFS="\t"}}{{if (NR == 1 || NR > 6) print $1,$2,$3,$4,$5}}' > {output}
		"""

rule download_roadmap_S25_segment:
	input:
		processed_raw_metadata_fn
	output:
		os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_imputed12marks_segments.bed.gz') # important note: these files are NOT sorted through sort -k1,1 -k2,2n to prepare for the format of bedtools map
	params:
		ct_out_folder=os.path.join(roadmap_S25_segment_folder, '{ct}')
	shell:
		"""
		link_prefix='https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/imputed12marks/jointModel/final/'
		link="${{link_prefix}}/{wildcards.ct}_25_imputed12marks_segments.bed.gz"
		echo $link
		wget $link -P {params.ct_out_folder}
		"""

rule sort_roadmap_S25_segment:
	input:
		os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_imputed12marks_segments.bed.gz') # from download_roadmap_S25_segment
	output:
		os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_segments_sorted.bed.gz')
	params:
		output_no_gz = os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_segments_sorted.bed')
	shell:
		"""
		zcat {input[0]} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule sample_genome_for_training: # sample 200bp window on the genome for training the model on pyro, resutling in one file chrom, start, end --> coordinates that we randomly chose to be in the training data
	input:	
		hg19_chrom_length,
	output:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'sample_genome_sorted.bed.gz')
	params:
		unsorted_fn = os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'sample_genome.bed.gz'),
		output_no_gz = os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'sample_genome_sorted.bed')
	shell:
		"""
		python sample_genome_for_training.py {input[0]} {wildcards.train_frac} {params.unsorted_fn}
		zcat {params.unsorted_fn} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule get_training_ref_segmentation:
	input:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'sample_genome_sorted.bed.gz'), # from sample_genome_for_training
		expand(os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_segments_sorted.bed.gz'), ct = get_ct_list(processed_raw_metadata_fn)), # from download_roadmap_S25_segment
	output:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'ref_epig_segment_for_training.bed.gz')
	params:
		ref_epig_fn_list_str = ' '.join(list(map(lambda x: os.path.join(roadmap_S25_segment_folder, '{}'.format(x), '{}_25_segments_sorted.bed.gz'.format(x)), get_ct_list(processed_raw_metadata_fn)))), # a little horrifying, excuse coder!
		output_no_gz = os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'ref_epig_segment_for_training.bed'),
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
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'sample_genome_sorted.bed.gz'), # from sample_genome_for_training
	output:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'observed_mark_signals_for_training.bed.gz')
	shell:
		"""
		python get_chrom_mark_signals_for_training.py {three_mark_signal_dir} {input[0]} {output[0]}
		"""

rule get_uniform_alpha_for_training:
	input:
		processed_raw_metadata_fn
	output:
		os.path.join(train_folder, 'train_data', 'frac_{train_frac}', 'uniform_alpha.txt')
	run:
		ct_list = get_ct_list(processed_raw_metadata_fn)
		alpha = np.array([1.5] * len(ct_list))
		alpha = pd.Series(alpha)
		alpha.index = ct_list
		alpha.to_csv(output[0], header = None, index = True, sep = '\t')
