import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
hg19_chromSize_fn = '../../../data/hg19.chrom-sizes'
CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # there was actually no data of chromatin state for chromosome Y
# CHROMOSOME_LIST = ['1']
ct_folder = '../../../data/hg19/K562'
ct_code = 'E123'
raw_logPval_dir = '../../../data/hg19/K562/roadmap_pval_signals'
binarized_for_chromHMM_dir = '../../../data/hg19/K562/three_mark_model_training/'
CHROM_MARK_LIST = ['DNase', 'H3K27ac', 'H3K4me3']
CHROM_MARK_LOGPVAL_THRESHOLD = {'DNase': 2.0, 'H2A.Z': 2.0, 'H3K27ac': 2.0, 'H3K27me3': 1.825, 'H3K36me3': 2.0, 'H3K4me1': 2.0, 'H3K4me2': 2.0, 'H3K4me3': 2.0, 'H3K79me2': 2.0, 'H3K9ac': 2.0, 'H3K9me1': 2.0, 'H3K9me3': 1.23, 'H4K20me1': 2.0} 
roadmap_segment_fn = '../../../data/hg19/K562/roadmap_published_chromHMM/E123_15_coreMarks_segments.bed.gz'
# NUM_STATE = 5
NUM_STATE_LIST = range(2,9) # the maximum number of states is 8, because we only have 3 marks here
NUM_BP_PER_WINDOW = 1000000
NUM_BP_PER_BIN = 200
NUM_BIN_PER_WINDOW = int(NUM_BP_PER_WINDOW / NUM_BP_PER_BIN)
model_outdir = '../../../model_analysis'
# for enrichment analysis
COORD_DIR_FOR_ENRICHMENT = '../../../data/hg19/genome_context_from_ha/for_enrichment'
COORD_DIR_FOR_NEIGHBORHOOD = '../../../program_source/ChromHMM/ChromHMM/ANCHORFILES/hg19'
def get_chromosome_length_dictionary():
	df = pd.read_csv(hg19_chromSize_fn, header = None, index_col = 0, sep = '\t')# 1 column: 1 --> length of the chromsome. indices are the chromosome
	df['length'] = np.ceil(df[1] / NUM_BP_PER_WINDOW).astype(int)
	results = pd.Series(df.length, index = df.index).to_dict()
	return results # keys: chr1, etc. values: # number of NUM_BP_PER_WINDOW-bp bins for this chromsomes
CHROMOSOME_NBINS_DICT = get_chromosome_length_dictionary()

def get_all_regions_binarized_fn_list():
	results = []
	for chrom in CHROMOSOME_LIST:
		# num_regions_this_chrom = CHROMOSOME_NBINS_DICT['chr' + chrom]
		num_regions_this_chrom = 3
		this_chrom_fn_list = list(map(lambda x: os.path.join(binarized_for_chromHMM_dir, 'genome_chr' + chrom + '.' + str(x) + '_binary.txt.gz'), range(num_regions_this_chrom)))
		results += this_chrom_fn_list
	return results

rule all:
	input:
		# get_all_regions_binarized_fn_list()
		# expand(os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'full_model','model_{num_state}.txt'), num_state = [8]), # to create the chromatin state model 
		# os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'eval_subset', 'sum_diag_all_marks.txt'), # to do eval subset of different marks
		expand(os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'overlap', 'overlap_enrichment_GC_gc.txt'), num_state = [8]), # to evalute and characterize the states
		expand(os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'neighborhood', 'neighborhood_enrichment_{context}.txt'), context = ['TES', 'TSS'], num_state = [8])

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

rule bigwigAverageOverBed:
	input:
		os.path.join(ct_folder, 'basic_bed', 'chr{chrom}.bed.gz'),
		os.path.join(raw_logPval_dir, ct_code + '-{mark}.pval.signal.bigwig')
	output:
		(os.path.join(raw_logPval_dir, '{mark}_chr{chrom}.tab')), 
		# 6 columns: name (same as the fourth column in get_one_chrom_basic_bed, size: 200bp, covered: mostly 200bp, sum, mean0: mean when non-covered bases are 0, mean: average of only covered bases)
	shell: 
		"""
		../../../program_source/ucsc_tools/bigWigAverageOverBed {input[1]} {input[0]} {output}
 		"""

rule binarize_multiple_bigwigAverageOverBed_output: # this rule will 
	input:
		expand(os.path.join(raw_logPval_dir, '{mark}_chr{{chrom}}.tab'), mark = CHROM_MARK_LIST), # from bigwigAverageOverBed
	output:
		(expand(os.path.join(binarized_for_chromHMM_dir, 'genome_chr{{chrom}}.{bin}_binary.txt'), bin = range(3)))
	run:	
		MEAN0_COLUMN_INDEX = 4
		all_region_df = pd.DataFrame()
		for fn in input:
			mark = fn.split('/')[-1].split('_')[0]
			mark_threshold = CHROM_MARK_LOGPVAL_THRESHOLD[mark]
			df = pd.read_csv(fn, header = None, sep = '\t', index_col = None)
			all_region_df[mark] = df[MEAN0_COLUMN_INDEX].apply(lambda x: 1 if x >= mark_threshold else 0)
		num_regions_this_chrom = CHROMOSOME_NBINS_DICT['chr' + wildcards.chrom]
		for region_index in range(num_regions_this_chrom):
			region_out_fn = os.path.join(binarized_for_chromHMM_dir, 'genome_chr{chrom}.{bin}_binary.txt'.format(chrom = wildcards.chrom, bin = region_index))
			start_index = int(region_index * NUM_BIN_PER_WINDOW)
			end_index = int(np.min([all_region_df.shape[0], start_index + NUM_BIN_PER_WINDOW - 1])) # I need the -1 because .loc in pandas actually include the end coordiate. Pandas is super weird, I tried it without -1 and it includes 1M+200 bp in each standard file instead of just 1M bp
			output_df = all_region_df.loc[start_index:end_index,:].copy()
			outF = open(region_out_fn, 'w')
			outF.write('genome\tchr{chrom}.{bin}\n'.format(chrom = wildcards.chrom, bin = region_index))
			output_df.to_csv(outF, header = True, index = False, sep = '\t')
			print("Done writing to for chr{chrom}.{bin}".format(chrom = wildcards.chrom, bin = region_index))

rule gzip_all_files:
	input: 
		expand(os.path.join(binarized_for_chromHMM_dir, 'genome_chr{{chrom}}.{bin}_binary.txt'), bin = range(3))
	output:
		expand(os.path.join(binarized_for_chromHMM_dir, 'genome_chr{{chrom}}.{bin}_binary.txt.gz'), bin = range(3))
	shell:
		"""	
		gzip -f {binarized_for_chromHMM_dir}/genome_chr{wildcards.chrom}.*_binary.txt
		"""

rule learn_model:
	input: 
		# get_all_regions_binarized_fn_list(),
		expand(os.path.join(binarized_for_chromHMM_dir, 'genome_chr{chrom}.{bin}_binary.txt.gz'), chrom = CHROMOSOME_LIST, bin = range(3))
	output:
		(os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'model_{num_state}.txt')),
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz')  # full_model is to notify that the model was trained based on the entire genome, and not on half of the genome for training and testing purposes in analysis of the optimal number of states
	shell:
		"""
		model_folder={model_outdir}/K562_hg19/three_mark_model/state_{wildcards.num_state}/full_model
		mkdir -p ${{model_folder}}
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar LearnModel -splitrows -holdcolumnorder -pseudo -many -p 6 -n 300 -d -1 -lowmem -gzip -noenrich -nobrowser {binarized_for_chromHMM_dir} ${{model_folder}} {wildcards.num_state} hg19
		"""


rule overlap_enrichment_GC: # overlap enrichment with genomcin contexts
	input:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz'),
	output: 
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model','overlap', 'overlap_enrichment_GC_gc.txt')
	params:
		output_no_tail = os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'overlap', 'overlap_enrichment_GC_gc')
	shell:
		"""	
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment {input[0]} {COORD_DIR_FOR_ENRICHMENT} {params.output_no_tail}
		"""

rule neighborhood_enrichment: # overlap with the neighboring regions of TSS and TES
	input:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz'),	
	output:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model','neighborhood', 'neighborhood_enrichment_{context}.txt')
	params:
		output_no_tail = os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'neighborhood', 'neighborhood_enrichment_{context}'),
		anchor_fn = os.path.join(COORD_DIR_FOR_NEIGHBORHOOD,'RefSeq{context}.hg19.txt.gz'),
	shell:
		"""
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar NeighborhoodEnrichment {input[0]} {params.anchor_fn} {params.output_no_tail}
		"""

#evalue different marks in how they recover the state segmentations
def helper_evalSubset_marks_to_include(wildcards):
	result = ['1']*len(CHROM_MARK_LIST)
	mark_index = CHROM_MARK_LIST.index(wildcards.mark)
	result[mark_index] = '0'
	return ''.join(result)

rule eval_subset:
	input:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'model_{num_state}.txt'),
		expand(os.path.join(binarized_for_chromHMM_dir, 'genome_chr{chrom}.{bin}_binary.txt.gz'), chrom = CHROMOSOME_LIST, bin = range(3))
	output:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'eval_subset', 'missing_{mark}.txt')
	params:
		output_no_tail = os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'eval_subset', 'missing_{mark}'),
		mark_to_include_string = helper_evalSubset_marks_to_include,
		segment_dir = os.path.join(model_outdir, 'K562_hg19', 'three_mark_model')
	shell:
		"""
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar EvalSubset {input[0]} {binarized_for_chromHMM_dir} {params.segment_dir} {params.output_no_tail} {params.mark_to_include_string}
		"""

rule combine_marks_eval_subset:
	input:
		expand(os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{{num_state}}', 'eval_subset', 'missing_{mark}.txt'), mark = CHROM_MARK_LIST) # from rule eval_subset
	output:
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'eval_subset', 'sum_diag_all_marks.txt'),
		os.path.join(model_outdir, 'K562_hg19', 'three_mark_model', 'state_{num_state}', 'full_model', 'eval_subset', 'sum_diag_all_marks.png')
	params:
	run:
		result_df = pd.DataFrame(columns = ['missing_mark', 'sum_diag'])
		for fn in input:
			missing_mark = fn.split('/')[-1].split('.')[0].split('_')[1]
			df = pd.read_csv(fn, skiprows = 1,header = 0, index_col = 0, sep = '\t')
			sum_diag_this_mark = np.sum(np.diag(df)) # calculate the sum of diagonals for this mark
			result_df.loc[result_df.shape[0]] = [missing_mark, sum_diag_this_mark]
		# now, we will report
		result_df.to_csv(output[0], header = True, index = False, sep = '\t')
		ax = sns.barplot(x = 'sum_diag', y = 'missing_mark', data = result_df, color = 'blue', ci = None)
		plt.tight_layout()
		plt.savefig(output[1])

