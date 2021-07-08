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
CHROM_MARK_LIST = ['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K27me3', 'H3K9me3']
roadmap_segment_fn = '../../../data/hg19/K562/roadmap_published_chromHMM/E123_15_coreMarks_segments.bed.gz'
NUM_STATE = 15

rule all:
	input:
		# get_one_chrom_basic_bed
		# bigwigAverageOverBed
		expand(os.path.join(raw_logPval_dir, 'test_binarization_procedure', 'logPval2', '{mark}_histogram.png'), mark = CHROM_MARK_LIST),

rule get_one_chrom_basic_bed:
	# get basic bed: chrom, start, end, index(0-based). Each segment 200bp apart based on the length of the chromosome
	input:
		hg19_chromSize_fn,
	output:
		os.path.join(ct_folder, 'basic_bed', 'chr{chrom}.bed.gz'),
	params: 
		NUM_BP_PER_BIN = 200
	run: 
		this_chrom = 'chr' + wildcards.chrom
		chromSize_df = pd.read_csv(input[0], header = None, sep = '\t', index_col = None)
		chrom_len = chromSize_df[chromSize_df[0] == this_chrom].iloc[0,1]
		num_bins = int(np.floor(chrom_len / params.NUM_BP_PER_BIN))
		result_df = pd.DataFrame({'chrom' : np.repeat(this_chrom, num_bins)})
		result_df['start'] = np.arange(num_bins) * params.NUM_BP_PER_BIN
		result_df['end'] = result_df['start'] + params.NUM_BP_PER_BIN
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

rule count_logPval_occurrences:
	input:
		expand(os.path.join(raw_logPval_dir, '{{mark}}_chr{chrom}.tab'), chrom = CHROMOSOME_LIST) # from bigwigAverageOverBed
	output:
		os.path.join(raw_logPval_dir, '{mark}_gw_count_logPval.tab')
	params:
		chrom_list_string = ' '.join(CHROMOSOME_LIST)
	shell:
		"""
		rm -f {output[0]} # so that we can write a new file
		for chrom in {params.chrom_list_string}
		do
			in_fn={raw_logPval_dir}/{wildcards.mark}_chr${{chrom}}.tab
			cat ${{in_fn}} | awk -F'\t' '{{print $5}}' |sort |uniq -c >> {output[0]}
		done
		"""

rule create_histogram_logPval:
	input:
		os.path.join(raw_logPval_dir, '{mark}_gw_count_logPval.tab')
	output:
		os.path.join(raw_logPval_dir, 'test_binarization_procedure', 'logPval2', '{mark}_histogram.png'),
		# os.path.join(raw_logPval_dir, 'test_binarization_procedure', 'logPval2', '{mark}_histogram.csv')
	params: 
		shortcut_fn = os.path.join(raw_logPval_dir, 'test_binarization_procedure', 'logPval2', '{mark}_histogram_bin01.csv')
	run:	
		bin_size = 1.0 / 100
		threshold_bin_index = int(2.0 / bin_size)
		# count_df = pd.read_csv(input[0], header = None, index_col = None) # 1 column of name 0 and it is a string of count<space>logPval. Whenenver I call logPval, it is actually -log10(pval)
		# count_df[0] = count_df[0].str.strip()
		# count_df['count'] = (count_df[0]).apply(lambda x: x.split()[0]).astype(int)
		# count_df['logPval'] = (count_df[0]).apply(lambda x: x.split()[1]).astype(float)
		# count_df.drop(0, axis = 1, inplace = True) # drop the first column
		# count_df['bin_index'] = np.floor(count_df['logPval'] / bin_size).astype(int)
		# count_df['bin_index'][count_df['bin_index'] > threshold_bin_index] = threshold_bin_index
		# count_df.drop('logPval', axis = 1, inplace = True) # drop the column logPval
		# count_df = count_df.groupby('bin_index').sum().reset_index() # 2 columns: bin_index, count
		# count_df.to_csv(output[1], header = True, index = False, sep = '\t')
		count_df = pd.read_csv(params.shortcut_fn, header = 0, index_col = None, sep = '\t')
		# now, onto drawing the histogram
		ax = sns.barplot(x = 'bin_index', y = 'count', color = 'blue', data = count_df, ci = None)
		plt.savefig(output[0])


