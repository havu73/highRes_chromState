import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
import glob
hg19_chromSize_fn = '../../../data/hg19.chrom-sizes'
TRAIN_CHROMOSOME_LIST = ['1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21', 'X']
TEST_CHROMOSOME_LIST = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22']

CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # there was actually no data of chromatin state for chromosome Y
# CHROMOSOME_LIST = ['1']
ct_folder = '../../../data/hg19/K562'
ct_code = 'E123'
raw_logPval_dir = '../../../data/hg19/K562/roadmap_pval_signals'
binarized_for_chromHMM_dir = '../../../data/hg19/K562/big_model_training/'
CHROM_MARK_LIST = ['DNase', 'H2A.Z', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me2', 'H3K9ac', 'H3K9me1', 'H3K9me3', 'H4K20me1'] # these marks are ordered exactly as the emission matrix
CHROM_MARK_LOGPVAL_THRESHOLD = {'DNase': 2.0, 'H2A.Z': 2.0, 'H3K27ac': 2.0, 'H3K27me3': 1.825, 'H3K36me3': 2.0, 'H3K4me1': 2.0, 'H3K4me2': 2.0, 'H3K4me3': 2.0, 'H3K79me2': 2.0, 'H3K9ac': 2.0, 'H3K9me1': 2.0, 'H3K9me3': 1.23, 'H4K20me1': 2.0} 
roadmap_segment_fn = '../../../data/hg19/K562/roadmap_published_chromHMM/E123_15_coreMarks_segments.bed.gz'
RANDOM_SEED_LIST = [ 95, 325, 319, 257, 685, 436, 958,  23, 397, 549, 348, 296, 557, 428, 168, 228, 982,  72, 584, 944, 318, 223, 781, 176, 429, 899,       847, 425, 894, 478]
# NUM_STATE = 5
NUM_STATE_LIST = list(range(2,26)) + list(range(30, 205, 5))# from 2 --> 25 states
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

def get_all_regions_binarized_fn_list(train_or_test):
	results = []
	if (train_or_test == 'train'):
		chrom_list_for_this_type = TRAIN_CHROMOSOME_LIST
	elif train_or_test == 'test':
		chrom_list_for_this_type = TEST_CHROMOSOME_LIST 
	elif train_or_test == 'full':
		chrom_list_for_this_type = CHROMOSOME_LIST
	else: return []
	for chrom in chrom_list_for_this_type:
		# num_regions_this_chrom = CHROMOSOME_NBINS_DICT['chr' + chrom]
		num_regions_this_chrom = 3
		this_chrom_fn_list = list(map(lambda x: os.path.join(binarized_for_chromHMM_dir, train_or_test, 'genome_chr' + chrom + '.' + str(x) + '_binary.txt.gz'), range(num_regions_this_chrom)))
		results += this_chrom_fn_list
	return results

rule all:
	input:
		# get_all_regions_binarized_fn_list()
		#expand(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'model_{num_state}.txt'), num_state = NUM_STATE_LIST), # to create the chromatin state model 
		# expand(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt'), num_state = NUM_STATE_LIST, chrom = TEST_CHROMOSOME_LIST),
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic_all_test_chrom.png'),
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'optimal_state_by_chrom.txt'),

##### SOME RULES TO BINARIZE THE SIGNALS AND PREPARE DATA FOR LEARNING THE CHROMHMM MODEL #####
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
		chrom_length_df = pd.read_csv(input[0], header = None, sep = '\t', index_col = None)
		chrom_len = chrom_length_df[chrom_length_df[0] == this_chrom].iloc[0,1]
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

##### LEARN MODEL ######
rule learn_model:
	input: 
		expand(os.path.join(binarized_for_chromHMM_dir, 'full', 'genome_chr{chrom}.{bin}_binary.txt.gz'), chrom = CHROMOSOME_LIST, bin = range(3))
	output:
		(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'full', 'model_{num_state}.txt')),
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'full', 'genome_{num_state}_segments.bed.gz') 
	params:
		input_binarized_dir = os.path.join(binarized_for_chromHMM_dir, 'full')
	shell:
		"""
		model_folder={model_outdir}/K562_hg19/all_mark_model/state_{wildcards.num_state}/full
		mkdir -p ${{model_folder}}
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar LearnModel -splitrows -holdcolumnorder -pseudo -many -p 6 -n 300 -d -1 -lowmem -gzip -noenrich -nobrowser {params.input_binarized_dir} ${{model_folder}} {wildcards.num_state} hg19
		"""

rule learn_model_train_test:
	input: 
		expand(os.path.join(binarized_for_chromHMM_dir, 'train', 'genome_chr{chrom}.{bin}_binary.txt.gz'), chrom = TRAIN_CHROMOSOME_LIST, bin = range(3))
	output:
		(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'model_{num_state}.txt')),
	params:
		input_binarized_dir = os.path.join(binarized_for_chromHMM_dir, 'train')
	shell:
		"""
		model_folder={model_outdir}/K562_hg19/all_mark_model/state_{wildcards.num_state}/half_for_model_evaluation
		mkdir -p ${{model_folder}}
		java -jar ../../../program_source/ChromHMM/ChromHMM/ChromHMM.jar LearnModel -splitrows -holdcolumnorder -pseudo -many -p 6 -n 300 -d -1 -lowmem -gzip -noenrich -nobrowser {params.input_binarized_dir} ${{model_folder}} {wildcards.num_state} hg19
		"""

##### PREPARE DATA FOR TRAIN AND TEST SEGMENTATION #####
###### CALCULATE THE AIC AND BIC FOR MODELS OF DIFFERENT NUMBER OF STATES #####

rule calcualte_log_llh_one_chrom:
	input:
		os.path.join(model_outdir,'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'model_{num_state}.txt'), # the model file
		get_all_regions_binarized_fn_list('test'), 
	output:
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt')
	params:
		binarized_for_test_dir = os.path.join(binarized_for_chromHMM_dir, 'test')
	shell:
		"""
		python calculate_log_likelihood_forward_alg_one_chrom.py {input[0]} {params.binarized_for_test_dir} {output[0]} {wildcards.chrom} 
		"""

def count_gw_bins():
	chrom_length_df = pd.read_csv(hg19_chromSize_fn, index_col = 0, header = None, sep = '\t', squeeze = False)
	chrom_length_df.columns = ['length_bp']
	chrom_length_df['num_bins'] = np.floor(chrom_length_df['length_bp'].astype(float) / NUM_BP_PER_BIN).astype(int)
	num_bin_test_chrom_list = list(map(lambda x: chrom_length_df.loc['chr{chrom}'.format(chrom = x)]['num_bins'], TEST_CHROMOSOME_LIST))
	return chrom_length_df, np.sum(num_bin_test_chrom_list).astype(int)

rule compare_aic_bic_models_all_test_chrom:
	input:
		expand(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt'), num_state = NUM_STATE_LIST, chrom = TEST_CHROMOSOME_LIST), # each file is produced by each time calling rule calcualte_log_llh_one_chrom
	output:
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic.txt'),
	params:
		out_dir = os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test')
	run:	
		aic_list = []
		bic_list = []
		gw_neg2_logLlh_list = []
		_ , gw_bins = count_gw_bins()
		print(gw_bins)
		for num_state in NUM_STATE_LIST: # for each model, calculate 
			llh_fn_list = list(map(lambda x: os.path.join(model_outdir, 'K562_hg19','all_mark_model', 'state_{num_state}'.format(num_state = num_state), 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt'.format(chrom = x)), TEST_CHROMOSOME_LIST)) 
			llh_list = list(map(lambda x: pd.read_csv(x, header = None, index_col = None, squeeze = True)[0], llh_fn_list)) # read a bunch of 1-line files , outputing a list of numbers, each representing the log llh in one chromosome
			gw_log_llh = np.sum(llh_list)
			gw_neg2_logLlh_list.append(- 2 * gw_log_llh)
			num_params = num_state + num_state ** 2 + num_state * len(CHROM_MARK_LIST) # init, transition, emission 
			aic = 2 * num_params - 2 * gw_log_llh
			aic_list.append(aic)
			bic = num_params * np.log(gw_bins) - 2 * gw_log_llh
			bic_list.append(bic)
		result_df = pd.DataFrame(data = {'num_state': NUM_STATE_LIST, 'aic': aic_list, 'bic': bic_list, 'log_llh': gw_neg2_logLlh_list}) 
		result_df.to_csv(output[0], header = True, index = False, sep = '\t')
		plot_df = pd.melt(result_df, id_vars = ['num_state'], value_vars = ['aic', 'bic', 'log_llh'])
		plot_df.columns = ['num_state', 'aic_or_bic', 'values']
		ax = sns.pointplot(data = plot_df, x = 'num_state', y = 'values', hue = 'aic_or_bic', plot_kws=dict(alpha=0.3))
		ax.set_xticklabels(plot_df['num_state'], size = 5, rotation = 270) 
		plt.tight_layout()
		plt.savefig(os.path.join(params.out_dir, 'aic_bic.png'))


def calculate_statistics_for_model_evaluation(num_state, chrom, chrom_length_df):
	# chrom_length_df: index are of forms chr1, etc.
	# columns: length and num_bins
	log_llh_fn = os.path.join(model_outdir, 'K562_hg19','all_mark_model', 'state_{num_state}'.format(num_state = num_state), 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt'.format(chrom = chrom)) # there file where we can get the calculation of log likelihood 
	log_llh = pd.read_csv(log_llh_fn, header = None, index_col = None, squeeze = True)[0]
	num_params = num_state + num_state ** 2 + num_state * len(CHROM_MARK_LIST)
	num_bins_this_chrom = chrom_length_df.loc['chr{chrom}'.format(chrom = chrom)]['num_bins']
	aic = 2 * num_params - 2 * log_llh
	bic = num_params * np.log(num_bins_this_chrom) - 2 * log_llh
	neg2_logLlh = -2 * log_llh
	return (aic, bic, neg2_logLlh)

rule compare_aic_bic_models_by_chrom:
	input:
		expand(os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'state_{num_state}', 'half_for_model_evaluation', 'log_llh', 'log_llh_chr{chrom}.txt'), num_state = NUM_STATE_LIST, chrom = TEST_CHROMOSOME_LIST), # each file is produced by each time calling rule calcualte_log_llh_one_chrom
	output:
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic_by_chrom.txt'),
	params:
		out_dir = os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test')
	run:	
		aic_list = []
		bic_list = []
		gw_neg2_logLlh_list = []
		chrom_length_df , _ = count_gw_bins()
		print(chrom_length_df.head())
		columns = ['num_state', 'test_chrom', 'aic', 'bic', 'neg2_logLlh'] 
		result_df = pd.DataFrame(columns = columns)
		for num_state in NUM_STATE_LIST: # for each model, calculate 
			for chrom in TEST_CHROMOSOME_LIST:
				aic, bic, neg2_logLlh = calculate_statistics_for_model_evaluation(num_state, chrom, chrom_length_df)
				row = [num_state, 'chr{chrom}'.format(chrom = chrom), aic, bic, neg2_logLlh]
				result_df.loc[result_df.shape[0]] = row # append one more row
		result_df.to_csv(output[0], header = True, index = False, sep = '\t')
		# plot_df = pd.melt(result_df, id_vars = ['num_state', 'test_chrom'], value_vars = ['aic', 'bic', 'neg2_logLlh'])
		# plot_df.columns = ['num_state', 'test_chrom', 'aic_or_bic', 'values']
		# ax = sns.pointplot(data = plot_df, x = 'num_state', y = 'values', hue = 'aic_or_bic', plot_kws=dict(alpha:0.3))
		# ax.set_xticklabels(plot_df['num_state'], size = 5, rotation = 270) 
		# plt.tight_layout()
		# plt.savefig(os.path.join(params.out_dir, 'aic_bic_all_test_chrom.png'))

rule get_summary_optimal_num_state_by_chrom:
	input:
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic_by_chrom.txt') # from rule compare_aic_bic_models_by_chrom
	output: 
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'optimal_state_by_chrom.txt'),
	run:
		df = pd.read_csv(input[0], header = 0, index_col = None, sep = '\t')
		grouped_df = df.groupby('test_chrom')
		result_df = pd.DataFrame(columns = ['test_chrom', 'aic', 'bic', 'neg2_logLlh'])
		for chrom, chrom_df in grouped_df:
			chrom_df = chrom_df.set_index('num_state')
			chrom_df = chrom_df.drop(['test_chrom'], axis = 1) # only 3 columns left: aic bic neg2_logLlh
			report_row = chrom_df.idxmin(axis = 0)
			report_row = report_row.append(pd.Series({'test_chrom': chrom}))
			result_df.loc[result_df.shape[0]] = report_row
		result_df.to_csv(output[0], header = True, index = False, sep = '\t')

rule plot_compare_aic_bic_models_by_chrom:
	input: 
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic_by_chrom.txt') # from rule compare_aic_bic_models_by_chrom
	output:
		os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test', 'aic_bic_all_test_chrom.png'),
	params:
		out_dir = os.path.join(model_outdir, 'K562_hg19', 'all_mark_model', 'aic_bic_compare_num_states', 'train_then_test')
	run:	
		plot_df = pd.read_csv(input[0], header = 0, index_col = None, sep = '\t')
		plot_df = pd.melt(plot_df, id_vars = ['num_state', 'test_chrom'], value_vars = ['aic', 'bic', 'neg2_logLlh'])
		plot_df.columns = ['num_state', 'test_chrom', 'aic_or_bic', 'values']
		plot_nrow = (np.floor(np.sqrt(len(TEST_CHROMOSOME_LIST)))).astype(int)
		plot_ncol = int(np.ceil((len(TEST_CHROMOSOME_LIST) + 1) / plot_nrow))
		fig, axes = plt.subplots(ncols = plot_ncol, nrows = plot_nrow, figsize = (18,9))
		for plot_index, chrom in enumerate(TEST_CHROMOSOME_LIST):
			ax = (axes.flat)[plot_index]
			this_chrom_df  = (plot_df[plot_df['test_chrom'] == 'chr{chrom}'.format(chrom = chrom)]).reset_index()
			sns.pointplot(data = this_chrom_df, x = 'num_state', y = 'values', ax = ax, hue = 'aic_or_bic')
			ax.set(ylim = (100000, 1550000))
			xlabels = ax.get_xticklabels() 
			for i, l in enumerate(xlabels):
				if i%2 == 0:
					xlabels[i] = ''
			ax.set_xticklabels(xlabels, rotation = 270, size = 5)
		fig.tight_layout()
		fig.savefig(output[0])
		# grid = sns.FacetGrid(plot_df, col = 'test_chrom', col_wrap = plot_nrow, hue = 'aic_or_bic', size = 4, aspect = 1.2)
		# grid.map(sns.pointplot, 'num_state', 'values')
		# grid.savefig(output[0])