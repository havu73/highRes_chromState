import pandas as pd
import numpy as np
import sys
import os
import helper
import glob 
def calculate_entropy_threshold_by_state(entropy_folder):
	entropy_fn_list = glob.glob(entropy_folder +  '/entropy_chr*.txt.gz')
	entropy_df_list = list(map(lambda x: pd.read_csv(x, header = None, index_col = None, sep = '\t'), entropy_fn_list[:1]))
	entropy_df_gw = pd.concat(entropy_df_list, ignore_index = True)
	entropy_df_gw.columns = ['entropy', 'state']
	grouped_df = entropy_df_gw.groupby('state')
	quantile_entropy_df = pd.DataFrame(columns = ['state', 'min_quantile', 'half_quantile', 'max_quantile'])
	for state, state_df in grouped_df:
		this_state_entropy_quantile = np.quantile(state_df['entropy'], [0, 0.5, 1])
		row = [state] + list(this_state_entropy_quantile)
		quantile_entropy_df.loc[quantile_entropy_df.shape[0]] = row
	return quantile_entropy_df

def get_state_based_on_entropy_one_row(row):# function applied to each row of the entropy_df object in get_segmentation_stratified_by_entropy. Each row has: # chrom, start, end, entropy, state, min_quantile, half_quantile, max_quantile
	result = int(row['state'][1:]) * 2 - 1 # if entropy is LESS THAN AND EQUAL to the median of this state, then the state label will be doubled, so that in the end we will get states that are numbereed according to how ChromHMM likes it. This is a strict requirement from ChromHMM. The indexing [1:] is to avoid the first character ('E') 
	if row['entropy'] > row['half_quantile']:
		result = int(row['state'][1:]) * 2
	result = 'E' + str(result)
	return result 

def ood_or_even_state(row):
	state_number = int(row['state_based_on_entropy'][1:])
	if (state_number % 2) == 0: # even state --> entropy is actually higher than the median
		return 'higher'
	else: # odd state --> entropy is lower than OR equal to the median
		return 'leq'

def get_segmentation_stratified_by_entropy(entropy_folder, output_folder, quantile_entropy_df):
	CHROMOSOME_LIST_ORDERED_BY_STRING = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '3', '4', '5', '6', '7', '8', '9', 'X'] # this oreder is how they ordered the chromosomes in the bed files that we will pass into bedtools map. 
	entropy_fn_list = list(map(lambda x: os.path.join(entropy_folder, 'entropy_chr{chrom}.txt.gz'.format(chrom = x)), CHROMOSOME_LIST_ORDERED_BY_STRING))
	segment_df = pd.DataFrame(columns = ['chrom', 'start', 'end', 'state_based_on_entropy', 'entropy'])
	for chrom_index, entropy_fn in enumerate(entropy_fn_list):
		entropy_df = pd.read_csv(entropy_fn, header = None, index_col = None, sep = '\t') # for only this chrom
		print(entropy_df.head())
		chrom = CHROMOSOME_LIST_ORDERED_BY_STRING[chrom_index] # this works beacuse we derived entropy_fn_list from CHROMOSOME_LIST_ORDERED_BY_STRING
		entropy_df.columns = ['entropy', 'state']
		entropy_df['chrom'] = 'chr' + chrom 
		entropy_df['start'] = helper.NUM_BP_PER_BIN * np.arange(entropy_df.shape[0])
		entropy_df['end'] = entropy_df['start'] + helper.NUM_BP_PER_BIN
		entropy_df = entropy_df.merge(quantile_entropy_df, how = 'left', left_on = 'state', right_on = 'state') # chrom, start, end, entropy, state, min_quantile, half_quantile, max_quantile
		entropy_df['state_based_on_entropy'] = entropy_df.apply(get_state_based_on_entropy_one_row, axis = 1)
		entropy_df = entropy_df[['chrom', 'start', 'end', 'state_based_on_entropy', 'entropy', 'state']] # state based on entropy is the transformed state, state by itself is the original state names
		segment_df = segment_df.append(entropy_df)
		print ("Done calculating for chrom: {chrom}".format(chrom = chrom))
	output_fn = os.path.join(output_folder, 'genome_segments_sorted.bed.gz')
	segment_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	# ADDED CODE FOR A NEW FUNCTIONALITY: stratify the genome into segments with high entropy and low entropy
	segment_df['compare_to_median_entropy'] = entropy_df.apply(ood_or_even_state, axis = 1)
	grouped_segment_df = segment_df.groupby('compare_to_median_entropy')
	for compare_name, group_df in grouped_segment_df:
		output_fn = os.path.join(output_folder, compare_name, 'genome_segments_sorted.bed.gz')
		helper.create_folder_for_file(output_fn)
		group_df = group_df[['chrom', 'start', 'end', 'state', 'entropy']]
		group_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	#### END OF ADDED CODE ####
	return 

def fix_existing_double_state_segmentation(output_folder):
	# this function is here because I want to fix the addition to the code without recomputing everything
	gw_segment_fn = os.path.join(output_folder, 'genome_segments_sorted.bed.gz')
	gw_df = pd.read_csv(gw_segment_fn, header = None, index_col = None, sep = '\t')
	gw_df.columns = ['chrom', 'start', 'end', 'state_based_on_entropy', 'entropy']
	gw_df['state'] = gw_df['state_based_on_entropy'].apply(lambda x: 'E' + (np.floor(int(x[1:]) / 2).astype(int).astype(str)))
	gw_df['compare_to_median_entropy'] = gw_df.apply(ood_or_even_state, axis = 1)
	for compare_name, group_df in grouped_segment_df:
		output_fn = os.path.join(output_folder, compare_name, 'genome_segments_sorted.bed.gz')
		helper.create_folder_for_file(output_fn)
		group_df = group_df[['chrom', 'start', 'end', 'state', 'entropy']]
		group_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	return 

def main():
	if len(sys.argv) != 3:
		usage()
	entropy_folder = sys.argv[1]
	helper.check_dir_exist(entropy_folder)
	output_folder = sys.argv[2]
	helper.make_dir(output_folder)
	print ("Done getting command line argument")
	quantile_entropy_df = calculate_entropy_threshold_by_state(entropy_folder) # state, min_quantile, half_quantile (median), max_quantile (max)
	get_segmentation_stratified_by_entropy(entropy_folder, output_folder, quantile_entropy_df)
	# fix_existing_double_state_segmentation(output_folder)
	print('Done')

def usage():
	print ('python change_segmentation_based_on_entropy.py')
	print ('entropy_folder: where the files outputted from calculate_entropy_posterior_chromHMM.py are stored')
	print ('output_folder: where we will store the segmentation values')
	exit(1)

main()