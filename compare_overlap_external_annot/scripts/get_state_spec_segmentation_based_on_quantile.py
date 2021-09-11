import pandas as pd 
import numpy as np 
import sys
import os
import glob
import helper
def process_quantile_one_state(quantile_df, state_0Index):# quantile_df has columns E1 --> E<num_state> --> 1-based index
	state_colname = 'E{}'.format(state_0Index+1)
	state_quantile_df = quantile_df[[state_colname, 'quantile_low_bound']]
	grouped_df = state_quantile_df.groupby(state_colname).agg({'quantile_low_bound': ['min', 'max']}).reset_index() # min is the lower bound of the lowest quantile, the max is the lower bound of the highest quantil that share this posterior probability ranges
	grouped_df.columns = [state_colname + '_lowerbound', 'min_quantile', 'max_quantile']
	if grouped_df.shape[0] > 1: # more than 1 row
		grouped_df[state_colname + '_upperbound'] = list(grouped_df[state_colname + '_lowerbound'])[1:] + [1]
	else:
		grouped_df[state_colname + '_upperbound'] = 1
	return grouped_df

def transform_one_state_segmentation_one_row(row, this_state_quantile_df, state_colname):
	result = state_colname + '_{}'.format(this_state_quantile_df.shape[0]) # by default, setting to the last bin
	for index, quantile_row in this_state_quantile_df.iterrows():
		if row[state_colname] < quantile_row[state_colname + '_upperbound']:
			return state_colname + '_{}'.format(index)
	return result

def process_one_state_slower_version(posterior_dir, quantile_fn, output_folder, state_0Index, num_state):
	quantile_df = pd.read_csv(quantile_fn, header = 0, index_col = None, sep = '\t')
	this_state_quantile_df = process_quantile_one_state(quantile_df, state_0Index) # columns example: E1_lowerbound, E1_upperbound, min_quantile, max_quantile
	print(this_state_quantile_df)
	result_df = pd.DataFrame(columns = ['chrom', 'start', 'end', 'state_spec_state'])
	state_colname = 'E{}'.format(state_0Index+1)
	for chrom in helper.CHROMOSOME_LIST[:1]:
		post_fn = os.path.join(posterior_dir, 'genome_{num_state}_chr{chrom}_posterior.txt.gz'.format(num_state = num_state, chrom = chrom))
		post_df = pd.read_csv(post_fn, header = 0, index_col = None, skiprows = 1, sep = '\t')
		post_df = post_df[[state_colname]] # only get infor for this one state
		if this_state_quantile_df.shape[0] == 1: # then we will have to divide the genome into >0 and =0 posterior for this state
			post_df['state_spec_state'] = post_df.apply(lambda x: state_colname + '_0' if x[state_colname] == 0 else state_colname + '_1', axis = 1)
		else: # then we will find the segmentation based on the range of posterior probability that this state belongs to
			post_df['state_spec_state'] = post_df.apply(lambda x: transform_one_state_segmentation_one_row(x, this_state_quantile_df, state_colname), axis = 1)
		post_df['chrom'] = 'chr{chrom}'.format(chrom = chrom)
		post_df['start'] = post_df.index * helper.NUM_BP_PER_BIN
		post_df['end'] = post_df['start'] + helper.NUM_BP_PER_BIN
		post_df = post_df[['chrom', 'start', 'end', 'state_spec_state']] # rearrange columns
		result_df = result_df.append(post_df)
	output_fn = os.path.join(output_folder, 'genome_{num_state}_{state_colname}_posterior_segment.txt.gz'.format(num_state = num_state, state_colname = state_colname))
	result_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	print ('Done processing for state {}'.format(state_colname))
	return

def process_one_state(posterior_dir, quantile_fn, output_folder, state_0Index, num_state):
	quantile_df = pd.read_csv(quantile_fn, header = 0, index_col = None, sep = '\t')
	this_state_quantile_df = process_quantile_one_state(quantile_df, state_0Index) # columns example: E1_lowerbound, E1_upperbound, min_quantile, max_quantile
	print(this_state_quantile_df)
	result_df = pd.DataFrame(columns = ['chrom', 'start', 'end', 'state_spec_state'])
	state_colname = 'E{}'.format(state_0Index+1)
	new_state_colname = 'S{}'.format(state_0Index+1)
	for chrom in helper.CHROMOSOME_LIST:
		post_fn = os.path.join(posterior_dir, 'genome_{num_state}_chr{chrom}_posterior.txt.gz'.format(num_state = num_state, chrom = chrom))
		post_df = pd.read_csv(post_fn, header = 0, index_col = None, skiprows = 1, sep = '\t')
		post_df = post_df[[state_colname]] # only get infor for this one state
		post_df['state_spec_state'] = state_colname
		for index, row in this_state_quantile_df.iterrows():
			lowerbound = row[state_colname + '_lowerbound']
			upperbound = row[state_colname + '_upperbound']
			post_df.loc[(post_df[state_colname]>=lowerbound) & (post_df[state_colname] <= upperbound), 'state_spec_state'] = new_state_colname + '-' + str(index)
		post_df['chrom'] = 'chr{chrom}'.format(chrom = chrom)
		post_df['start'] = post_df.index * helper.NUM_BP_PER_BIN
		post_df['end'] = post_df['start'] + helper.NUM_BP_PER_BIN
		post_df = post_df[['chrom', 'start', 'end', 'state_spec_state']] # rearrange columns
		result_df = result_df.append(post_df)
	output_fn = os.path.join(output_folder, 'genome_{state_colname}_posterior_segment.txt.gz'.format(num_state = num_state, state_colname = state_colname))
	result_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	print ('Done processing for state {}'.format(state_colname))
	return


def main():
	if len(sys.argv) != 5:
		usage()
	posterior_dir = sys.argv[1]
	helper.check_dir_exist(posterior_dir)
	quantile_fn = sys.argv[2]
	helper.check_file_exist(quantile_fn)
	output_folder = sys.argv[3]
	helper.make_dir(output_folder)
	num_state = helper.get_command_line_integer(sys.argv[4])
	print ('Done getting command line argument')
	for state_0Index in range(num_state):
		process_one_state(posterior_dir, quantile_fn, output_folder, state_0Index, num_state)
	print ("Done!")

def usage():
	print ('python get_state_spec_segmentation_based_on_quantile.py')
	print ('posterior_dir: directory of posterior probability of state_assignment in one chrom')
	print ('quantile_fn: where we see the quantile of posterior probability for each state')
	print ('output_folder: where each file corresponds to a state segmentation')
	print ('num_state')
	exit(1)

main()