import pandas as pd 
import numpy as np 
import sys
import os
import glob
import helper

def get_quantiles_rep_scores(chrom_posterior_fn, num_bins, output_fn, num_chromHMM_state):
	# first assemble the score data for the whole genome. No need to put it in order of the whole genome
	posterior_score_df = pd.read_csv(chrom_posterior_fn, sep = '\t', skiprows = 1, header = 0)
	print ("Done compiling whole-genome data")
	# second, calculate the quantile for each state
	quantile_lower_bound_list = np.arange(0, 1, 1.0 / num_bins)
	quantile_upper_bound_list = quantile_lower_bound_list + 1.0 / num_bins
	quantile_df = posterior_score_df.apply(lambda x: np.quantile(x, quantile_lower_bound_list), axis = 0) # apply function to each column --> each function call returns a pandas Series of size num_bins --> rows: quantiles, columns: states, values: the quantiles values in each state. Each row will correspond to one quantile window. If num_bins = 10 --> [0, .. 0.9], which is good for calculation of ROC later
	quantile_with_ends = np.append(quantile_lower_bound_list,  1.0) # this is 1 element longer than the number of rows of the quantile_df
	gw_quantile = np.quantile(posterior_score_df, quantile_with_ends) # calculate the quantiles the genome, regardless of states
	quantile_df['gw_low_bound'] = gw_quantile[:-1] #values of quantiles 0, 0.1, 0.2, ..., 0.9
	quantile_df['gw_high_bound'] = gw_quantile[1:] # values of quantiles 0.1, 0.2, ...., 1
	quantile_df['quantile_low_bound'] = quantile_lower_bound_list
	quantile_df.to_csv(output_fn, header = True, index = False, sep = '\t')
	print ("Done calculating quantiles")
	return

def main():
	if len(sys.argv) != 5:
		usage()
	chrom_posterior_fn = sys.argv[1]
	helper.check_file_exist(chrom_posterior_fn)
	num_bins = helper.get_command_line_integer(sys.argv[2])
	output_fn = sys.argv[3]
	helper.create_folder_for_file(output_fn)
	num_chromHMM_state = helper.get_command_line_integer(sys.argv[4])
	print('Done with command line argument')
	get_quantiles_rep_scores(chrom_posterior_fn, num_bins, output_fn, num_chromHMM_state)

def usage():
	print('python get_quantiles_rep_scores.py')
	print('chrom_posterior_fn: File where we get the posterior probability of state assignment in one chromosome')
	print('num_bins: number of bins that we will divide 100\% into so that we can draw the quantile functions to.')
	print('output_fn')
	print('num_chromHMM_state')
	exit(1)
main()