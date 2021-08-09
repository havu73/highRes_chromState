import pandas as pd
import numpy as np
import torch 
import pybedtools as bed # cite: https://academic.oup.com/bioinformatics/article/27/24/3423/304825
import os 
import sys
import helper
import itertools

def calculate_join_emission_multiple_marks(row, binary_tuple, chrom_mark_list):
	# this will process each row in the emission matrix (each state)
	# binary tuple will be a tuple of length #num_mark, each element in the tuple is 0/1 --> presence/absence call of chromatin mark. The order of chromatin marks will be given in chrom_mark_list. Ex: binary_tuple = (0,0,1), chrom_mark_list = [m1, m2, m3] --> m3 is present and others are absent. This function will return the probability of observing binary_tuple given each of the state. 
	# function tested on 08/03/2021
	base = row[chrom_mark_list]
	exponent = pd.Series(binary_tuple, index = chrom_mark_list)
	return np.prod(base**exponent * (1-base)**(1-exponent))

def read_emission_matrix_into_categorical_prob(emission_fn, chrom_mark_list): # exactly the same function as in train_model_using_pyro.py. May need to rearrange my code later but keep this for now
	emission_df = pd.read_csv(emission_fn, header = 0,index_col = 0, sep = '\t') # row indices are states 
	all_possible_obs_marks = list(itertools.product(range(2), repeat = len(chrom_mark_list))) # list of tuples, each of length # num_marks --> all possible observations of marks 
	all_possible_obs_marks_str = list(map(lambda x: ''.join(list(map(str, x))), all_possible_obs_marks)) # convert (0,0,0) --> '000'
	for obs_pattern in all_possible_obs_marks:
		obs_string = ''.join(list(map(str, obs_pattern)))
		emission_df[obs_string] = emission_df.apply(lambda x: calculate_join_emission_multiple_marks(x, obs_pattern, chrom_mark_list), axis = 1) # apply function to each row
	result_df = emission_df[all_possible_obs_marks_str].copy() # columns are all the possible chromatin mark sequences for the chrom_mark_list. Right now, we assume that the assays being profiled are a subset of the 12 marks in the 25-state roadmap model, We can care about the case where the profiled marks for sample of interest are not among the 12 marks later.
	result_df.columns = list(map(lambda x: int(x, 2), result_df.columns))
	result_df = result_df[np.arange(len(all_possible_obs_marks))] # rearrange so that if # marks = 3 --> columns will be 0 --> 7, correpsonding to the 8 possible combination of observed marks 000 --> 111
	result_tt = torch.tensor(result_df.values) # tensor with rows: states, columns: possible combinations of chromatin marks 
	return result_tt

def get_parameters(parameter_folder, emission_fn, obs_chromMark_signal_fn):
	# 1. emission probabilities
	obs_mark_df = pd.read_csv(obs_chromMark_signal_fn, header = 0, index_col = None, sep = '\t', nrows = 1) # only need to read the column names so we know how to process the emission probabilities 
	chrom_mark_list = obs_mark_df.columns
	emission_tt = read_emission_matrix_into_categorical_prob(emission_fn, chrom_mark_list) # rows: states, columns: if len(chrom_mark_list) == 3, then columns will be 0 --> 7, meaning the different combinations of 0/1 signals of different marks
	# 2. Pi: mixture probabilities of different reference epigenomes
	pi_fn = os.path.join(parameter_folder, 'posterior_pi.txt')
	pi_df = pd.read_csv(pi_fn, header = 0, index_col = 0, sep = '\t', squeeze = True)
	ref_epig_name_list = pi_df.index
	pi_tt = torch.tensor(pi_df.values)
	# 3. beta matrix: rows: states in reference epigenome, columns: states in the sample of interest
	beta_fn = os.path.join(parameter_folder, 'beta_state_transition.txt')
	beta_df = pd.read_csv(beta_fn, header = None, index_col = None, sep = '\t')
	assert beta_df.shape[0] == beta_df.shape[1], 'Number of columns and rows in beta matrix are not similar'
	beta_tt = torch.tensor(beta_df.values)
	return emission_tt, pi_tt, ref_epig_name_list, beta_tt

def read_chrom_mark_observed_signals(obs_chromMark_signal_fn):
	# obs_chromMark_signa_fn contains binarized signals of different chromatin marks in sample of interest. Columns: marks, rows: different 200bp windows across the chromosome. There are no columns associated with chrom, start, end in bed format. We will transform this data such that each positions' signals will be presented by one number (ex: if there are 3 marks, then 000 --> 0 and 111 --> 7, the numbers (0--> 7) will be matching with the emission probabilities processed in read_emission_matrix_into_categorical_prob)
	chrom_signal_df = pd.read_csv(obs_chromMark_signal_fn, header = 0, index_col = None, sep = '\t')
	chrom_mark_list = chrom_signal_df.columns
	chrom_signal_df = chrom_signal_df.apply(lambda x: x.astype(int).astype(str), axis = 0) # convert the data from 0.0, 1.0 to 0 and 1 integers
	chrom_signal_df['combined_obs_int'] = chrom_signal_df.apply(lambda x: int(''.join(x), 2), axis = 1) # apply function to each row
	obs_signal_tt = torch.tensor(chrom_signal_df['combined_obs_int'].values) # 1D array each element is the observed data at each postion. If we have 3 marks, the the observed values can be 0-7.
	return obs_signal_tt, chrom_mark_list 

def prepare_ref_epig_state_df(all_refEpig_segment_folder, ref_epig_name_list, chrom_basic_bed):
	all_epig_df = bed.BedTool(chrom_basic_bed)
	names = ['chrom', 'start', 'end', 'dummy']
	for ref_epig in ref_epig_name_list:
		ref_epig_fn = os.path.join(all_refEpig_segment_folder, ref_epig, '{}_25_segments_sorted.bed.gz'.format(ref_epig))
		ref_epig_df = bed.BedTool(ref_epig_fn)
		names += [ref_epig]
		all_epig_df = all_epig_df.map(ref_epig_df, c=4, o='collapse')
	all_epig_df.delete_temporary_history(ask = False) # based on tutorial http://daler.github.io/pybedtools/history.html#deleting-temp-files-specific-to-a-single-bedtool
	all_epig_df = all_epig_df.to_dataframe() # convert from bedtool to dataframe. Now columns: chrom, start, end, dummy_column, <ref_epig> --> each column show the chromatin state map for a ref_epig
	# the folllowing linds of code is to fix some stupid complication with pybedtools because otherwise, the column names would be the first bin in the chromsoome's data
	wrong_first_row = all_epig_df.columns
	correct_first_row = list(wrong_first_row[:4]) + list(map(lambda x: x.split('.')[0], wrong_first_row[4:]))
	correct_first_row_df = pd.DataFrame(columns = np.arange(all_epig_df.shape[1]))
	correct_first_row_df.loc[0] = correct_first_row
	all_epig_df.columns = np.arange(all_epig_df.shape[1]) # change the column names to just numbers, next we will drop the first 4 columns
	all_epig_df = pd.concat([correct_first_row_df, all_epig_df]).reset_index(drop = True)
	####### End of that fixing stupid mistake #####
	print(all_epig_df.head())
	print(all_epig_df.columns)
	all_epig_df.drop(labels = np.arange(4), axis = 'columns', inplace = True)
	all_epig_df = all_epig_df.applymap(lambda x: int(x[1:]) - 1) # E1 --> 0, E25 --> 24. Convert chromHMM state names to 0-based indices
	all_epig_df.columns = np.arange(all_epig_df.shape[1]) # rename the columns so that now all the ct_index are 0-based, and matching with alpha --> we can generate the ct picked at each position later.
	return all_epig_df #r ows are different genomic positions, columns are the ref_epig_name_list. values are 0-based state at each position for each ref_epig. The columns are ordered as in ref_epig_name_list, so the resulting alpha (prior for ref_epig similarity) should have the same order. 

def calculate_4D_tensor_all_variable_combo(emission_tt, pi_tt, beta_tt):
	# given the parameters of the model, this function will construct 4D tensor [ref_epig][state in ref_epig][state in sample of interest][observed combo of marks' signals] --> values: probability of Z = ref_epig, R = state in ref_epig, S = state in sample of interest, M = observed marks --> joint probabilities of hidden and observed variables. 
	# emission_tt: rows: state, columns: index of the binarized combination of signals of marks.
	# pi_tt: 1D, mixture probability of different reference epigenome
	# beta_tt: 2D, rows: states in ref_epig, columns: states in sample of interest
	num_ct = len(pi_tt)
	num_state = beta_tt.shape[0]
	num_mark_combo = emission_tt.shape[1]
	d4_tt = torch.zeros(size = (num_ct, num_state, num_state, num_mark_combo)) # initialize the output tensor
	d3_tt_state_to_marks = torch.zeros(size = (num_state, num_state, num_mark_combo)) # initialize the 3D matrix [state in ref_epig][state  in sample of interest][observed pattern of marks]
	for obs_i in range(emission_tt.shape[1]):
		d3_tt_state_to_marks[:,:,obs_i] = beta_tt * emission_tt[:,obs_i] # multiply each row of beta_tt with each column of emission_tt. In each row of beta_tt, each element corresponds to a state in sample of interest. In each column of emission_tt, each column corresponds to a state in sample of interest
	for ref_epig_id in range(len(pi_tt)):
		d4_tt[ref_epig_id,:,:,:] = pi_tt[ref_epig_id] * d3_tt_state_to_marks # multiply each d3_tt_state_to_marks (a tensor of: [state in ref_epig] [state in sample of interest] [obs_marks]) with each reference epig mixture probability. --> adding a new dimension to the tensor
	return d4_tt

def calculate_P_state_and_mark(row, d4_tt, state_index, num_ct):
	# this function gets apply to each row of all_epig_df with columns: 0-based indices of ref_epig and obs. It calculates P(S = state_index, M = obs) = sum over all reference epigenome {p(Z=ref_epig_index) * P (S=state_index | R_Z) * P(M=obs | S = state_index)}
	obs = row['obs'] # 4th index of d4_tt
	result = 0
	for refE_i in range(num_ct): #summing over all reference epigenome
		R_Z = row[refE_i]
		result += d4_tt[refE_i, R_Z, state_index, obs]
	return result 

def calculate_state_posterior_probabilities(emission_tt, pi_tt, beta_tt, obs_signal_tt, all_epig_df, output_fn):
	d4_tt = calculate_4D_tensor_all_variable_combo(emission_tt, pi_tt, beta_tt) # precalculated joint probablities of random variables [ref_epig][state in ref_epig][state in sample of interest][observed combo of marks' signals]
	num_ct = len(pi_tt)
	num_state = beta_tt.shape[0]
	num_mark_combo = emission_tt.shape[1]
	all_epig_df['obs'] = obs_signal_tt # add one column showing the observed data at each genomic position (row). Now, this dataframe as columns: 0-based indices of reference epigenome and 'obs'
	for state_index in range(num_state):
		all_epig_df['E{}'.format(state_index+1)] = all_epig_df.apply(lambda x: calculate_P_state_and_mark(x, d4_tt, state_index, num_ct), axis = 1) # apply function to each row of the dataframe
	result_colnames = list(map(lambda x: 'E{}'.format(x+1), range(num_state)))
	result_df = all_epig_df[result_colnames].copy()
	result_df = result_df.applymap(lambda x: x.item()) # convert each cell from tensor(0.1000) to 0.1
	print(result_df.head())
	result_df['max_state'] = result_df.idxmax(axis = 1) # apply function idxmax to each row
	result_df.to_csv(output_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	return 


def main():
	if len(sys.argv) != 7:
		usage()
	parameter_folder = sys.argv[1]
	helper.check_dir_exist(parameter_folder)
	emission_fn = sys.argv[2]
	helper.check_file_exist(emission_fn)
	all_refEpig_segment_folder = sys.argv[3]
	helper.check_dir_exist(all_refEpig_segment_folder)
	obs_chromMark_signal_fn = sys.argv[4]
	helper.check_file_exist(obs_chromMark_signal_fn)
	output_fn = sys.argv[5]
	helper.create_folder_for_file(output_fn)
	chrom_basic_bed = sys.argv[6]
	helper.check_file_exist(chrom_basic_bed)
	logger = helper.argument_log(command = sys.argv, output_folder= os.path.dirname(output_fn)) # to save the command line arguments that led to this 
	logger.write_log()
	print ('Done getting command line arguments')
	# 1. Process all the model parameters
	emission_tt, pi_tt, ref_epig_name_list, beta_tt = get_parameters(parameter_folder, emission_fn, obs_chromMark_signal_fn)
	print('Done getting model parameters')
	# 2. Prepare the input data of ref_epig's chromatin state maps and observed chromatin mark signals
	obs_signal_tt, chrom_mark_list = read_chrom_mark_observed_signals(obs_chromMark_signal_fn) # obs_signal_tt: 1D tensor, each element corresponds to a genomic position (200bp), values are the 0-based numbers representing different combinations of presence/ absence signals of marks. Ex: 3 marks --> values 0...7 representing 8 combinations.
	all_epig_df = prepare_ref_epig_state_df(all_refEpig_segment_folder, ref_epig_name_list, chrom_basic_bed) # all_epig_df: rows: genomic positions (200bp), columns: reference epigenome ordered similarly to those in pi, column names are 0-based indices. values: 0-based state assignment at each genomic position at each ref_epig
	print("Done reading input observed data")
	# 3. Calculate the posterior probabilities and write the output
	calculate_state_posterior_probabilities(emission_tt, pi_tt, beta_tt, obs_signal_tt, all_epig_df, output_fn)
	print('Done')

def usage():
	print ('python calculate_gw_state_posterior_prob.py')
	print ('parameter_folder: output from training the model, two files should be here: beta_state_transition.txt and posterior_pi.txt. First column of posterior_pi.txt show the ref_epig_id, and these will be used to extract the chromatin ref_epig\'s chromatin state segmentations to predict the chromatin state assignments in our sample of interests')
	print ('emission_fn: the emission probabilities matrix of state(rows) and marks (columns) --> We use the emission parameters from ROADMAP')
	print ('all_refEpig_segment_foler: where we save the sorted chromatin state segmentations of reference epigenomes. We will not require all the reference epigenomes\' data be used to calculate the posterior probabilities of state assignments for sample of interest')
	print ('obs_chromMark_signal_fn: file where we store the observed chromatin mark signals for the sample of interest. This file should correspond to one chromosome. First line corresponds to marks\' names. Following lines correspond to 200bp across the chromosome. Right now our model can only support the case where the available chromatin marks\' signals are among those used as input for the 25-state chromatin state model.' )
	print ('output_fn: where we store the output, rows: genomic regions (200bp each), columns: states --> values: posterior probabilities that each positions are in each state, based on the trained model parameters outputted by pyro')
	print ('chrom_basic_bed: a bed file with 3 columns: chromosome, start, end where each line corresponds to one segment in the chromosome. This file is useful in getting the chromatin state maps for the reference epigenomes. ') # there must be another way of extracting this data, but I keep this for now. 
	exit(1)
main()