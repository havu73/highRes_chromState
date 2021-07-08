# the forward algorithm used to calcualte the log likelihood for observed chromatin mark signals in one chromosome. References: https://web.stanford.edu/~jurafsky/slp3/A.pdf
# the scaling that involves alpha_hat and cn are based on https://web.stanford.edu/~jurafsky/slp3/A.pdf
import pandas as pd 
import numpy as np 
import os
import sys
import helper
import glob

def process_model_fn(model_fn):
	inF = open(model_fn, 'r')
	first_line = inF.readline().strip().split()
	num_state = int(first_line[0])
	num_mark = int(first_line[1])
	estimated_llh = float(first_line[3])
	num_iterations = int(first_line[4])
	init_probs = pd.read_csv(model_fn, header = None, index_col = None, skiprows = 1, skipfooter = num_state * num_mark * 2 + num_state**2, sep = '\t') # 3 columns 1: probinit, 2: state index 1 based, 3 init prob. 
	init_probs = pd.Series(init_probs[2]) # only retain the column of states' initial probabilities
	transition_probs = pd.read_csv(model_fn, header = None, index_col = None, skiprows = num_state + 1, skipfooter = num_state * num_mark * 2, sep = '\t')
	transition_probs = pd.pivot_table(transition_probs, values = 3, index = 1, columns = 2) # row sum will be 1.0 --> rows to columns showing the transition probabilities from states in rows to states in columns
	transition_probs.index = range(num_state)
	transition_probs.columns = range(num_state)
	emission_probs = pd.read_csv(model_fn, header = None, index_col = None, skiprows = num_state + 1 + num_state**2, sep = '\t') # 0< emissionprobs, 1: state_index 1-based, 2: mark index 0-based, 3: mark name, 4: 0 or 1 presence or absence, 5: emission probabilities
	emission_probs.columns = ['to_drop', 'state_index_1Based', 'mark_index_0Based', 'mark_name', 'zero_or_one', 'emission_probs']
	emission_probs.drop(['to_drop', 'mark_index_0Based'], axis = 1, inplace = True)
	emission_probs['state_index_1Based'] = (emission_probs['state_index_1Based']).apply(lambda x: x - 1) 
	emission_probs = emission_probs.rename(columns = {'state_index_1Based' : 'state_index_0Based'})
	return num_state, num_mark, init_probs, transition_probs, emission_probs

def calculate_joint_emission_for_uniqRows(unique_signal_df, emission_probs, num_state): #tested
	# uniq_signal_df has columns that correspond to the marks
	result_columns = ['signal_key'] + list(range(num_state)) # the 0 and 1 will correspond to absernce/presence call of marks at that positoin. The marks are exactly ordered as the columns in unique_signal_df
	result_df = pd.DataFrame(0, columns = result_columns, index = range(unique_signal_df.shape[0])) # signal_key, 0 --> n-1  where n is the number of states
	result_df['signal_key'] = unique_signal_df.apply(lambda x: ''.join(x.astype(str)), axis = 1) # apply function to each row
	for row_index, row in unique_signal_df.iterrows():
		this_row_emission = np.ones(num_state) # number of states as the length
		for mark in unique_signal_df.columns:
			this_mark_emission = emission_probs[(emission_probs['mark_name'] == mark) & (emission_probs['zero_or_one'] == row[mark])] # a dataframe with #rows = #states. columns: state_index_0Based, mark_name, zero_or_one, emission_probs
			this_mark_emission.index = this_mark_emission['state_index_0Based'] # set the index, so that we can easily multiply with this_row_emission in the future
			this_row_emission = (this_row_emission * np.array(this_mark_emission['emission_probs']))
		# this_row_emission now is a vector of size #state, each showing the iteratively normalzied joint probability of emission of marks at each state. Now, we will record the joint emission probabilities given each state for this sequence of obseravtion
		result_df.loc[row_index, range(num_state)] = this_row_emission
	return result_df # signal_key,  0 --> n-1 where n: #state 


def calculate_log_likelihood_one_region(num_state, num_mark, transition_probs, emission_probs, init_probs, region_fn, start_index_in_region, alpha_hat_vec, current_sum_logCn): # tested
	# start_index_in_region (0-based): except for when the region is the first region of the chromosome, the start_index_in_region is always 0. Otherwise, 1.
	signal_df = pd.read_csv(region_fn, header = 0, skiprows = 1, index_col = None, sep = '\t') # skiprows = 1 so that we skip the first line with the format "genome chr2.207", all columns correspond to different chromatin marks available in the system
	mark_colnames = signal_df.columns
	signal_df = signal_df.loc[start_index_in_region:,:] # filter the rows that we do not want to look at the signals for
	unique_signal_df = signal_df.drop_duplicates().reset_index().drop('index', axis = 1) # we only need to calculate emission probabilities for regions with the same types of signals
	uniqSig_joint_emission_df = calculate_joint_emission_for_uniqRows(unique_signal_df, emission_probs, num_state) ## signal_key: strings of 0 and 1 corresponds to different patterns for presence/absence calls for each mark (marks ordered as in signal_df) ,  0 --> n-1 where n: #state, data showing the probabilty of observed presence/absence of marks given the state
	signal_df['signal_key'] = signal_df.apply(lambda x: ''.join(x.astype(str)), axis = 1) # apply function to each row
	signal_df = signal_df.merge(uniqSig_joint_emission_df, left_on = 'signal_key',  right_on = 'signal_key').drop(mark_colnames, axis = 1) # now this df will have columns signal_key, 0 --> n-1
	# now, we will use the forward algorithm to calculate the log llh of observed patterns in this region, given the existing log likelihood from previous region
	for row_index, row in signal_df.iterrows(): # at each new genomic position, we will update the current likelihood of observed data.
		emission_this_row = row[1:] # not looking at the 'signal_key' entry, only the states
		transition_times_emission = transition_probs.multiply(emission_this_row, axis = 1) # transition: from states in rows to states in columns. transition*emission will mulitply each row tot he emisison_this_row --> probability of jumping from row state to column state and observing the signals in the column state
		alpha_hat_vec = transition_times_emission.multiply(alpha_hat_vec, axis = 0) # multily element-wise, column-wise: each column of transition_times_emission mulitiply element-wise to alpha_hat_vec
		alpha_hat_vec = alpha_hat_vec.sum(axis = 0) # column sum, because the columns correspond to the destination state of this genomic position
		this_cn = np.sum(alpha_hat_vec)
		alpha_hat_vec = alpha_hat_vec / this_cn
		current_sum_logCn += np.log(this_cn)
	return alpha_hat_vec, current_sum_logCn

def calculate_log_likelihood_multiple_files(num_state, num_mark, init_probs, transition_probs, emission_probs, signal_dir, output_fn, chrom):
	region_fn_list = glob.glob(signal_dir + '/genome_chr{chrom}.*.txt.gz'.format(chrom = chrom))
	num_regions_this_chrom = len(region_fn_list)
	ordered_region_fn_list = list(map(lambda x: os.path.join(signal_dir, 'genome_chr{chrom}.{region_index}_binary.txt.gz'.format(chrom = chrom, region_index = x)), list(range(num_regions_this_chrom))))
	print(num_regions_this_chrom)
	print(len(ordered_region_fn_list))
	first_region_fn = ordered_region_fn_list[0]
	first_region_df = pd.read_csv(first_region_fn, header = 0, skiprows = 1, index_col = None, sep = '\t')
	first_bin_df = first_region_df.loc[0:0,:] # only get data of the first bin, columns: mark, row: just the first bin, data: 0/1
	first_bin_df = calculate_joint_emission_for_uniqRows(first_bin_df, emission_probs, num_state) # columns: signal_key and different states --> emission probabilities of the first bin
	first_alpha_vec = init_probs * first_bin_df.iloc[0,1:] # init prob times the emission probabilities of the first bin, each entry correspond to a state
	first_cn = np.sum(first_alpha_vec)
	alpha_hat_vec = first_alpha_vec / first_cn
	current_sum_logCn = np.log(first_cn)
	# Done with first bin in the first region, now process the first region
	start_index_in_region = 1
	alpha_hat_vec, current_sum_logCn = calculate_log_likelihood_one_region(num_state, num_mark, transition_probs, emission_probs, init_probs, first_region_fn, start_index_in_region, alpha_hat_vec, current_sum_logCn)
	# after done with the first region, we will move on to calculate for the following region
	start_index_in_region = 0
	for region_fn in (ordered_region_fn_list[1:]):
		alpha_hat_vec, current_sum_logCn = calculate_log_likelihood_one_region(num_state, num_mark, transition_probs, emission_probs, init_probs, region_fn, start_index_in_region, alpha_hat_vec, current_sum_logCn)
		print ('Done calculating for region: {fn}'.format(fn = region_fn))
	with open(output_fn, 'w') as f:
		f.write(str(current_sum_logCn))
		f.close()
	return 

def main():
	if len(sys.argv) != 5: 
		usage()
	model_fn = sys.argv[1]
	helper.check_file_exist(model_fn)
	signal_dir = sys.argv[2]
	helper.check_dir_exist(signal_dir)
	output_fn  = sys.argv[3]
	helper.create_folder_for_file(output_fn)
	chrom = sys.argv[4]
	print('Done getting command line arguments')
	num_state, num_mark, init_probs, transition_probs, emission_probs = process_model_fn(model_fn)
	print ('Done reading in model file')
	calculate_log_likelihood_multiple_files(num_state, num_mark, init_probs, transition_probs, emission_probs, signal_dir, output_fn, chrom)
	print('Done')

def usage():
	print('python calculate_log_likelihood_forward_alg.py')
	print('model_fn: output of the ChromHMM with all the model parameters')
	print('signal_dir: directory of the binarized signals. This code is assuming that the files are for format: genome_chr<chrom>.<region index 0-based>_binary.txt.gz')
	print ('output_fn: we only report one number which is the log likelihood of the data')
	print('chrom: the chromosome that we are trying to calcualate the log likelihood. It can be chr1 or 1')

main()