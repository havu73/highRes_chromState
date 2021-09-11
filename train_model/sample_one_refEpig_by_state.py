# sample the segmentation such that we have a more even representation of the states in the training data
import pandas as pd 
import numpy as np 
import sys
import os
import helper 


def sample_one_state_df(state_df, MAX_BIN_TO_SAMPLE_PER_STATE):
	# function called inside sample_one_chrom_equal_state_rep
	# state_df: chrom, start, end, state, num_bins, start_bin, end_bin. All state are assocated with one state
	# MAX_BIN_TO_SAMPLE_PER_STATE: number of bins to sample out of all of those associated with this state in state_df
	# return a list of indices: 0-based 200bp regions that we sample for this state
	all_bin_indices_in_state = []
	for row_index, row in state_df.iterrows(): 
		all_bin_indices_in_state += list(np.arange(row['start_bin'], row['end_bin']))
	sample_bins_this_state = np.random.choice(all_bin_indices_in_state, size = MAX_BIN_TO_SAMPLE_PER_STATE, replace = False)
	return list(sample_bins_this_state)

def sample_one_chrom_equal_state_rep(segment_fn, chrom, MAX_BIN_TO_SAMPLE_PER_STATE):
	df = pd.read_csv(segment_fn, header = None, sep = '\t', index_col = None)
	df.columns = ['chrom', 'start', 'end', 'state']
	df = df[df['chrom'] ==  chrom]
	df.reset_index(inplace = True)
	df['num_bins'] = ((df['end'] - df['start'])/ helper.NUM_BP_PER_BIN).astype(int)
	df['start_bin'] = (df['start'] / helper.NUM_BP_PER_BIN).astype(int)
	df['end_bin'] = (df['end'] / helper.NUM_BP_PER_BIN).astype(int)
	grouped_df = df.groupby('state')
	numBins_byState = grouped_df['num_bins'].sum() # series: index: state (ex: E1--> E25), values: numbers of 200bp bins that are in each state
	state_to_collect_all = numBins_byState.index[numBins_byState < MAX_BIN_TO_SAMPLE_PER_STATE] # we will sample from all places where the state has fewer bins than MAX_BIN_TO_SAMPLE_PER_STATE
	# first we will get index of regions that are associated with states that we will sample everything from 
	sample_indices_list = []
	for state in state_to_collect_all:
		state_df = grouped_df.get_group(state)
		for row_index, row in state_df.iterrows(): # iterate through all the rows
			sample_indices_list += list(np.arange(row['start_bin'], row['end_bin'])) 
	# next we will get index of bins that we sample from those associated with states that have more than MAX_BIN_TO_SAMPLE_PER_STATE bins in this chrom
	states_to_sample_max_bin = numBins_byState.index[numBins_byState >= MAX_BIN_TO_SAMPLE_PER_STATE]
	print(states_to_sample_max_bin)
	for state in states_to_sample_max_bin:
		state_df = grouped_df.get_group(state)
		sample_indices_list += sample_one_state_df(state_df, MAX_BIN_TO_SAMPLE_PER_STATE)
	# now, we will list out all the places that we sampled to assure the best equal state representation
	sample_indices_list = np.array(sample_indices_list)
	sample_indices_list.sort()
	sample_df = pd.DataFrame()
	sample_df['start'] = (helper.NUM_BP_PER_BIN * np.array(sample_indices_list)).astype(int)
	sample_df['end'] = sample_df['start'] + helper.NUM_BP_PER_BIN
	sample_df['chrom'] = chrom
	sample_df = sample_df[['chrom', 'start', 'end']] # rearrange columns to look like bed file
	# sample_df.to_csv(output_fn, header = False, index = False, sep ='\t', compression = 'gzip')
	return sample_df

def sample_genome_equal_state_representation(segment_fn, MAX_BIN_TO_SAMPLE_PER_STATE, chrom_list, output_fn):
	result_df = pd.DataFrame(columns = ['chrom', 'start', 'end'])
	for chrom in chrom_list:
		if not chrom.startswith('chr'):
			chrom = 'chr{}'.format(chrom)
		this_chrom_df = sample_one_chrom_equal_state_rep(segment_fn, chrom, MAX_BIN_TO_SAMPLE_PER_STATE)
		result_df = result_df.append(this_chrom_df)
	result_df.to_csv(output_fn, header = False, index = False, sep ='\t', compression = 'gzip')
	return 

def main():
	if len(sys.argv) < 4:
		usage()
	segment_fn = sys.argv[1]
	helper.check_file_exist(segment_fn)
	MAX_BIN_TO_SAMPLE_PER_STATE = helper.get_command_line_integer(sys.argv[2])
	assert MAX_BIN_TO_SAMPLE_PER_STATE > 0, 'MAX_BIN_TO_SAMPLE_PER_STATE must be positive'
	output_fn = sys.argv[3]
	try: 
		chrom_list = sys.argv[4:]
	except:
		chrom_list = helper.CHROMOSOME_LIST
	assert set(chrom_list).issubset(set(helper.CHROMOSOME_LIST)), "The chromosome list should be a subset of all the chomosomes {}".format(chrom_list) 
	helper.create_folder_for_file(output_fn)
	print ("Done getting command line arguments")
	# select regions on the genome that we will sample from
	sample_genome_equal_state_representation(segment_fn, MAX_BIN_TO_SAMPLE_PER_STATE, chrom_list, output_fn) # --> a dataframe of 3 columns: "chromosome", "start_bp", 'end_bp'
	print('Done')
	return

	
def usage():
	print ("python sample_genome.py")
	print ("segment_fn: a bed file of state segmentation from a reference epigenome")
	print ("MAX_BIN_TO_SAMPLE_PER_STATE: the max number of bins per state per chromsome that we will sample from.")
	print ("output_fn: where the data of sampled regions and state segementation will be stored for all the cell types that we chose")
	print ('chromosome (1, 2, ... X, Y, etc): what chromosome we restrict our sampling from. If not provided, then we sample whole-genome')
	print ("The result should give us around 1518140 200-bp bins, genome-wide, 10%")
	exit(1)
main()