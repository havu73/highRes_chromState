import pandas as pd 
import numpy as np 
import sys
import os
import helper 

def get_chrom_mark_signals_for_training_from_chromHMMInput(binarized_signal_folder, train_sorted_coord_fn, output_fn):
	sample_df = pd.read_csv(train_sorted_coord_fn, header = None, sep = '\t')
	sample_df.columns = ['chrom', 'start', 'end']
	NUM_BP_PER_BIN = 200
	NUM_BIN_PER_REGION = 5000
	sample_df['region'] = np.floor(sample_df['start'] / (NUM_BP_PER_BIN * NUM_BIN_PER_REGION)).astype(int)
	sample_df['index_within_region'] = np.floor((sample_df['start'] - (sample_df['region'] * NUM_BP_PER_BIN * NUM_BIN_PER_REGION)) / NUM_BP_PER_BIN).astype(int)
	grouped_df = sample_df.groupby(['chrom', 'region'])
	result_df_list = []
	for region_tuple, region_df in grouped_df:
		# region_tuple example: ('chrX', 262)
		signal_result_fn = os.path.join(binarized_signal_folder, 'genome_{chrom}.{reg_index}_binary.txt.gz'.format(chrom = region_tuple[0], reg_index = region_tuple[1]))
		signal_df = pd.read_csv(signal_result_fn, skiprows = 1, header = 0, index_col = None, sep = '\t') # first row is just chrom.region_index --> skip 
		region_df = region_df.merge(signal_df, how = 'left', left_on = 'index_within_region', right_index = True)
		region_df.drop(labels = ['region', 'index_within_region'], axis = 'columns', inplace = True)
		result_df_list.append(region_df)
	result_df = pd.concat(result_df_list, ignore_index = True)
	result_df.to_csv(output_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	return

def get_chrom_mark_signals_for_training_no_splitrows(binarized_signal_folder, train_sorted_coord_fn, output_fn): # each file in binarized_signal_folder corresponds to one chromosome. Currently 08062021, we would like to restrict training and testing to one chrom for training, one chrom for testing, so there will likely be one file from this folder that is used during this function. 
	sample_df = pd.read_csv(train_sorted_coord_fn, header = None, sep = '\t')
	sample_df.columns = ['chrom', 'start', 'end']
	NUM_BP_PER_BIN = 200
	sample_df['bin_index'] = np.floor(sample_df['start'] / NUM_BP_PER_BIN).astype(int)
	grouped_df = sample_df.groupby(['chrom'])
	result_df_list = [] # each element corresponds to one chromosome with rows: regions in the training sample
	for chrom, region_df in grouped_df:
		signal_result_fn = os.path.join(binarized_signal_folder, 'genome_{chrom}_binary.txt.gz'.format(chrom = chrom))
		signal_df = pd.read_csv(signal_result_fn, header = 0, index_col = None, sep = '\t') # columns: marks, rows: genomic position in the training sample
		region_df = region_df.merge(signal_df, how = 'left', left_on = 'bin_index', right_index = True)
		region_df.drop(labels = ['bin_index'], axis = 'columns', inplace = True)
		result_df_list.append(region_df)
	result_df = pd.concat(result_df_list, ignore_index = True)
	result_df.to_csv(output_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	return

def main():
	if len(sys.argv) != 4:
		usage()
	binarized_signal_folder = sys.argv[1] # files in this folder are of forms: genome_chr6.102_binary.txt.gz
	helper.check_dir_exist(binarized_signal_folder)
	train_sorted_coord_fn = sys.argv[2]
	helper.check_file_exist(train_sorted_coord_fn)
	output_fn = sys.argv[3]
	helper.create_folder_for_file(output_fn)
	print ("Done getting command line arguments")
	# get_chrom_mark_signals_for_training_from_chromHMMInput(binarized_signal_folder, train_sorted_coord_fn, output_fn)
	get_chrom_mark_signals_for_training_no_splitrows(binarized_signal_folder, train_sorted_coord_fn, output_fn)
	print ('Done')

def usage():
	print ("python get_chrom_mark_signals_for_training_from_chromHMMInput.py")
	print ("binarized_signal_folder: where we store the signals of binarized marks that have been created in the pipeline. Each file in this folder should correspond to one chromosome")
	print ("train_sorted_coord_fn: 3 columns chrom, start, end --> coordinates of genomic regions that are in the training data set")
	print ("output_fn: chrom, start, end, <mark> --> genomic coodiantes of training regions, and the binarized mark signals for the marks that we get from experiments")
	exit(1)
main()