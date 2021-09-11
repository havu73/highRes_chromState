import pandas as pd 
import numpy as np 
import os 
import sys
import helper
import glob
from tqdm import tqdm

def jaccard_binary(x,y): 
	# function taken from blog https://www.learndatasci.com/glossary/jaccard-similarity/
	"""A function for finding the similarity between two binary vectors"""
	intersection = np.logical_and(x, y)
	union = np.logical_or(x, y)
	similarity = intersection.sum() / float(union.sum())
	return similarity

def mutual_info_binary(x, y):
	result = 0 
	X0 = np.logical_not(x)
	# X1 = np.logical(x)
	Y0 = np.logical_not(y)
	# Y1 = np.logical(y)
	X_val_list = [X0, x]
	Y_val_list = [Y0, y]
	total_obs = len(x) # also = len(y)
	for xi in [0,1]:
		this_X = X_val_list[xi]
		PX = (this_X.sum()+1) / (total_obs+1) # to avoid log of 0 later
		for yi in [0,1]:
			this_Y = Y_val_list[yi]
			PY = (this_Y.sum()+1) / (total_obs+1) # to avoid log of 0 later
			XY = np.logical_and(this_X, this_Y)
			P_XY = (XY.sum()+1) / (total_obs+1) # to avoid log of 0 later
			result += P_XY * (np.log(P_XY) - np.log(PX) - np.log(PY))
	return result

def read_ct_signal(fn):
	ct_series = pd.read_csv(fn, header = 0, index_col = None, sep = '\t', squeeze = True)
	ct_series = np.array(ct_series)
	return(ct_series)

def calculate_jaccard_index_across_ct(all_ct_signal_folder, output_prefix, chromosome, mark):
	search_pattern = all_ct_signal_folder + "/*/binarized_signals/indv_marks/genome_{chromosome}_binary_{mark}.txt.gz".format(chromosome = chromosome, mark = mark)
	signal_ct_fn_list = glob.glob(search_pattern)
	ct_list = list(map(lambda x: x.split('/')[-4], signal_ct_fn_list)) # from /gstore/project/gepiviz_data/vuh6/roadmap/marks_signal/E001/binarized_signals/indv_marks/genome_chr21_binary_H3K4me3.txt.gz to E001
	num_ct = len(ct_list)
	result_df = np.ones((num_ct, num_ct)) # initialize all to 0
	for ct_index in tqdm(range(num_ct)):
		ct_series = read_ct_signal(signal_ct_fn_list[ct_index]) # a numpy array fo binarized signals for this ct, each element in the array corresponds to one genomic position
		for other_ct_index in range(ct_index, num_ct):
			other_ct_series = read_ct_signal(signal_ct_fn_list[other_ct_index]) # a numpy array fo binarized signals for this ct, each element in the array corresponds to one genomic position
			jaccard_index = jaccard_binary(ct_series, other_ct_series)
			result_df[ct_index, other_ct_index] = jaccard_index
			result_df[other_ct_index, ct_index] = jaccard_index
	result_df = pd.DataFrame(result_df, columns = ct_list)
	result_df.index = ct_list
	result_df.to_csv(output_prefix + '_jaccard.txt.gz', header = True, index = True, sep = '\t', compression = 'gzip')
	return

def calculate_mututal_info_across_ct(all_ct_signal_folder, output_prefix, chromosome, mark):
	search_pattern = all_ct_signal_folder + "/*/binarized_signals/indv_marks/genome_{chromosome}_binary_{mark}.txt.gz".format(chromosome = chromosome, mark = mark)
	signal_ct_fn_list = glob.glob(search_pattern)
	ct_list = list(map(lambda x: x.split('/')[-4], signal_ct_fn_list)) # from /gstore/project/gepiviz_data/vuh6/roadmap/marks_signal/E001/binarized_signals/indv_marks/genome_chr21_binary_H3K4me3.txt.gz to E001
	num_ct = len(ct_list)
	result_df = np.zeros((num_ct, num_ct)) # initialize all to 0
	for ct_index in tqdm(range(num_ct)):
		ct_series = read_ct_signal(signal_ct_fn_list[ct_index]) # a numpy array fo binarized signals for this ct, each element in the array corresponds to one genomic position
		for other_ct_index in range(ct_index, num_ct):
			other_ct_series = read_ct_signal(signal_ct_fn_list[other_ct_index]) # a numpy array fo binarized signals for this ct, each element in the array corresponds to one genomic position
			jaccard_index = mutual_info_binary(ct_series, other_ct_series)
			result_df[ct_index, other_ct_index] = jaccard_index
			result_df[other_ct_index, ct_index] = jaccard_index
	result_df = pd.DataFrame(result_df, columns = ct_list)
	result_df.index = ct_list
	result_df.to_csv(output_prefix + '_MI.txt.gz', header = True, index = True, sep = '\t', compression = 'gzip')
	return
 
def main():
	if len(sys.argv) != 5:
		usage()
	all_ct_signal_folder = sys.argv[1]
	helper.check_dir_exist(all_ct_signal_folder)
	output_prefix = sys.argv[2]
	helper.create_folder_for_file(output_prefix)
	chromosome = sys.argv[3]
	if not chromosome.startswith('chr'):
		chromosome = 'chr{}'.format(chromosome)
	mark = sys.argv[4]
	logger = helper.argument_log(command = sys.argv, output_folder= os.path.dirname(output_prefix)) # to save the command line arguments that led to this 
	logger.write_log()
	print('Done getting command line arguments')
	# calculate_jaccard_index_across_ct(all_ct_signal_folder, output_prefix, chromosome, mark)
	calculate_mututal_info_across_ct(all_ct_signal_folder, output_prefix, chromosome, mark)
	print('Done')

def usage():
	print ('python calculate_jaccard_index_all_ref_ct.py')
	print ('all_ct_signal_folder: each ct data is in \'{all_ct_signal_folder\'}/\'{ct\'}/indv_marks/genome_chr<chrom>_binary_<mark>.txt.gz')
	print ('output_prefix: where the matrix of jaccard index or mutual information is stored')
	print ('chromosome: the chromosome that we will calculate the jaccard index for all cell types from')
	print ('mark: the chromatin mark that we get signals from to calculate the jaccard index from')
	print ('Note: Right now the function is only supporting the calculation for only one mark that is available for all the ct')
	exit(1)

main()