import pandas as pd
import numpy as np
import sys
import os
import helper
import glob 
from scipy.stats import entropy
def calculate_entropy_one_file(input_fn, output_fn):
	df = pd.read_csv(input_fn, header = 0, skiprows = 1, sep = '\t', index_col = None) # rows: genomic positions, columns: states
	entropy_df = pd.DataFrame()
	entropy_df['entropy'] = df.apply(entropy, axis = 1) # calcualte the entropy of each row (genomic positions)
	entropy_df['state'] = df.idxmax(axis = 1) # for each position, get the state with highest posterior probability
	entropy_df.to_csv(output_fn, header = False, index = False, sep = '\t', compression = 'gzip')
	return 

def calculate_entropy_multiple_files(posterior_dir, output_folder):
	posterior_fn_list = glob.glob(posterior_dir + '/*_chr*_posterior.txt.gz')
	for posterior_fn in posterior_fn_list:
		chrom = posterior_fn.split('/')[-1].split('_posterior.txt.gz')[0].split('chr')[1]
		output_fn = os.path.join(output_folder, 'entropy_chr{chrom}.txt.gz'.format(chrom = chrom))
		calculate_entropy_one_file(posterior_fn, output_fn)
		print ("Done calculating for chrom: {chrom}".format(chrom = chrom))
	return 

def main():
	if len(sys.argv) != 3:
		usage()
	posterior_dir = sys.argv[1]
	helper.check_dir_exist(posterior_dir)
	output_folder = sys.argv[2]
	helper.make_dir(output_folder)
	print ("Done getting command line argument")
	calculate_entropy_multiple_files(posterior_dir, output_folder)
	print('Done')

def usage():
	print ('python calculate_entropy_posterior_chromHMM.py')
	print ('posterior_dir: where the files outputted from ChromHMM MakeSegmentation -printposterior are stored')
	print ('output_folder')
	exit(1)

main()