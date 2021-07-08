import pandas as pd
import numpy as np
import os
import sys
import helper 
NUM_BP_PER_BIN = 200

def write_coord_data(chrom_index, sample_bin_indices):
	sample_this_chrom_df = pd.DataFrame() 
	sample_this_chrom_df['chrom'] = ['chr' + chrom_index] * len(sample_bin_indices)
	sample_this_chrom_df['start_bp'] = sample_bin_indices * NUM_BP_PER_BIN
	sample_this_chrom_df['end_bp'] = sample_this_chrom_df['start_bp'] + NUM_BP_PER_BIN
	return sample_this_chrom_df

def sample_genome_positions(chrom_length_fn, sample_fraction, out_dir):
	chr_length_df = pd.read_table(chrom_length_fn, sep = '\t', header = None)
	chr_length_df = chr_length_df[[0,2]] # only get the first and third column: chromosome and length. THe second column can go because it is just 0
	chr_length_df.columns = ['chrom', 'length'] 
	# chr_length_df has the following columns: chrom, length
	train_sample_df = pd.DataFrame(columns = ['chrom', 'start_bp', 'end_bp']) # all bp index are zero-based --> a dataframe of all the positions that we will sample
	test_sample_df = pd.DataFrame(columns = ['chrom', 'start_bp', 'end_bp']) # all bp index are zero-based --> a dataframe of all the positions that we will sample
	for chrom_index in helper.CHROMOSOME_LIST:
		this_chrom_length = (chr_length_df[chr_length_df['chrom'] == 'chr' + chrom_index])['length']
		num_bin_this_chrom = int(this_chrom_length / NUM_BP_PER_BIN )
		num_bin_to_sample = int(num_bin_this_chrom * sample_fraction) 
		train_bins_this_chrom = np.random.choice(range(num_bin_this_chrom), size = num_bin_to_sample, replace = False)
		test_bins_this_chrom = np.setdiff1d(range(num_bin_this_chrom), train_bins_this_chrom)
		train_sample_df = train_sample_df.append(write_coord_data(chrom_index, train_bins_this_chrom))
		test_sample_df = test_sample_df.append(write_coord_data(chrom_index, test_bins_this_chrom))
		print ("Done with chromosome: " + chrom_index)
	# save to file 
	train_fn = os.path.join(out_dir, 'train_segments.bed.gz')
	test_fn = os.path.join(out_dir, 'test_segments.bed.gz')
	train_sample_df.to_csv(train_fn, sep = '\t', header = None, index = False, compression = 'gzip')
	test_sample_df.to_csv(test_fn, sep = '\t', header = None, index = False, compression = 'gzip')
	return 

def main():
	if len(sys.argv) != 4: 
		usage()
	chrom_length_fn = sys.argv[1]
	helper.check_file_exist(chrom_length_fn)
	sample_fraction = helper.get_command_line_float(sys.argv[2])
	out_dir = sys.argv[3]
	helper.make_dir(out_dir)
	print("Done getting command line arguments")
	sample_genome_positions(chrom_length_fn, sample_fraction, out_dir)
	print('Done')

def usage():
	print ("get_sample_genome_bed.py")
	print ('chrom_length_fn: where we get the length of the chromosomes')
	print ('sample_fraction: 0 to 1')
	print ('out_dir: where the train and test bed files are stored')
	print ('Notes: we will randomly pick the 200-bp bins in the genomes')
	exit(1)
main()