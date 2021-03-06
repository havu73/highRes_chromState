# tested 02/25/2021
import pandas as pd 
import numpy as np 
import sys
import os
import helper 


def sample_genome_positions(chrom_length_fn, sample_fraction, output_fn, chrom_list):
	NUM_BP_PER_BIN = 200
	chr_length_df = pd.read_table(chrom_length_fn, sep = '\t', header = None)
	if chr_length_df.shape[1] == 3: # if there are 3 columns: chrom, start (0), end
		chr_length_df = chr_length_df[[0,2]] # get rid of the second column because they are all 1's
	elif chr_length_df.shape[1] == 2: # if there are 2 columns: chrom, length
		pass
	else:
		print("chr_length_df should have 2/3 columns, observed: {}".format(chr_length_df.shape[1]))
		usage()
	chr_length_df.columns = ['chrom', 'length'] 
	sample_df = pd.DataFrame(columns = ['chrom', 'start_bp', 'end_bp']) # all bp index are zero-based --> a dataframe of all the positions that we will sample
	for chrom_index in chrom_list: #user's choice of chromsomes, or all of them 1 --> 22, X
		this_chrom_length = (chr_length_df[chr_length_df['chrom'] == 'chr' + chrom_index])['length']
		num_bin_this_chrom = int(this_chrom_length / NUM_BP_PER_BIN )
		num_bin_to_sample = int(num_bin_this_chrom * sample_fraction) 
		sample_bins_this_chrom = np.random.choice(range(num_bin_this_chrom), size = num_bin_to_sample, replace = False)
		sample_bins_this_chrom.sort()
		sample_this_chrom_df = pd.DataFrame() 
		sample_this_chrom_df['chrom'] = ['chr' + chrom_index] * len(sample_bins_this_chrom)
		sample_this_chrom_df['start_bp'] = sample_bins_this_chrom * NUM_BP_PER_BIN
		sample_this_chrom_df['end_bp'] = sample_this_chrom_df['start_bp'] + NUM_BP_PER_BIN
		sample_df = sample_df.append(sample_this_chrom_df)
		print ("Done with chromosome: " + chrom_index)
	# save to file 
	sample_df.to_csv(output_fn, sep = '\t', header = None, index = False, compression = 'gzip')
	return sample_df


def main():
	if len(sys.argv) < 4:
		usage()
	chrom_length_fn = sys.argv[1]
	helper.check_file_exist(chrom_length_fn)
	sample_fraction = sys.argv[2]
	try: 
		sample_fraction = float(sample_fraction)
	except: 
		print("sample_fraction should be a float number")
		usage()
	assert sample_fraction > 0 and sample_fraction < 1.0, "sample_fraction should be greater than 0 and smaller than 1"
	output_fn = sys.argv[3]
	try: 
		chrom_list = sys.argv[4:]
	except:
		chrom_list = helper.CHROMOSOME_LIST
	assert set(chrom_list).issubset(set(helper.CHROMOSOME_LIST)), "The chromosome list should be a subset of all the chomosomes {}".format(chrom_list) 
	helper.create_folder_for_file(output_fn)
	print ("Done getting command line arguments")
	# select regions on the genome that we will sample from
	genome_sample_df = sample_genome_positions(chrom_length_fn, sample_fraction, output_fn, chrom_list) # --> a dataframe of 3 columns: "chromosome", "start_bp", 'end_bp'
	print('Done')
	return

	
def usage():
	print ("python sample_genome.py")
	print ("chrom_length_fn: a bed file with 2 or 3 columns, no headers. If 3 columns: Columns should correspond to: chromsoome (chr1, chr2, etc.), start_bp (0 in all chromsomes), end_bp (the length of the chromosome, which will be a multiple of 200 because it's the resolution of the chromatin state annotation). If 2 columns (chrom length from ucsc genome browser: chromosome, length (not multiple of 200)")
	print ("sampling fraction: the fraction of the genome that we want to sample. Remember fractions are not percentages.")
	print ("output_fn: where the data of sampled regions and state segementation will be stored for all the cell types that we chose")
	print ('chromosome (1, 2, ... X, Y, etc): what chromosome we restrict our sampling from. If not provided, then we sample whole-genome')
	print ("The result should give us around 1518140 200-bp bins, genome-wide, 10%")
	exit(1)
main()