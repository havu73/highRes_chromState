import pandas as pd 
import numpy as np 
import os 
import sys

def main():
	NUM_MANDATORY_ARGS = 4
	if len(sys.argv) < NUM_MANDATORY_ARGS:
		usage()
	num_bigwig_to_binarize = helper.get_command_line_integer(sys.argv[1])
	output_folder = sys.argv[2]
	helper.make_dir(output_folder)
	chromsize_fn = sys.argv[3]
	helper.check_file_exist(chromsize_fn)
	if len(sys.argv) != (NUM_MANDATORY_ARGS + num_bigwig_to_binarize):
		print("Number of specified bigwig files do not match with number of input arguments")
		usage()
	bigwig_fn_list = sys.argv[NUM_MANDATORY_ARGS:] 
	map(lambda x: helper.check_file_exist(x), bigwig_fn_list)
	print("Done processing command line arguments")

def usage():
	print ("python binarize_logPval_signal.py")
	print ("num_bigwig_to_binarize: number of bigwig files that we will binarize the data for ChromHMM LearnModel")
	print ("output_folder: where each genomic region will be stored as a separate file")
	print ("bigwig_fn_list: Files are separated by space")
	exit(1)
# import argparse
# parser = argparse.ArgumentParser(desctiption = 'This file will takes in the signals from pval.signal.wig (transformed from bigwig files) from roadmap and binary the data in a format usable to ChromHMM', epilog = 'The output folder can be inputted into ChromHMM LearnModel function', add_help = True)
# parser.add_argument('--chrom_size', help = 'path to file that specify the size of the chromosomes that we want to binarize the data for')
