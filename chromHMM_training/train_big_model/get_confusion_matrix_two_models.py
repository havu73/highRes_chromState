import pandas as pd
import numpy as np 
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import helper
import pybedtools as pbed

def count_num_segment_each_intersect(row): # row is a row from the join_df after intersecting the two segment files, in function get_confusion_matrix_data
	start = np.max([row['startF1'], row['startF2']]) # the larger start coordinate
	end = np.min([row['endF1'], row['endF2']]) # the smaller end coordinate
	num_bins = int((end - start) / helper.NUM_BP_PER_BIN)
	return num_bins

def get_confusion_matrix_data(segment_fn1, segment_fn2, confusionOut_prefix):
	# we will produce two files <confusionOut_prefix>.txt and <confusionOut_prefix>.png
	df1 = pbed.BedTool(segment_fn1)
	df2 = pbed.BedTool(segment_fn2)
	joint_df = df1.intersect(df2, wa= True, wb = True).to_dataframe()
	joint_df.columns = ['chrF1', 'startF1', 'endF1', 'stateF1', 'chrF2', 'startF2', 'endF2', 'stateF2']
	joint_df['num_intersect_bins'] = joint_df.apply(count_num_segment_each_intersect, axis = 'columns') # apply function to each row
	result_df = joint_df.groupby(['stateF1', 'stateF2'])['num_intersect_bins'].agg('sum').to_frame().reset_index() # --> stateF1, stateF2, num_intersect_bins
	result_df = result_df.pivot_table(values = 'num_intersect_bins', index = 'stateF1', columns = 'stateF2', aggfunc = np.sum, dropna = False, fill_value = 0, margins = True)
	rowsum = result_df.sum(axis = 1) # vector of row sums
	result_df = result_df.div(rowsum, axis = 0) # each column will get divied by the rowsum vector --> after this, rowsum of result+df is 1.0 for all rows
	result_df.to_csv(confusionOut_prefix + '.txt.gz', index = True, header = True, sep = '\t', compression = 'gzip')
	ax = sns.heatmap(result_df, vmin = 0, vmax = 1, cmap = sns.color_palette('Blues', n_colors = 30), annot = True, fmt='.2f')
	plt.savefig(confusionOut_prefix + '.png')
	print('Done!')
	return 

def main():
	if len(sys.argv) != 4:
		usage()
	segment_fn1 = sys.argv[1]
	helper.check_file_exist(segment_fn1)
	segment_fn2 = sys.argv[2]
	helper.check_file_exist(segment_fn2)
	confusionOut_prefix = sys.argv[3]
	helper.create_folder_for_file(confusionOut_prefix)
	print ('Done getting command line arguments')
	get_confusion_matrix_data(segment_fn1, segment_fn2, confusionOut_prefix)	
	return 

def usage():
	print ('python get_confusion_matrix_two_models.py')	
	print ('segment_fn1: recommend the path to the bigger model segment file (corrected)')
	print ('segment_fn2: recommend the path to the smaller model segment file (corrected)')
	print ('confusionOut_prefix: path to the files, minus the extensions')
	exit(1)

main()