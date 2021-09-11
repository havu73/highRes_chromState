import pandas as pd 
import numpy as np
import pybedtools as bed # cite: https://academic.oup.com/bioinformatics/article/27/24/3423/304825
import os 
import sys
import helper
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def read_pred_output_fn(pred_output_fn, pred_chrom):# pred_chrom is always of form 'chr<chrom_index>'
	pred_df = pd.read_csv(pred_output_fn, header = 0, index_col = None, sep = '\t')
	pred_state_out_df = (pred_df[['max_state']]).copy() # we only select one column that's important for this analysis
	pred_posterior_df = pred_df[pred_df.columns[pd.Series(pred_df.columns).str.startswith('E')]]
	pred_state_out_df['chrom'] = pred_chrom
	pred_state_out_df['start'] = pred_state_out_df.index * helper.NUM_BP_PER_BIN
	pred_state_out_df['end'] = pred_state_out_df['start'] + helper.NUM_BP_PER_BIN
	pred_state_out_df = pred_state_out_df.rename(columns = {'max_state': 'pred_state_max_posterior'})
	pred_state_out_df = pred_state_out_df[['chrom', 'start', 'end', 'pred_state_max_posterior']] # rearrange columns
	pred_state_out_df = bed.BedTool.from_dataframe(pred_state_out_df) # convert to a bedtool object for comparing with the true annotation later
	return pred_state_out_df, pred_posterior_df


def rearranged_state_orders(names): 
	# names is a list of E1, E10, E11, ..., E2, E20, ...
	state_numbers = list(map(lambda x: int(x[1:]), names))
	state_numbers.sort()
	renames = list(map(lambda x: 'E{}'.format(x), state_numbers))
	return renames

def calculate_confusion_matrix(combined_state_df, output_folder):
	count_occ_df = combined_state_df.groupby(['pred_state', 'true_state']).size().reset_index(name='count').pivot(index = 'true_state', columns = 'pred_state', values = 'count').fillna(0)#  --> row indices: true_state, columns: pred_state, values: counts of occurrences
	count_occ_df = count_occ_df.div(count_occ_df.sum(axis = 1), axis = 0) # divide each row by its sum --> row sum: 1 --> values: from each state in the true_state, probabilities of being predicted to another state in the model
	rearranged_colnames = rearranged_state_orders(count_occ_df.columns)
	rearranged_rownames = rearranged_state_orders(count_occ_df.index)
	count_occ_df = count_occ_df[rearranged_colnames] # rearrange columns
	count_occ_df = count_occ_df.reindex(rearranged_rownames) # rearrange rows
	count_occ_df.to_csv(output_folder + '/confusion_matrix_pred_true_segment.txt', header = True, index = True, sep = '\t')
	ax = sns.heatmap(count_occ_df, vmin = 0, vmax = 1, cmap = sns.color_palette('Blues', n_colors = 30), annot = False, fmt='.2f')
	plt.savefig(output_folder + '/confusion_matrix_pred_true_segment.png')
	return 

def count_num_start_with_larger_posterior_than_true_state(row):
	true_state_posterior = row[row['true_state']] # posterior of the state that is the true state assigned at each position
	return (row[row.index[:-1]] > true_state_posterior).sum()

def rearranged_num_wrong_state(names):
	# names: a list of number of states that have higher posterior probabiliities compared to the true state
	renames = list(map(lambda x: 'nS_wrong_{}'.format(x), names))
	return renames

def calculate_num_state_higher_posterior_than_true_state(pred_posterior_df, combined_state_df, output_folder):
	pred_posterior_df = pred_posterior_df.merge(combined_state_df, how = 'left', left_index = True, right_index = True) # columns: E<1--> 25>, pred_state, true_state, chrom_start_end
	pred_posterior_df.drop(labels = ['chrom', 'start', 'end', 'pred_state'], axis = 1, inplace = True)
	pred_posterior_df['count_state_wrong'] = pred_posterior_df.apply(count_num_start_with_larger_posterior_than_true_state, axis = 1) # apply function to each row
	count_wrong_df = pred_posterior_df[['true_state', 'count_state_wrong']] # count_state_wrong can only take values 0 --> 24
	count_wrong_df = count_wrong_df.groupby(['true_state', 'count_state_wrong']).size().reset_index(name = 'count').pivot(index = 'true_state', columns = 'count_state_wrong', values = 'count').fillna(0) # #  --> row indices: true_state, columns: number of states that have higher posterior probabiliities than the true state, values: counts of occurrences
	count_wrong_df = count_wrong_df.div(count_wrong_df.sum(axis = 1), axis = 0) # divide each row by its sum --> row sum: 1 --> values: from each state in the true_state, probabilities of havung n number of states with higher posterior probabilities than the true state
	rearranged_rownames = rearranged_state_orders(count_wrong_df.index)
	count_wrong_df = count_wrong_df[count_wrong_df.columns.sort_values()]
	count_wrong_df.columns = rearranged_num_wrong_state(count_wrong_df.columns)
	count_wrong_df = count_wrong_df.reindex(rearranged_rownames)
	count_wrong_df.to_csv(output_folder + '/dist_nState_higher_posterior_than_true_state.txt', header = True, index = True, sep = '\t')
	ax = sns.heatmap(count_wrong_df, vmin = 0, vmax = 1, cmap = sns.color_palette('Blues', n_colors = 30), annot = False, fmt='.2f')
	plt.savefig(output_folder + '/dist_nState_higher_posterior_than_true_state.png')
	return 

def evaluate_state_posterior(pred_output_fn, pred_chrom, true_segment_fn, output_folder):
	pred_state_out_df, pred_posterior_df = read_pred_output_fn(pred_output_fn, pred_chrom)
	true_segment_df = bed.BedTool(true_segment_fn)
	combined_state_df = pred_state_out_df.map(true_segment_df, c = 4, o = 'collapse')
	combined_state_df = combined_state_df.to_dataframe() 
	combined_state_df.columns = ['chrom', 'start', 'end', 'pred_state', 'true_state']
	# calculate_confusion_matrix(combined_state_df, output_folder)
	print("Done getting confusion matrix")
	calculate_num_state_higher_posterior_than_true_state(pred_posterior_df, combined_state_df, output_folder)
	print("Done getting distribution of number of states that show higher predicted posterior probabilities than the true state")
	return 

def main():
	if len(sys.argv) != 5: 
		usage()
	pred_output_fn = sys.argv[1]
	helper.check_file_exist(pred_output_fn)
	pred_chrom = sys.argv[2]
	if not pred_chrom.startswith('chr'):
		pred_chrom = 'chr{}'.format(pred_chrom)
	true_segment_fn = sys.argv[3]
	helper.check_file_exist(true_segment_fn)
	output_folder = sys.argv[4]
	helper.make_dir(output_folder)
	print("Done getting command line arguments")
	evaluate_state_posterior(pred_output_fn, pred_chrom, true_segment_fn, output_folder)
	print ('Done')

def usage():
	print('python eval_confusion_matrxi_pred_true_segment.py')
	print('pred_output_fn: output from calculate_gw_state_posterior_prob.py. Rows: E<state_index_1based>, max_state')
	print('pred_chrom: the chromosome that the pred_output_fn shows results for. Each chromosome should have their own pred_output_fn, based on our pipeline')
	print('true_segment_fn: data of true segmentation, based on the format from ChromHMM> chrom, start, end, state (E<state_index_1based>)')
	print('output_fn: where we have the confusion matrix. Rows: Columns. ')
	exit(1)
main()