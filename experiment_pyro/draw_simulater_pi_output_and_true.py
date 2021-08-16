import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
import sys
import helper
import pandas as pd 
import numpy as np

def visualize_pi()
def main():
	true_pi_fn = sys.argv[1]
	helper.check_file_exist(true_pi_fn)
	pred_pi_fn = sys.argv[2]
	helper.check_file_exist(pred_pi_fn)
	true_beta_fn = sys.argv[3]
	helper.check_file_exist(true_beta_fn)
	pred_beta_fn = sys.argv[4]
	helper.check_file_exist(pred_beta_fn)
	output_fn = sys.argv[5]
	helper.create_folder_for_file(output_fn)
	print ('Done getting command line arguments')

def usage():
	print('python draw_compare_params_true_and_pred.py')
	print("true_pi_fn")
	print('pred_pi_fn: output of toy_model_chrom_mark_model.py')
	print('true_beta_fn')
	print('pred_beta_fn: output of toy_model_chrom_mark_model.py')
	print('output_fn')