import pandas as pd
import numpy as np
import sys
import os
import helper

def get_rid_of_stupid_file_tail(context_name):
	return context_name.split('/')[-1].split('.')[0]

def get_enrichment_df(enrichment_fn):
    enrichment_df = pd.read_csv(enrichment_fn, sep = "\t", header = 0)
    enrichment_df = enrichment_df.rename(columns = {u'state (Emission order)': 'state', u'Genome %' : 'percent_in_genome', u'State (Emission order)': 'state'})
    enrichment_df.columns = map(get_rid_of_stupid_file_tail, enrichment_df.columns)
    (num_state, num_enr_cont) = (enrichment_df.shape[0] - 1, enrichment_df.shape[1] - 2) # substract the first two columns: state and percent_in_genome 
    percent_genome_of_cont = enrichment_df.iloc[num_state, 2:]
    enrichment_df = enrichment_df.loc[:(num_state - 1)]
    # num_enr_cont : number of enrichment context (TSS, exon, etc.)
    return enrichment_df, num_state, num_enr_cont, percent_genome_of_cont


def calculate_percent_background_of_foreground(fg_df, percent_genome_of_cont_fg, fg_cont_name, percent_genome_of_cont_bg, bg_cont_name): # this is redundant function, we dont need it anymore
	# this function will calculate the percentage of the background context that one foregrounc context occupies
	# percent_genome_of_cont_bg: percent of the genome that are of the background context, as list with each element corresponding to one context
	# fg_cont_name: column name of the foreground context that we are trying to calculate
	calculate_df = (fg_df[['percent_in_genome', fg_cont_name]]).copy() 
	calculate_df['frac_gen_in_state'] = calculate_df['percent_in_genome'] / 100.0 # fraction of the genome that is in the state
	calculate_df['frac_cont_in_state'] = calculate_df[fg_cont_name] * calculate_df['frac_gen_in_state'] # #SM/#M = FE * #S/#G --> fraction of the context that is in the state
	calculate_df['frac_gen_in_state_in_cont'] = calculate_df['frac_cont_in_state'] * percent_genome_of_cont_fg[fg_cont_name]  / 100 # #SM/#G = (#SM/#M) * ((#M*100)/#G) / 100
	frac_gen_in_fg_cont = np.sum(calculate_df['frac_gen_in_state_in_cont']) # fraction of gene in the fg context 
	percent_bg_in_fg_cont = frac_gen_in_fg_cont / percent_genome_of_cont_bg[bg_cont_name] * 100 * 100 # #Mf/#Mb = #Mf/#G * #G/(#Mb*100) * 100 * 100
	return percent_bg_in_fg_cont

def calculate_fraction_background_in_state(bg_df, percent_genome_of_cont_bg): # tested
	background_cont_name = bg_df.columns[2:] # the last column is the column of the background fold enrichment
	result_df = pd.DataFrame()
	result_df['state'] = bg_df['state']
	result_df['frac_gen_in_state'] = bg_df['percent_in_genome'] / 100.0 
	for cont_name in background_cont_name:
		result_df[cont_name] = bg_df[cont_name] * result_df['frac_gen_in_state']
	return result_df # state, frac_gene_in_state, <context>: each column is named after each of the background context in the overlap file, and the values show the frac_bg_in_state for each state and each background

def calculate_FE_against_background(background_fn, foreground_fn, output_fn, fg_bg_name_dict):
	bg_df, num_state_bg, num_enr_cont_bg, percent_genome_of_cont_bg = get_enrichment_df(background_fn)
	fg_df, num_state_fg, num_enr_cont_fg, percent_genome_of_cont_fg = get_enrichment_df(foreground_fn)
	assert num_state_fg == num_state_bg, 'Number of states between the foreground and background data is not the same. Some thing went wrong befor this step. Check your pipeline'
	assert np.array_equal(bg_df['percent_in_genome'], fg_df['percent_in_genome']), 'The percentage of each state in the genome in the foreground and background datasets are different. This is not good since we suppose you calculate foreground and background enrichment based on the same segmentation file'
	bg_cont_name_list = bg_df.columns[2:] # list of the column name of fold enrichment with the background 
	FE_against_bg_df = pd.DataFrame()
	FE_against_bg_df['state'] = bg_df['state']
	frac_bg_in_state_df = calculate_fraction_background_in_state(bg_df, percent_genome_of_cont_bg) # # state, frac_gene_in_state, <context>: each column is named after each of the background context in the overlap file, and the values show the frac_bg_in_state for each state and each background
	fg_cont_name_list = list(fg_df.columns[2:])
	percent_bg_in_fg_cont = pd.Series([])
	for fg_cont in fg_cont_name_list:
		matched_bg_cont = fg_bg_name_dict[fg_cont] 
		FE_against_bg_df[fg_cont] = (fg_df[fg_cont]) / bg_df[matched_bg_cont] # divide each enrichment context in the foreground by the enrichment of background. This is dividing each column from one dataframe by the same series (in this case, the enrichment from background)
		FE_against_bg_df['frac_bg_in_state_' + fg_cont] = frac_bg_in_state_df[matched_bg_cont]
		# get the percentage of the background that are in  each of the foreground context	
		percent_bg_in_fg_this_cont = percent_genome_of_cont_fg[fg_cont] / percent_genome_of_cont_bg[matched_bg_cont] * 100  # no need to use the function calculate_percent_background_of_foreground, waste of time, this calculation is much faster and still correct
		percent_bg_in_fg_cont[fg_cont] = percent_bg_in_fg_this_cont
		percent_bg_in_fg_cont['frac_bg_in_state_' + fg_cont] = ''
	(num_state, num_col) = FE_against_bg_df.shape # get the current shape, so that we can add one last row to the dataframe
	FE_against_bg_df.loc[num_state] = ['Base'] + list(percent_bg_in_fg_cont)
	FE_against_bg_df.to_csv(output_fn, index = False, header = True, sep = '\t')
	print ("Done calculating all the necessary information and printing the fold enrichment of forground against background after: " )
	return 

def read_column_match_fn(column_match_fn):
	df = pd.read_csv(column_match_fn, header = None, index_col = 0, sep = '\t', squeeze = True).to_dict()
	return df

def main():
	if len(sys.argv) != 5:
		usage()
	background_fn = sys.argv[1]
	helper.check_file_exist(background_fn)
	foreground_fn = sys.argv[2]
	helper.check_file_exist(foreground_fn)
	column_match_fn = sys.argv[3]
	helper.check_file_exist(column_match_fn)
	output_fn = sys.argv[4]
	helper.create_folder_for_file(output_fn)
	fg_bg_name_dict = read_column_match_fn(column_match_fn) # dictionary keys: the column names of contexts in the foreground, values: the column names of contexts in the background
	print ("Done getting command line argument after: ")
	calculate_FE_against_background(background_fn, foreground_fn, output_fn, fg_bg_name_dict)


def usage():
	print ("python calculate_fold_enrichment_against_background.py")
	print ("background_fn: fn of background enrichment")
	print ("foreground_fn: fn of foreground enrichment")
	print ("column_match_fn: the file with 2 column, first is the column name in the foreground of the overlap file, second is the column name in the background of the overlap file")
	print ("output_fn: where the output of FE against background is stored")
	exit(1)

main()