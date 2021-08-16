import numpy as np 
import pandas as pd 
import string
import os
import sys
import time
CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # there was actually no data of chromatin state for chromosome Y
NUM_BP_PER_WINDOW = 1000000
NUM_BP_PER_BIN = 200
NUM_BIN_PER_WINDOW = int(NUM_BP_PER_WINDOW / NUM_BP_PER_BIN)

class argument_log(object):
	def __init__(self, command, output_folder, log_prefix):
		self.command = command
		self.output_folder = output_folder
		self.log_prefix = log_prefix
	def write_log(self):
		output_fn = os.path.join(self.output_folder, 'logger_{}.txt'.format(self.log_prefix))
		outF = open(output_fn, 'w')
		outF.write('\n'.join(self.command))
		outF.close()
		return 

def make_dir(directory):
	try:
		os.makedirs(directory)
	except:
		print ( 'Folder' + directory + ' is already created')



def check_file_exist(fn):
	if not os.path.isfile(fn):
		print ( "File: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 

def check_dir_exist(fn):
	if not os.path.isdir(fn):
		print ( "Directory: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 
	
def create_folder_for_file(fn):
	last_slash_index = fn.rfind('/')
	if last_slash_index != -1: # path contains folder
		make_dir(fn[:last_slash_index])
	return 

def get_command_line_integer(arg):
	try: 
		arg = int(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID")
		exit(1)

def get_command_line_int_with_default(arg, default_value):
	try: 
		arg = int(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID. Changing to default value {}".format(default_value))
		return default_value

def get_command_line_float(arg):
	try: 
		arg = float(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID")
		exit(1)

		
def get_enrichment_df (enrichment_fn): # enrichment_fn follows the format of ChromHMM OverlapEnrichment's format
	enrichment_df = pd.read_csv(enrichment_fn, sep = "\t")
	# rename the org_enrichment_df so that it's easier to work with
	enrichment_df =	enrichment_df.rename(columns = {"state (Emission order)": "state", "Genome %": "percent_in_genome"})
	return enrichment_df

def get_non_coding_enrichment_df (non_coding_enrichment_fn):
	nc_enrichment_df = pd.read_csv(non_coding_enrichment_fn, sep = '\t')
	if len(nc_enrichment_df.columns) != 3:
		print ( "Number of columns in a non_coding_enrichment_fn should be 3. The provided file has " + str(len(nc_enrichment_df.columns)) + " columns.")
		print ( "Exiting, from ChromHMM_untilities_common_functions_helper.py")
		exit(1)
	# Now, we know that the nc_enrichment_df has exactly 3 columns
	# change the column names
	nc_enrichment_df.columns = ["state", "percent_in_genome", "non_coding"]
	return (nc_enrichment_df)