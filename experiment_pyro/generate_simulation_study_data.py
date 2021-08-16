import os
import numpy as np
import pandas as pd 
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from tqdm import tqdm
import sys
import helper
def write_alpha_or_pi(output_folder, vector, alpha_or_pi):
	num_ref_epig = len(vector)
	vector = pd.Series(vector)
	vector.index = list(map(lambda x: 'Z{}'.format(x), range(num_ref_epig))) 
	save_fn = os.path.join(output_folder, '{}.txt'.format(alpha_or_pi))
	vector.to_csv(save_fn, header = False, index = True, sep = '\t')

def write_ground_truth_data(output_folder, alpha, pi, ref_state_df, emission_df, transition_mat, S, mark_data):
	num_ref_epig = len(pi)
	num_state = emission_df.shape[0]
	num_mark = emission_df.shape[1]
	mark_name_list = list(map(lambda x: 'M{}'.format(x), range(num_mark)))
	state_name_list = list(map(lambda x: 'S{}'.format(x), range(num_state)))
	# 1. Write pi and alpha
	write_alpha_or_pi(output_folder, pi, 'pi')
	write_alpha_or_pi(output_folder, alpha, 'alpha')
	# 2. Write ref_state_df 
	ref_state_df = pd.DataFrame(ref_state_df)
	ref_state_df.columns = list(map(lambda x: 'Z{}'.format(x), range(num_ref_epig)))
	ref_state_save_fn = os.path.join(output_folder, 'ref_state_df.txt.gz')
	ref_state_df.to_csv(ref_state_save_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	# 3. Write mark data
	mark_data = pd.DataFrame(mark_data, columns = mark_name_list)
	mark_save_fn = os.path.join(output_folder, 'mark_data.txt.gz')
	mark_data.to_csv(mark_save_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	# 4. write emission_df
	emission_df = pd.DataFrame(emission_df, columns = mark_name_list, index = state_name_list)
	emission_fn = os.path.join(output_folder, 'emission.txt')
	emission_df.to_csv(emission_fn, header = True, index = True, sep = '\t')
	# 5. write transition_mat
	transition_mat = pd.DataFrame(transition_mat, columns = state_name_list, index = state_name_list)
	transition_mat = transition_mat.applymap(lambda x: x.item()) # convert from tensor(0.1) to 0.1	
	transition_mat_fn = os.path.join(output_folder, 'beta_state_transition.txt')
	transition_mat.to_csv(transition_mat_fn, header = True, index = True, sep = '\t')
	# 6. write state
	S = pd.Series(S).astype(int).apply(lambda x: 'S{}'.format(x))
	S_fn = os.path.join(output_folder, 'state.txt.gz')
	S.to_csv(S_fn, header = False, index = False, compression = 'gzip')
	return

def generate_data(num_state, num_ref_epig, num_mark, num_obs, output_folder): 
	### 1. Generate mixture parameters
	alpha = np.random.uniform(1, 6, num_ref_epig) # generate randome alpha as prior to generate pi later. No strong rationales on how to generate this
	pi = pyro.sample('pi', dist.Dirichlet(torch.tensor(alpha)))
	### 2. Generate random states in reference epigenomes
	ref_state_df = np.random.choice(num_state, size = num_obs * num_ref_epig, replace = True).reshape(num_obs, num_ref_epig) # observed data of the reference epig's state maps in N positions. Rows; positions, columns: ref. epig. This dataframe should have dimension num_obs * num_ref_epig
	### 3. Generate the emission matrix
	emission_df = np.random.uniform(0,1, num_state * num_mark).reshape(num_state, num_mark)
	### 4. Generate the beta transition matrix
	transition_mat = torch.zeros((num_state, num_state))
	for i in tqdm(pyro.plate('state_loop', num_state)):
		transition_mat[i,:] = pyro.param('beta_{}'.format(i), torch.randn(num_state).exp(), constraint = constraints.simplex) # beta_<state_index> : a vector of probability that sum up to one --> probability of being from one state in the reference epig to being an another state in the sample of interest. We name each of the parameter as beta_<state_index>
	### 5. Generate the hidden data Z, S, observed data of marks' binarized signals across the genome
	mark_data = np.zeros((num_obs, num_mark)) # a list of N numbers, each can be 0 or 1 representing the presence/absence call of the 1 chormatin mark at this position. For this toy example, we only assume that there is only one chromatin mark data at each genomic position. Right now we initialize all to 0 and we will fill it up later for the sake of 
	Z = torch.zeros(num_obs) # initilize an array of length N, each will correspond to the zero-based index of ref_epig chosen at each position
	S = torch.zeros(num_obs) # initilize an array of length N, each will correspond to the zero-based index of hidden state at each position
	for i in pyro.plate('genome_loop', num_obs):
		Z[i] = (pyro.sample('z_{}'.format(i), dist.Categorical(pi))).type(torch.long) # sample reference epig from pi at this genomic position 
		R_i = ref_state_df[i,int(Z[i])] # index of state that is observed at the pick refernece epigenome at the current position
		S[i] = pyro.sample('S_{}'.format(i), dist.Categorical(pyro.param('beta_{}'.format(R_i)))).type(torch.long) # We can get access to parameters by just using pyro.param('<param_name>')
		for j in pyro.plate('mark_loop', num_mark):
			mark_data[i,j] = pyro.sample('M_{}_{}'.format(i,j), dist.Bernoulli(emission_df[S[i].type(torch.long),j])).item()
	write_ground_truth_data(output_folder, alpha, pi, ref_state_df, emission_df, transition_mat, S, mark_data)
	return 

def generate_tiny_toy_data(output_folder):
	num_ref_epig = 5
	num_state = 3
	num_obs = 1000000
	alpha = np.random.uniform(1, 6, num_ref_epig)
	pi = pyro.sample('pi', dist.Dirichlet(torch.tensor(alpha)))
	ref_state_df = np.random.choice(num_state, size = num_obs * num_ref_epig, replace = True).reshape(num_obs, num_ref_epig)
	emission_df = np.array([[0.1,0.1,0.8], [0.2,0.1,0.7], [0.3, 0.1,0.6]])
	transition_mat = np.array([[0.8,0.1,0.1], [0.3,0.5,0.2], [0.2,0.2,0.6]])
	mark_data = np.zeros((num_obs, num_mark))
	Z = torch.zeros(num_obs)
	S = torch.zeros(num_obs)

def main():
	if len(sys.argv) != 6:
		usage()
	num_state = helper.get_command_line_integer(sys.argv[1])
	num_ref_epig = helper.get_command_line_integer(sys.argv[2])
	num_mark = helper.get_command_line_integer(sys.argv[3])
	num_obs = helper.get_command_line_integer(sys.argv[4])
	output_folder = sys.argv[5]
	helper.make_dir(output_folder)
	logger = helper.argument_log(sys.argv, output_folder, 'generate_simData') # to save the command line arguments that led to this 
	logger.write_log()
	print('Done getting command line arguments')
	generate_data(num_state, num_ref_epig, num_mark, num_obs, output_folder)
	print('Done!')

def usage():
	print ("python generate_simulation_data.py")
	print ('num_state')
	print ('num_ref_epig')
	print ('num_mark')
	print ('num_obs')
	print ('output_folder')
	exit(1)

main()