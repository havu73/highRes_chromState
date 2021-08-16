import os
import sys
import helper
import itertools
import numpy as np
import pandas as pd 
import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from torch.distributions import constraints
from tqdm import tqdm
import torch.nn.functional as f
######## PARAMTERS RELATED TO THE TRAINING OF THE MODEL ########
DEFAULT_NUM_TRAIN_ITERATIONS = 5000
DEFAULT_NUM_BINS_SAMPLE_PER_ITER = 1000

######## FOLLOWING WE DEFINE A CLASS THAT WILL KEEP TRACK OF THE SAMPLE POSTERIOR DATA ########
class Posterior_pi_log(object):
	def __init__(self, ref_epig_name_list):
		self.ref_epig_name_list = ref_epig_name_list
		self.pi = torch.zeros(len(ref_epig_name_list))
	def update(self, updated_pi):
		self.pi = updated_pi
	def write_pi(self, output_folder):
		last_pi = pd.Series(self.pi.detach().numpy())
		last_pi.index = self.ref_epig_name_list
		save_fn = os.path.join(output_folder, 'posterior_pi.txt')
		last_pi.to_csv(save_fn, header = False, index = True, sep = '\t')
###########################################

######## FOLLOWING ARE FUNCTIONS TO PROCESS THE INPUT DATA FILES ########
def read_ref_state_df(ref_state_fn, ref_epig_name_list):
	ref_state_df = pd.read_csv(ref_state_fn, header = 0, index_col = None, sep = '\t')
	ref_state_df = ref_state_df[ref_epig_name_list] # select only columns corresponding to the reference epigenomes that are input to the model to predict the chromatin state maps in sample of interests. Usually, we  use all the reference epigenomes available (127), but as of right now 08062021 this may not be the case. ref_epige_name_list is from reading alpha file.
	ref_state_tt = np.array(ref_state_df.values) # tt: torch_tensor, rows are different genomic positions, columns are the ref_epig_name_list. values are 0-based state at each position for each ref_epig. The columns are ordered as in ref_epig_name_list, so the resulting alpha (prior for ref_epig similarity) should have the same order. 
	return ref_state_tt

def read_chrom_mark_observed_signals(chrom_mark_binarized_fn):
	chrom_signal_df = pd.read_csv(chrom_mark_binarized_fn, header = 0, index_col = None, sep = '\t')
	chrom_mark_list = chrom_signal_df.columns
	chrom_signal_df = chrom_signal_df.apply(lambda x: x.astype(int).astype(str), axis = 0) # convert the data from 0.0, 1.0 to 0 and 1 integers
	chrom_signal_df['combined_obs_int'] = chrom_signal_df.apply(lambda x: int(''.join(x), 2), axis = 1) # apply function to each row
	obs_signal_tt = torch.tensor(chrom_signal_df['combined_obs_int'].values) # 1D array each element is the observed data at each postion. If we have 3 marks, the the observed values can be 0-7.
	return obs_signal_tt, chrom_mark_list 

def calculate_join_emission_multiple_marks(row, binary_tuple, chrom_mark_list):
	# this will process each row in the emission matrix (each state)
	# binary tuple will be a tuple of length #num_mark, each element in the tuple is 0/1 --> presence/absence call of chromatin mark. The order of chromatin marks will be given in chrom_mark_list. Ex: binary_tuple = (0,0,1), chrom_mark_list = [m1, m2, m3] --> m3 is present and others are absent. This function will return the probability of observing binary_tuple given each of the state. 
	# function tested on 08/03/2021
	base = row[chrom_mark_list]
	exponent = pd.Series(binary_tuple, index = chrom_mark_list)
	return np.prod(base**exponent * (1-base)**(1-exponent))

def read_emission_matrix_into_categorical_prob(emission_fn, chrom_mark_list):
	emission_df = pd.read_csv(emission_fn, header = 0,index_col = 0, sep = '\t') # row indices are states 
	all_possible_obs_marks = list(itertools.product(range(2), repeat = len(chrom_mark_list))) # list of tuples, each of length # num_marks --> all possible observations of marks 
	all_possible_obs_marks_str = list(map(lambda x: ''.join(list(map(str, x))), all_possible_obs_marks)) # convert (0,0,0) --> '000'
	for obs_pattern in all_possible_obs_marks:
		obs_string = ''.join(list(map(str, obs_pattern)))
		emission_df[obs_string] = emission_df.apply(lambda x: calculate_join_emission_multiple_marks(x, obs_pattern, chrom_mark_list), axis = 1) # apply function to each row
	result_df = emission_df[all_possible_obs_marks_str].copy() # columns are all the possible chromatin mark sequences for the chrom_mark_list. Right now, we assume that the assays being profiled are a subset of the 12 marks in the 25-state roadmap model, We can care about the case where the profiled marks for sample of interest are not among the 12 marks later.
	result_df.columns = list(map(lambda x: int(x, 2), result_df.columns))
	result_df = result_df[np.arange(len(all_possible_obs_marks))] # rearrange so that if # marks = 3 --> columns will be 0 --> 7, correpsonding to the 8 possible combination of observed marks 000 --> 111
	result_df = torch.tensor(result_df.values) # tensor with rows: states, columns: possible combinations of chromatin marks 
	return result_df

def read_alpha(prior_alpha_fn):
	alpha = pd.read_csv(prior_alpha_fn, header = None, index_col = 0, squeeze = True, sep = '\t') # --> a pd Series with index: ref_epig_id (E001 --> E129). Values: alpha values
	ref_epig_name_list = alpha.index
	return torch.tensor(alpha), ref_epig_name_list

def read_beta(beta_fn):
	beta = pd.read_csv(beta_fn, header = 0, index_col = 0, sep = '\t') # this is assuming that the states are order 0 --> num_state-1
	beta = torch.tensor(beta.values)
	return beta
###########################################

##### BELOW ARE FUNCTIONS TO TRAIN PYRO MODEL #########
@pyro.infer.config_enumerate
def model(alpha, state_emission_tt, ref_state_tt, obs_signal_tt, pi_log, num_obs, num_state, beta_tt, NUM_BINS_SAMPLE_PER_ITER): 
	num_ct = len(alpha)
	pi = pyro.sample('pi', dist.Dirichlet(alpha)) # sample mixture probabilities of reference epigenome
	for i in pyro.plate('genome_loop', num_obs):
		z_i = pyro.sample('z_{}'.format(i), dist.Categorical(pi))
		R_i = (ref_state_tt[i,z_i]).astype(int) # Ha also checked that when doing subsampling, the model still got the exact data as expected
		S_i = pyro.sample('S_{}'.format(i), dist.Categorical(beta_tt[R_i,:])) # We can get access to parameters by just using pyro.param('<param_name>')
		pyro.sample('M_{}'.format(i), dist.Categorical(state_emission_tt[S_i.type(torch.long)]), obs = obs_signal_tt[i])


def guide(alpha, state_emission_tt, ref_state_tt, obs_signal_tt, pi_log, num_obs, num_state, beta_tt, NUM_BINS_SAMPLE_PER_ITER):
	num_ct = len(alpha)
	q_lambda = pyro.param('q_lambda', alpha, constraint = constraints.positive)
	pi = pyro.sample('pi', dist.Dirichlet(q_lambda))
	pi_log.update(pi) # to keep track of the pi values that we sampled. This will be used later to store as the actual mixture of reference epigenomes. 
	for i in pyro.plate('genome_loop', num_obs, subsample_size = NUM_BINS_SAMPLE_PER_ITER): # for subsampling, we only need to specify the subsampling in guide function, not in the model function
		z_probs = pyro.param("q_z_{}".format(i), torch.randn(num_ct).exp(), constraint=constraints.simplex) # i added .exp() as suggested by https://www.programcreek.com/python/example/123171/torch.distributions.constraints.positive, constraints.simplex is to guarantee that they sum up to 1, based on https://pytorch.org/docs/stable/distributions.html (search for simplex in this page)
		z_i = pyro.sample('z_{}'.format(i), dist.Categorical(z_probs))
		R_i = (ref_state_tt[i,z_i]).astype(int) # Ha also checked that when doing subsampling, the model still got the exact data as expected
		S_i = pyro.sample('S_{}'.format(i), dist.Categorical(beta_tt[R_i,:])) # We can get access to parameters by just using pyro.param('<param_name>')
		# state_probs = pyro.param('q_s_{}'.format(i), torch.randn(num_state).exp(), constraint=constraints.simplex) 
		# pyro.sample('S_{}'.format(i), dist.Categorical(state_probs))

def train(alpha, ref_state_tt, ref_epig_name_list, obs_signal_tt, chrom_mark_list, state_emission_tt, num_state, num_obs, output_folder, beta_tt, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER):
	pyro.clear_param_store()
	pi_log = Posterior_pi_log(ref_epig_name_list)
	loss_func = pyro.infer.TraceGraph_ELBO(max_plate_nesting=1)
	svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=loss_func)
	losses = []
	for _ in tqdm(range(NUM_TRAIN_ITERATIONS)):
		loss = svi.step(alpha, state_emission_tt, ref_state_tt, obs_signal_tt, pi_log, num_obs, num_state, beta_tt, NUM_BINS_SAMPLE_PER_ITER)
		losses.append(loss)
	pi_log.write_pi(output_folder) # update the pi vector after all training iterations
	posterior_params = {k: np.array(v.data) for k, v in pyro.get_param_store().items()}
	return posterior_params

###########################################

##### BELOW ARE FUNCTIONS TO WRITE MODEL PARAMETERS #########
def write_transition_matrix(posterior_params, output_folder, num_state):
	result_df = np.zeros((num_state, num_state))
	for i in range(num_state):
		result_df[i,:] = posterior_params['beta_{}'.format(i)]
	save_fn = os.path.join(output_folder, 'beta_state_transition.txt')
	result_df = pd.DataFrame(result_df)
	result_df.columns = list(map(lambda x: 'S{}'.format(x), range(num_state)))
	result_df.index = list(map(lambda x: 'S{}'.format(x), range(num_state)))
	result_df.to_csv(save_fn, header = True, index = True, sep = '\t')
	return 

def write_alpha(posterior_params, output_folder, ref_epig_name_list):
	print(posterior_params)
	result_df = np.zeros(len(ref_epig_name_list))
	q_lambda = pd.Series(posterior_params['q_lambda'])
	q_lambda.index = ref_epig_name_list
	save_fn = os.path.join(output_folder, 'q_lambda.txt')
	q_lambda.to_csv(save_fn, header = False, index = True, sep = '\t')
	return 
###########################################

def main():
	if len(sys.argv) < 7:
		usage()
	ref_state_fn = sys.argv[1]
	helper.check_file_exist(ref_state_fn)
	chrom_mark_binarized_fn = sys.argv[2]
	helper.check_file_exist(chrom_mark_binarized_fn)
	prior_alpha_fn = sys.argv[3]
	helper.check_file_exist(prior_alpha_fn)
	emission_fn = sys.argv[4]
	helper.check_file_exist(emission_fn)
	beta_fn = sys.argv[5]
	helper.check_file_exist(beta_fn)
	output_folder = sys.argv[6]
	helper.make_dir(output_folder)
	# the following two lines of code is a little wrong. to fix: change to using argparse later
	NUM_TRAIN_ITERATIONS = helper.get_command_line_int_with_default(sys.argv[7], DEFAULT_NUM_TRAIN_ITERATIONS)
	NUM_BINS_SAMPLE_PER_ITER = helper.get_command_line_int_with_default(sys.argv[8], DEFAULT_NUM_TRAIN_ITERATIONS)
	logger = helper.argument_log(sys.argv, output_folder, 'train_toy_model') # to save the command line arguments that led to this 
	logger.write_log()
	print("Done getting command line arguments")
	# 1. Take in the input data
	alpha, ref_epig_name_list = read_alpha(prior_alpha_fn) # tensor 1D each element is the alpha for the corresponding ref_epig. Ordered similarly to ref_epig_name_list
	print('Done reading alpha')
	beta_tt = read_beta(beta_fn) 
	print('Done reading beta')
	ref_state_tt = read_ref_state_df(ref_state_fn, ref_epig_name_list) # tt: torch_tensor, rows are different genomic positions, columns are the ref_epig_name_list. values are 0-based state at each position for each ref_epig. The columns are ordered as in ref_epig_name_list, so the resulting alpha (prior for ref_epig similarity) should have the same order. 
	print("Done reading ref_state_df")
	obs_signal_tt, chrom_mark_list = read_chrom_mark_observed_signals(chrom_mark_binarized_fn) # chrom_mark_list is needed so that we can figure out what each of the value in observed_signal_tt mean --> needed for calculation of the emission probabilities for different combination of observations
	state_emission_tt = read_emission_matrix_into_categorical_prob(emission_fn, chrom_mark_list) # a tensor that's state_emission_tt[state_index] --> a vector showing the probabiliities of observing the sequence of marks, given state state_index. state_emission_tt.sum(axis = 1) --> a vector of 1.0
	print('Done reading emission_df')
	# 2. Train the model
	num_state = state_emission_tt.shape[0]
	num_obs = ref_state_tt.shape[0] # number of genomic position
	num_ref_epig = 	len(ref_epig_name_list)
	posterior_params = train(alpha, ref_state_tt, ref_epig_name_list, obs_signal_tt, chrom_mark_list, state_emission_tt, num_state, num_obs, output_folder, beta_tt, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER)
	# 3. Write output
	# write_transition_matrix(posterior_params, output_folder, num_state)
	write_alpha(posterior_params, output_folder, ref_epig_name_list)
	print ('Done!')

def usage():
	print ("python toy_model_chrom_mark_model.py")
	print ("ref_state_fn: the fn of observed chromatin states in reference epigenomes. Headers: <ref_epig_id>: Z1 --> Z<num_ref_epig in generate_simulation_study_data.py>. Format for chromatin states in this file will be 0-based state index")
	print ("chrom_mark_binarized_fn: the fn where we store observed binarized chromatin mark signals in sample of interest. The genomic position will also align with positions in ref_state_fn. Headers: <mark_name: in simulation data, M0--> M<something>>. Each row corresponds to a genomic position")
	print ('prior_alpha_fn: prior parameters for the mixture probabilities (pi). Each line corresponds to a ref_epig_id in ref_state_fn. Format of each line: <ref_epig_id><tab><alpha value>')
	print ('emission_fn: emission probabilities of marks given states. This file should be the ground-truth generated by generate_simulation_study_data.py. Row indces are S0 --> S<num_state-1>, column names are M0 --> M<num_mark-1>')
	print ('beta_fn: beta matrix that we keep from the data generation process')
	print ('output_folder: where we will have the output to this program')
	print ('NUM_TRAIN_ITERATIONS: default {}'.format(DEFAULT_NUM_TRAIN_ITERATIONS))
	print ('NUM_BINS_SAMPLE_PER_ITER: default {}'.format(DEFAULT_NUM_BINS_SAMPLE_PER_ITER))
	exit(1)

main()