import argparse
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
import itertools
from scipy import stats
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

def generate_tiny_toy_data(num_obs):
	num_ref_epig = 5
	num_state = 3
	num_mark = 3
	alpha = np.array([9, 0.5, 0.5])
	pi = torch.tensor(alpha / np.sum(alpha)) # expectation of Dirichlet distribution with parameter alpha
	# now we sample the chromatin state in reference epigenome
	sample_state_probability = np.eye(num_state) - 0.075 * np.eye(num_state) + 0.025 # a matrix with 0.95 on the diagonal and 0.025 on the off-diag. Each row is the prob of sampling from the corresponding (column) state in the corresponding (row) ref epig. 
	ref_state_np = np.empty((num_obs, num_ref_epig))
	for i in range(num_ref_epig):
		main_state_in_this_ref_epig = np.random.randint(num_state)
		ref_state_np[:,i] = np.random.choice(np.arange(num_state), num_obs, replace = True, p = sample_state_probability[:,main_state_in_this_ref_epig])
	ref_state_np = ref_state_np.astype(int)
	# point of samling ref_state_np is that we will allow each ref_epig to be mostly annotated as a state, with a prob of 0.05 of being from another random state rather than from the main state
	# now we let the emission probabilty to also be most one mark per state
	emission_np = sample_state_probability.copy() # this is because num_state = num_mark in our case
	emission_df = pd.DataFrame(emission_np, columns = list(map(lambda x: 'M{}'.format(x), range(emission_np.shape[1])))) # columns: M0 --> M...
	transition_mat = torch.eye(num_state)
	mark_data = np.zeros((num_obs, num_mark))
	Z = torch.zeros(num_obs)
	S = torch.zeros(num_obs)
	for i in pyro.plate('genome_loop', num_obs):
		Z[i] = (pyro.sample('z_{}'.format(i), dist.Categorical(pi))).type(torch.long) # sample reference epig from pi at this genomic position 
		R_i = ref_state_np[i,int(Z[i])] # index of state that is observed at the pick refernece epigenome at the current position
		S[i] = pyro.sample('S_{}'.format(i), dist.Categorical(transition_mat[R_i,:])) # We can get access to parameters by just using pyro.param('<param_name>')
		for j in pyro.plate('mark_loop', num_mark):
			mark_data[i,j] = pyro.sample('M_{}_{}'.format(i,j), dist.Bernoulli(emission_np[S[i].type(torch.long),j])).item()
	mark_data = pd.DataFrame(mark_data, columns = emission_df.columns)
	return alpha, pi, ref_state_np, emission_df, transition_mat, mark_data

# @pyro.infer.config_enumerate : this is never needed because we it is only used when all the hidden variables are discrete. In our case, pi is not discrete.
def model(alpha, transformed_emission_tt, ref_state_np, transformed_mark_data, num_obs, num_state, NUM_BINS_SAMPLE_PER_ITER): 
	num_ct = len(alpha)
	pi = pyro.sample('pi', dist.Dirichlet(alpha)) # sample mixture probabilities of reference epigenome
	for i in pyro.plate('state_loop', num_state):
		trans_from_state = pyro.param('beta_{}'.format(i), torch.randn(num_state).exp(), constraint = constraints.simplex) 
	for i in pyro.plate('genome_loop', num_obs):
		z_i = pyro.sample('z_{}'.format(i), dist.Categorical(pi))
		R_i = ref_state_np[i,z_i] 
		S_i = pyro.sample('S_{}'.format(i), dist.Categorical(pyro.param('beta_{}'.format(R_i)))) # We can get access to parameters by just using pyro.param('<param_name>')
		pyro.sample('M_{}'.format(i), dist.Categorical(transformed_emission_tt[S_i.type(torch.long)]), obs = transformed_mark_data[i])


def guide(alpha, transformed_emission_tt, ref_state_np, transformed_mark_data, num_obs, num_state, NUM_BINS_SAMPLE_PER_ITER):
	# in this guide, we assume that pi and z are independent
	# transformed_mark_data: a 1D tensor, each position corresponding to a genomic position
	num_ct = len(alpha)
	q_lambda = pyro.param('q_lambda', alpha, constraint = constraints.positive)
	pi = pyro.sample('pi', dist.Dirichlet(q_lambda))
	for i in pyro.plate('genome_loop', num_obs, subsample_size = NUM_BINS_SAMPLE_PER_ITER): # for subsampling, we only need to specify the subsampling in guide function, not in the model function
		z_probs = pyro.param("q_z_{}".format(i), torch.randn(num_ct).exp(), constraint=constraints.simplex) 
		# i added .exp() as suggested by https://www.programcreek.com/python/example/123171/torch.distributions.constraints.positive, constraints.simplex is to guarantee that they sum up to 1, based on https://pytorch.org/docs/stable/distributions.html (search for simplex in this page)
		z_i = pyro.sample('z_{}'.format(i), dist.Categorical(z_probs))
		R_i = (ref_state_np[i,z_i]).astype(int) # Ha also checked that when doing subsampling, the model still got the exact data as expected
		state_probs = pyro.param('q_s_{}'.format(i), torch.randn(num_state).exp(), constraint=constraints.simplex) 
		pyro.sample('S_{}'.format(i), dist.Categorical(state_probs))

def train(alpha, ref_state_np, transformed_mark_data, transformed_emission_tt, num_state, num_obs, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER):
	pyro.clear_param_store()
	loss_func = pyro.infer.TraceGraph_ELBO(max_plate_nesting=1)
	svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=loss_func)
	losses = []
	for _ in tqdm(range(NUM_TRAIN_ITERATIONS)):
		loss = svi.step(alpha, transformed_emission_tt, ref_state_np, transformed_mark_data, num_obs, num_state, NUM_BINS_SAMPLE_PER_ITER)
		losses.append(loss)
	posterior_params = {k: np.array(v.data) for k, v in pyro.get_param_store().items()}
	return posterior_params

def read_chrom_mark_observed_signals(mark_data):
	chrom_mark_list = mark_data.columns
	mark_data = mark_data.apply(lambda x: x.astype(int).astype(str), axis = 0) # convert the data from 0.0, 1.0 to 0 and 1 integers
	mark_data['combined_obs_int'] = mark_data.apply(lambda x: int(''.join(x), 2), axis = 1) # apply function to each row
	transformed_mark_data = torch.tensor(mark_data['combined_obs_int'].values) # 1D tensor each element is the observed data at each postion. If we have 3 marks, the the observed values can be 0-7.
	return transformed_mark_data, chrom_mark_list 

def calculate_join_emission_multiple_marks(row, binary_tuple, chrom_mark_list):
	# this will process each row in the emission matrix (each state)
	# binary tuple will be a tuple of length #num_mark, each element in the tuple is 0/1 --> presence/absence call of chromatin mark. The order of chromatin marks will be given in chrom_mark_list. Ex: binary_tuple = (0,0,1), chrom_mark_list = [m1, m2, m3] --> m3 is present and others are absent. This function will return the probability of observing binary_tuple given each of the state. 
	# function tested on 08/03/2021
	base = row[chrom_mark_list]
	exponent = pd.Series(binary_tuple, index = chrom_mark_list)
	return np.prod(base**exponent * (1-base)**(1-exponent))

def read_emission_matrix_into_categorical_prob(emission_df, chrom_mark_list):
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

def evaluate(alpha, pi, transition_mat, num_state, posterior_params, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER, output_fn):
	result_df = pd.DataFrame(columns = ['param_pred', 'metric', 'value'])
	print(alpha)
	q_lambda = posterior_params['q_lambda']	
	alpha_corr = stats.pearsonr(alpha, q_lambda) 
	alpha_diff = np.average(q_lambda) - np.average(alpha)
	# (correlation, p-value)
	print(posterior_params['q_lambda'])
	result_df.loc[result_df.shape[0]] = ['alpha_qLabmda', 'pearsonr', alpha_corr[0]]
	result_df.loc[result_df.shape[0]] = ['alpha_qLabmda', 'avg_pred_minus_real', alpha_diff]
	print("pi")
	print(pi)
	pred_p = q_lambda / q_lambda.sum() # expected valye of pi given q_lambda
	print(pred_p)
	result_df.loc[result_df.shape[0]] = ['pi_expLambda', 'pearsonr', stats.pearsonr(pi.numpy(), pred_p)[0]]
	print('beta')
	print(transition_mat)
	for i in range(num_state):
		pearsonr = stats.pearsonr((transition_mat[i,:]).numpy(), posterior_params['beta_{}'.format(i)])
		result_df.loc[result_df.shape[0]] = ['beta_'.format(i), 'pearsonr', pearsonr]
		print(posterior_params['beta_{}'.format(i)])
	result_df['NUM_TRAIN_ITERATIONS'] = NUM_TRAIN_ITERATIONS
	result_df['NUM_BINS_SAMPLE_PER_ITER'] = NUM_BINS_SAMPLE_PER_ITER
	result_df.to_csv(output_fn, header = True, index = False, sep = '\t')
	return 

def main(args):
	NUM_TRAIN_ITERATIONS = args.NUM_TRAIN_ITERATIONS
	NUM_BINS_SAMPLE_PER_ITER = args.NUM_BINS_SAMPLE_PER_ITER
	output_fn = args.output_fn
	helper.create_folder_for_file(output_fn)
	num_obs = 10000
	alpha, pi, ref_state_np, emission_df, transition_mat, mark_data = generate_tiny_toy_data(num_obs)
	# mark_data and emission_df are pd.DataFrame that share the same column names
	print ('Done generating data')
	alpha = torch.tensor(alpha) # to make it implementable for pyro
	num_state = emission_df.shape[0]
	transformed_mark_data, chrom_mark_list = read_chrom_mark_observed_signals(mark_data) 
	# transformed_mark_data: 1D tensor, each element is the observed data at each postion.
	# If we have 3 marks, the the observed values can be 0-7.
	transformed_emission_tt = read_emission_matrix_into_categorical_prob(emission_df, chrom_mark_list) # tested
	# 2D tensor with rows: states, columns: possible combinations of chromatin marks 
	print(transformed_emission_tt)
	posterior_params = train(alpha, ref_state_np, transformed_mark_data, transformed_emission_tt, num_state, num_obs, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER)
	evaluate(alpha, pi, transition_mat, num_state, posterior_params, NUM_TRAIN_ITERATIONS, NUM_BINS_SAMPLE_PER_ITER, output_fn)

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    parser = argparse.ArgumentParser(description="Tiny toy example")
    parser.add_argument("-n", "--NUM_TRAIN_ITERATIONS", default=4000, type=int)
    parser.add_argument("-b", "--NUM_BINS_SAMPLE_PER_ITER", default=1000, type=int)
    parser.add_argument('-out', '--output_fn', type = str)
    args = parser.parse_args()
    main(args)

