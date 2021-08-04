import os
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
def generate_data(N, num_state): # N: number of genomic positions, num_pos
	### first we define some toy parameters to the model
	num_mark = 1
	num_ref_epig = 3
	alpha = torch.tensor([0.3,0.3,0.4])
	ref_state_df = np.random.choice(num_state, size = N * num_ref_epig, replace = True).reshape(N, num_ref_epig)#np.array([[0,0,0],[1,0,1], [2,2,0], [1,1,1], [0,1,2]])  # observed data of the reference epig's state maps in N positions. Rows; positions, columns: ref. epig. This dataframe should have dimension N * num_ref_epig
	bernouli_mark = torch.tensor([0.1,0.9,0.5]) # probabilty of the one mark being present given each state. indices correspond to states. We are currently assuming there is only one mark as observed data.
	pi = pyro.sample('pi', dist.Dirichlet(alpha))
	transiton_mat = torch.zeros((num_state, num_state))
	for i in pyro.plate('state_loop', num_state):
		transiton_mat[i,:] = pyro.param('beta_{}'.format(i), torch.randn(num_state).exp(), constraint = constraints.simplex) # beta_<state_index> : a vector of probability that sum up to one --> probability of being from one state in the reference epig to being an another state in the sample of interest. We name each of the parameter as beta_<state_index>
	mark_data = torch.tensor(np.zeros(N)) # a list of N numbers, each can be 0 or 1 representing the presence/absence call of the 1 chormatin mark at this position. For this toy example, we only assume that there is only one chromatin mark data at each genomic position. Right now we initialize all to 0 and we will fill it up later for the sake of 
	Z = torch.zeros(N) # initilize an array of length N, each will correspond to the zero-based index of ref_epig chosen at each position
	S = torch.zeros(N) # initilize an array of length N, each will correspond to the zero-based index of hidden state at each position
	for i in pyro.plate('genome_loop', N):
		Z[i] = (pyro.sample('z_{}'.format(i), dist.Categorical(pi))).type(torch.long) # sample reference epig from pi at this genomic position 
		R_i = ref_state_df[i,int(Z[i])] # index of state that is observed at the pick refernece epigenome at the current position
		S[i] = pyro.sample('S_{}'.format(i), dist.Categorical(pyro.param('beta_{}'.format(R_i)))) # We can get access to parameters by just using pyro.param('<param_name>')
		mark_data[i] = pyro.sample('M_{}'.format(i), dist.Bernoulli(bernouli_mark[S[i].type(torch.long)]))
	return alpha, ref_state_df, bernouli_mark, pi, Z, S, transiton_mat, mark_data



def model(alpha, bernouli_mark, ref_state_df, obs, num_obs, num_state): # bernouli_mark: an array indexed by the state
	num_ct = len(alpha)
	pi = pyro.sample('pi', dist.Dirichlet(alpha))
	for i in pyro.plate('state_loop', num_state):
		trans_from_state = pyro.param('beta_{}'.format(i), torch.randn(num_state).exp(), constraint = constraints.simplex)
	for i in pyro.plate('genome_loop', num_obs):
		z_i = pyro.sample('z_{}'.format(i), dist.Categorical(pi))
		R_i = ref_state_df[i,z_i]
		S_i = pyro.sample('S_{}'.format(i), dist.Categorical(pyro.param('beta_{}'.format(R_i)))) # We can get access to parameters by just using pyro.param('<param_name>')
		pyro.sample('M_{}'.format(i), dist.Bernoulli(bernouli_mark[S_i.type(torch.long)]), obs = obs[i])


def guide(alpha, bernouli_mark, ref_state_df, obs, num_obs, num_state):
	num_ct = len(alpha)
	q_lambda = pyro.param('q_lambda', alpha, constraint = constraints.positive)
	pyro.sample('pi', dist.Dirichlet(q_lambda))
	for i in pyro.plate('genome_loop', num_obs, subsample_size = 10):
		z_probs = pyro.param("q_z_{}".format(i), torch.randn(num_ct).exp(), constraint=constraints.simplex) # i added .exp() as suggested by https://www.programcreek.com/python/example/123171/torch.distributions.constraints.positive, constraints.simplex is to guarantee that they sum up to 1, based on https://pytorch.org/docs/stable/distributions.html (search for simplex in this page)
		pyro.sample('z_{}'.format(i), dist.Categorical(z_probs))
		state_probs = pyro.param('q_s_{}'.format(i), torch.randn(num_state).exp(), constraint=constraints.simplex) 
		pyro.sample('S_{}'.format(i), dist.Categorical(state_probs))

def train(alpha, bernouli_mark, ref_state_df, obs, num_state):
	num_steps = 3
	pyro.clear_param_store()
	num_obs = ref_state_df.shape[0] # ref_state_df: rows: genome positions, columns: reference epig
	loss_func = pyro.infer.TraceGraph_ELBO(max_plate_nesting=1)
	svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=loss_func)
	losses = []
	for _ in tqdm(range(num_steps)):
		loss = svi.step(alpha, bernouli_mark, ref_state_df, obs, num_obs, num_state)
		losses.append(loss)
	posterior_params = {k: np.array(v.data) for k, v in pyro.get_param_store().items()}
	return posterior_params

def write_transition_matrix(posterior_params, output_folder, num_state):
	result_df = np.zeros((num_state, num_state))
	for i in range(num_state):
		result_df[i,:] = posterior_params['beta_{}'.format(i)]
	save_fn = os.path.join(output_folder, 'beta_state_transition.txt')
	result_df = pd.DataFrame(result_df)
	result_df.to_csv(save_fn, header = False, index = False, sep = '\t')
	return 

def write_pi(posterior_params, output_folder, ref_epig_name_list):
	posterior_pi = pd.Series(posterior_params['q_lambda'])
	posterior_pi.index = ref_epig_name_list # because this is how the original alpha (prior of pi) is constructed
	save_fn = os.path.join(output_folder, 'posterior_pi.txt')
	posterior_pi.to_csv(save_fn, header = False, index = True, sep = '\t')
	return

def main():
	num_obs = 100
	num_state = 3
	alpha, ref_state_df, bernouli_mark, pi, Z, S, transiton_mat, mark_data = generate_data(num_obs, num_state) # this will need to be changed so that we can get ref_state_df from pandas dataframes outside
	print()
	transiton_mat = torch.tensor(transiton_mat)
	posterior_params = train(alpha, bernouli_mark, ref_state_df, mark_data, num_state) # turns out, posterior_param is a dictionary of parameters, keys are the names of the parameters, values are the parameter values at the end of all iterations 
	print(posterior_params) 
	output_folder = './'
	write_transition_matrix(posterior_params, output_folder, num_state)
	write_pi(posterior_params, output_folder, ref_epig_name_list = np.arange(len(pi)))


main()
