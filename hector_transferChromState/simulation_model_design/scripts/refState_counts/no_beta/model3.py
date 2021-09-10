import pyro
import pyro.distributions as dist
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, num_signals, num_states, num_groups, hidden_sig, hidden_ref, hidden_comb, dropout):
		super().__init__()
		self.drop = nn.Dropout(dropout)
		self.fc1 = nn.Linear(num_signals, hidden_sig)
		self.fc2 = nn.Linear(num_states*num_groups, hidden_ref)
		self.fc_comb = nn.Linear(hidden_sig+hidden_ref, hidden_comb)
		self.fcmu = nn.Linear(hidden_comb, num_states)
		self.fclv = nn.Linear(hidden_comb, num_states)

	def forward(self, m, r):
		r = r.reshape(r.shape[0], -1) # flatten from bins,groups,states --> bins, groups*states
		h1 = F.softplus(self.fc1(m)) # hidden_sig
		h2 = F.softplus(self.fc2(r)) # hidden_ref
		h2 = self.drop(h2)
		h = torch.cat((h1, h2), 1) # concat to increase # columns # hidden_sig + hidden_ref
		h = F.softplus(self.fc_comb(h)) # hidden_comb
		logpi_loc = self.fcmu(h) # num_states
		logpi_logvar = self.fclv(h) # num_states
		logpi_scale = (0.5 * logpi_logvar).exp()
		return logpi_loc, logpi_scale

class Decoder(nn.Module):
	def __init__(self, num_states, num_signals, num_groups, hidden_sig, hidden_ref, hidden_comb, dropout):
		super().__init__()
		self.num_states = num_states
		self.num_signals = num_signals
		self.num_groups = num_groups
		self.drop = nn.Dropout(dropout)
		self.fcih = nn.Linear(num_states, hidden_comb) # input (state probabilities) --> hidden
		self.fchh_sig = nn.Linear(hidden_comb, hidden_sig) # hiddent --> hidden
		self.fchs_sig = nn.Linear(hidden_sig, self.num_signals) # hidden --> signals
		self.fchh_ref = nn.Linear(hidden_comb, hidden_ref) # hidden --> hidden_ref
		self.fchr_ref = nn.Linear(hidden_ref, self.num_groups * self.num_states)


	def forward(self, inputs):
		# takes in the values of collapsed pi: probabilities of state 
		# assignments at each positions, and then apply a linear trans
		# to get the probabilities of observing signals at each position
		# --> vector size #signals
		# used as parameters for bernoulli dist. to get obs. signals
		# create multiple layers
		# inputs: bins, state probabilities
		# h: bins, hidden 
		# signal_param: bins, signals
		# ref_param: bins, num_groups, num_states
		h = F.softplus(self.fcih(inputs)) # states --> hidden element vector
		h = self.drop(h)
		h_sig = F.softplus(self.fchh_sig(h)) # --> hidden --> hidden_sign
		signal_param = torch.sigmoid(self.fchs_sig(h_sig)) # hidden_sig --> marks
		h_ref = F.softplus(self.fchh_ref(h)) # hidden_comb --> hidden_ref
		ref_param = torch.sigmoid(self.fchr_ref(h_ref))
		ref_param = ref_param.reshape((ref_param.shape[0], self.num_groups, self.num_states)) 
		# hidden --> num_ref*num_states
		ref_param = F.normalize(ref_param, p = 1.0, dim = 2, eps=1e-6) # row normalize, sum over states per ref is 1
		return signal_param, ref_param

class Model_three(nn.Module):
	def __init__(self, num_signals, num_groups, num_ref_per_groups, num_states, hidden_sig, hidden_ref, hidden_comb, dropout):
		super().__init__()
		self.num_signals = num_signals
		self.num_groups = num_groups
		self.num_ref_per_groups = num_ref_per_groups
		self.num_states = num_states
		self.hidden_sig = hidden_sig
		self.hidden_ref = hidden_ref
		self.hidden_comb = hidden_comb
		self.dropout = dropout
		self.encoder = Encoder(num_signals, num_states, num_groups, hidden_sig, hidden_ref, hidden_comb, dropout)
		self.decoder = Decoder(num_states, num_signals, num_groups, hidden_sig, hidden_ref, hidden_comb, dropout)

	# shapes: 
	#  m: (bins x signals) signal matrix
	#  r: (bins x reference x state) indicator matrix
	def model(self, m, r):
		# flatten out the r indicator matrix
		pyro.module("decoder", self.decoder)
		with pyro.plate('bins', m.shape[0]):
			logCpi_loc = m.new_zeros((m.shape[0], self.num_states))
			logCpi_scale = m.new_ones((m.shape[0], self.num_states))
			logCpi = pyro.sample('log_collapsedPi', dist.Normal(logCpi_loc, logCpi_scale).to_event(1))
			Cpi = F.softmax(logCpi, -1) 
			# the softmax function should be used here because Cpi is lognormal
			signal_param, ref_param = self.decoder(Cpi) # vector of probabilities. 
			# signal_param: bins, signals
			# ref_param: bins, references, states
			# first num_signals elements: bernoulli params
			# each of the following num_states elements: multinomial params
			# for the state segmentation in a reference    
			pyro.sample('m', dist.Bernoulli(signal_param).to_event(1), obs=m)
			# plate across references
			with pyro.plate('refs', self.num_groups):
				try:
					pyro.sample('r', dist.Multinomial(self.num_ref_per_groups, ref_param).to_event(1), obs = r)
					# multinomial for non-homogeneous total_counts are not supported yet
				except:
					print(ref_param)
	            	
	def guide(self, m, r):
		pyro.module("encoder", self.encoder)
		with pyro.plate('bins', m.shape[0]):
			logpi_loc, logpi_scale = self.encoder(m, r)
			try:
				logpi = pyro.sample('log_collapsedPi', dist.Normal(logpi_loc, logpi_scale).to_event(1))
			except:
				torch.save(ref_param, 'problematic_logpi_loc.txt')
				exit(1)

	def predict_state_assignment(self, m, r):
		logpi_loc, logpi_scale = self.encoder(m, r)
		Cpi = F.softmax(logpi_loc, -1)
		return(Cpi)

	def write_predicted_state_assignment(self, m, r, output_fn):
		Cpi = self.predict_state_assignment(m, r)
		df = pd.DataFrame(Cpi.detach().numpy())
		df['max_state'] = df.idxmax(axis = 1)
		df.to_csv(output_fn, header = True, index = False, sep = '\t', compression = 'gzip')
		return

	def generate_reconstructed_data(self, m, r):
		logpi_loc, logpi_scale = self.encoder(m, r)
		Cpi = F.softmax(logpi_loc, -1)
		signal_param, ref_param = self.decoder(Cpi) # vector of probabilities. 
		re_m = pyro.sample('re_m', dist.Bernoulli(signal_param).to_event(1))
		re_r = pyro.sample('re_r', dist.Multinomial(1, ref_param).to_event(1))
		return(re_m, re_r)


	def get_percentage_correct_reconstruct(self, m, r):
		# m and r can be different from the m and r used in training
		re_m, re_r = self.generate_reconstructed_data(m,r)
		# re_r: bins, groups, states --> counts
		total_m_entries = re_m.shape[0] * re_m.shape[1]
		signals_CR = (re_m==m).sum() # correct reconstruct entries of signals
		total_r_entries = re_r.shape[0] * self.num_groups
		# for each reference at each position, if the state assignment is different between re_r and r, there are 2 out of num_states entries that are different between re_r and r
		wrong_r = torch.abs(re_r - r)/(self.num_ref_per_groups*2) # # bins and average references per group that re_r got wrong
		r_CR = total_r_entries - wrong_r.sum()
		ratio_m_CR = (signals_CR / total_m_entries).item()
		ratio_r_CR = (r_CR / total_r_entries).item()
		return ratio_m_CR, ratio_r_CR




