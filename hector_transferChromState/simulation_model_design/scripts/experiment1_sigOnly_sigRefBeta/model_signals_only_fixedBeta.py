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
import generate_toy_data as data
class Encoder(nn.Module):
	def __init__(self, num_signals, num_states, num_references, hidden, dropout):
		super().__init__()
		self.drop = nn.Dropout(dropout)
		input_dim = num_signals + num_states * num_references
		self.fc1 = nn.Linear(num_signals, hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.fcmu = nn.Linear(hidden, num_states)
		self.fclv = nn.Linear(hidden, num_states)

	def forward(self, m):
		inputs = m
		h = F.softplus(self.fc1(inputs))
		h = F.softplus(self.fc2(h))
		h = self.drop(h)
		logpi_loc = F.softplus(self.fcmu(h))
		logpi_logvar = self.fclv(h)
		logpi_loc = self.drop(logpi_loc)
		logpi_scale = (0.5 * logpi_logvar).exp()
		return logpi_loc, logpi_scale

class Decoder(nn.Module):
	def __init__(self, num_states, num_signals, hidden, dropout, fixed_signalP):
		super().__init__()
		self.drop = nn.Dropout(dropout)
		self.beta = nn.Linear(num_states, num_signals, bias=False)
		self.bn = nn.BatchNorm1d(num_signals, affine=True)
		self.fixed_signalP = fixed_signalP

	def forward(self, inputs):
		# takes in the values of collapsed pi (inputs): probabilities of state 
		# assignments at each positions, and then multiply by fixed_signalP (beta)
		# to get the probabilities of observing signals at each position
		# --> vector size #signals
		# used as parameters for bernoulli dist. to get obs. signals
		# fixed_signalP: #states, #marks, from the random number generator
		signal_param = torch.matmul(inputs, self.fixed_signalP)
		return torch.sigmoid(signal_param) # to transform to [0,1]
    
class model_signals_only_fixedBeta(nn.Module):
	def __init__(self, num_signals, num_references, num_states, hidden, dropout, fixed_signalP):
		super().__init__()
		self.num_signals = num_signals
		self.num_references = num_references
		self.num_states = num_states
		self.fixed_signalP = fixed_signalP
		self.hidden = hidden
		self.dropout = dropout
		self.encoder = Encoder(num_signals, num_states, num_references, hidden, dropout)
		self.decoder = Decoder(num_states, num_signals, hidden, dropout, fixed_signalP)

	# shapes: 
	#  m: (bins x signals) signal matrix
	#  r: (bins x reference x state) indicator matrix
	def model(self, m):
		# flatten out the r indicator matrix
		pyro.module("decoder", self.decoder)
		with pyro.plate('bins', m.shape[0]):
			logCpi_loc = m.new_zeros((m.shape[0], self.num_states))
			logCpi_scale = m.new_ones((m.shape[0], self.num_states))
			logCpi = pyro.sample('log_collapsedPi', dist.Normal(logCpi_loc, logCpi_scale).to_event(1))
			Cpi = logCpi.exp()			
			Cpi = F.softmax(logCpi, -1)
			signal_param = self.decoder(Cpi)          
			pyro.sample('m', dist.Bernoulli(signal_param).to_event(1), obs=m)
	            
	def guide(self, m):
		pyro.module("encoder", self.encoder)
		with pyro.plate('bins', m.shape[0]):
			logpi_loc, logpi_scale = self.encoder(m)
			logpi = pyro.sample('log_collapsedPi', dist.Normal(logpi_loc, logpi_scale).to_event(1))

	def predict_state_assignment(self, m):
		logpi_loc, logpi_scale = self.encoder(m)
		Cpi = F.softmax(logpi_loc, -1)
		return(Cpi)

	def write_predicted_state_assignment(self, m, output_fn):
		Cpi = self.predict_state_assignment(m)
		df = pd.DataFrame(Cpi.detach().numpy())
		df['max_state'] = df.idxmax(axis =1)
		df.to_csv(output_fn, header = True, index = False, sep = '\t', compression = 'gzip')
		return

	def generate_reconstructed_data(self, m):
		'''
		m: num_bins, num_signals
		logpi_loc, logpi_scale: num_bins, num_states
		signal_param: num_bins, num_signals
		'''
		logpi_loc, logpi_scale = self.encoder(m)
		Cpi = F.softmax(logpi_loc, -1)
		signal_param = self.decoder(Cpi) 
		re_m = pyro.sample('reconstructed_m', dist.Bernoulli(signal_param).to_event(1))
		return (re_m)

	def get_percentage_correct_reconstruct(self, m):
		# m can be different from the m used in training
		re_m = self.generate_reconstructed_data(m)
		total_m_entries = re_m.shape[0] * re_m.shape[1]
		signals_CR = (re_m==m).sum() # correct reconstruct entries of signals
		ratio_m_CR = (signals_CR / total_m_entries).item()
		return ratio_m_CR
