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

toy_parms = {
    'num_bins': 25,
    'num_references': 10,
    'num_signals': 3,
    'num_states': 4
}

'''
M: # regions
N: # bins per region
L: # signals (marks)
alpha: params of dirichlet prior over reference ethetagenomics
beta: ref --> sample state categorical distribution
p: state --> signal bernoulli distribution 
r: reference state at each bin. one-hot encoding, matrix size : #bins * #ref * #states
theta: the mixture probabilities of reference ethetagenome
'''

class CircularStateGenerator:
    # Within the number of references, there is a group of references that will be similar to the 
    # sample of interests in terms of state assignments
    def __init__(self,  
                 num_bins=5, 
                 num_references=10, 
                 num_groups=3,
                 state_vary_rate=0.01, 
                 # fraction of the genome where the state assignments among references of the same group are diff
                 num_signals=3,
                 num_states=5,
                 high_w=100):
        self.num_bins = num_bins
        self.num_references = num_references
        self.num_groups = num_groups
        self.state_vary_rate = state_vary_rate
        self.num_signals = num_signals
        self.num_states = num_states
        self.high_w = high_w
        self.sample = None
        self.params = self.set_params()
    
        
    # parameter of state->signal distributions
    # shape is (num_states, num_signals)
    def generate_param_p(self):
        p = torch.zeros((self.num_states, self.num_signals))
        for i in range(self.num_states):
            w = -self.high_w * torch.ones(self.num_signals)
            w[i % self.num_signals] = self.high_w
            p[i,:] = w
        return p
    
    # generate a state assignment tensor
    # shape is (num_regions, num_bins_per_region, num_references)
    def generate_ref_states(self):
        # this is code for the case where we want varied state patterns from each reference
        # and that there are actually groups of references that are similar to each other
        num_ref_per_groups = np.ceil(self.num_references/self.num_groups).astype(int)
        sample_r = torch.zeros(self.num_states, self.num_groups)
        for i in range(self.num_groups):
            sample_r[:,i] = torch.arange(self.num_states).roll(i)
            # each group has a circular permutation of states that are characteristics to that group
        sample_r = sample_r.repeat(np.ceil(self.num_bins / self.num_states).astype(int), 1)
        # now r is just a repeated sequence of sample_r
        r = torch.zeros(sample_r.shape[0], self.num_references)
        for i in range(self.num_references):
            r[:,i] = sample_r[:, i % self.num_groups]
        # now we will start to introduce some random changes to the state assignments among references from
        # the same groups
        num_change = int(self.state_vary_rate * self.num_bins)
        for i in range(self.num_states, self.num_references): 
            # for the first num_states columns, keep all the state assignments
            # if num_references < num_states, this loop will not be called
            org_r = r[:,i]
            indices_to_change = np.random.choice(self.num_bins, num_change)
            indices_to_change = torch.tensor(indices_to_change).type(torch.LongTensor)
            states_to_change = torch.tensor(np.random.choice(self.num_states, num_change)).float()
            r[indices_to_change,i] = states_to_change
        r = r[:self.num_bins,:self.num_references]
        return r.long() # num_bins, num_references --> values: state-0-based 
    
    # set parameters of the data generator
    def set_params(self):
        # parameters of the dirichlet over references
        # same one for every region
        # very high probability that generated sample looks like
        # reference 0
        # shape is (num_references,)
        alpha = torch.ones(self.num_references)
        num_ref_per_groups = np.ceil(self.num_references/self.num_groups).astype(int)
        for i in range(self.num_references):
            if i % self.num_groups == 0:
                alpha[i] = self.high_w # all refs in group 1 will be more similar to sample of interest
        
        # parameters of bernoulli distribution for each signal
        # for each state
        # shape is (num_states, num_signals)
        p = self.generate_param_p()
        
        # an indicator matrix along genome of the state for 
        # each refenrece
        # shape is (num_regions, num_bins_per_region, num_states, num_references)
        ref_states_indicator = F.one_hot(self.generate_ref_states(), self.num_states)
        params = {
            'alpha': alpha,
            'p': p,
            'ref_states_indicator': ref_states_indicator
        }
        self.params = params
        return params
        
    # collapse a prob vector over references to a prob vector over states
    # takes the cross product of prob vector theta and reference state indicator matrix r
    # shapes:
    #  theta: (None, num_references)
    #  r: (None, num_references, num_states)
    #  out: (None, num_states)
    def collapse_theta(self, theta, r=None):
        if r is None:
            assert self.params is not None
            r = self.params['ref_states_indicator']
            
        r = r.float()
        collapsed_theta = torch.zeros(theta.shape[0], r.shape[2])
        for i in range(theta.shape[0]):
            collapsed_theta[i,:] = torch.matmul(r[i,:,:].T, theta[i,:])
        return collapsed_theta
    
    def generate_sample(self):
        if self.params is None:
            self.set_params()
            
        r = self.params['ref_states_indicator']
                
        # generate reference distribution for each region
        with pyro.plate('bins', self.num_bins):
            # theta is shape (num_regions, num_references)
            theta = pyro.sample('theta', dist.Dirichlet(self.params['alpha']))
            # collapse the reference distribution for each bin to a 
            # state distribution 
            collapsed_theta = self.collapse_theta(theta, r)

            signal_params = torch.sigmoid(torch.matmul(collapsed_theta, self.params['p']))
            m = pyro.sample('m', dist.Bernoulli(signal_params).to_event(1))

        result = {
            'theta': theta,
            'm': m
        }
        self.sample = result
        return self.sample
    

    def get_sampled_collapsed_theta(self):
        if self.sample is None:
            self.generate_sample()
        theta = self.sample['theta']
        return self.collapse_theta(theta)
    
    def save_collapsed_theta(self, output_fn):
        collapsed_theta = self.get_sampled_collapsed_theta()
        collapsed_theta = pd.DataFrame(collapsed_theta.numpy())
        collapsed_theta['max_state'] = collapsed_theta.idxmax(axis = 1)
        collapsed_theta.to_csv(output_fn, header = True, index = False, sep = '\t')
        return 

    def get_sampled_signals(self):
        if self.sample is None:
            self.generate_sample()
        return self.sample['m']
    
    def get_sampled_theta(self):
        if self.sample is None:
            self.generate_sample()
        return self.sample['theta']
    
    def get_signal_parms(self):
        collapsed_theta = self.get_sampled_collapsed_theta()
        return torch.sigmoid(torch.matmul(collapsed_theta, self.params['p']))
    
    def get_ref_state_indicators(self):
        if self.params is None:
            self.set_params()
        return self.params['ref_states_indicator']
    

class CircularStateGenerator:
    # Within the number of references, there is a group of references that will be similar to the 
    # sample of interests in terms of state assignments
    def __init__(self,  
                 num_bins=5, 
                 num_references=10, 
                 num_groups=3,
                 state_vary_rate=0.01, 
                 # fraction of the genome where the state assignments among references of the same group are diff
                 num_signals=3,
                 num_states=5,
                 high_w=100):
        self.num_bins = num_bins
        self.num_references = num_references
        self.num_groups = num_groups
        self.state_vary_rate = state_vary_rate
        self.num_signals = num_signals
        self.num_states = num_states
        self.high_w = high_w
        self.sample = None
        self.params = self.set_params()
    
        
    # parameter of state->signal distributions
    # shape is (num_states, num_signals)
    def generate_param_p(self):
        p = torch.zeros((self.num_states, self.num_signals))
        for i in range(self.num_states):
            w = -self.high_w * torch.ones(self.num_signals)
            w[i % self.num_signals] = self.high_w
            p[i,:] = w
        return p
    
    # generate a state assignment tensor
    # shape is (num_regions, num_bins_per_region, num_references)
    def generate_ref_states(self):
        # this is code for the case where we want varied state patterns from each reference
        # and that there are actually groups of references that are similar to each other
        num_ref_per_groups = np.ceil(self.num_references/self.num_groups).astype(int)
        sample_r = torch.zeros(self.num_states, self.num_groups)
        for i in range(self.num_groups):
            sample_r[:,i] = torch.arange(self.num_states).roll(i)
            # each group has a circular permutation of states that are characteristics to that group
        sample_r = sample_r.repeat(np.ceil(self.num_bins / self.num_states).astype(int), 1)
        # now r is just a repeated sequence of sample_r
        r = torch.zeros(sample_r.shape[0], self.num_references)
        for i in range(self.num_references):
            r[:,i] = sample_r[:, i % self.num_groups]
        # now we will start to introduce some random changes to the state assignments among references from
        # the same groups
        num_change = int(self.state_vary_rate * self.num_bins)
        for i in range(self.num_states, self.num_references): 
            # for the first num_states columns, keep all the state assignments
            # if num_references < num_states, this loop will not be called
            org_r = r[:,i]
            indices_to_change = np.random.choice(self.num_bins, num_change)
            indices_to_change = torch.tensor(indices_to_change).type(torch.LongTensor)
            states_to_change = torch.tensor(np.random.choice(self.num_states, num_change)).float()
            r[indices_to_change,i] = states_to_change
        r = r[:self.num_bins,:self.num_references]
        return r.long() # num_bins, num_references --> values: state-0-based 
    
    # set parameters of the data generator
    def set_params(self):
        # parameters of the dirichlet over references
        # same one for every region
        # very high probability that generated sample looks like
        # reference 0
        # shape is (num_references,)
        alpha = torch.ones(self.num_references)
        num_ref_per_groups = np.ceil(self.num_references/self.num_groups).astype(int)
        for i in range(self.num_references):
            if i % self.num_groups == 0:
                alpha[i] = self.high_w # all refs in group 1 will be more similar to sample of interest
        
        # parameters of bernoulli distribution for each signal
        # for each state
        # shape is (num_states, num_signals)
        p = self.generate_param_p()
        
        # an indicator matrix along genome of the state for 
        # each refenrece
        # shape is (num_regions, num_bins_per_region, num_states, num_references)
        ref_states_indicator = F.one_hot(self.generate_ref_states(), self.num_states)
        params = {
            'alpha': alpha,
            'p': p,
            'ref_states_indicator': ref_states_indicator
        }
        self.params = params
        return params
        
    # collapse a prob vector over references to a prob vector over states
    # takes the cross product of prob vector theta and reference state indicator matrix r
    # shapes:
    #  theta: (None, num_references)
    #  r: (None, num_references, num_states)
    #  out: (None, num_states)
    def collapse_theta(self, theta, r=None):
        if r is None:
            assert self.params is not None
            r = self.params['ref_states_indicator']
            
        r = r.float()
        collapsed_theta = torch.zeros(theta.shape[0], r.shape[2])
        for i in range(theta.shape[0]):
            collapsed_theta[i,:] = torch.matmul(r[i,:,:].T, theta[i,:])
        return collapsed_theta
    
    def generate_sample(self):
        if self.params is None:
            self.set_params()
            
        r = self.params['ref_states_indicator']
                
        # generate reference distribution for each region
        with pyro.plate('bins', self.num_bins):
            # theta is shape (num_regions, num_references)
            theta = pyro.sample('theta', dist.Dirichlet(self.params['alpha']))
            # collapse the reference distribution for each bin to a 
            # state distribution 
            collapsed_theta = self.collapse_theta(theta, r)

            signal_params = torch.sigmoid(torch.matmul(collapsed_theta, self.params['p']))
            m = pyro.sample('m', dist.Bernoulli(signal_params).to_event(1))

        result = {
            'theta': theta,
            'm': m
        }
        self.sample = result
        return self.sample
    
    def get_sampled_collapsed_theta(self):
        if self.sample is None:
            self.generate_sample()
        theta = self.sample['theta']
        return self.collapse_theta(theta)

    def save_collapsed_theta(self, output_fn):
        collapsed_theta = self.get_sampled_collapsed_theta()
        collapsed_theta = pd.DataFrame(collapsed_theta.numpy())
        collapsed_theta['max_state'] = collapsed_theta.idxmax(axis = 1)
        collapsed_theta.to_csv(output_fn, header = True, index = False, sep = '\t')
        return 
    
    def get_sampled_signals(self):
        if self.sample is None:
            self.generate_sample()
        return self.sample['m']
    
    def get_sampled_theta(self):
        if self.sample is None:
            self.generate_sample()
        return self.sample['theta']
    
    def get_signal_parms(self):
        collapsed_theta = self.get_sampled_collapsed_theta()
        return torch.sigmoid(torch.matmul(collapsed_theta, self.params['p']))
    
    def get_ref_state_indicators(self):
        if self.params is None:
            self.set_params()
        return self.params['ref_states_indicator']
    