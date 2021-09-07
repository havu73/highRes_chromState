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
'''
M: # regions
N: # bins per region
L: # signals (signals)
alpha: params of dirichlet prior over reference epigenomics
beta: ref --> sample state categorical distribution
p: state --> signal bernoulli distribution 
r: reference state at each bin. one-hot encoding, matrix size : #bins * #ref * #states
theta: the mixture probabilities of reference ethetagenome
'''
class real_simulation_mark_mixture:
    def __init__(self):
        self.num_bins = 10000
        self.num_signals = 5
        self.num_obs_signals = 3
        self.num_references = 15
        self.num_groups = 5
        self.num_ref_per_groups = int(self.num_references/self.num_groups)
        self.num_states = 8
        self.num_const_states = 5
        self.num_ctSpec_states = 3
        self.state_vary_rate = 0.03
        self.high_w = 100
        self.sample = None
        self.params = self.set_params()
        
    def generate_param_p(self):
        '''
        M1 --> 5: H3K4me3, H3K27ac, DNase, H3K4me1 (TSS like), RepressiveM
        S0: H3K4me3, constitutive
        S1: quiescent, constitutive
        S2: RepressiveM, constitutive
        S3: DNase, const
        S4: Dnase + K4me1, const
        S5: K27ac, ct-spec
        S6: K27ac + RepressiveM, ct-spec
        S7: poised TSS, K4me3 + repressiveM, ct-spec
        '''
        p = torch.zeros((self.num_states, self.num_signals)) - self.high_w
        p[0,0] = self.high_w # K4me3
        p[2,4] = self.high_w # RepressiveM
        p[3,2] = self.high_w # Dnase
        p[4,2] = self.high_w # Dnase
        p[4,3] = self.high_w # K4me1
        p[5,1] = self.high_w # K27ac
        p[6,1] = self.high_w # K27ac
        p[6,4] = self.high_w # RepressiveM
        p[7,0] = self.high_w # K4me3
        p[7,4] = self.high_w # RepressiveM
        return p

    # generate a state assignment tensor
    # shape is (num_bins, num_references)
    def generate_ref_states(self):
        group_r = torch.ones(self.num_bins, self.num_groups)
        const_ratio = 0.05 * self.num_const_states # 5 constitutive states
        const_nBins = int(self.num_bins * const_ratio)
        ctSpec_ratio = 0.05 * self.num_groups * self.num_ctSpec_states # 3 ct-spec states
        ctSpec_nBins = self.num_bins - const_nBins
        # sample constitutive position
        const_bins = np.random.choice(self.num_bins, const_nBins, replace = False)
        numBins_per_state = int(0.05 * self.num_bins)
        for i in range(self.num_const_states): 
            # first num_const_states are const_states
            bins_indices = torch.tensor(const_bins[i*numBins_per_state:(i+1)*numBins_per_state]).type(torch.LongTensor)
            group_r[bins_indices,:] = i
        # sample ctSpec positions
        ctSpec_bins = torch.tensor(np.setdiff1d(np.arange(self.num_bins), const_bins)).type(torch.LongTensor)
        num_ctSpec_bins_per_group = self.num_ref_per_groups*numBins_per_state
        for i in range(self.num_groups):
            indices_for_group = ctSpec_bins[i*num_ctSpec_bins_per_group:(i+1)*num_ctSpec_bins_per_group]
            other_groupI = list(range(self.num_groups))
            other_groupI.remove(i)
            other_groupI = torch.tensor(other_groupI).type(torch.LongTensor)
            for j in range(self.num_ctSpec_states):
                # after all constitutive states have been filled, 
                # we shall fill the ctSpec states
                stateI = self.num_const_states + j
                indices_for_state = indices_for_group[j*numBins_per_state:(j+1)*numBins_per_state]
                group_r[indices_for_state, i] = stateI # ct_spec state in this group
                # quiescent state in other groups by default because of how group_r is initiated
        # now randomly introduce some noise by varying states among references of the same group, to be implemented
        
        # create a new dimension: num_ref_per_group --> num_ref_per_group, num_bins, num_group: --> values: state
        r = group_r.unsqueeze(0).repeat(self.num_ref_per_groups,1,1) 
        # unsqueeze(0) to add 1 dimension to the front
        # repeat to repeat along each each dimension
        return r.long() # convert to long because they will be used as index later
    
    # set parameters of the data generator
    def set_params(self):
        # parameters of the dirichlet over references
        # same one for every region
        # very high probability that generated sample looks like
        # reference 0
        # shape is (num_references,)
        alpha = torch.ones(self.num_groups)
        alpha[0] = self.high_w # we assume group 1 has the highest prob of being similar to sample of interest
        
        # parameters of bernoulli distribution for each signal
        # for each state
        # shape is (num_states, num_signals)
        p = self.generate_param_p()
        
        # an indicator matrix along genome of the state for 
        # each refenrece
        # shape is (num_ref_per_groups, num_bins, num_group, num_states)
        ref_states_indicator = F.one_hot(self.generate_ref_states(), self.num_states)
        # count of num_ref in each group that are annotated as each state
        # shape is (num_bins, num_groups, num_states)
        ref_states_count = ref_states_indicator.sum(axis = 0) 
        params = {
            'alpha': alpha,
            'p': p,
            'ref_states_indicator': ref_states_indicator,
            'ref_states_count' : ref_states_count
        }
        self.params = params
        return params
    
    # collapse a prob vector over references to a prob vector over states
    # takes the cross product of prob vector theta and reference state indicator matrix r
    # shapes:
    #  theta: (None, num_groups)
    #  r: (None, num_groups, num_states)
    #  out: (None, num_states)
    def collapse_theta(self, theta, r=None):
        if r is None:
            assert self.params is not None
            r = self.params['ref_states_count']
        r = r.float()
        collapsed_theta = torch.zeros(theta.shape[0], r.shape[2]) # bins, states
        for i in range(theta.shape[0]):
            collapsed_theta[i,:] = torch.matmul(r[i,:,:].T, theta[i,:])
        collapsed_theta = collapsed_theta / float(self.num_ref_per_groups) 
        # added for the case of state count
        # instead of state indicators
        return collapsed_theta
    
    def generate_sample(self):
        if self.params is None:
            self.set_params()
            
        r = self.params['ref_states_count']
                
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
    
    def get_ref_state_counts(self):
        if self.params is None:
            self.set_params()
        return self.params['ref_states_count']

'''
M: # regions
N: # bins per region
L: # signals (signals)
alpha: params of dirichlet prior over reference epigenomics
beta: ref --> sample state categorical distribution
p: state --> signal bernoulli distribution 
r: reference state at each bin. one-hot encoding, matrix size : #bins * #ref * #states
theta: the mixture probabilities of reference ethetagenome
'''
class real_simulation_mark_sampled_state:
    def __init__(self):
        self.num_bins = 10000
        self.num_signals = 5
        self.num_obs_signals = 3
        self.num_references = 15
        self.num_groups = 5
        self.num_ref_per_groups = int(self.num_references/self.num_groups)
        self.num_states = 8
        self.num_const_states = 5
        self.num_ctSpec_states = 3
        self.state_vary_rate = 0.03
        self.high_w = 10
        self.sample = None
        self.params = None #self.set_params()
        
    def generate_param_p(self):
        '''
        M1 --> 5: H3K4me3, H3K27ac, DNase, H3K4me1 (TSS like), RepressiveM
        S0: H3K4me3, constitutive
        S1: quiescent, constitutive
        S2: RepressiveM, constitutive
        S3: DNase, const
        S4: Dnase + K4me1, const
        S5: K27ac, ct-spec
        S6: K27ac + RepressiveM, ct-spec
        S7: poised TSS, K4me3 + repressiveM, ct-spec
        '''
        p = torch.zeros((self.num_states, self.num_signals)) - self.high_w
        p[0,0] = self.high_w # K4me3
        p[2,4] = self.high_w # RepressiveM
        p[3,2] = self.high_w # Dnase
        p[4,2] = self.high_w # Dnase
        p[4,3] = self.high_w # K4me1
        p[5,1] = self.high_w # K27ac
        p[6,1] = self.high_w # K27ac
        p[6,4] = self.high_w # RepressiveM
        p[7,0] = self.high_w # K4me3
        p[7,4] = self.high_w # RepressiveM
        p = torch.sigmoid(p) # each row is a probability distribution over marks, sigmoid to convert to [0,1]
        return p

    # generate a state assignment tensor
    # shape is (num_bins, num_references)
    def generate_ref_states(self):
        group_r = torch.ones(self.num_bins, self.num_groups)
        const_ratio = 0.05 * self.num_const_states # 5 constitutive states
        const_nBins = int(self.num_bins * const_ratio)
        ctSpec_ratio = 0.05 * self.num_groups * self.num_ctSpec_states # 3 ct-spec states
        ctSpec_nBins = self.num_bins - const_nBins
        # sample constitutive position
        const_bins = np.random.choice(self.num_bins, const_nBins, replace = False)
        numBins_per_state = int(0.05 * self.num_bins)
        for i in range(self.num_const_states): 
            # first num_const_states are const_states
            bins_indices = torch.tensor(const_bins[i*numBins_per_state:(i+1)*numBins_per_state]).type(torch.LongTensor)
            group_r[bins_indices,:] = i
        # sample ctSpec positions
        ctSpec_bins = torch.tensor(np.setdiff1d(np.arange(self.num_bins), const_bins)).type(torch.LongTensor)
        num_ctSpec_bins_per_group = self.num_ref_per_groups*numBins_per_state
        for i in range(self.num_groups):
            indices_for_group = np.random.choice(ctSpec_bins, num_ctSpec_bins_per_group, replace = False)
            ctSpec_bins = torch.tensor(np.setdiff1d(ctSpec_bins, indices_for_group)).type(torch.LongTensor)
            # ctSpec_bins will shrink after each iteration to assign bins to each group
            for j in range(self.num_ctSpec_states):
                # after all constitutive states have been filled, 
                # we shall fill the ctSpec states
                stateI = self.num_const_states + j
                indices_for_state = np.random.choice(indices_for_group, numBins_per_state, replace = False)
                group_r[indices_for_state, i] = stateI # ct_spec state in this group
                indices_for_group = torch.tensor(np.setdiff1d(indices_for_group, indices_for_state)).type(torch.LongTensor)
                # indices_for_group will shrink after each iteration to assign bins to state withing the group
                # quiescent state in other groups by default because of how group_r is initiated
        # now randomly introduce some noise by varying states among references of the same group, to be implemented
        # create a new dimension: num_ref_per_group --> num_ref_per_group, num_bins, num_group: --> values: state
        r = group_r.unsqueeze(0).repeat(self.num_ref_per_groups,1,1) 
        # unsqueeze(0) to add 1 dimension to the front
        # repeat to repeat along each each dimension
        return r.long() # convert to long because they will be used as index later
    
    # set parameters of the data generator
    def set_params(self):
        # parameters of the dirichlet over references
        # same one for every region
        # very high probability that generated sample looks like
        # reference 0
        # shape is (num_references,)
        alpha = torch.ones(self.num_groups)
        alpha[0] = self.high_w # we assume group 1 has the highest prob of being similar to sample of interest
        
        # parameters of bernoulli distribution for each signal
        # for each state
        # shape is (num_states, num_signals)
        p = self.generate_param_p()
        
        # an indicator matrix along genome of the state for 
        # each refenrece
        # shape is (num_ref_per_groups, num_bins, num_group, num_states)
        ref_states_indicator = F.one_hot(self.generate_ref_states(), self.num_states)
        # count of num_ref in each group that are annotated as each state
        # shape is (num_bins, num_groups, num_states)
        ref_states_count = ref_states_indicator.sum(axis = 0) 
        params = {
            'alpha': alpha,
            'p': p,
            'ref_states_indicator': ref_states_indicator,
            'ref_states_count' : ref_states_count
        }
        self.params = params
        return params
    
    # collapse a prob vector over references to a prob vector over states
    # takes the cross product of prob vector theta and reference state indicator matrix r
    # shapes:
    #  theta: (None, num_groups)
    #  r: (None, num_groups, num_states)
    #  out: (None, num_states)
    def collapse_theta(self, theta, r=None):
        if r is None:
            assert self.params is not None
            r = self.params['ref_states_count']
        r = r.float()
        collapsed_theta = torch.zeros(theta.shape[0], r.shape[2]) # bins, states
        for i in range(theta.shape[0]):
            collapsed_theta[i,:] = torch.matmul(r[i,:,:].T, theta[i,:])
        collapsed_theta = collapsed_theta / float(self.num_ref_per_groups) 
        # the above line is added for the case of state count
        # instead of state indicators
        return collapsed_theta
    
    def generate_sample(self):
        if self.params is None:
            self.set_params()
            
        r = self.params['ref_states_count']
                
        # generate reference distribution for each region
        with pyro.plate('bins', self.num_bins):
            # theta is shape (num_regions, num_references)
            theta = pyro.sample('theta', dist.Dirichlet(self.params['alpha']))
            # collapse the reference distribution for each bin to a 
            # state distribution 
            collapsed_theta = self.collapse_theta(theta, r)
        state = pyro.sample('state', dist.Categorical(collapsed_theta).to_event(1)) # 1D tensor of state indices
        signal_params = self.params['p'][state,:] # each row shows the probabilities of signals given the selected state at the genomic bin
        m = pyro.sample('m', dist.Bernoulli(signal_params).to_event(1))

        result = {
            'theta': theta,
            'state': state,
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
        collapsed_theta['state'] = self.sample['state'].numpy()
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
        if self.sample is None:
            self.generate_sample()
        collapsed_theta = self.get_sampled_collapsed_theta()
        state = pyro.sample('state', dist.Categorical(collapsed_theta).to_event(1)) # 1D tensor of state indices
        signal_params = self.params['p'][state,:]
        return signal_params
    
    def get_state(self):
        if self.sample is None:
            self.generate_sample()
        return self.sample['state']
    
    def get_ref_state_indicators(self):
        if self.params is None:
            self.set_params()
        return self.params['ref_states_indicator']
    
    def get_ref_state_counts(self):
        if self.params is None:
            self.set_params()
        return self.params['ref_states_count']
        