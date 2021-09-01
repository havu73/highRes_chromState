# import packages
import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.nn.functional as F
import generate_toy_data as data
import math
from tqdm import trange

# import classes
from pyro.infer import SVI, TraceMeanField_ELBO
import model_signals_only as MSO
import model_signals_only_fixedBeta as MSB
import model_signals_and_refState as MSR
import model_signals_refStates_fixedBeta as MSRB

import argparse
import os 
import sys
import helper
###### PROCESSING INPUT PARAMETERS BY USERS ######
parser = argparse.ArgumentParser(description='Testing different models with different settings of data simulation')
parser.add_argument('--num_bins', default=10000, type=int,
                    help='number of genomic bins on the genome')
parser.add_argument('--num_references', default=3, type=int,
                    help='number of reference epigenomes')
parser.add_argument('--num_groups', default = 3, type = int,
                    help = 'number of distrinct groups among the reference epigenomes')
parser.add_argument('--state_vary_rate', default = 0.01, type = float,
                    help = 'fraction of the genome in which there are variablilty in states from two ref of the same group')
parser.add_argument('--num_signals', default=3, type=int,
                    help='number of observed signal tracks')
parser.add_argument('--num_states', default=3, type=int,
                    help='number of hidden states')
parser.add_argument('--output_folder', type = str,
                    help='where reported results are stored')
args = parser.parse_args()
data_params = vars(args)
keys_to_extract = ['num_bins', 'num_references', 'num_groups', 'state_vary_rate', 'num_signals', 'num_states']
generator_params = {key: data_params[key] for key in keys_to_extract} # extract those associated with the generator
print('Params for data generation:')
print(generator_params)
# write the command out for our record
helper.make_dir(args.output_folder)
logger = helper.argument_log(sys.argv, args.output_folder, 'simulated_model')

####### SETTING UP DEVICES #######
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#### STEP 1: GENERATE DATA ####
# generator = data.ToyGenerator(**generator_params, high_w=100)
generator = data.CircularStateGenerator(**generator_params, high_w=100)

'''
M: # regions
N: # bins per region
L: # signals (marks)
alpha: params of dirichlet prior over reference epigenomics
beta: ref --> sample state categorical distribution
p: state --> signal bernoulli distribution 
r: reference state at each bin. one-hot encoding, matrix size : #bins * #ref * #states
theta: the mixture probabilities of reference epigenome
'''
m = generator.get_sampled_signals()
r = generator.get_ref_state_indicators()
p = generator.params['p']
# this will save the simulated data of state assignment 
# (ground truth state assignment probabilities)
generator.save_collapsed_theta(os.path.join(args.output_folder, 'collapsed_theta.txt.gz'))

#### STEP 2: RUN DIFFERENT MODELS #####
hidden = 32
dropout = 0.2
# declare 4 models with similar parameters
m_SigOnly = MSO.model_Signals(args.num_signals, args.num_references, args.num_states, hidden, dropout)
m_SigBeta = MSB.model_signals_only_fixedBeta(args.num_signals, args.num_references, args.num_states, hidden, dropout, p)
m_SigRef = MSR.model_signals_refStates(args.num_signals, args.num_references, args.num_states, hidden, dropout)
m_SigRefBeta = MSRB.model_signals_refStates_fixedBeta(args.num_signals, args.num_references, args.num_states, hidden, dropout, p)
m_SigOnly.to(device)
m_SigBeta.to(device)
m_SigRef.to(device)
m_SigRefBeta.to(device)
print(isinstance(m_SigOnly, (type, MSO.model_Signals)))
print(isinstance(m_SigBeta, (type, MSB.model_signals_only_fixedBeta)))
print(isinstance(m_SigRef, (type, MSR.model_signals_refStates)))
print(isinstance(m_SigRefBeta, (type, MSRB.model_signals_refStates_fixedBeta)))

# define a function to learn and reconstruct the data, and gives the results of ratios of the data that gets reconstructed
def learn_and_reconstruct_input(state_model, m, r, p, posterior_fn):
    batch_size = 200
    learning_rate = 1e-3
    num_epochs = 1000
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    num_batches = int(math.ceil(m.shape[0] / batch_size))
    bar = trange(num_epochs)
    withRef = True # some models needs reference states data, others don't
    if isinstance(state_model, (type, MSO.model_Signals)) or isinstance(state_model, (type, MSB.model_signals_only_fixedBeta)):
        withRef = False 
    svi = SVI(state_model.model, state_model.guide, optimizer, loss=TraceMeanField_ELBO())
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_m = m[i * batch_size:(i+1) * batch_size, :]
            if withRef:
                batch_r = r[i * batch_size:(i+1) * batch_size, :, :]
                loss = svi.step(batch_m, batch_r)
            else:
                loss = svi.step(batch_m)
            running_loss += loss / batch_m.size(0)
    bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    if withRef:
        ratio_m_CR, ratio_r_CR = state_model.get_percentage_correct_reconstruct(m,r)
        state_model.write_predicted_state_assignment(m, r, posterior_fn)
        return ratio_m_CR, ratio_r_CR
    else:
        ratio_m_CR = state_model.get_percentage_correct_reconstruct(m)
        state_model.write_predicted_state_assignment(m, posterior_fn)
        return ratio_m_CR, 0 # return 0 as a placeholder for ratio_r_CR that is not applicable for this case
    

#### REPORT THE DATA ######
result_df = pd.DataFrame(columns = ['model', 'num_signals', 'num_references', 'num_states', 'hidden', 'dropout', 'ratio_m_CR', 'ratio_r_CR'])
def get_one_line_to_report(state_model, model_name):
    results = [model_name, state_model.num_signals, state_model.num_references, state_model.num_states, state_model.hidden, state_model.dropout]
    posterior_fn = os.path.join(args.output_folder, '{}_pos.txt.gz'.format(model_name))
    ratio_m_CR, ratio_r_CR = learn_and_reconstruct_input(state_model, m, r, p, posterior_fn)
    results += [ratio_m_CR, ratio_r_CR]
    return results

result_df.loc[0] = get_one_line_to_report(m_SigOnly, 'SigOnly')
result_df.loc[1] = get_one_line_to_report(m_SigBeta, 'SigBeta')
result_df.loc[2] = get_one_line_to_report(m_SigRef, 'SigRef')
result_df.loc[3] = get_one_line_to_report(m_SigRefBeta, 'SigRefBeta')
report_fn = os.path.join(args.output_folder, 'report_ratio_CR.txt')
result_df.to_csv(report_fn, header = True, index = False, sep = '\t')