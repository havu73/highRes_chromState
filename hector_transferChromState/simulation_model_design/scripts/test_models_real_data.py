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
import math
from tqdm import trange

# import classes that we wrote with different models
from pyro.infer import SVI, TraceMeanField_ELBO
import model_signals_only as MSO
import model_signals_only_fixedBeta as MSB
import model_signals_and_refState as MSR
import model_signals_refStates_fixedBeta as MSRB

# modules to process input data
import argparse
import os 
import helper
import sys
###### PROCESSING INPUT PARAMETERS BY USERS ######
parser = argparse.ArgumentParser(description='Testing different models with different settings of data simulation')
parser.add_argument('--mark_fn', type=str,
                    help='fn of marks signals. Chrom, start, end, <mark name>')
parser.add_argument('--ref_fn', type=str,
                    help='reference epigenomes state maps. chrom, start, end, <EID>')
parser.add_argument('--output_folder', type=str,
                    help='where files of reconstruction results, and files of posterior probabilities are stored')
parser.add_argument('--emission_fn', type=str,
                    help='emission probabilities from roadmap')
parser.add_argument('--emission_scale', default=100, type=str,
                    help='multiply emission probabilities with this number to get p (fixed_signalP)')
parser.add_argument('--num_states', default=25, type=int,
                    help='number of hidden states')
parser.add_argument('--batch_size', default=200, type=int,
                    help='batch_size per iteration')
parser.add_argument('--num_epochs', default = 1000, type=int,
                    help='num_epochs')
parser.add_argument('--num_hidden', default = 32, type=int,
                    help='number of hidden nodes')
parser.add_argument('--dropout', default=0.2, type=float,
                    help='droput ratio in decoder and encoder')
args = parser.parse_args()

####### SETTING UP DEVICES #######
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#### STEP 1: GET INPUT DATA ####
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
def get_mark_signals(mark_fn):
    helper.check_file_exist(mark_fn)
    m = pd.read_csv(mark_fn, header = 0, index_col = None, sep = '\t')
    m.drop(['chrom', 'start', 'end'], axis = 1, inplace = True)
    m = torch.tensor(m.values).float()
    return m

def get_ref_state_indicators(ref_fn, num_states):
    helper.check_file_exist(ref_fn)
    ref_df = pd.read_csv(ref_fn, header = 0, index_col = None, sep = '\t')
    ref_df.drop(['chrom', 'start', 'end'], axis = 1, inplace = True)
    ref_df = ref_df.applymap(lambda x: int(x[1:])-1)
    r = torch.tensor(ref_df.values)
    r = F.one_hot(r, num_states).float()
    return r

def get_signals_params_p(emission_fn, emission_scale, mark_fn):
    helper.check_file_exist(emission_fn)
    helper.check_file_exist(mark_fn)
    m = pd.read_csv(mark_fn, header = 0, index_col = None, sep = '\t')
    mark_name_list = m.columns[3:] # first three colnames are chrom, start, end, followed by mark names
    emission_df = pd.read_csv(emission_fn, header = 0, index_col = 0, sep = '\t')
    emission_df = emission_df[mark_name_list] # get only emissions of marks associated with observed data
    p = torch.tensor(emission_df.values).float() * float(emission_scale)
    return p



# define a function to learn and reconstruct the data, and gives the results of ratios of the data that gets reconstructed
def learn_and_reconstruct_input(state_model, m, r, p, posterior_fn, batch_size, num_epochs):
    # batch_size = 200
    learning_rate = 1e-3
    # num_epochs = 1000
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
def get_one_line_to_report(state_model, model_name, m, r, p, output_folder, batch_size, num_epochs):
    helper.make_dir(output_folder)
    results = [model_name, state_model.num_signals, state_model.num_references, state_model.num_states, state_model.hidden, state_model.dropout]
    posterior_fn = os.path.join(output_folder, '{}_pos.txt.gz'.format(model_name))
    ratio_m_CR, ratio_r_CR = learn_and_reconstruct_input(state_model, m, r, p, posterior_fn, batch_size, num_epochs)
    results += [ratio_m_CR, ratio_r_CR]
    print('Done getting results from model: {}'.format(model_name))
    return results

if __name__ == '__main__':
    logger = helper.argument_log(command = sys.argv, output_folder= args.output_folder, log_prefix='test_models') 
    # to save the command line arguments that led to this   
    m = get_mark_signals(args.mark_fn) 
    # num_bins, num_marks
    num_signals = m.shape[1]
    r = get_ref_state_indicators(args.ref_fn, args.num_states) # num_bins, num_ref, num_states
    num_references = r.shape[1]
    p = get_signals_params_p(args.emission_fn, args.emission_scale, args.mark_fn)
    print("Done getting m, r, p")
    # num_states, num_marks
    hidden = args.num_hidden # default = 32
    dropout = args.dropout # default = 0.2
    #### STEP 2: RUN DIFFERENT MODELS #####
    # declare 4 models with similar parameters
    m_SigOnly = MSO.model_Signals(num_signals, num_references, args.num_states, hidden, dropout)
    m_SigBeta = MSB.model_signals_only_fixedBeta(num_signals, num_references, args.num_states, hidden, dropout, p)
    m_SigRef = MSR.model_signals_refStates(num_signals, num_references, args.num_states, hidden, dropout)
    m_SigRefBeta = MSRB.model_signals_refStates_fixedBeta(num_signals, num_references, args.num_states, hidden, dropout, p)
    m_SigOnly.to(device)
    m_SigBeta.to(device)
    m_SigRef.to(device)
    m_SigRefBeta.to(device)
    print(isinstance(m_SigOnly, (type, MSO.model_Signals)))
    print(isinstance(m_SigBeta, (type, MSB.model_signals_only_fixedBeta)))
    print(isinstance(m_SigRef, (type, MSR.model_signals_refStates)))
    print(isinstance(m_SigRefBeta, (type, MSRB.model_signals_refStates_fixedBeta)))
    result_df = pd.DataFrame(columns = ['model', 'num_signals', 'num_references', 'num_states', 'hidden', 'dropout', 'ratio_m_CR', 'ratio_r_CR'])
    result_df.loc[0] = get_one_line_to_report(m_SigOnly, 'SigOnly', m, r, p, args.output_folder, args.batch_size, args.num_epochs)
    result_df.loc[1] = get_one_line_to_report(m_SigBeta, 'SigBeta', m, r, p, args.output_folder, args.batch_size, args.num_epochs)
    # result_df.loc[2] = get_one_line_to_report(m_SigRef, 'SigRef', m, r, p, args.output_folder, arg.batch_size, args.num_epochs)
    # result_df.loc[3] = get_one_line_to_report(m_SigRefBeta, 'SigRefBeta', m, r, p, args.output_folder, arg.batch_size, args.num_epochs)
    output_fn = os.path.join(args.output_folder, 'test_models_CR.txt')
    result_df.to_csv(output_fn, header = True, index = False, sep = '\t')



