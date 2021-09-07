# import packages
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
import generate_data_realSce as data
import math
from tqdm import trange

# import classes
from pyro.infer import SVI, TraceMeanField_ELBO
from with_beta import model1
from with_beta import model2
from with_beta import model3
from with_beta import model4

# classes for processing input data
import argparse
import os 
import sys
import helper

# classes for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

###### PROCESSING INPUT PARAMETERS BY USERS ######
parser = argparse.ArgumentParser(description='Testing different models with different settings of data simulation')
parser.add_argument('--output_folder', type = str,
                    help='where reported results are stored')
parser.add_argument('--batch_size', default=200, type=int,
                    help='batch_size per iteration')
parser.add_argument('--num_epochs', default = 1000, type=int,
                    help='num_epochs')
parser.add_argument('--num_hidden_sig', default = 1, type=int,
                    help='num_hidden_sig, model with separate signal and ref states')
parser.add_argument('--num_hidden_ref', default = 30, type=int,
                    help='num_hidden_ref, model with separate signal and ref states')
parser.add_argument('--num_hidden_comb', default = 15, type=int,
                    help='num_hidden_comb, model with separate signal and ref states')
parser.add_argument('--num_hidden', default=32, type=int,
                    help='num_hidden, model with joint hidden layer for signal and ref states')
parser.add_argument('--dropout', default=0.2, type=float,
                    help='ratio of dropout in training')
args = parser.parse_args()
num_bins = 10000
num_signals = 3 # raw signals is 5, but we will only provide 3 marks to the models
num_obs_signals = 3
num_references = 15
num_groups = 5
num_ref_per_groups = int(num_references/num_groups)
num_states = 8
data_params = vars(args)

# write the command out for our record
helper.make_dir(args.output_folder)
logger = helper.argument_log(sys.argv, args.output_folder, 'simulated_model')

####### SETTING UP DEVICES #######
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#### STEP 1: GENERATE DATA ####
generator = data.real_simulation_mark_sampled_state() 

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
raw_m = generator.get_sampled_signals() # num_bins, # num_signals (5), but in the data given to the model, 
# we will only include the first 3 signals
m=raw_m[:,:num_obs_signals]
r = generator.get_ref_state_counts().float()
raw_p = generator.params['p']
p = raw_p[:,:num_obs_signals] # states, signals (5), but in the data given to the model, we will only include the first 3 models
# this will save the simulated data of state assignment 
# (ground truth state assignment probabilities)
generator.save_collapsed_theta(os.path.join(args.output_folder, 'collapsed_theta.txt.gz'))

#### STEP 2: RUN DIFFERENT MODELS #####
hidden = args.num_hidden
hidden_sig = args.num_hidden_sig
hidden_ref = args.num_hidden_ref
hidden_comb = args.num_hidden_comb
dropout = args.dropout
# declare 4 models with similar parameters
m_one = model1.Model_one(num_signals, num_groups, num_ref_per_groups, num_states, hidden, dropout, p)
# m_two = model2.Model_two(num_signals, num_groups, num_ref_per_groups, num_states, hidden, dropout, p)
m_three = model3.Model_three(num_signals, num_groups, num_ref_per_groups, num_states, hidden_sig, hidden_ref, hidden_comb, dropout, p)
# m_four = model4.Model_four(num_signals, num_groups, num_ref_per_groups, num_states, hidden_sig, hidden_ref, hidden_comb, dropout, p)
m_one.to(device)
# m_two.to(device)
m_three.to(device)
# m_four.to(device)
print(isinstance(m_one, (type, model1.Model_one)))
# print(isinstance(m_two, (type, model2.Model_two)))
print(isinstance(m_three, (type, model3.Model_three)))
# print(isinstance(m_four, (type, model4.Model_four)))

# define a function to learn and reconstruct the data, and gives the results of ratios of the data that gets reconstructed
def learn_and_reconstruct_input(state_model, m, r, p, posterior_fn):
    batch_size = args.batch_size
    learning_rate = 1e-3
    num_epochs = args.num_epochs
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    num_batches = int(math.ceil(m.shape[0] / batch_size))
    bar = trange(num_epochs)
    svi = SVI(state_model.model, state_model.guide, optimizer, loss=TraceMeanField_ELBO())
    running_loss_list = []
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_m = m[i * batch_size:(i+1) * batch_size, :]
            batch_r = r[i * batch_size:(i+1) * batch_size, :, :]
            loss = svi.step(batch_m, batch_r)
            running_loss += loss / batch_m.size(0)
        running_loss_list.append(running_loss)
    bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    ratio_m_CR, ratio_r_CR = state_model.get_percentage_correct_reconstruct(m,r)
    state_model.write_predicted_state_assignment(m, r, posterior_fn)
    return ratio_m_CR, ratio_r_CR, running_loss_list
    

#### REPORT THE DATA ######
def get_one_line_to_report(state_model, model_name):
    results = [model_name, state_model.num_signals, state_model.num_groups, state_model.num_ref_per_groups, state_model.num_states, args.num_hidden_sig, args.num_hidden_ref, args.num_hidden, state_model.dropout]
    posterior_fn = os.path.join(args.output_folder, '{}_pos.txt.gz'.format(model_name))
    ratio_m_CR, ratio_r_CR, running_loss_list = learn_and_reconstruct_input(state_model, m, r, p, posterior_fn)
    results += [ratio_m_CR, ratio_r_CR]
    loss_fn = os.path.join(args.output_folder, '{}_loss.txt.gz'.format(model_name))
    loss_df = pd.Series(running_loss_list)
    loss_df.to_csv(loss_fn, index= False, header = False, compression = 'gzip', sep = '\n')
    return results

def call_models_and_report_reconstruction():
    result_df = pd.DataFrame(columns = ['model', 'num_signals', 'num_groups', 'num_ref_per_groups', 'num_states', 'hidden_sig', 'hidden_ref', 'hidden', 'dropout', 'ratio_m_CR', 'ratio_r_CR'])
    result_df.loc[0] = get_one_line_to_report(m_one, 'm_one')
    # result_df.loc[1] = get_one_line_to_report(m_two, 'm_two')
    result_df.loc[1] = get_one_line_to_report(m_three, 'm_three')
    # result_df.loc[3] = get_one_line_to_report(m_four, 'm_four')
    report_fn = os.path.join(args.output_folder, 'report_ratio_CR.txt')
    result_df.to_csv(report_fn, header = True, index = False, sep = '\t')

#### CALCULATING THE MODEL PERFORMANCE BASED ON STATE ASSIGNMENTS COMPARISION ####
def read_state_df(fn, model_name):
    df = pd.read_csv(fn, header = 0, index_col = None, sep = '\t')
    df['max_prob'] = df.apply(lambda x: np.max(x[:-1]), axis = 1)
    df.columns = list(map(lambda x: '{}|{}'.format(model_name, x), df.columns))
    return df

def report_result_latent_state():
    truth_fn = os.path.join(args.output_folder, 'collapsed_theta.txt.gz')
    truth_df = read_state_df(truth_fn, 'truth')
    model_name_list = ['m_one', 'm_three'] # ['m_one', 'm_two', 'm_three', 'm_four']
    model_fn_list = list(map(lambda x: os.path.join(args.output_folder, '{}_pos.txt.gz'.format(x)), model_name_list))
    model_df_list = list(map(lambda x: read_state_df(model_fn_list[x], model_name_list[x]), range(len(model_name_list))))
    all_df  = pd.concat([truth_df] + model_df_list, axis = 1)
    state_df = all_df.filter(regex='state',axis=1) # get all columns ending with state
    plot_nrow = 1
    plot_ncol = 2
    fig, axes = plt.subplots(ncols = plot_ncol, nrows = plot_nrow, figsize = (9,9)) 
    for model_index, model_name in enumerate(model_name_list):
        colnames = ['truth|state', '{}|max_state'.format(model_name)]
        df = state_df[colnames]
        df = df.groupby(colnames).size().to_frame(name = 'size').reset_index()
        df = df.pivot(colnames[0], colnames[1], 'size')
        df = df.div(df.sum(axis = 1), axis = 0) # row normalize
        ax = (axes.flat)[model_index] 
        sns.heatmap(df, cbar=True, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues', ax=ax).set_title(model_name)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'confusion_matrix.png'))


call_models_and_report_reconstruction()
report_result_latent_state()