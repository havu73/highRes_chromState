# following the stick-breaking methods
# tutorial: https://pyro.ai/examples/dirichlet_process_mixture.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam

assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(0)

N = 5
num_state = 3
num_mark = 1
num_ct = 3
alpha = torch.tensor([0.3,0.3,0.4])
data = torch.tensor(np.random.randint(0, 2, 5)) # a list of 5 numbers, each can be 0 or 1
bernouli_mark = torch.tensor([0.1,0.9,0.5])

def model(data):
    with pyro.plate('data', N):
        beta = pyro.sample('beta', Dirichlet(alpha))
        z = pyro.sample('z', Categorical(beta))
        pyro.sample('obs', Bernouli(bernouli_mark[z]))

def guide(data):
    with pyro.plate('data', N):
        
data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([50]), MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))
                  
def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def model(data):
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("lambda_plate", T):
        lmbda = pyro.sample("lambda", Gamma(3, 0.05))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", Poisson(lmbda[z]), obs=data)

def guide(data):
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau_0 = pyro.param('tau_0', lambda: Uniform(0, 5).sample([T]), constraint=constraints.positive)
    tau_1 = pyro.param('tau_1', lambda: LogNormal(-1, 1).sample([T]), constraint=constraints.positive)
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("lambda_plate", T):
        q_lambda = pyro.sample("lambda", Gamma(tau_0, tau_1))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))
