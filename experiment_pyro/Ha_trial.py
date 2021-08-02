# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
 A toy mixture model to provide a simple example for implementing discrete enumeration.

 (A) -> [B] -> (C)

 A is an observed Bernoulli variable with Beta prior.
 B is a hidden variable which is a mixture of two Bernoulli distributions (with Beta priors),
 chosen by A being true or false.
 C is observed, and like B, is a mixture of two Bernoulli distributions (with Beta priors),
 chosen by B being true or false.
 There is a plate over the three variables for n independent observations of data.

 Because B is hidden and discrete we wish to marginalize it out of the model.
 This is done by:
    1) marking the model method with `@pyro.infer.config_enumerate`
    2) marking the B sample site in the model with `infer={"enumerate": "parallel"}`
    3) passing `pyro.infer.SVI` the `pyro.infer.TraceEnum_ELBO` loss function
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from tqdm import tqdm

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.ops.indexing import Vindex


def main(args):
    num_obs = args.num_obs
    num_steps = args.num_steps
    bernouli_mark, pi, z, state, transiton_mat, mark = generate_data(num_obs) # this will need to be changed so that we can get ref_state_df from pandas dataframes outside
    posterior_params = train(prior, data, num_steps, num_obs)
    evaluate(CPDs, posterior_params)

def toy():
    print("Hello")
def generate_data(N): # N: number of genomic positions
    N = 5
    num_state = 3
    num_mark = 1
    num_ct = 3
    alpha = torch.tensor([0.3,0.3,0.4])
    data = torch.tensor(np.random.randint(0, 2, 5)) # a list of 5 numbers, each can be 0 or 1 # this is observed data of the mark presence/absence calls
    ref_state_df = np.array([[0,0,0],[1,0,1], [2,2,0], [1,1,1], [0,1,2]])  # observed data of the reference epig's state maps in N positions. Rows; positions, columns: ref. epig
    bernouli_mark = torch.tensor([0.1,0.9,0.5]) # probabilty of the one mark being present given each state. indices correspond to states
    # domain = [False, True]
    pi = dist.Dirichlet(alpha).sample()
    with pyro.plate('z_plate', N): 
        z = pyro.sample('z', dist.Categorical(pi)) # sample the reference cell type for each position, independently
    R = np.choose(z, ref_state_df.T) # get the state of the reference epigenome picked at each genomic position
    transiton_mat = np.random.rand(num_state, num_state)
    transiton_mat = transiton_mat / np.linalg.norm(transiton_mat, ord=1, axis=1, keepdims=True) # normalize so that sum of each row is 1, rows: state from ref epig, columns: state in the sample of interest
    state = torch.zeros(N)
    for i in pyro.plate('genome_loop', N):
        state[i] = dist.Categorical(torch.tensor(transiton_mat[R[i],:])).sample()
    state = state.type(torch.long)
    mark = torch.zeros(N)
    for i in pyro.plate('mark_across_genome', N):
        mark[i] = dist.Bernoulli(bernouli_mark[state[i]]).sample()
    return bernouli_mark, pi, z, state, transiton_mat, mark
    prior = {
        "A": torch.tensor([1.0, 10.0]),
        "B": torch.tensor([[10.0, 1.0], [1.0, 10.0]]),
        "C": torch.tensor([[10.0, 1.0], [1.0, 10.0]]),
    }
    CPDs = {
        "p_A": Beta(prior["A"][0], prior["A"][1]).sample(),
        "p_B": Beta(prior["B"][:, 0], prior["B"][:, 1]).sample(),
        "p_C": Beta(prior["C"][:, 0], prior["C"][:, 1]).sample(),
    }
    data = {"A": Bernoulli(torch.ones(num_obs) * CPDs["p_A"]).sample()}
    data["B"] = Bernoulli(
        torch.gather(CPDs["p_B"], 0, data["A"].type(torch.long))
    ).sample()
    data["C"] = Bernoulli(
        torch.gather(CPDs["p_C"], 0, data["B"].type(torch.long))
    ).sample()
    return prior, CPDs, data


@pyro.infer.config_enumerate
def model(alpha, bernouli_mark, ref_state_df, obs, num_obs):
pi = Dirichlet(alpha).sample()
with pyro.plate('genome_loop', num_obs):
    Z = pyro.sample('Z', dist.Categorical(pi))
R = np.choose(z, ref_state_df.T) # get the state of the reference epigenome picked at each genomic position

    p_A = pyro.sample("p_A", dist.Beta(1, 1))
    p_B = pyro.sample("p_B", dist.Beta(torch.ones(2), torch.ones(2)).to_event(1))
    p_C = pyro.sample("p_C", dist.Beta(torch.ones(2), torch.ones(2)).to_event(1))
    with pyro.plate("data_plate", num_obs):
        A = pyro.sample("A", dist.Bernoulli(p_A.expand(num_obs)), obs=obs["A"])
        # Vindex used to ensure proper indexing into the enumerated sample sites
        B = pyro.sample(
            "B",
            dist.Bernoulli(Vindex(p_B)[A.type(torch.long)]),
            infer={"enumerate": "parallel"},
        )
        pyro.sample("C", dist.Bernoulli(Vindex(p_C)[B.type(torch.long)]), obs=obs["C"])


def guide(prior, obs, num_obs):
    a = pyro.param("a", prior["A"], constraint=constraints.positive)
    pyro.sample("p_A", dist.Beta(a[0], a[1]))
    b = pyro.param("b", prior["B"], constraint=constraints.positive)
    pyro.sample("p_B", dist.Beta(b[:, 0], b[:, 1]).to_event(1))
    c = pyro.param("c", prior["C"], constraint=constraints.positive)
    pyro.sample("p_C", dist.Beta(c[:, 0], c[:, 1]).to_event(1))


def train(prior, data, num_steps, num_obs):
    pyro.clear_param_store()
    # max_plate_nesting = 1 because there is a single plate in the model
    loss_func = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=loss_func)
    losses = []
    for _ in tqdm(range(num_steps)):
        loss = svi.step(prior, data, num_obs)
        losses.append(loss)
    plt.figure()
    plt.plot(losses)
    plt.show()
    posterior_params = {k: np.array(v.data) for k, v in pyro.get_param_store().items()}
    posterior_params["a"] = posterior_params["a"][
        None, :
    ]  # reshape to same as other variables
    return posterior_params


def evaluate(CPDs, posterior_params):
    true_p_A, pred_p_A = get_true_pred_CPDs(CPDs["p_A"], posterior_params["a"])
    true_p_B, pred_p_B = get_true_pred_CPDs(CPDs["p_B"], posterior_params["b"])
    true_p_C, pred_p_C = get_true_pred_CPDs(CPDs["p_C"], posterior_params["c"])
    print("\np_A = True")
    print("actual:   ", true_p_A)
    print("predicted:", pred_p_A)
    print("\np_B = True | A = False/True")
    print("actual:   ", true_p_B)
    print("predicted:", pred_p_B)
    print("\np_C = True | B = False/True")
    print("actual:   ", true_p_C)
    print("predicted:", pred_p_C)


def get_true_pred_CPDs(CPD, posterior_param):
    true_p = CPD.numpy()
    pred_p = posterior_param[:, 0] / np.sum(posterior_param, axis=1)
    return true_p, pred_p


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    parser = argparse.ArgumentParser(description="Toy mixture model")
    parser.add_argument("-n", "--num-steps", default=4000, type=int)
    parser.add_argument("-o", "--num-obs", default=10000, type=int)
    args = parser.parse_args()
    main(args)