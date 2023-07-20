import numpy as np
from .sde import SDE
import torch

# black scholes in one dimension
name = 'multi Black Scholes'

dim = 2

r = 0.06
sig = 0.3
K = 1.
X0 = 1.

def drift(X):
    return r*X

def sigma(X):
    return np.diag(sig*X)

def batch_sigma(X):
    return torch.diag_embed(sig*X)

multi_bs_sde = SDE(
    dim=dim,
    dim_noise=dim,
    drift=drift,
    sigma=sigma,
    X0=np.array(dim*[X0]),
    batch_drift=drift,
    batch_sigma=batch_sigma,
    name=name
)

def multi_bs_payoff(X):
    return np.mean(np.maximum(X-K,0))

def multi_bs_payoff_2(X):
    return np.maximum(np.mean(X-K), 0)

def multi_bs_batch_payoff(X):
    return torch.mean(torch.relu(X-K), axis=1, keepdims=True)

