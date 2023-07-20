import numpy as np
from .sde import SDE
import torch

# black scholes in one dimension
name = 'Black-Scholes'

r = 0.06
sig = 0.3
K = 1.
X0 = 1.


def drift(X):
    return r*X

def sigma(X):
    return np.reshape(sig*X, [1, 1])

def batch_sigma(X):
    return torch.reshape(sig*X, [-1, 1, 1])

bs_sde = SDE(
    dim=1,
    dim_noise=1,
    drift=drift,
    sigma=sigma,
    X0=np.array([X0]),
    batch_drift=drift,
    batch_sigma=batch_sigma,
    name=name,
)

def bs_payoff(X):
    return np.mean(np.maximum(X-K,0))

def bs_batch_payoff(X):
    return torch.relu(X-K)



