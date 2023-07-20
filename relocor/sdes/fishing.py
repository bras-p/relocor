import torch
import numpy as np
from .sde import SDE

sigmoid_np = lambda z : 1./(1. + np.exp(-z))

name = 'fishing'

dim = 5

r = 2.
eta = 0.3
u_m = 0.1
u_M = 1.
a = 0.2


kappa_np = np.array([
    [1.2, -0.1, 0., 0., -0.1],
    [0.2, 1.2, 0., 0., -0.1],
    [0., 0.2, 1.2, -0.1, 0.],
    [0., 0., 0.1, 1.2, 0.],
    [0.1, 0.1, 0., 0., 1.2] ])

kappa_torch = torch.Tensor(kappa_np)

def drift(X):
    return r*X - X*np.matmul(kappa_np, X) - a*X*sigmoid_np(X)

def batch_drift(X):
    return r*X - X*torch.einsum('ij,bj->bi', kappa_torch, X) - a*X*torch.sigmoid(X)

def sigma(X):
    return eta*np.eye(dim)

def batch_sigma(X):
    return eta*torch.tile(torch.eye(dim), [len(X), 1, 1])


fishing_sde = SDE(
    dim=dim,
    dim_noise=dim,
    drift=drift,
    sigma=sigma,
    X0=np.array([1., 0.8, 0.6, 1.5, 1.1]),
    batch_drift=batch_drift,
    batch_sigma=batch_sigma,
    name=name
)


def fishing_payoff(X):
    return np.mean(X)

def fishing_batch_payoff(X):
    return torch.mean(X, axis=1, keepdims=True)


