import numpy as np
from .sde import SDE
import torch

# Heston (two dimensions)

name = 'Heston 2d'

dim = 2
dim_noise = dim

a = 0.5 # 1.
b = 0.04
rho = -0.7
eta = 0.5

s0 = 1.
K = s0
v0 = 0.1


def drift(X):
    return np.array([0., a*(b-X[1])])

def batch_drift(X):
    return torch.concat(
        [0.*X[:,0:1], a*(b-X[:,1:])], axis=1)


def sigma(X):
    vol = np.sqrt(np.maximum(X[1], 0.))
    return np.array([
        [vol*X[0], 0.],
        [rho*eta*vol, eta*vol*np.sqrt(1.-rho**2)]
    ])

def batch_sigma(X):
    vol = torch.sqrt(torch.relu(X[:,1:])) # use ReLU for numerical stability
    matrix = torch.concat([
            vol*X[:,0:1], 0.*X[:,0:1],
            rho*eta*vol, np.sqrt(1-rho**2)*eta*vol
        ], axis=1)
    matrix = torch.reshape(matrix, [-1, 2, 2])
    return matrix

heston_sde = SDE(
    dim=dim,
    dim_noise=dim_noise,
    drift=drift,
    sigma=sigma,
    X0=np.array([s0, v0]),
    batch_drift=batch_drift,
    batch_sigma=batch_sigma,
    name=name,
)

def heston_payoff(X):
    return np.mean(np.maximum(X[0]-K, 0))

def heston_batch_payoff(X):
    return torch.relu(X[:,0:1]-K)