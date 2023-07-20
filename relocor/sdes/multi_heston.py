import torch
import numpy as np
from .sde import SDE

# multi-Heston: multiple (uncorrelated) Heston models
name = 'multi Heston'

multi_dim = 2

dim = 2*multi_dim
dim_noise = 2*multi_dim

a = 0.5 # 1.
b = 0.04
rho = -0.7
eta = 0.5

s0 = 1.
K = s0
v0 = 0.1


vect_S_np = np.array(multi_dim*[1., 0.])
vect_V_np = np.array(multi_dim*[0., 1.])

vect_S_torch = torch.Tensor(multi_dim*[1., 0.])
vect_V_torch = torch.Tensor(multi_dim*[0., 1.])

def drift(X):
    return vect_S_np * a * (b - X)
# !!! only if a, b are defined as scalars

def batch_drift(X):
    return vect_S_torch * a * (b - X)


def sigma(X):
    vol = np.sqrt(np.maximum(X[1::2], 0.))
    s = X[::2]
    diag = np.stack([vol*s, np.sqrt(1-rho**2)*eta*vol], axis=1)
    diag = diag.reshape((dim,))
    sub_diag = np.stack([rho*eta*vol, np.zeros((multi_dim,))], axis=1)
    sub_diag = sub_diag.reshape((dim,))[:-1]

    return np.diag(diag) + np.diag(sub_diag, k=-1)


def batch_sigma(X):
    vol = torch.sqrt(torch.relu(X[:,1::2]))
    s = X[:,::2]
    diag = torch.stack([vol*s, np.sqrt(1-rho**2)*eta*vol], axis=2)
    diag = diag.reshape([-1, dim])
    sub_diag = torch.stack([rho*eta*vol, torch.zeros_like(vol)], axis=2)
    sub_diag = sub_diag.reshape([-1, dim])[:,:-1]

    return torch.diag_embed(diag) + torch.diag_embed(sub_diag, -1)



multi_heston_sde = SDE(
    dim=dim,
    dim_noise=dim_noise,
    drift=drift,
    sigma=sigma,
    X0=np.array(multi_dim*[s0, v0]),
    batch_drift=batch_drift,
    batch_sigma=batch_sigma,
    name=name
)


def multi_heston_payoff(X):
    return np.maximum(np.mean(vect_S_np*X)-K, 0)

# def multi_heston_batch_payoff(X):
#     return torch.relu(torch.mean(vect_S_torch*X, axis=1, keepdims=True) - K)

def multi_heston_batch_payoff(X):
    return torch.mean(torch.relu(vect_S_torch*X-K), axis=1, keepdims=True)

