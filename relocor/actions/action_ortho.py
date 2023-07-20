import numpy as np
import gymnasium as gym
import torch
from scipy.linalg import block_diag

from .action_param import ActionParam

"""
Action in any dimension, the rotation matrix is parametrized
as a matrix with 2x2 rotation diagonal blocks; if the dimension
is odd then we add 1 at the end.
"""


class ActionOrtho(ActionParam):
        
    def __init__(self, dim_noise):
        self.dim_noise = dim_noise
        self.action_space = gym.spaces.Box(
            low = np.array( [-1.]*dim_noise + [0.]*(dim_noise//2) ),
            high = np .array( [1.]*dim_noise + [1.]*(dim_noise//2) ),
            shape = (dim_noise + dim_noise//2,),
            dtype=np.float32)
        # dim_noise first coordinates for diagonal matrix and dim_noise//2 for cos(rotation)'s

        self.baseline_action = np.array([0.]*dim_noise + [1.]*(dim_noise//2))
        self.antithetic_action = np.array([-1.]*dim_noise + [1.]*(dim_noise//2))
        # mp action only for even dim_noise
        if dim_noise % 2 == 0:
            self.minus_plus_action = np.array([-1., 1.]*(dim_noise//2) + [1.]*(dim_noise//2))

        self.dim_diag = dim_noise
        self.dim_cosine = dim_noise//2

    def get_matrix(self, action):
        ortho = self.get_ortho(action[self.dim_noise:])
        diag = np.diag(action[:self.dim_noise])
        return np.dot(np.dot(ortho, diag), np.transpose(ortho))

    def get_reverse_matrix(self, action):
        ortho = self.get_ortho(action[self.dim_noise:])
        diag = np.diag(action[:self.dim_noise])
        reverse_diag = np.sqrt(np.eye(self.dim_noise) - diag**2)
        return np.dot(np.dot(ortho, reverse_diag), np.transpose(ortho))
    
    def get_ortho(self, cosine):
        sine = np.sqrt(1.-cosine**2)
        blocks = []

        for k in range(self.dim_noise//2):
            blocks.append(np.array([
                [cosine[k], -sine[k]],
                [sine[k], cosine[k]]]))
            
        if self.dim_noise % 2 == 1:
            blocks.append(np.array([1.]))
        
        return block_diag(*blocks)



class BatchActionOrtho(ActionOrtho):

    def __init__(self, dim_noise):
        super().__init__(dim_noise)

    def get_matrix(self, action):
        ortho = self.get_ortho(action[:, self.dim_noise:])
        diag = torch.diag_embed(action[:,:self.dim_noise])
        return torch.matmul(torch.matmul(ortho, diag), torch.transpose(ortho, 1, 2))
    
    def get_reverse_matrix(self, action):
        ortho = self.get_ortho(action[:, self.dim_noise:])
        diag = torch.diag_embed(action[:,:self.dim_noise])
        reverse_diag = torch.sqrt(torch.eye(self.dim_noise) - diag**2)
        return torch.matmul(torch.matmul(ortho, reverse_diag), torch.transpose(ortho, 1, 2))
    
    def get_ortho(self, cosine):
        sine = torch.sqrt(1.-cosine**2)

        v = torch.Tensor([1., 0.])
        sub_diag = torch.einsum('bk,i->bki', sine, v)
        sub_diag = torch.reshape(sub_diag, [-1, 2*(self.dim_noise//2)])

        w = torch.Tensor([1., 1.])
        diag = torch.einsum('bk,i->bki', cosine, w)
        diag = torch.reshape(diag, [-1, 2*(self.dim_noise//2)])

        if self.dim_noise % 2 == 1:
            diag = torch.concat([diag, 1.*torch.ones((len(cosine), 1))], axis=1)

        if self.dim_noise % 2 == 0:
            ortho = torch.diag_embed(diag) + torch.diag_embed(sub_diag[:,:-1], -1) - torch.diag_embed(sub_diag[:,:-1], +1)
        else:
            ortho = torch.diag_embed(diag) + torch.diag_embed(sub_diag, -1) - torch.diag_embed(sub_diag, +1)

        return ortho

