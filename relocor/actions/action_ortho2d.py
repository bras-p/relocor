import numpy as np
import gymnasium as gym
import torch

from .action_param import ActionParam


class ActionOrtho2d(ActionParam): # we parametrize the rotation by cos(rot) instead of rot

    def __init__(
        self, # only for dim_noise=2 for now
    ):
        self.dim_noise = 2
        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1., 0.]),
            high=np.array([1., 1., 1.]),
            shape=(self.dim_noise +1,),
            dtype=np.float32)
        # or low=[-1,-1,-1] ?
        # 2 first coordinates for diagonal matrix and last coordinate for cos(rotation)

        self.baseline_action = np.array([0.,0.,1.])
        self.antithetic_action = np.array([-1.,-1.,1.])
        self.minus_plus_action = np.array([-1.,1.,1.])

        self.dim_diag = 2
        self.dim_cosine = 1

    def get_matrix(self, action): # careful action must be as action_space
        ortho = self.get_ortho(action[2:][0])
        diag = np.diag(action[:2])
        return np.dot(np.dot(ortho, diag), np.transpose(ortho))

    def get_reverse_matrix(self, action):
        ortho = self.get_ortho(action[2:][0])
        diag = np.diag(action[:2])
        reverse_diag = np.sqrt(np.eye(self.dim_noise) - diag**2)
        return np.dot(np.dot(ortho, reverse_diag), np.transpose(ortho))
    
    def get_ortho(self, cosine):
        sine = np.sqrt(1.-cosine**2)
        ortho = np.array([
            [cosine, -sine],
            [sine, cosine]
        ])
        return ortho
    


class BatchActionOrtho2d(ActionOrtho2d):

    def __init__(self):
        super().__init__()

    def get_matrix(self, action):
        ortho = self.get_ortho(action[:,2:])
        diag = torch.diag_embed(action[:,:2])
        return torch.matmul(torch.matmul(ortho, diag), torch.transpose(ortho, 1, 2))
    
    def get_reverse_matrix(self, action):
        ortho = self.get_ortho(action[:,2:])
        diag = torch.diag_embed(action[:,:2])
        reverse_diag = torch.sqrt(torch.eye(self.dim_noise) - diag**2)
        return torch.matmul(torch.matmul(ortho, reverse_diag), torch.transpose(ortho, 1, 2))
    
    def get_ortho(self, cosine):
        sine = torch.sqrt(1.-cosine**2)
        ortho = torch.concat(
            [cosine, -sine, sine, cosine],
            axis=1)
        ortho = torch.reshape(ortho, [-1, 2, 2])
        return ortho






    # def get_baseline(self):
    #     return np.array([0.,0.,1.])
    
    # def get_antithetic(self):
    #     return np.array([-1.,-1.,1.])
    
    # def get_minus_plus_one(self):
    #     return np.array([-1.,1.,1.]) 


    # check the implementation of baseline and others...
    # def get_baseline(self):
    #     return torch.Tensor([0., 0., 1.])
    
    # def get_antithetic(self):
    #     return torch.Tensor([-1., -1., 1.])

    # def get_minus_plus(self):
    #     return torch.Tensor([-1., 1., 1.])


