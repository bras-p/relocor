import numpy as np
import gymnasium as gym
import torch

from .action_param import ActionParam



class ActionDiag(ActionParam):

    def __init__(self, dim_noise):
        self.dim_noise = dim_noise
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(dim_noise,), dtype=np.float32)
        self.baseline_action = np.zeros((self.dim_noise,))
        self.antithetic_action = -1.*np.ones((self.dim_noise,))
        self.dim_diag = dim_noise
        self.dim_cosine = 0

    def get_matrix(self, action):
        return np.diag(action)
    
    def get_reverse_matrix(self, action):
        return np.diag(np.sqrt(1.-action**2))



class BatchActionDiag(ActionDiag):

    def __init__(self, dim_noise):
        super().__init__(dim_noise)

    def get_matrix(self, action):
        return torch.diag_embed(action)
    
    def get_reverse_matrix(self, action):
        return torch.diag_embed(torch.sqrt(1.-action**2))





    # def get_baseline(self):
    #     return np.zeros((self.dim_noise,))
    
    # def get_antithetic(self):
    #     return -1.*np.ones((self.dim_noise,))
    
    # def get_baseline(self):
    #     return torch.zeros((self.dim_noise,))
    
    # def get_antithetic(self):
    #     return -1.*torch.ones((self.dim_noise,))