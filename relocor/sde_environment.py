import numpy as np
import gymnasium as gym
import torch



class SDEEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(
            self,
            sde,
            T,
            N_euler,
            test_function = None,
            action_param = None,
            render_mode="console"
    ):
        super().__init__()
        self.sde = sde
        self.T, self.N_euler, self.h = T, N_euler, T/N_euler
        self.test_function = test_function
        self.action_param = action_param

        self.render_mode = render_mode
        
        self.action_space = action_param.action_space
        self.observation_space = gym.spaces.Box(low=0., high=np.inf, shape=(2*sde.dim + 1,), dtype=np.float32)
        # here we consider only positive sdes
        self.X1, self.X2, self.step_n = self.sde.X0, self.sde.X0, 0
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.X1, self.X2, self.step_n = self.sde.X0, self.sde.X0, 0
        return np.concatenate([self.X1, self.X2, np.zeros(1)], axis=0).astype(np.float32), {} # return the observation space and info


    def step(self, action):
        value = self.test_function(self.X1) * self.test_function(self.X2)

        rho1 = self.action_param.get_matrix(action)
        rho2 = self.action_param.get_reverse_matrix(action)

        dW1 = np.sqrt(self.h) * np.random.normal(size=(self.sde.dim_noise,))
        dW2 = np.matmul(rho1, dW1) + np.sqrt(self.h)*np.matmul(rho2, np.random.normal(size=(self.sde.dim_noise,)))

        self.X1 = self.X1 + self.h*self.sde.drift(self.X1) + np.matmul(self.sde.sigma(self.X1), dW1)
        self.X2 = self.X2 + self.h*self.sde.drift(self.X2) + np.matmul(self.sde.sigma(self.X2), dW2)
        self.step_n += 1

        reward = - (self.test_function(self.X1) * self.test_function(self.X2) - value)
        
        terminated = bool(self.step_n >= self.N_euler)
        truncated = False
        info = {}

        return (
            np.concatenate([self.X1, self.X2, self.h*self.step_n*np.ones(1)], axis=0).astype(np.float32),
            reward,
            terminated,
            truncated,
            info
        )


    def render(self):
        if self.render_mode == "console":
            print("Current position: X1: {}, X2: {}, time: {}"
                  .format(self.X1, self.X2, self.step_n*self.h))

    def close(self):
        pass





class BatchSDEEnvironment:

    def __init__(
            self,
            sde,
            T,
            N_euler,
            batch_test_function = None,
            batch_action_param = None,
    ):
        super().__init__()
        self.sde = sde
        self.T, self.N_euler, self.h = T, N_euler, T/N_euler
        self.test_function = batch_test_function
        self.action_param = batch_action_param

        self.step_n = 0
        self.X1, self.X2 = None, None
    

    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        self.step_n = 0
        self.X1 = torch.tile(torch.Tensor(self.sde.X0), [self.batch_size, 1])
        self.X2 = torch.tile(torch.Tensor(self.sde.X0), [self.batch_size, 1])
        return torch.concat([self.X1, self.X2, torch.zeros((self.batch_size,1))], axis=1), {}


    def step(self, action):
        batch_size = len(self.X1)

        value = self.test_function(self.X1) * self.test_function(self.X2)

        rho1 = self.action_param.get_matrix(action)
        rho2 = self.action_param.get_reverse_matrix(action)

        dW1 = np.sqrt(self.h) * torch.normal(0., 1., size=(batch_size, self.sde.dim_noise))
        dW2 = torch_matvec(rho1, dW1) + np.sqrt(self.h)*torch_matvec(rho2, torch.normal(0., 1., size=(batch_size, self.sde.dim_noise)))

        self.X1 = self.X1 + self.h*self.sde.batch_drift(self.X1) + torch_matvec(self.sde.batch_sigma(self.X1), dW1)
        self.X2 = self.X2 + self.h*self.sde.batch_drift(self.X2) + torch_matvec(self.sde.batch_sigma(self.X2), dW2)
        self.step_n += 1

        reward = - (self.test_function(self.X1) * self.test_function(self.X2) - value)
        terminated = bool(self.step_n >= self.N_euler)

        return (
            torch.concat([self.X1, self.X2, self.h*self.step_n*torch.ones((len(self.X1),1))], axis=1),
            reward,
            terminated,
            False,
            {}
        )
    



def torch_matvec(mat, vec):
    return torch.einsum('ijk,ik->ij', mat, vec)




# same but reward only at the end
# class SDEEnvironment2(SDEEnvironment):

#     def step(self, action):

#         rho1 = self.action_param.get_matrix(action)
#         rho2 = self.action_param.get_reverse_matrix(action)

#         dW1 = np.sqrt(self.h) * np.random.normal(size=(self.sde.dim_noise,))
#         dW2 = np.matmul(rho1, dW1) + np.sqrt(self.h)*np.matmul(rho2, np.random.normal(size=(self.sde.dim_noise,)))

#         self.X1 = self.X1 + self.h*self.sde.drift(self.X1) + np.matmul(self.sde.sigma(self.X1), dW1)
#         self.X2 = self.X2 + self.h*self.sde.drift(self.X2) + np.matmul(self.sde.sigma(self.X2), dW2)
#         self.step_n += 1

#         if self.step_n >= self.N_euler:
#             reward = self.test_function(self.X1) * self.test_function(self.X2)
#         else:
#             reward = 0.
        
#         terminated = bool(self.step_n >= self.N_euler)
#         truncated = False
#         info = {}

#         return (
#             np.concatenate([self.X1, self.X2, self.h*self.step_n*np.ones(1)], axis=0).astype(np.float32),
#             reward,
#             terminated,
#             truncated,
#             info
#         )



# would be better with Dict but maybe not supported by StableBaselines...
# my_spaces = {
#     'X1': gym.spaces.Box(low=0., high=np.inf, shape=(sde.dim,), dtype=np.float32),
#     'X2': gym.spaces.Box(low=0., high=np.inf, shape=(sde.dim,), dtype=np.float32),
#     'time': gym.spaces.Box(low=0., high=T, shape=(sde.dim,), dtype=np.float32)
#     }
# self.observation_space = gym.spaces.Dict(my_spaces)

# https://stackoverflow.com/questions/58964267/how-to-create-an-openai-gym-observation-space-with-multiple-features