import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import mkdir
import csv
from time import time as measure_time
from stable_baselines3 import PPO, A2C, TD3

from .agents.policy_gradient import PG




class Experiment:

    def __init__(
            self,
            env,
            batch_env,
            AgentClass,
    ):
        self.env = env
        self.batch_env = batch_env
        self.AgentClass = AgentClass # agent class from stable_baselines3, typically A2C, PPO etc

        self.variances_log, self.total_rewards_log = [], []
        self.d_time = 0.

    
    def train(self, total_timesteps, verbose=0, batch_eval=0, epoch_eval_freq=1, batch_size=512):
        self.variances_log, self.total_rewards_log = [], []
        self.batch_eval = batch_eval
        self.d_time = 0.

        try:
            self.model = self.AgentClass("MlpPolicy", env=self.env, verbose=verbose)
        except:
            self.model = self.AgentClass("MlpPolicy", batch_env=self.batch_env, verbose=verbose, batch_size=batch_size)
        
        start_time = measure_time()

        if batch_eval == 0:
            self.model.learn(total_timesteps, progress_bar=True)
        else:
            callback = BatchEvalCallback(
                experiment=self, eval_freq=epoch_eval_freq*self.env.N_euler, batch_eval=batch_eval)
            self.model.learn(total_timesteps, progress_bar=True, callback=callback)
            # is it ok if we use both callback and progress bar ?

        end_time = measure_time()
        self.d_time += end_time - start_time


    
    def save_experiment(self, path='./results/my_experiment', notes=''):
        mkdir(path)
        EPOCHS = len(self.total_rewards_log)
        x = np.arange(EPOCHS)
        arrays = {
            'EPOCHS': x,
            'variances_val': np.array(self.variances_log),
            'rewards': np.array(self.total_rewards_log),
            }
        save_np_dict('{}/var.csv'.format(path), arrays)

        f = open("{}/experiment_settings.txt".format(path), "w+")
        txt_dict = {
            'SDE': self.env.sde.name,
            'Action Param': type(self.env.action_param).__name__,
            'Agent': self.AgentClass.__name__,
            'Horizon': 'T: {}, N_euler: {}'.format(self.env.T, self.env.N_euler),
            'Batch Eval size': str(self.batch_eval),
            'Computation time (s)': str(self.d_time),
            'Notes': notes,
        }

        if hasattr(self.model, 'batch_agent') and self.model.batch_agent:
            txt_dict['Batch size'] = str(self.model.batch_size)
        
        for key, value in txt_dict.items():
            f.write('{}: {}\n'.format(key, value))
        f.close()


    def plot_train_variance(self):
        plt.plot(self.variances_log, label='train variance')
        plt.legend()
        plt.title("Evolution of variance given by the policy during training.")
        plt.show()


    def evaluate(self, nb_episodes, batch_size, policy_action = None, use_tqdm=True): # maybe we need to use .todevice ???
        # policy_action: if None then we take the action defined by self.model
        # else it is constant array or tensor defining the policy to evaluate 
        total_reward, f_mean, f2_mean = 0., 0., 0.
        iterator = tqdm(range(nb_episodes)) if use_tqdm else range(nb_episodes)
        for episode in iterator:
            obs, _ = self.batch_env.reset(batch_size)
            for k in range(self.env.N_euler):
                if policy_action is None:
                    action, _ = self.model.predict(obs, deterministic=True)
                    action = torch.Tensor(action)
                else:
                    action = torch.tile(torch.Tensor(policy_action), [batch_size, 1])
                obs, reward, terminated, truncated, info = self.batch_env.step(action)
                total_reward += torch.mean(reward).detach().numpy()[()]
            f_mean += 0.5 * torch.mean(self.batch_env.test_function(self.batch_env.X1.detach()) + self.batch_env.test_function(self.batch_env.X2.detach())).numpy()[()]
            f2_mean += 0.25 * torch.mean((self.batch_env.test_function(self.batch_env.X1.detach()) + self.batch_env.test_function(self.batch_env.X2.detach()))**2).numpy()[()]
        total_reward = total_reward/nb_episodes
        f_mean, f2_mean = f_mean/nb_episodes, f2_mean/nb_episodes
        return f2_mean - f_mean**2, total_reward, f_mean



    def run_trajectory(self): # no need to use batch_env here
        self.traj_X1 = np.zeros([self.env.N_euler+1, self.env.sde.dim])
        self.traj_X2 = np.zeros([self.env.N_euler+1, self.env.sde.dim])
        # traj_action is taken as one concatenated vector
        self.traj_action = np.zeros([self.env.N_euler+1, self.env.action_param.action_space.shape[0]])

        obs, _ = self.env.reset()
        self.traj_X1[0], self.traj_X2[0] = self.env.X1, self.env.X2

        for k in range(self.env.N_euler):
            if hasattr(self.model, 'batch_agent') and self.model.batch_agent:
                action, _ = self.model.predict(torch.Tensor(obs)[None,:], deterministic=True)
                action = np.array(action[0, :].detach())
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            self.traj_action[k] = action
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.traj_X1[k+1], self.traj_X2[k+1] = self.env.X1, self.env.X2

        # fill with a last action to have same length
        # action, _ = self.model.predict(obs, deterministic=True)
        self.traj_action[self.env.N_euler] = action
    

    def display_trajectory(self, state_idxs=[0], action_idxs=[0]):    
        time = np.linspace(0., self.env.T, self.env.N_euler+1)
        fig, axs = plt.subplots(2)

        axs[0].plot(time, self.traj_X1[:,state_idxs], label='X1')
        axs[0].plot(time, self.traj_X2[:,state_idxs], label='X2')
        axs[0].set_title('Trajectories')
        axs[0].legend()

        axs[1].plot(time, self.traj_action[:, action_idxs], label='rho')
        axs[1].set_title('Action')
        axs[1].legend()

        plt.show()

    
    def save_trajectory(self, path='./results/my_experiment'):
        X1s = {'X1{}'.format(k): self.traj_X1[:,k] for k in range(self.traj_X1.shape[1])}
        X2s = {'X2{}'.format(k): self.traj_X2[:,k] for k in range(self.traj_X2.shape[1])}
        rhos = {'rho{}'.format(k): self.traj_action[:,k] for k in range(self.traj_action.shape[1])}
        arrays = {'time': np.linspace(0., self.env.T, self.env.N_euler+1)}
        arrays.update(X1s); arrays.update(X2s); arrays.update(rhos)
        save_np_dict('{}/traj.csv'.format(path), arrays)




def save_np_dict(path, np_dict):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(np_dict.keys())
        for row in zip(*np_dict.values()):
            writer.writerow(row)




from stable_baselines3.common.callbacks import BaseCallback

class BatchEvalCallback(BaseCallback):

    def __init__(
            self,
            experiment,
            eval_freq,
            batch_eval,
            verbose=0
            ):
        super().__init__(verbose)
        self.experiment = experiment
        self.eval_freq = eval_freq
        self.batch_eval = batch_eval

    def _on_step(self):
        start_time = measure_time()
        if self.num_timesteps % self.eval_freq == 0:
            variance, total_reward, mean = self.experiment.evaluate(
                nb_episodes=1, batch_size=self.batch_eval, use_tqdm=False)
            self.experiment.variances_log.append(variance)
            self.experiment.total_rewards_log.append(total_reward)
        end_time = measure_time()
        self.experiment.d_time -= end_time - start_time
        return True




# from stable_baselines3.common.callbacks import EvalCallback
        # else:
        #     eval_callback = EvalCallback(
        #         self.eval_env, log_path='./results/logs/',
        #         eval_freq=self.env.N_euler, deterministic=True, render=False)
        #     self.model.learn(total_timesteps, progress_bar=True, callback=eval_callback)