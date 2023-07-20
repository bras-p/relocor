import torch
from tqdm import tqdm
from copy import deepcopy




class PG:

    def __init__(
            self,
            policy = 'MlpPolicy',
            batch_env = None, # must be batch_env
            batch_size = 512,
            hidden_units_actor = [64, 64],
            ActorOptimizerClass = torch.optim.Adam,
            verbose = 0
            ):
        self.batch_env = deepcopy(batch_env)
        obs, _ = self.batch_env.reset(batch_size)
        self.actor_network = ActorNetwork(
            input_dim = obs.shape[1],
            hidden_units = hidden_units_actor,
            dim_diag = batch_env.action_param.dim_diag,
            dim_cosine = batch_env.action_param.dim_cosine)
        self.actor_optimizer = ActorOptimizerClass(params=self.actor_network.parameters())
        self.batch_size = batch_size
        self.batch_agent = True


    def predict(self, obs, deterministic=True):
        return self.actor_network(obs), None
    
    def learn(self, total_timesteps, progress_bar = True, callback = None, batch_size = None):
        if batch_size is not None:
            self.batch_size = batch_size
        obs, _ = self.batch_env.reset(self.batch_size)

        if progress_bar:
            iterator = tqdm(range(total_timesteps))
        else:
            iterator = range(total_timesteps)

        for k in iterator:

            if callback is not None:
                callback._on_step()
                callback.num_timesteps += 1

            if k != 0:
                obs = obs.detach()
                self.batch_env.X1 = self.batch_env.X1.detach()
                self.batch_env.X2 = self.batch_env.X2.detach()

            self.actor_optimizer.zero_grad()
            action, _ = self.predict(obs)
            obs, reward, terminated, truncated, info = self.batch_env.step(action)
            loss = - torch.mean(reward)
            loss.backward()
            self.actor_optimizer.step()

            if terminated:
                obs, _ = self.batch_env.reset(self.batch_size)







class ActorNetwork(torch.nn.Module):

    def __init__(
            self,
            input_dim,
            dim_diag,
            dim_cosine,
            hidden_units = [64, 64]
            ):
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for k in range(len(hidden_units)-1):
            hidden_layers.append(torch.nn.Linear(hidden_units[k], hidden_units[k+1]))
            hidden_layers.append(torch.nn.ReLU())
        self.sq = torch.nn.Sequential(*hidden_layers)
        self.fc_out_diag = torch.nn.Linear(hidden_units[-1], dim_diag)
        self.fc_out_cosine = torch.nn.Linear(hidden_units[-1], dim_cosine)


    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        x = self.sq(x)
        diag = 2.*torch.sigmoid(self.fc_out_diag(x))  - 1.
        cosine = torch.sigmoid(self.fc_out_cosine(x))
        return torch.concat([diag, cosine], axis=1)



