from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from cs285.infrastructure.sac_utils import SquashedNormal

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.action_scale=(self.action_range[1]-self.action_range[0])/2
        self.action_bias=(self.action_range[1]+self.action_range[0])/2
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    # @property
    def alpha(self):
        # TODO: get this from previous HW
        return self.log_alpha.detach().exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW
        obs = ptu.from_numpy(obs)
        mean = self.mean_net(obs)
        if sample:
            log_std = torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1])
            action_distribution = SquashedNormal(mean, torch.exp(log_std)*torch.ones_like(mean))
            action = action_distribution.sample()
        else:
            action = mean
        return ptu.to_numpy(action * self.action_scale + self.action_bias)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        mean=self.mean_net.forward(observation)
        log_std=torch.clamp(self.logstd,self.log_std_bounds[0],self.log_std_bounds[1])
        action_distribution = SquashedNormal(mean,torch.exp(log_std)*torch.ones_like(mean))
        action=action_distribution.rsample()
        log_prob=torch.sum(action_distribution.log_prob(action),dim=1,keepdim=True)
        return action*self.action_scale+self.action_bias,log_prob

    def update(self, obs, critic):
        # TODO: get this from previous HW
        obs = ptu.from_numpy(obs)
        act_pred, log_prob = self.forward(obs)
        log_prob = log_prob.reshape(-1, 1)
        with torch.no_grad():
            ent_coef=self.log_alpha.exp()
        min_qf,_=torch.min(critic.forward(obs, act_pred),dim=1,keepdim=True)
        self.actor_loss = (ent_coef*log_prob-min_qf).mean()

        self.optimizer.zero_grad()
        self.actor_loss.backward()
        self.optimizer.step()

        self.alpha_loss = -(self.log_alpha*(log_prob+self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        self.alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return self.actor_loss, self.alpha_loss, self.alpha()