from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from torch.distributions import Normal

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
        self.action_range = action_range#TOD
        self.action_scale = (action_range[1]-action_range[0])/2
        self.action_bias = (action_range[1]+action_range[0])/2
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    # @property
    def alpha(self):
        # ODO: Formulate entropy term
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # f=MLPPolicySAC.alpha
        # ODO: return sample from distribution if sampling
        obs = ptu.from_numpy(obs)
        # if not sampling return the mean of the distribution
        dist = super().forward(obs)
        logstd = torch.clamp(self.logstd,self.log_std_bounds[0],self.log_std_bounds[1])
        scale = torch.exp(logstd)
        action_distribution = sac_utils.SquashedNormal(loc=dist.mean,scale=scale)
        if sample:
            action = action_distribution.sample()
        else:
            action = action_distribution.mean
        return ptu.to_numpy(action*self.action_scale + self.action_bias)#.cpu().detach()

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing action_range
        # dist = super().forward(observation)
        # mu,self.log_std=dist.mean,dist.logs
        # log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        # std = log_std.exp()
        # dist = Normal(0, 1)
        # e = dist.sample().to(device)
        # action = torch.tanh(mu + e * std)
        # log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        #V2
        batch_mean = self.mean_net(observation)
        logstd = torch.clamp(self.logstd,self.log_std_bounds[0],self.log_std_bounds[1])
        scale = torch.exp(logstd)*torch.ones_like(batch_mean)
        action_distribution = sac_utils.SquashedNormal(loc=batch_mean,scale=scale)
        act_pred = action_distribution.rsample()
        log_prob=torch.sum(action_distribution.log_prob(act_pred),dim=1,keepdim=True)

        return act_pred*self.action_scale + self.action_bias, log_prob

        #alternativ
        # mu,std=dist.mean,dist.stddev
        # dist = Normal(0, 1)
        # e = dist.sample()
        # act_pred = torch.tanh(mu + e * std)
        # log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - act_pred.pow(2) + 1e-6)
        # return act_pred,log_prob

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
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