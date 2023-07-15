from collections import OrderedDict
from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch
from cs285.infrastructure.sac_utils import soft_update_params

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()
        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.001 # 0.005
        self.learning_rate = self.agent_params['learning_rate']
        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=self.agent_params['buffer_size'])
        # self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # ODO:
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        for i in range(self.critic_target_update_frequency):
            with torch.no_grad():
                next_action,next_log_prob = self.actor.forward(next_ob_no)
                q_nex,_ = torch.min(self.critic_target.forward(next_ob_no,next_action),1, keepdim=True)
                H = -self.actor.log_alpha.exp()*next_log_prob#torch.sum(next_log_prob,dim=1,keepdim=True)#.mean(1).unsqueeze(1)
                target = self.gamma*(1-terminal_n)*(q_nex+H).squeeze(1)+reward_n
            v_critic = self.critic.forward(ob_no, ac_na)
            self.critic_loss = 0.5*self.critic.loss(target,v_critic[:,0]) + 0.5*self.critic.loss(target,v_critic[:,1])
            # critic_loss/=2
            self.critic.optimizer.zero_grad()
            self.critic_loss.backward()
            self.critic.optimizer.step()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        self.training_step+=1
        loss = OrderedDict()
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        if self.training_step%self.critic_target_update_frequency==0:
            soft_update_params(self.critic,self.critic_target,self.critic_tau)
            # print('updated /n')
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        if self.training_step%self.agent_params['actor_update_frequency']==0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                self.actor.update(ob_no,self.critic)
        # 4. gather losses for logging
        # loss['Temperature'] = self.actor
        loss['Critic_Loss'] = self.critic_loss.item()
        loss['Actor_Loss'], loss['Alpha_Loss'], loss['Temperature'] = self.actor.actor_loss.item(),self.actor.alpha_loss.item(),self.actor.log_alpha
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)