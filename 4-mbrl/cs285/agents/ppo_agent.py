from collections import OrderedDict
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure.utils import *
import torch as th
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
        self.critic_tau = 0.005
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
        self.replay_buffer = ReplayBuffer(max_size=100000)

        self.batch_size = self.agent_params['batch_size']
        self.n_epochs = self.agent_params['n_epochs']
        self.clip_range = self.agent_params['clip_range']
        self.clip_range_vf = self.agent_params['clip_range_vf']
        self.normalize_advantage = self.agent_params['normalize_advantage']
        self.target_kl = self.agent_params['target_kl']

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # ODO: get this from previous HW
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        for i in range(self.critic_target_update_frequency):
            with th.no_grad():
                next_action, next_log_prob = self.actor.forward(next_ob_no)
                q_nex, _ = th.min(self.critic_target.forward(next_ob_no, next_action), 1, keepdim=True)
                H = -self.actor.log_alpha.exp() * next_log_prob  # th.sum(next_log_prob,dim=1,keepdim=True)#.mean(1).unsqueeze(1)
                target = self.gamma * (1 - terminal_n) * (q_nex + H).squeeze(1) + reward_n
            v_critic = self.critic.forward(ob_no, ac_na)
            self.critic_loss = 0.5 * self.critic.loss(target, v_critic[:, 0]) + 0.5 * self.critic.loss(target,
                                                                                                       v_critic[:, 1])
            # critic_loss/=2
            self.critic.optimizer.zero_grad()
            self.critic_loss.backward()
            self.critic.optimizer.step()

    def train(self, observations, actions, re_n, next_ob_no, terminal_n):


        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = log_prob - rollout_data.old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            continue_training = False
            if self.verbose >= 1:
                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
            break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        #old
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
