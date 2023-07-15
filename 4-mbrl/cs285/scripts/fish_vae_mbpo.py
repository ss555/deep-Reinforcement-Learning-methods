'''
script for running mbpo on fish env
'''
import os
import time
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.mbpo_agent import MBPOAgent
from rlutils.VAE_model import Encoder,std_config_vae
import torch as th
import gym
class MBPO_Trainer(object):
    def __init__(self, params, trial=None,encoder=None):
        self.encoder=encoder
        #####################
        ## SET AGENT PARAMS
        #####################

        mb_computation_graph_args = {
            'ensemble_size': params['ensemble_size'],
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
        }
        
        sac_computation_graph_args = {
            'n_layers': params['sac_n_layers'],
            'size': params['sac_size'],
            'learning_rate': params['sac_learning_rate'],
            'init_temperature': params['sac_init_temperature'],
            'actor_update_frequency': params['sac_actor_update_frequency'],
            'critic_target_update_frequency': params['sac_critic_target_update_frequency']
        }
        
        mb_train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        sac_train_args = {
            'num_agent_train_steps_per_iter': params['sac_num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['sac_num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['sac_num_actor_updates_per_agent_update'],
            'n_iter': params['sac_n_iter'],
            'train_batch_size': params['sac_train_batch_size']
        }

        estimate_advantage_args = {
            'gamma': params['sac_discount'],
        }

        controller_args = {
            'mpc_horizon': params['mpc_horizon'],
            'mpc_num_action_sequences': params['mpc_num_action_sequences'],
            'mpc_action_sampling_strategy': params['mpc_action_sampling_strategy'],
            'cem_iterations': params['cem_iterations'],
            'cem_num_elites': params['cem_num_elites'],
            'cem_alpha': params['cem_alpha'],
        }

        mb_agent_params = {**mb_computation_graph_args, **mb_train_args, **controller_args}
        sac_agent_params = {**sac_computation_graph_args, **estimate_advantage_args, **sac_train_args}
        agent_params = {**mb_agent_params}
        agent_params['sac_params'] = sac_agent_params

        self.params = params
        self.params['agent_class'] = MBPOAgent
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################
        from rlutils.env_wrappers import make_cnn_render_env_fish
        self.rl_trainer = RL_Trainer(self.params,trial)
        self.env,_ = make_cnn_render_env_fish(env=gym.make(self.params['env_name']), encoder=encoder)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():
    import yaml
    # Parse command line arguments
    with open("./../params/mbpo_config.yml", "r") as f:
        dic = yaml.safe_load(f)
        kwargs = {'sac_train_batch_size': 1024, 'sac_discount': 0.95, 'lr': 0.0022923266888452394, 'sac_n_layers': 2, 'sac_size': 256, 'size': 200, 'train_batch_size': 1024, 'mpc_horizon': 37.026879682283884, 'ensemble_size': 3, 'learning_rate': 0.00424382474371494, 'batch_size_initial': 5000, 'mpc_num_action_sequences': 100, 'cem_alpha': 0.1, 'num_agent_train_steps_per_iter': 5000}
        #dic['mbpo_stationary_env']
        config = dic['mbpo_config']
        config.update(kwargs)


    logdir_prefix = 'hw4_'  # keep for autograder
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + '_' + config['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    config['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################
    VAE_MODEL_PATH = '/home/sardor/1-THESE/2-Robotic_Fish/2-DDPG/deepFish/docs/weightsParams/study_vae/beta/0.01-best-visual-least-decoupled/best_vae_model.pt'  # c-256
    config_vae = std_config_vae()
    encoder = Encoder(**config_vae)
    encoder.load_state_dict(th.load(VAE_MODEL_PATH, map_location=th.device("cuda:0" if th.cuda.is_available() else "cpu"))['enc'])
    print(f'num params vae : {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}')
    encoder.eval()
    trainer = MBPO_Trainer(config, encoder=encoder)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
