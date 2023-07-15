'''
script for running mbpo on fish env
'''
import os
import time
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.mbpo_agent import MBPOAgent
from cs285.scripts.run_hw4_mbpo import MBPO_Trainer
import torch as th
import gym
import numpy as np
import cv2
import socket
from rlutils.VAE_model import Encoder, std_config_vae
from rlutils.linear_expe import make_red_yellow_env_speed
def main():
    import yaml
    # Parse command line arguments
    try:
        with open("./../params/mbpo_config.yml", "r") as f:
            dic = yaml.safe_load(f)
            kwargs = {'sac_train_batch_size': 1024, 'sac_discount': 0.95, 'init_temperature':0, 'lr': 0.0022923266888452394, 'sac_n_layers': 2, 'sac_size': 256, 'size': 200, 'train_batch_size': 1024, 'mpc_horizon': 37.026879682283884, 'ensemble_size': 3, 'learning_rate': 0.00424382474371494, 'batch_size_initial': 5000, 'mpc_num_action_sequences': 100, 'cem_alpha': 0.1, 'num_agent_train_steps_per_iter': 5000}
            #dic['mbpo_stationary_env']
            config = dic['mbpo_config']
            config.update(kwargs)
    except:
        with open("./params/mbpo_config.yml", "r") as f:
            dic = yaml.safe_load(f)
            kwargs = {'sac_train_batch_size': 1024, 'sac_discount': 0.95, 'init_temperature':0, 'lr': 0.0022923266888452394, 'sac_n_layers': 2, 'sac_size': 256, 'size': 200, 'train_batch_size': 1024, 'mpc_horizon': 37.026879682283884, 'ensemble_size': 3, 'learning_rate': 0.00424382474371494, 'batch_size_initial': 5000, 'mpc_num_action_sequences': 100, 'cem_alpha': 0.1, 'num_agent_train_steps_per_iter': 5000}
            #dic['mbpo_stationary_env']
            config = dic['mbpo_config']
            config.update(kwargs)

    params=config
    logdir_prefix = 'hw4_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    HOST = 'raspberrypi.local'  # '192.168.0.10'  # IP address of Raspberry Pi
    PORT = 8080  # same arbitrary port as on server
    try:
        vid = cv2.VideoCapture(1)
        _, obs = vid.read()
        assert obs.any() != None
    except:
        vid = cv2.VideoCapture(0)
        _, obs = vid.read()
        assert obs.any() != None
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # connect to the server
    s.connect((HOST, PORT))
    env, params = make_red_yellow_env_speed(vid, s, params['logdir'], len_episode=128)
    params['env_name'] = env
    ###################
    ### RUN TRAINING
    ###################
    trainer = MBPO_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
