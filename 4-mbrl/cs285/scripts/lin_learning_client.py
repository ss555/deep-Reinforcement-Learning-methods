'''
for learning to swim in the vein direction.
This is the client side of the learning experiment. It connects to the server
'''
from glob import glob
import os
import sys
from rlutils.env_wrappers import Monitor, normalize, makeMultiEnvMonitor, LoggerWrap
import socket
import time
import cv2
from RobotFishEnv import RobotFishEnv, RobotFishGoal, Fish_obs_Fy_Fyd, Fish_Omega_obs_Fy_Fyd_action
from stable_baselines3 import DQN, PPO, SAC
from rlutils.utils import plot_html_eps_from_inference_experiment, play_episodes, read_hyperparameters, save_plot_best_episode, all_files_from_extension, plot_data_from_dir, save_env_any_config, CustomCNN
from rlutils.env_wrappers import make_dir_exp
from rlutils.utils import inference_from_path
from stable_baselines3.common.atari_wrappers import AtariWrapper
from rlutils.custom_callbacks import CheckPointEpisode, ProgressBarManager, SaveOnBestTrainingRewardCallback, EvalCustomCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from rlutils.env_wrappers import make_cnn_render_env_fish, vaeEncoderWrapper, HistoryWrapper, make_dir_exp, makeMultiEnvMonitor
from rlutils.VAE_model import * #ini_encoder_inference
from datetime import datetime
#preload
from gym.wrappers import TimeLimit
import numpy as np
from RobotFishEnv import * #fish_exp_vae_multi,camera_linear
import traceback
import matplotlib.pyplot as plt
from linear_expe import *

# learn_mode = 'img_crop'#'img','img_crop'
inference_path = '/home/install/Project/deepFish/servo-experiment/logs/142'
# model_path = '/home/install/Project/deepFish/servo-experiment/logs/99/checkpoint780.zip' #'/home/install/Project/deepFish/servo-experiment/logs/20/checkpoint60.zip'
model_path = None
IMITATION = True
STEPS_TO_TRAIN = 100000
MANUAL_SEED = 0

HOST = 'raspberrypi.local'#'192.168.0.10'  # IP address of Raspberry Pi
PORT = 8080  # same arbitrary port as on server
vid = cv2.VideoCapture(0)
_, obs = vid.read()
assert obs.any()!=None
# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# connect to the server
s.connect((HOST, PORT))


monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs'))) #'../docs/weightsParams/ppo.yml')
env, params = make_red_yellow_env_speed(vid, s, monitor_dir, len_episode=256)

# env = make_encoder_env(vid,s,monitor_dir)
# env = make_encoder_env_speed(vid,s,monitor_dir)
#40.199153900146484

policy_str = 'MlpPolicy' if len(env.observation_space.shape)<=2 else 'CnnPolicy'
try:
    if inference_path is None:
        #env
        checkpoint = CheckPointEpisode(save_path=monitor_dir, save_freq_ep=10, episodes_init=0)
        model = PPO(policy_str, seed=MANUAL_SEED, env=env, tensorboard_log=f'./CNN/fish-{str(env)}{datetime.now()}', **params, verbose=0)
        #LOADING PRETRAINED MODEL
        if model_path!=None:
            model.load(model_path)
            print(f'loaded model-{model_path}')

        #imitation learning
        if IMITATION:
            print('imitation learning')
            imitation_bc(model,env)

        #TRAINING
        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, tb_log_name=f'./CNN/fish-{str(env)}{datetime.now()}',callback=[cus_callback, checkpoint])

        # inference_from_path(inference_path=inference_path, env=env, model_type=PPO,every=5)
    else:#INFERENCE
        inference_from_path(inference_path=inference_path, env=env, model_type=PPO,every=1)
except:
    traceback.print_exc()
finally:
    print('closing env')
    env.close()
    try:
        s.close()
    except:
        print('conn problem')
    vid.close()

