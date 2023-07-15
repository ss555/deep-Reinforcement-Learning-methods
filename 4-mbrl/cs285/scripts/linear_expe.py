'''
behavioral cloning improves the speed of convergence
uses VAE in latent space

onpolicy:
dones=False
        while not dones:
        # while n_steps < n_rollout_steps:
        RolloutBuffer:

        if not self.full and self.n_envs==1:
            self.buffer_size=self.pos
            self.actions=self.actions[:self.pos]
            self.rewards=self.rewards[:self.pos]
            self.episode_starts=self.episode_starts[:self.pos]
            self.values=self.values[:self.pos]
            self.log_probs=self.log_probs[:self.pos]
            self.advantages=self.advantages[:self.pos]
            self.returns=self.returns[:self.pos]
'''

import sys
import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from rlutils.imitation.algorithms import bc
from rlutils.imitation.data import rollout
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from rlutils.imitation.data.wrappers import RolloutInfoWrapper
from rlutils.envs import FishStationary
from rlutils.policies import SwingPolicy, bangBangControl
from rlutils.utils import make_dir_exp, plot_data_from_dirs, read_hyperparameters,read_video_to_frames
from stable_baselines3 import DQN, PPO, SAC
from rlutils.utils import plot_html_eps_from_inference_experiment, play_episodes, read_hyperparameters, save_plot_best_episode, all_files_from_extension, plot_data_from_dir, save_env_any_config, CustomCNN
from rlutils.env_wrappers import make_dir_exp
from rlutils.utils import inference_from_path
from rlutils.custom_callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback, EvalCustomCallback,CheckPointEpisode
from rlutils.env_wrappers import make_cnn_render_env_fish, normalize, LoggerWrap, Monitor
from RobotFishEnv import * #RobotFishEnv,  Fish_obs_Fy_Fyd, Fish_Omega_obs_Fy_Fyd_action,
import os
from rlutils.envs import register_fish_envs
from glob import glob
import cv2
from rlutils.imitation.data.rollout import  TrajectoryAccumulator,flatten_trajectories_with_rew
from RobotFishEnv import camera_linear
import socket
from rlutils.vision import *
from rlutils.VAE_model import * #ini_encoder_inference
from rlutils.env_wrappers import make_cnn_render_env_fish, vaeEncoderWrapper, HistoryWrapper,make_dir_exp, makeMultiEnvMonitor
from datetime import datetime
from copy import deepcopy
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


def imitation_bc(model, env, buffer_traj=None,semi_period=8):
    #imitation
    expert_score=0
    def train_expert(venvv, policy=bangBangControl):
        # expert = policy(env=venvv, semi_period=None)#1-2hz 10 times
        expert = policy(env=venvv, semi_period=semi_period)
        return expert

    # if semi_period==None:#steps
    #     episodes_c=10
    #     n_epochs=1
    # else: #OVERFIT?
    #     episodes_c=3
    #     n_epochs=5
    episodes_c = 1
    n_epochs = 10

    rng = np.random.default_rng(0)
    def sample_expert_transitions():
        print("Sampling expert transitions.")
        # venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)]) if ENV_NAME != 'stationary_imgs' else RolloutInfoWrapper(env)
        venv = model.get_env()  # DummyVecEnv([lambda: RolloutInfoWrapper(env)]) #RolloutInfoWrapper(env)
        # if type(venv)==DummyVecEnv:
        #     venv=venv.unwrapped.envs[0]
        # venv = RolloutInfoWrapper(env)
        expert = train_expert(venv)
        rollouts = rollout.rollout(
            expert,
            venv,
            rollout.make_sample_until(min_timesteps=None, min_episodes=episodes_c),
            rng=rng,
            unwrap=False
        )
        expert_score = rollouts[0].rews.sum()
        print(f'expert score: {expert_score}, shape ')

        return rollout.flatten_trajectories(rollouts)
    with open(f'./trajs_{datetime.now()}.pkl', 'wb+') as file:
        pickle.dump(np.array([0]), file)
    trajectories = sample_expert_transitions()
    with open(f'./trajs_{datetime.now()}-expert-{expert_score}-e-{episodes_c}.pkl', 'wb+') as file:
        pickle.dump(trajectories, file)
    bc_trainer = bc.BC(
                policy=model.policy,
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=trajectories,
                rng=rng,
            )
    bc_trainer.train(n_epochs=n_epochs)

def make_cnn_env(learn_mode,vid,s):
    print('learning with CNN on imgs')
    env = camera_linear(cap=vid, pi_conn=s, reward_mode='speed', img_crop=(learn_mode=='img_crop'))
    # env = LoggerWrap(env,save_only_reward=True)
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)
    env = VecFrameStack(env, 4)
    env = VecTransposeImage(env)
    params = read_hyperparameters(name='fish', path=os.path.abspath('/home/sardor/1-THESE/2-Robotic_Fish/2-DDPG/deepFish/docs/weightsParams/ppo.yml'))
    return env


def make_encoder_env_speed(vid, s, monitor_dir,max_episode_steps=256):
    print('learning with VAE on imgs')
    encoder = ini_encoder_inference('./../docs/weightsParams/linear-fish/zoom4/best_vae_model.pt', encoder_type=EncoderVar)
    # env = camera_linear(cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)
    # # env = LoggerWrap(env, save_without_obs=True, pickle_images=False)
    # env = vaeEncoderWrapper(env, encoder, mode_vae='mean')  # sample #raw
    #multimodal env
    env = fish_exp_vae_multi_speed(encoder=encoder, cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)  # sample #raw
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    print('using encoder env')
    env = HistoryWrapper(env, 4, concatenate_actions=True)
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)

    params = read_hyperparameters(name='fish_vae_v1', path='../docs/weightsParams/ppo.yml')
    return env,params

def make_encoder_env(vid, s, monitor_dir):
    print('learning with VAE on imgs')
    encoder = ini_encoder_inference('./../docs/weightsParams/linear-fish/zoom4/best_vae_model.pt', encoder_type=EncoderVar)
    # env = camera_linear(cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)
    # # env = LoggerWrap(env, save_without_obs=True, pickle_images=False)
    # env = vaeEncoderWrapper(env, encoder, mode_vae='mean')  # sample #raw
    #multimodal env
    env = fish_exp_vae_multi(encoder=encoder, cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)  # sample #raw
    env = TimeLimit(env, max_episode_steps=768)
    print('using encoder env')
    env = HistoryWrapper(env, 4, concatenate_actions=True)
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)

    params = read_hyperparameters(name='fish_vae_v1', path='../docs/weightsParams/ppo.yml')
    return env,params

def make_sensor_env(vid,s,monitor_dir):
    pass
    print('QQQ learning with sensor values')
    env = fish_exp_vae_multi(encoder=encoder, cap=vid, pi_conn=s, reward_mode='speed')
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)
    return env #sensor based cx, cy,pwm

def make_red_yellow_env(vid, s, monitor_dir):
    print('learning with 2pts,action values')
    env = fish_exp_red_yellow(cap=vid, pi_conn=s)
    env = TimeLimit(env, max_episode_steps=768)
    print('using make_red_yellow_env env')
    env = HistoryWrapper(env, 4, concatenate_actions=True)
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)
    params = read_hyperparameters(name='fish_fast', path='../docs/weightsParams/ppo.yml')
    return env, params #sensor based cx, cy,pwm

def make_red_yellow_env_speed(vid, s, monitor_dir,len_episode=256):
    print('learning with 2pts,action values')
    env = fish_exp_red_yellow_speed(cap=vid, pi_conn=s)
    env = TimeLimit(env, max_episode_steps=len_episode)
    print('using make_red_yellow_env env')
    env = HistoryWrapper(env, 4, concatenate_actions=True)
    env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)
    params = read_hyperparameters(name='fish_fast', path='../docs/weightsParams/ppo.yml')
    return env, params #sensor based cx, cy,pwm


if __name__ == '__main__':
    monitor_dir, COUNTER = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs'))) #'../docs/weightsParams/ppo.yml')

    HOST = 'raspberrypi.local'#'192.168.0.10'  # IP address of Raspberry Pi
    PORT = 8080  # same arbitrary port as on server
    inference_path = None#'/home/sardor/1-THESE/2-Robotic_Fish/2-DDPG/deepFish/servo-experiment/logs/76' #vae :li
    vae_path = './../docs/weightsParams/linear-fish/zoom4/best_vae_model.pt' #'./archs/v1/252/best_vae_model.pt'
    #var1
    # encoder=ini_encoder_inference(vae_path,encoder_type=EncoderVar)
    #variant2
    config = std_config_vae()
    vae = VariationalAutoencoder(encoder_model=EncoderVar, decoder_model=DecoderVar,
                                load_path=os.path.join(os.path.dirname(__file__), vae_path),
                                **config)
    encoder = deepcopy(vae.encoder)

    STEPS_TO_TRAIN = 200000
    MANUAL_SEED = 0
    IMITATION=True #have 1 shot imitation
    # DEBUG=True
    DEBUG=False
    if not DEBUG:
        # ENV DEFINITION EXPERIMENT
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect to the server
        s.connect((HOST, PORT))
        vid = cv2.VideoCapture(0)
        # env = camera_linear(cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)
        # env = LoggerWrap(env, save_without_obs=True, pickle_images=False)
        # env = vaeEncoderWrapper(env, encoder, mode_vae='mean')  # sample #raw
        #VAE ENV multi
        env = fish_exp_vae_multi(encoder=encoder, cap=vid, pi_conn=s, reward_mode='speed', img_crop=True)
        env = HistoryWrapper(env, 4)
        env = makeMultiEnvMonitor(env, n_envs=1, monitor_dir=monitor_dir)
    else:
        #debug
        env = Fish_Omega_obs_Fy_Fyd_action(discrete_actions=True)
        env = LoggerWrap(env, save_without_obs=False, pickle_images=True)
        env, env_name = make_cnn_render_env_fish(env, monitor_dir=monitor_dir)
        
    params = read_hyperparameters(name='fish_vae_v1', path=os.path.join(os.path.dirname(__file__), './../docs/weightsParams/ppo.yml'))


    if inference_path is None:
        model = PPO("MlpPolicy", seed=0, tensorboard_log=f'./CNN/fish-{str(env)}{datetime.now()}',env=env, **params, verbose=0)
        #
        # #eval random
        # obs, rews, acts = play_episodes(env, model, episodes_num=1,return_data=True,deterministic = False)
        # rews = np.array(rews).reshape(-1,)
        # rews = rews[~np.isnan(rews)]
        # print(np.sum(rews))
        #
        if IMITATION:
            imitation_bc(model)
        print("Training a policy using Behavior Cloning")
        checkpoint = CheckPointEpisode(save_path=monitor_dir, save_freq_ep=30, episodes_init=0)

        with ProgressBarManager(STEPS_TO_TRAIN) as cus_callback:
            model.learn(total_timesteps=STEPS_TO_TRAIN, tb_log_name=f'./CNN/fish-{str(env)}{datetime.now()}', callback=[cus_callback,checkpoint])
            # model.learn(total_timesteps=STEPS_TO_TRAIN, tensorboard_log=f'./CNN/fish-{str(env)}{datetime.now()}', callback=[cus_callback,checkpoint])
        # plot_data_from_dir(monitor_dir)

    else:#INFERENCE
        print(f'started inference')
        inference_from_path(inference_path=inference_path, env=env, model_type=PPO,every=5)

