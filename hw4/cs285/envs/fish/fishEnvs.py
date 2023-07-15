#%%
import math
import time
import cv2
import numpy as np
import timeit
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from collections import deque
#%%

class forceSensorWrapEnv(gym.Env):
    def __init__(self, pi_conn, act_scale=60, EP_STEPS=768):
        self.act_scale = act_scale
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_width, self.resize_height, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.conn = pi_conn
        self.N = EP_STEPS
        self.buf_counter = 0
        self.step_counter = 0
        self.FORMAT = 'utf-8'
        print('connected')

    def step(self, action):
        self.conn.sendall(str(action).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        rew, Fy, Fy_dot = np.array(sData.split(',')).astype(np.float32)
        self.step_counter += 1
        done = True if self.step_counter >= self.N else False
        return [Fy,Fy_dot,action], rew, done, {}

    def reset(self):
        self.conn.sendall(str('RESET').encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        self.step_counter = 0
        return sData


class FishStationary(gym.Env):
    '''

    '''
    def __init__(self,
                 Gamma=60,
                 tau=0.05,
                 seed=0,
                 rwd_method=2,
                 _max_episode_steps=800,
                 omega0=13,
                 Omega=7,  ## rad/s
                 delta_phi=0.26,  ## rad
                 eta=0.5,
                 startCondition='random',
                 gap=None,
                 FyBias=None,
                 sensorRot=0,
                 noiseFx=None,
                 targetFxtype='Fx',
                 discrete_actions=True
                 ):
        super(FishStationary, self).__init__()
        ### Control parameters
        self.rwd_method = rwd_method
        self.Gamma = Gamma
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(90)
        self.FyBias = FyBias
        self.targetFxtype = targetFxtype

        self.startCondition = startCondition
        self.seed(seed)
        self.viewer = None
        self._max_episode_steps = _max_episode_steps
        self.target_counter = 0
        self.step_current = 0
        self.start = timeit.default_timer()
        self.timesteps_current = timeit.default_timer()

        ### physical parameters
        ## Servo motor to fish
        self.Omega = Omega
        self.delta_phi = delta_phi
        self.ratio_phi_alpha = 0.55
        self.ratio_non_linear = 1.6  ## rad-2
        self.gap = gap
        self.sensorRot = sensorRot
        ## Fish
        self.mass = 0.5
        self.length = 0.08  ## in m

        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 11.3e-3  # K_\alpha'' in N.rad-1.s2
        self.Ka1 = 147e-3  # K_\alpha'
        # self.Ct = 0.04  ## Ct = rho l**4 = 1e3*(0.08)**4 =0.041
        self.Ct = self.Ka2

        low = np.array([-3.0,  ## Fy
                        -3.0 / self.tau,  ## Fy_dot
                        # np.deg2rad(-90), ## alpha, in rad
                        # -np.deg2rad(360), ## alpha_dot in rad/s
                        np.deg2rad(-self.Gamma),  ## action
                        ],
                       dtype=np.float32)

        high = np.array([3.0,  ## Fy
                         3.0 / self.tau,  ## Fy_dot
                         # np.deg2rad(90),
                         # np.deg2rad(360),
                         np.deg2rad(self.Gamma),
                         ],
                        dtype=np.float32)
        # does it need float32?, maybe float16 is sufficient
        self.discrete_actions=discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initiation of state and intermedian variables
        self.state = None
        self.alpha = None
        self.alpha_dot = None
        self.angleState = []
        self.phi = None
        self.Fy_last = None
        self.avgFx = None
        self.dList = []

        self.reward = self.reward_2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reward_2(self, done, Fx):
        if not done:
            reward = Fx
        else:
            reward = 0
        return reward

    def step(self, action):
        if len(action.shape)>0:
            action=action[0]
        # sensing
        Fy, Fy_dot, act = self.state
        alpha, alpha_dot = self.angleState
        phi = self.phi
        if self.discrete_actions:
            phi_c= (action-1)*np.deg2rad(self.Gamma)
        else:
            phi_c = action * np.deg2rad(self.Gamma)
        act = phi_c
        # Physical model
        ### Servo motor
        phi_dot = self.Omega * np.tanh((phi_c - phi) / self.delta_phi)
        phi = phi + phi_dot * self.tau  ## or do the integration

        if self.gap is not None:
            if phi > np.deg2rad(self.gap) or phi < np.deg2rad(-self.gap):
                alpha_c = self.ratio_phi_alpha * phi
            else:
                alpha_c = 0
        else:
            alpha_c = self.ratio_phi_alpha * phi

        ### 1.linear approximation
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
        Fy = - self.Ka2 * alpha_dd - self.Ka1 * alpha_dot
        if self.FyBias is not None:
            Fy += self.FyBias

        Fx = Fy * np.sin(alpha)
        Fx2 = - self.Ka2 * alpha_dd * np.sin(alpha)
        Fx3 = - self.Ka2 * alpha_dd * alpha

        ### small angle beta between the sensor's axis and the fish's axis
        beta = np.deg2rad(self.sensorRot)
        trans = np.array([
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)]
        ])

        if self.targetFxtype == 'Fx':
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx, Fy]))
        elif self.targetFxtype == 'Fx3':
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx3, Fy]))

        self.avgFx = (self.avgFx * (self.step_current) + Fx_mesure) / (self.step_current + 1)
        Fy_dot = (Fy_mesure - self.Fy_last) / self.tau
        ### Alpha update
        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        ### renew the state and memory
        self.state = np.array([Fy_mesure, Fy_dot, act], dtype=np.float32)  # QA: np.float16!?
        self.angleState = [alpha, alpha_dot]
        self.phi = phi
        self.Fy_last = Fy_mesure

        done = bool(
            # self.target_counter >= self.target_number
            self.step_current >= self._max_episode_steps
            # or abs(alpha) >= 2
        )
        reward = self.reward(done, Fx_mesure)
        self.step_current += 1
        self.timesteps_current = timeit.default_timer() - self.start

        return np.array(self.state), reward, done, {'TimeLimit.truncated': self.step_current >= self._max_episode_steps}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.state = np.zeros(shape=(3,))
        self.state[2] = 0
        if self.startCondition == 'random':
            self.alpha = self.np_random.uniform(low=-np.deg2rad(self.Gamma), high=np.deg2rad(self.Gamma))
        else:
            self.alpha = 0
        # print(self.state)
        # self.alpha_dot = 0  # self.np_random.uniform(low=-0.1, high=0.1, size=(1,))
        self.angleState = [self.alpha, 0]
        self.phi = 0
        self.avgFx = 0
        self.Fy_last = 0
        self.steps_beyond_done = None
        self.step_current = 0
        self.start = timeit.default_timer()
        self.timesteps_current = timeit.default_timer()
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class FishMoving(gym.Env):
    '''
    reward 1300-1500 learned
    '''
    def __init__(self,
                 Gamma=60,
                 tau=0.02,
                 seed=0,
                 rwd_method=1,
                 EP_STEPS=768,
                 omega0=11.8,
                 eta=1.13 / 2,
                 startCondition='random',
                 targetU=1,
                 kPunish=0.1,
                 discrete_actions = False
                 ):

        super(FishMoving, self).__init__()
        ### Control parameters
        self.rwd_method = rwd_method
        self.Gamma = np.deg2rad(Gamma)
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(90)
        self.targetU = targetU
        self.kPunish = kPunish

        self.startCondition = startCondition
        self.seed(seed)
        self.viewer = None
        self.N_STEPS = EP_STEPS
        self.target_counter = 0
        self.step_current = 0
        self.start = timeit.default_timer()
        self.discrete_actions = discrete_actions

        ## Fish
        self.mass = 1
        self.length = 0.08  ## in m
        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 12.9e-3  # K_\alpha'' in N.rad-1.s2
        self.Ka1 = 39.9e-3  # K_\alpha'
        self.Ct = self.Ka2
        self.Cd = 0.254  ## from Jesus's PRL paper

        low = np.array([-0.1,  ## x, Umax = 0.1, 0.1*768*tau ~ 1.6
                        -100.0,
                        -np.deg2rad(75),  ## alpha, in rad
                        -100.0,
                        ],
                       dtype=np.float32)
        self.x_threshold = 50
        high = np.array([self.x_threshold,
                         100.0,
                         np.deg2rad(75),
                         100.0,
                         ],
                        dtype=np.float32)
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1,1,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        # Initiation of state and intermedian variables
        self.state = None
        self.avgAlpha = None
        self.dList = []
        self.viewer = None
        # Determination of reward
        self.reward = self.rew_map(counter=rwd_method)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def rew_map(self,counter):
        def reward_1(done):
            if not done:
                reward = self.state[1]
            else:
                reward = 0
            return reward

        def reward_2(done):
            if not done:
                reward = - abs(self.state[1] - self.targetU)
            else:
                reward = 0
            return reward

        def reward_3(done):
            if not done:
                reward = - abs(self.state[1] - self.targetU) - self.kPunish * abs(self.avgAlpha)
            else:
                reward = 0
            return reward
        if counter==1:
            return reward_1
        elif counter==2:
            return reward_2
        elif counter==3:
            return reward_3
        else:
            raise EnvironmentError

    def get_reward(self, obs,candidate_action_sequences):
        return obs[:,1].copy()

    def step(self, action):
        # if len(action)==2:
        #     action=action[0]
        err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(np.array(action, dtype=np.float64)), err_msg
        # assert self.observation_space.contains(self.state), f'state {self.state}'
        [x, x_dot, alpha, alpha_dot] = self.state

        if self.discrete_actions:
            alpha_c = (action-1)*(self.Gamma)
        else:
            alpha_c = action[0] * (self.Gamma)

        # if alpha >= self.Gamma and alpha_c>=0:
        #     alpha_dd = 0
        #     alpha_dot=0
        # elif alpha<=-self.Gamma and alpha_c<=0:
        #     alpha_dd = 0
        #     alpha_dot=0
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
        x_dd = (- self.Cd * x_dot * np.abs(x_dot) - self.Ct * alpha_dd * alpha) / self.mass

        x_dot = x_dot + self.tau * x_dd
        x = x + self.tau * x_dot
        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        self.state = [x, x_dot, alpha, alpha_dot]

        done = bool(
            self.step_current >= self.N_STEPS
        )
        reward = self.reward(done)

        self.step_current += 1
        self.avgAlpha = (self.avgAlpha * (self.step_current) + alpha) / (self.step_current + 1)

        return np.array(self.state), reward, done, {'ep_length':self.step_current}

    def reset(self):
            #self.alpha = self.np_random.uniform(low=-0.05, high=0.05)
        self.state = np.zeros(shape=(4,),dtype=np.float32)
        self.avgAlpha = self.state[2]
        self.step_current = 0
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


