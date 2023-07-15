#%%
import math
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import timeit
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
from collections import deque
from rlutils.cnn_utils import save_buffer
from rlutils.utils import rungekutta4
from rlutils.vision import red_point, blue_line,mark_episode,crop_red_point
from timeit import default_timer as timer

#%%
def fish_dynamics(state, t, action,x_dd):  # [_, x_dot, _, alpha_dot] = state # x_dd = (- abs(x_dot) * x_dot - self.K2 * action * alpha) / self.K1
    return np.array([state[1], x_dd, state[3], action])

class RobotFishEnv(gym.Env):
    """
    Description:
        A rotot fish swimming in the water

    Source:
        The environment is built based on the Jesús Sánchez-Rodríguez's model

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Position x                           
        1       Velocity x'        
        2       alpha               
        3       alpha'                  

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Rest still(nee rien faire)
		1	  Tail to the positif +Omega = +alpha''
        2     Tail to the negatif -Omega = -alpha''

        Note: 

    Reward:
         Reward of 1 is awarded if the agent reached the target velocity Ut.
         Reward of -1 is awarded if the alpha is superior to A/L = 0.3
         Reward of 0 is awarded for other situations

    Starting State:
         alpha, x_dot (all) are assigned a uniform random value in [-0.05..0.05]
         # The starting velocity of the fish is always assigned to 0.

    Episode Termination:
         The velocity of fish is more than the goal_velocity, for
		 after _max_episode_steps=3000 timesteps

	a=0.2-0.3
	a'=50
	a''=50

	Physics Model: y/Cp=x
	Eq Finale:
	x’’=-Cp*(x’)^2-K*alpha’’*alpha
	
	
	1/Cp*x’’=-(x’)^2-K*alpha’’*alpha	
	Eq Simplifie:
	y''=-(y')^2 - G*alpha’’*alpha
	G=
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0.2, seed=0):
        #self.L=0.1
        self.Cp = 0.5 # 0.5*1000*(self.L)**2*0.1 #
        self.K = 0.1 #1000*(self.L)**4
        #self.massfish = 1.0
        #self.tail_length = 0.5  
        self.angular_force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.goal_velocity = goal_velocity
        self.forceScale = 5 #TODO: non-linear
        # Angle limit set to 2 * alpha_threshold_radians so failing observation
        # is still within bounds.
        self.alpha_threshold_radians = 0.2
        self.x_threshold = 5
        self.x_dot_threshold = 50
        self.alpha_dot_threshold = 100
        high = np.array([self.x_threshold, ## x
                         self.x_dot_threshold, ## x_dot, maybe we can set some limits for x_dot
                         self.alpha_threshold_radians *2, ## alpha is limited by the A/L ratio
                         self.alpha_dot_threshold],
                        dtype=np.float32)
        # does it need float32?, maybe float16 is sufficient
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self._max_episode_steps = 1000
        self.target_counter = 0
        self.timesteps_current = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, alpha, alpha_dot = self.state
        if action == 0: 
            angular_force = 0 
        elif action == 1:
            angular_force = self.angular_force_mag 
        elif action == 2:
            angular_force = -self.angular_force_mag
        else:
            print('invalid action chosen ')
        alpha_dd = self.forceScale*angular_force
        ## so here I have questions about this relation:
        ## the alpha_dd is just the angular_force that the fish (or the servo motor) exerts on itself, or it's the sum of the exerted openFish-13-01 plus the external openFish-13-01 (hydrodynamic), i.e., Fy?
        ## in the later case, if this external openFish-13-01 should be included in the observation space ?

        ## The model is : x_dd = -Cp * x_dot**2 - K * alpha_dd * alpha
        x_dd = -self.Cp * x_dot**2 - self.K * alpha_dd * alpha

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_dd
            alpha = alpha + self.tau * alpha_dot
            alpha_dot = alpha_dot + self.tau * alpha_dd
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_dd
            x = x + self.tau * x_dot
            alpha_dot = alpha_dot + self.tau * alpha_dd
            alpha = alpha + self.tau * alpha_dot

        self.state = np.array([x, x_dot, alpha, alpha_dot], dtype=np.float32) #QA: np.float16!?

        if abs(alpha) >= self.alpha_threshold_radians:
            reward = -10.0
        elif x_dot >= self.goal_velocity:
            reward = 100.0
        else:
            reward = -1.0 + x_dot/self.goal_velocity
        

        done = bool(
            x_dot >= self.goal_velocity or
            abs(alpha) >= self.alpha_threshold_radians*1.5 or
            self.timesteps_current>self._max_episode_steps
        )
        self.timesteps_current+=1
        return np.array(self.state), reward, done, {}

    def reset(self):

        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.state[0] = 0.0
        # self.state[1] = 0.0
        self.state = np.zeros((4,))
        self.steps_beyond_done = None
        self.timesteps_current = 0
        return np.array(self.state)

    

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400
        fishy=200


        world_width = self.x_threshold * 2
        scale = screen_width / (self.x_threshold*2)
        fishwidth=50.0
        fishheight=30.0
        talewidth=10.0
        talelen=50.0

        if self.viewer is None:
            from rlutils import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -fishwidth / 2, fishwidth / 2, fishheight / 2, -fishheight / 2
            eyeoffset = fishheight / 4.0
            #track
            self.track = rendering.Line((0, fishy), (screen_width, fishy))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            #body
            fish = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.fishtrans = rendering.Transform()
            fish.add_attr(self.fishtrans)
            self.viewer.add_geom(fish)
            #tale
            l, r, t, b = -talewidth / 2, talewidth / 2, talelen - talewidth / 2, -talewidth / 2
            tale = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            tale.set_color(.8, .6, .4)
            self.taletrans = rendering.Transform(translation=(-fishwidth / 2, 0))
            tale.add_attr(self.taletrans)
            tale.add_attr(self.fishtrans)
            self.viewer.add_geom(tale)
            #eye1s
            self.eye1trans= rendering.Transform(translation=(fishwidth / 2, eyeoffset))
            self.eye1 = rendering.make_circle(talewidth / 2)
            self.eye1.add_attr(self.eye1trans)
            self.eye1.add_attr(self.fishtrans)
            self.eye1.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.eye1)
            #eye2
            self.eye2trans = rendering.Transform(translation=(fishwidth / 2, -eyeoffset))
            self.eye2 = rendering.make_circle(talewidth / 2)
            self.eye2.add_attr(self.eye2trans)
            self.eye2.add_attr(self.fishtrans)
            self.eye2.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.eye2)




        if self.state is None: return None

        x, _, alpha, _ = self.state
        fishx = x * scale + screen_width / 2.0  # MIDDLE OF fish
        self.fishtrans.set_translation(fishx, fishy)
        self.taletrans.set_rotation(alpha+math.pi/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class Fish_obs_Fy_Fyd(gym.Env):

    def __init__(self, targetF=0.2, Gamma=60, tau=0.05, seed=0, rwd_method=1, threshold=0.2, target_number=100,
                 _max_episode_steps=768, omega0=10, eta=0.5,
                 startCondition='alpha_n',
                 ):
        super(Fish_obs_Fy_Fyd, self).__init__()

        self.rwd_method = rwd_method
        self.Gamma = Gamma
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(70)
        self.targetF = targetF
        self.threshold = threshold
        self.target_number = target_number  ## target_number steps, keep the velocity around the target value
        self.forceScale = 1  # TODO: non-linear
        self.startCondition = startCondition
        self.seed(seed)
        self.viewer = None
        self._max_episode_steps = _max_episode_steps
        self.target_counter = 0
        self.step_current = 0
        self.start = timeit.default_timer()
        self.timesteps_current = timeit.default_timer()

        ### physical parameters
        self.mass = 0.5
        self.length = 0.08  ## in m
        self.Ct = 0.04  ## Ct = rho l**4 = 1e3*(0.08)**4 =0.041
        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 11.3e-3  # K_\alpha''
        self.Ka1 = 147e-3  # K_\alpha'
        ## m x'' = -cp|x'|x' - k alpha'' alpha -cl alpha alpha |x'|x'
        ## K1 x'' = -x'|x'| - K2 alpha'' alpha - K3 alpha alpha |x'|x'

        low = np.array([-3.0,  ## Fy
                        -60,  ## Fy_dot
                        # np.deg2rad(-90), ## alpha, in rad
                        # -np.deg2rad(360), ## alpha_dot in rad/s
                        ],
                       dtype=np.float32)

        high = np.array([3.0,  ## Fy
                         60,  ## Fy_dot
                         # np.deg2rad(90),
                         # np.deg2rad(360),
                         ],
                        dtype=np.float32)
        # does it need float32?, maybe float16 is sufficient
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.Fy_last = 0
        self.state = None
        self.angleState = None

        self.dList = []
        if self.rwd_method == 1:
            self.reward = self.reward_1
        elif self.rwd_method == 2:
            self.reward = self.reward_2
        elif self.rwd_method == 3:
            self.reward = self.reward_3

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_1(self, done, Fx):
        if not done:
            reward = 0
            if np.abs(Fx - self.targetF) <= np.abs(self.targetF) * self.threshold:
                reward = 1
        else:
            if self.target_counter >= self.target_number:
                reward = 100
            else:
                reward = 0
        return reward

    def reward_2(self, done, Fx):
        if not done:
            reward = Fx
        else:
            reward = 0
        return reward

    def reward_3(self, done, Fx):
        if not done:
            reward = -Fx
        else:
            reward = 0
        return reward

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        Fy, Fy_dot = self.state
        alpha, alpha_dot = self.angleState

        if action == 0:
            thetaC = np.deg2rad(0)  ## degree2rad
        elif action == 1:
            thetaC = np.deg2rad(self.Gamma)
        elif action == 2:
            thetaC = np.deg2rad(-self.Gamma)
        else:
            print('invalid action chosen ')

        # thetaC = thetaC * self.forceScale
        # K_m * x'' = - a''*a - x'**2
        # alpha'' + 2*eta * omega * alpha' + omega**2(alpha-Gamma) = 0

        ## virtual env
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - thetaC)
        ##
        ## Fy = -Ka''*alpha_dd - Ka'*alpha_d - Ka*alpha
        # Fx = (- self.Cd*x_dot * np.abs(x_dot) - self.Ct* alpha_dd * alpha)/self.mass
        Fx = - (self.Ct * alpha_dd * alpha)  # [F]=[ML][1/t*2]
        Fy = - self.Ka2 * alpha_dd - self.Ka1 * alpha_dot
        Fy_dot = (Fy - self.Fy_last) / self.tau

        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        self.Fy_last = Fy
        self.state = np.array([Fy, Fy_dot], dtype=np.float32)  # QA: np.float16!?
        self.angleState = np.array([alpha, alpha_dot], dtype=np.float32)

        if np.abs(Fx - self.targetF) <= self.targetF * self.threshold:
            self.target_counter += 1
        else:
            self.target_counter = 0

        if self.target_counter >= self.target_number:
            self.target_counter = 0

        done = bool(
            # self.target_counter >= self.target_number
            self.step_current >= self._max_episode_steps
            # or abs(alpha) >= 2
        )
        reward = self.reward(done, Fx)
        self.step_current += 1
        self.timesteps_current = timeit.default_timer() - self.start

        dData = {'step': self.step_current, 'time': self.timesteps_current, 'Fx': Fx, 'Fy': self.state[0],
                 'alpha': self.angleState[0], 'alpha_dot': self.angleState[1], 'action': action}
        self.dList.append(dData)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.01, high=0.01, size=(2,))
        if self.startCondition == 'alpha_n':
            self.angleState = np.array([np.deg2rad(-30), np.deg2rad(0.1)], dtype=np.float32)
        else:
            self.angleState = np.array([np.deg2rad(0), np.deg2rad(0.1)], dtype=np.float32)
        self.Fy_last = 0
        self.steps_beyond_done = None
        self.step_current = 0
        self.start = timeit.default_timer()
        self.timesteps_current = timeit.default_timer()
        self.dList = [{'step': 0, 'time': 0, 'Fx': 0, 'Fy': self.state[0], 'alpha': self.angleState[0],
                       'alpha_dot': self.angleState[1], 'action': 0}]
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, title="title"):
        self.df = pd.DataFrame.from_dict(self.dList)

        self.df.loc[self.df['action'] < 2, 'action_readable'] = self.df['action'] * np.deg2rad(self.Gamma)
        self.df.loc[self.df['action'] == 2, 'action_readable'] = -1 * np.deg2rad(self.Gamma)
        self.df.to_csv(title + '.csv')
def rw1(x,x_dot,alpha,alpha_dot , goal_position = 1.0):
#recompense
    reward = -(goal_position-x)**2
    if abs(alpha) >= 1:
        reward -= 100
    if x >= goal_position:
        reward += 3000
    return reward

def rw1_sparse(x,x_dot,alpha,alpha_dot , goal_position = 1.0):
#recompense
    reward = -1#abs(goal_position-x)
    if abs(alpha) >= 1:
        reward -= 100
    if x >= goal_position:
        reward += 3000
    return reward
def rw2(x,x_dot,alpha,alpha_dot):
    # recompense
    reward = -abs(x - 1.0)
    if abs(alpha) >= 1:
        reward -= 2200
    if x >= 1.0:
        reward += 2500
    return reward






class RobotFishGoal(gym.Env):


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self, goal_position=10, seed=0, action_space_is_discrete=True, render_window=False):
        self.Cp = 0.5  # 0.5*1000*(self.L)**2*0.1 #
        self.K = 0.1  # 1000*(self.L)**4
        # self.massfish = 1.0
        # self.tail_length = 0.5
        self.angular_force_mag = 5.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.alpha_threshold_done = 0.3
        self.alpha_threshold_penalize = 0.2
        self.goal_position = goal_position
        self.render_window = render_window

        high = np.array([goal_position*2,  ## x 1
                         100,  ## x_dot, maybe we can set some limits for x_dot 1
                         self.alpha_threshold_done * 2,  ## alpha is limited by the A/L ratio 1
                         100], ## alpha_dot limit if changed put the condition in def step for termination
                        dtype=np.float32)

        self.discrete = action_space_is_discrete
        if action_space_is_discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.x_threshold=self.goal_position+1
        self.seed(seed)
        self.viewer = None
        self.state = None
        self._max_episode_steps = 1000
        self.target_counter = 0
        self.timesteps_current = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, alpha, alpha_dot = self.state
        if self.discrete:
            alpha_dd = self.angular_force_mag * (action-1.0)
        else:
            alpha_dd = self.angular_force_mag * action[0]

        x_dd = -self.Cp * x_dot ** 2 - self.K * alpha_dd * alpha

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_dd
            alpha = alpha + self.tau * alpha_dot
            alpha_dot = alpha_dot + self.tau * alpha_dd
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_dd
            x = x + self.tau * x_dot
            alpha_dot = alpha_dot + self.tau * alpha_dd
            alpha = alpha + self.tau * alpha_dot

        self.state = np.array([x, x_dot, alpha, alpha_dot], dtype=np.float32)  # QA: np.float16!?
        #recompense
        reward = x_dot #-(self.goal_position-x)/self.goal_position
        if abs(alpha) >= self.alpha_threshold_penalize:
            reward -= 10
        if x >= self.goal_position:
            reward += 1000

        done = bool(
            x > self.goal_position or
            abs(alpha) >= self.alpha_threshold_done or
            self.timesteps_current > self._max_episode_steps
        )
        self.timesteps_current += 1
        return np.array(self.state), reward, done, {'ep_length': self.timesteps_current}

    def reset(self):
        self.state = np.zeros((4,))#self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.timesteps_current = 0
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human', goal=True):
        screen_width = 800
        screen_height = 400
        fishy = 200

        world_width = self.x_threshold * 2
        scale = screen_width / (self.x_threshold * 2)
        fishwidth = 50.0
        fishheight = 30.0
        talewidth = 10.0
        talelen = 60.0

        if self.viewer is None:
            if self.render_window:
                from gym.envs.classic_control import rendering
            else:
                from rlutils import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -fishwidth / 2, fishwidth / 2, fishheight / 2, -fishheight / 2
            eyeoffset = fishheight / 4.0
            # track
            self.track = rendering.Line((0, fishy), (screen_width, fishy))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            #goal
            # track
            xCoord=self.goal_position*scale + screen_width / 2.0
            self.track = rendering.Line((xCoord, fishy-10), (xCoord, fishy+10))
            self.track.set_color(1.0, 0, 0)
            self.viewer.add_geom(self.track)
            # body
            fish = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.fishtrans = rendering.Transform()
            fish.add_attr(self.fishtrans)
            self.viewer.add_geom(fish)
            # tale
            l, r, t, b = -talewidth / 2, talewidth / 2, talelen - talewidth / 2, -talewidth / 2
            tale = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            tale.set_color(.8, .6, .4)
            self.taletrans = rendering.Transform(translation=(-fishwidth / 2, 0))
            tale.add_attr(self.taletrans)
            tale.add_attr(self.fishtrans)
            self.viewer.add_geom(tale)
            # eye1s
            self.eye1trans = rendering.Transform(translation=(fishwidth / 2, eyeoffset))
            self.eye1 = rendering.make_circle(talewidth / 2)
            self.eye1.add_attr(self.eye1trans)
            self.eye1.add_attr(self.fishtrans)
            self.eye1.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.eye1)
            # eye2
            self.eye2trans = rendering.Transform(translation=(fishwidth / 2, -eyeoffset))
            self.eye2 = rendering.make_circle(talewidth / 2)
            self.eye2.add_attr(self.eye2trans)
            self.eye2.add_attr(self.fishtrans)
            self.eye2.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.eye2)

        if self.state is None: return None

        x, _, alpha, _ = self.state
        fishx = x * scale + screen_width / 8.0  # MIDDLE OF fish
        self.fishtrans.set_translation(fishx, fishy)
        self.taletrans.set_rotation(alpha + math.pi / 2)
        if goal and x>=self.goal_position:
            print('Reached the goal={}'.format(self.goal_position))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

class cameraDiscrete(gym.Env):
    def __init__(self, cap: cv2.VideoCapture, pi_conn, act_scale=60, _max_episode_steps = 768):
        self.act_scale = act_scale
        MIN_ANGLE = 30
        MAX_ANGLE = 150
        MIN_WIDTH = 950  ## µs
        MAX_WIDTH = 2150  ## µs
        print('MIN_WIDTH={}, MAX_WIDTH={}'.format(MIN_WIDTH, MAX_WIDTH))
        self.MID_WIDTH = (MIN_WIDTH+MAX_WIDTH)/2
        self.servo_scale = (MAX_WIDTH-MIN_WIDTH)/(MAX_ANGLE-MIN_ANGLE)
        self.cap = cap
        self.resize_width = 84
        self.resize_height = 84
        # self.mode = mode
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_width, self.resize_height, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.conn = pi_conn
        self.N = _max_episode_steps
        # self.buffer = np.zeros(shape=(self.N+1, self.resize_width,self.resize_height,1))
        self.buf_counter = 0
        self.step_counter = 0
        self.FORMAT = 'utf-8'
        print('connected')

    def step(self, action):
        pwm = (action - 1.0)*self.act_scale*self.servo_scale + self.MID_WIDTH
        self.conn.sendall(str(pwm).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        rew = state[0]
        ret, obs = self.cap.read()
        obs = self.process(obs)
        self.step_counter += 1
        done = True if self.step_counter >= self.N else False
        return obs, rew, done, {}

    def reset(self):
        self.conn.sendall(str('RESET').encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        ret, obs = self.cap.read()
        self.step_counter = 0
        obs = self.process(obs)
        return obs
    def process(self, img):
        obs = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (self.resize_height, self.resize_width), interpolation = cv2.INTER_CUBIC)
        return np.expand_dims(obs, axis=-1)

class sensor_tcp_general(gym.Env):
    def __init__(self, pi_conn, _max_episode_steps = 768, act_scale=60):
        MIN_ANGLE = 50
        MAX_ANGLE = 130
        MIN_WIDTH = 1100  ## µs
        MAX_WIDTH = 1900  ## µs
        self.MID_WIDTH = (MIN_WIDTH+MAX_WIDTH)/2
        self.servo_scale = (MAX_WIDTH-MIN_WIDTH)/(MAX_ANGLE-MIN_ANGLE)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_width, self.resize_height, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.conn = pi_conn
        self.N = _max_episode_steps
        self.step_counter = 0
        self.FORMAT = 'utf-8'
        print('connected')

    def step(self, action):
        pwm = (action - 1.0)*self.act_scale*self.servo_scale + self.MID_WIDTH
        self.conn.sendall(str(pwm).encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        state = np.array(sData.split(',')).astype(np.float32)
        rew = state[0]
        self.step_counter += 1
        done = True if self.step_counter >= self.N else False
        return state[1:], rew, done, {}

    def reset(self):
        self.conn.sendall(str('RESET').encode(self.FORMAT))
        sData = self.conn.recv(124).decode(self.FORMAT)
        self.step_counter = 0

        return sData

class camera_linear(gym.Env):
    '''
    This is a fish linear movement camera environment
    1. tracks red points in the image and gives reward accordingly to the advancement along the blue line of the tank.
    '''
    def __init__(self, cap: cv2.VideoCapture, pi_conn, _max_episode_steps = 768, act_scale=40, reward_mode='speed', speed_buf_len=20, tau=0.05, img_crop=False):
        MIN_ANGLE = 50
        MAX_ANGLE = 130
        MIN_WIDTH = 1100  ## µs
        MAX_WIDTH = 1900  ## µs

        self.img_crop = img_crop
        self.cap = cap
        self.reward_mode = reward_mode
        self.speed_buf_len = speed_buf_len
        self.speed_buf = deque(np.zeros(speed_buf_len), maxlen=self.speed_buf_len)
        self.tau=tau
        self.pos = 0
        self.last_pos = 0
        self.coun = 0 #counter for reset process
        self.MID_WIDTH = (MIN_WIDTH+MAX_WIDTH)/2
        self.servo_scale = (MAX_WIDTH-MIN_WIDTH)/(MAX_ANGLE-MIN_ANGLE)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.conn = pi_conn
        self.N = _max_episode_steps
        self.step_counter = 0
        self.FORMAT = 'utf-8'
        self.act_scale = act_scale
        _, obs = self.cap.read()
        self.bx, self.by, _ = blue_line(obs)
        print(f'reward mode{self.reward_mode}')# and ini rew :{self.calculate_rew_speed_flag(obs)}')
        self.last_timer=timer()
        print('connected')
        self.reset_comms=[1600,1400]
        self.imu=True
        if self.img_crop:
            print(f'using centered images')

        self.proccesed_obs=self.process(obs)
        plt.imshow(self.proccesed_obs)
        plt.show()

    def step(self, action):
        #pwm and send/receive
        pwm = (action - 1.0)*self.act_scale*self.servo_scale + self.MID_WIDTH
        self.conn.sendall(str(pwm).encode(self.FORMAT))
        _, obs = self.cap.read()
        if self.imu:
            sData = self.conn.recv(124).decode(self.FORMAT)
        #WAIT tau seconds for action to take effect
        if timer()-self.last_timer<self.tau:
            try:
                time.sleep(timer()-self.last_timer)
            except:
                print('err time sleep')
        self.last_timer = timer()
        self.step_counter += 1
        # reward and done
        rew, flag = self.calculate_rew_speed_flag(obs)
        done = (True if self.step_counter >= self.N else False) or flag == 'done'
        return self.process(obs), rew, done, {}


    def ini_pos(self,obs):
        self.cx, self.cy = red_point(obs)
        self.calc_pose()
        self.last_pos = self.pos
        self.speed_buf = deque(maxlen=self.speed_buf_len)
        # self.speed_buf = deque(np.zeros(speed_buf_len), maxlen=self.speed_buf_len)
        self.step_counter = 0
    def calc_pose(self):
        self.pos = np.dot([self.cx, self.cy], np.asarray(self.bx) - np.asarray(self.by)) / 250000
        
    def calculate_rew_speed_flag(self,obs):
        self.cx, self.cy = red_point(obs)
        _, flag = mark_episode(obs, self.cx, self.cy)
        self.calc_pose()
        self.speed_buf.append((self.pos-self.last_pos)/self.tau)
        self.last_pos = self.pos
        rew = np.mean(self.speed_buf) # projection/norm
        rew -= 0.01  # timestep penality
        if flag=='done':#self.cx<0:
            rew += 8
        return rew, flag

    def process(self, img):
        if self.img_crop:
            img = crop_red_point(img, self.cx, self.cy)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (84, 84), interpolation = cv2.INTER_CUBIC)
        return np.expand_dims(img, axis=-1)
    
    def send_com(self,com):
        self.conn.sendall(str(com).encode(self.FORMAT))
        if self.imu:
            sData = self.conn.recv(124).decode(self.FORMAT)
            return sData
    def reset_raspi(self):
        self.conn.sendall(str('RESET').encode(self.FORMAT))
        if self.imu:
            sData = self.conn.recv(124).decode(self.FORMAT)
        flag=''
        while flag!='reset':
            ret, obs = self.cap.read()
            self.cx, self.cy = red_point(obs)
            _, flag = mark_episode(obs, self.cx, self.cy)
            print('reset in progress')
            self.send_com(self.reset_comms[self.coun % 2])
            self.coun+=1
            time.sleep(3)

            # time.sleep(1)
        self.send_com(1500)
        #for stability
        time.sleep(3)

    def reset(self):
        self.reset_raspi()
        ret, self.obs = self.cap.read()
        self.ini_pos(self.obs)
        return self.process(self.obs)
    # def calculate_rew_pos_servoing(self,obs):
    #     self.cx, self.cy = red_point(obs)
    #     _, flag = mark_episode(obs, self.cx,self.cy)
    #     return (np.dot([self.cx,self.cy], np.asarray(self.bx)-np.asarray(self.by))/250000-1.5)**2, flag #projection/norm


class fish_exp_vae_multi(camera_linear):
    '''
    concatenate latent space and action of servo and position cx
    '''
    def __init__(self,encoder,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.encoder = encoder
        self.state_len = 10
        #mean pixels measured on f=2 phi=30deg mode=sin to be used for NORMALISE
        self.mean_pix = 202
        self.std_cx =570
        self.std_cy =30
        self.state = deque(np.zeros(self.state_len), maxlen=self.state_len)
        high = np.array(np.ones_like(self.state), dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def ini_state(self):#worth to init with cx,cy!?
        ret, self.obs = self.cap.read()
        self.ini_pos(self.obs)
        self.state = deque(np.zeros(self.state_len),maxlen=self.state_len)
        self.state.extend([self.encoder(self.obs) ,(self.cx-self.mean_pix)/self.std_cx, 0])

    def step(self, action):
        #pwm and send/receive
        self.action = action
        self.pwm = ( self.action - 1.0)*self.act_scale*self.servo_scale + self.MID_WIDTH
        self.conn.sendall(str(self.pwm).encode(self.FORMAT))
        _, obs = self.cap.read()
        if self.imu:
            sData = self.conn.recv(124).decode(self.FORMAT)
        #WAIT tau seconds for action to take effect
        if timer()-self.last_timer<0.05:
            try:
                time.sleep(timer()-self.last_timer)
            except:
                print('err time sleep')

        self.last_timer=timer()
        self.step_counter += 1
        # reward and done
        rew, flag = self.calculate_rew_speed_flag_state(obs) #self.calc_pose()
        done = (True if self.step_counter >= self.N else False) or flag == 'done'
        return self.state, rew, done, {}

    def reset(self):
        self.reset_raspi()
        self.ini_state()
        return self.state

    def calculate_rew_speed_flag_state(self,obs):
        rew, flag=super().calculate_rew_speed_flag(obs)     #calculate speed in super
        self.state.extend([self.encoder(obs) ,(self.cx-self.mean_pix)/self.std_cx, self.action-1])
        return rew, flag


class Fish_Omega_obs_Fy_Fyd_action(gym.Env):
    '''
    phi_dot = self.Omega * np.tanh((phi_c - phi) / self.delta_phi)
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
        super(Fish_Omega_obs_Fy_Fyd_action, self).__init__()
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
        # action = int(action+1) if type(action) == float
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
        if len(np.array(action).shape)>0:
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

class Fish_ammorti(gym.Env):
    '''
    moving fish env
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
                 discrete_actions = True
                 ):
        super(Fish_ammorti, self).__init__()
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
                        -100,
                        -np.deg2rad(75),  ## alpha, in rad
                        -100,
                        ],
                       dtype=np.float32)
        self.x_threshold = 50
        high = np.array([self.x_threshold,
                         100,
                         np.deg2rad(75),
                         100,
                         ],
                        dtype=np.float32)
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1,1,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

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
        def reward_0(done):
            if not done:
                reward = self.x_dd
            else:
                reward = 0
            return reward

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
        if counter==0:
            return reward_0
        if counter==1:
            return reward_1
        elif counter==2:
            return reward_2
        elif counter==3:
            return reward_3
        else:
            raise EnvironmentError
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.observation_space.contains(self.state), f'state {self.state}'
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
        self.x_dd = x_dd

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
        self.x_dd = 0
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class Fish_Omega_obs_Fy_Fyd_frugal(Fish_Omega_obs_Fy_Fyd_action):
    '''
    inheritence from Fish_Omega_obs_Fy_Fyd_action and implementing frugal-servo-model reward: based on mean force and punish on phi_dot (optional)
    '''
    def __init__(self,
                 targetFx=1,#0.05,
                 tau=0.05,
                 _max_episode_steps=768,#*0.05/tau,
                 sparse=False,
                 frugal_rew=False,
                 discrete_actions=True,#False
                 ):
        '''
        :param targetFx: target force in x direction
        :param tau: time step
        :param _max_episode_steps: number of steps in one episode
        :param sparse: if sparse reward
        :param frugal_rew: if frugal-servo-model reward i.e punish the agent for moving the tail (alpha)
        '''
        super(Fish_Omega_obs_Fy_Fyd_frugal, self).__init__(tau=tau, _max_episode_steps=_max_episode_steps*int(0.05/tau),discrete_actions=discrete_actions)#768*0.05/tau)
        self.targetFx = targetFx
        self.sparse = sparse
        self.frugal_rew = frugal_rew
        # print(self.observation_space.shape)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0]+1,), dtype=np.float32)

    def rew(self):
        if self.step_current != self._max_episode_steps:
            return 0
        else:
            return -np.abs(self.avgFx - self.targetFx) / self.targetFx
    def step(self, action):
        obs, fx, dones, infos = super().step(action)
        alpha, alpha_dot = self.angleState
        self.mean_fx = (self.mean_fx * self.step_current + fx) / (self.step_current + 1)
        self.mean_alpha_dot = (self.mean_alpha_dot * self.step_current + alpha_dot) / (self.step_current + 1)
        if not self.sparse:
            reward = -np.abs(self.mean_fx-self.targetFx)/self.targetFx #- self.mean_fx*alpha_dot
            if self.frugal_rew:
                reward -= self.mean_alpha_dot**2
        else:
            reward = 0 if self.step_current != self._max_episode_steps else -np.abs(self.mean_fx-self.targetFx)/self.targetFx#*self._max_episode_steps
            if self.frugal_rew:
                reward -= alpha_dot**2
        obs = np.concatenate((obs, [fx]))
        # obs = np.concatenate((obs, [self.mean_fx]))
        return obs, reward, dones, infos
    def reset(self):
        obs = super().reset()
        self.mean_alpha_dot=0
        self.mean_fx = 0
        obs = np.concatenate((obs, [self.mean_fx]))
        return obs


class Fish_Omega_obs_Fy_Fyd_action_all(gym.Env):
    '''
    like Fish_Omega_obs_Fy_Fyd_action but with all the possible values in
    - observation [fy,fy_d,act,a,a_dot]
    - state_his_len: the length of the state history
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
                 state_his_len=1,
                 discrete_actions=True,
                 ):

        super(Fish_Omega_obs_Fy_Fyd_action_all, self).__init__()
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
                        np.deg2rad(-90), ## alpha, in rad
                        -np.deg2rad(360), ## alpha_dot in rad/s
                        np.deg2rad(-self.Gamma),  ## action
                        ],
                       dtype=np.float32)

        high = np.array([3.0,  ## Fy
                         3.0 / self.tau,  ## Fy_dot
                         np.deg2rad(90),
                         np.deg2rad(360),
                         np.deg2rad(self.Gamma),
                         ],
                        dtype=np.float32)
        self.state_size = len(high)
        self.state_his_len = state_his_len
        high = np.tile(high, self.state_his_len)
        low = np.tile(low, self.state_his_len)

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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reward(self, done, Fx):
        if not done:
            reward = Fx
        else:
            reward = 0
        return reward

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action)) #USED FOR DEV ALGO
        # assert self.action_space.contains(action.astype(np.float32)), err_msg
        if len(action.shape)>0:
            action=action[0]
        # sensing
        Fy, Fy_dot, act, alpha, alpha_dot = np.array(self.state)[-5:]
        phi = self.phi

        # translate action
        # if self.action_space.
        if action == 0:
            phi_c = np.deg2rad(0)  ## degree2rad
        elif action == 1:
            phi_c = np.deg2rad(self.Gamma)
        elif action == 2:
            phi_c = np.deg2rad(-self.Gamma)
        else:
            print('invalid action chosen')
        act = phi_c
        # Physical model
        ### Servo motor
        phi_dot = self.Omega * np.tanh((phi_c - phi) / self.delta_phi)
        phi = phi + phi_dot * self.tau  ## or do the integration

        ### Servo motor to fin
        # alpha_c = self.ratio_phi_alpha*phi ##
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

        Fx = Fy * np.sin(alpha)
        Fx2 = - self.Ka2 * alpha_dd * np.sin(alpha)
        Fx3 = - self.Ka2 * alpha_dd * alpha

        ### small angle beta between the sensor's axis and the fish's axis
        beta = np.deg2rad(self.sensorRot)
        trans = np.array([
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)]
        ])

        if self.targetFxtype == 'Fx':#DIFFERENT TYPES OF MODELS
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx, Fy]))
        elif self.targetFxtype == 'Fx3':
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx3, Fy]))

        self.avgFx = (self.avgFx * (self.step_current) + Fx_mesure) / (self.step_current + 1)
        Fy_dot = (Fy_mesure - self.Fy_last) / self.tau
        ### Alpha update
        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        ### renew the state and memory

        self.state.extend(np.array([Fy_mesure, Fy_dot, act, alpha, alpha_dot], dtype=np.float32))  # QA: np.float16!?
        self.angleState = [alpha, alpha_dot]
        self.phi = phi
        self.Fy_last = Fy_mesure

        done = bool(
            self.step_current >= self._max_episode_steps
        )
        reward = self.reward(done, Fx_mesure)
        self.step_current += 1
        self.timesteps_current = timeit.default_timer() - self.start
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = deque(np.zeros(self.state_size * self.state_his_len), maxlen=self.state_size * self.state_his_len)
        self.s = np.zeros(shape=(3,))
        self.s[2] = 0
        if self.startCondition == 'random':
            self.alpha = self.np_random.uniform(low=-np.deg2rad(self.Gamma), high=np.deg2rad(self.Gamma))
        else:
            self.alpha = 0
        self.angleState = np.array([self.alpha, 0])
        self.state.extend(np.hstack((self.s, self.angleState)))
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

    def render(self, title="title"):
        pass


class Fish_Omega_obs_Fy_Fyd_action_final_param(gym.Env):
    """
    This env uses the final physical parameters determined with Jesus on 14/01/2022.
    We use the definition Fx3 = - self.Ct * alpha_dd * alpha to do the optimization.
    The observable Fy = - self.Ka2 * alpha_dd - self.Ka1 * alpha_dot.
    """

    def __init__(self,
                 Gamma=60,
                 tau=0.02,
                 seed=0,
                 rwd_method=2,
                 _max_episode_steps=800,
                 omega0=11.8,
                 Omega=6,  ## rad/s
                 delta_phi=0.26,  ## rad
                 eta=1.13 / 2,
                 startCondition='random',
                 gap=None,
                 FyBias=None,
                 sensorRot=0,
                 noiseFx=None,
                 targetFxtype=3,
                 targetFx=0.03,
                 Fx_max=0.08,
                 ):

        super(Fish_Omega_obs_Fy_Fyd_action_final_param, self).__init__()
        ### Control parameters
        self.rwd_method = rwd_method
        self.Gamma = Gamma
        self.tau = tau
        self.kinematics_integrator = 'symplectic'
        self.action_max = np.deg2rad(90)
        self.FyBias = FyBias
        self.targetFxtype = targetFxtype
        self.targetFx = targetFx

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
        self.ratio_phi_alpha = 0.48
        self.ratio_phic_phi = 0.85
        self.ratio_non_linear = 1.6  ## rad-2
        self.gap = gap
        self.sensorRot = sensorRot
        ## Fish
        self.mass = 0.5
        self.length = 0.08  ## in m
        self.omega0 = omega0  # * self.tau  # ~ 10
        self.eta = eta  # ~ 0.5 conjugé, harmonique + exp
        self.Ka2 = 12.9e-3  # K_\alpha'' in N.rad-1.s2
        self.Ka1 = 39.9e-3  # K_\alpha'
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
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initiation of state and intermedian variables
        self.state = None
        self.alpha = None
        self.alpha_dot = None
        self.phi = None
        self.Fy_last = None
        self.Fx = None
        self.avgFx = None
        self.dList = []
        self.Fx_max = Fx_max

        # Determination of reward
        if self.rwd_method == 1:
            self.reward = self.reward_1
        elif self.rwd_method == 2:
            self.reward = self.reward_2
        elif self.rwd_method == 3:
            self.reward = self.reward_3
        elif self.rwd_method == 4:
            self.reward = self.reward_4
        elif self.rwd_method == 5:
            self.reward = self.reward_5
        elif self.rwd_method == 6:
            self.reward = self.reward_6
        elif self.rwd_method == 7:
            self.reward = self.reward_7

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_1(self, done):
        if not done:
            reward = self.avgFx
        else:
            reward = 0
        return reward

    def reward_2(self, done):
        if not done:
            reward = self.Fx
        else:
            reward = 0
        return reward

    def reward_3(self, done):
        if not done:
            reward = - (self.avgFx - self.targetFx) ** 2
        else:
            reward = 0
        return reward

    def reward_4(self, done):
        if not done:
            reward = - (self.Fx - self.targetFx) ** 2
        else:
            reward = 0
        return reward

    def reward_5(self, done):
        if not done:
            reward = self.Fx_max ** 2 - (self.Fx - self.targetFx) ** 2
        else:
            reward = 0
        return reward

    def reward_6(self, done):
        if not done:
            reward = - np.abs(self.Fx - self.targetFx)
        else:
            reward = 0
        return reward

    def reward_7(self, done):
        if not done:
            reward = - np.abs(self.avgFx - self.targetFx)
        else:
            reward = 0
        return reward

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # sensing
        Fy, Fy_dot, act = self.state
        alpha = self.alpha
        alpha_dot = self.alpha_dot
        phi = self.phi

        # translate action
        if action == 0:
            phi_c = np.deg2rad(0) * self.ratio_phic_phi  ## degree2rad
        elif action == 1:
            phi_c = np.deg2rad(self.Gamma) * self.ratio_phic_phi
        elif action == 2:
            phi_c = np.deg2rad(-self.Gamma) * self.ratio_phic_phi
        else:
            print('invalid action chosen')
        act = phi_c
        # Physical model
        ### Servo motor
        phi_dot = self.Omega * np.tanh((phi_c - phi) / self.delta_phi)
        ## phi_dot = self.Omega*np.tanh((self.ratio_phi*phi_c-phi)/self.delta_phi) ## self.ratio_phi links the phi_c and real phi in static regime
        phi = phi + phi_dot * self.tau  ## or do the integration

        ### Servo motor to fin
        # alpha_c = self.ratio_phi_alpha*phi ##
        if self.gap is not None:
            if phi > np.deg2rad(self.gap) or phi < np.deg2rad(-self.gap):
                alpha_c = self.ratio_phi_alpha * phi
            else:
                alpha_c = 0
        else:
            alpha_c = self.ratio_phi_alpha * phi

        ### 1.linear approximation
        alpha_dd = -2 * self.eta * self.omega0 * alpha_dot - self.omega0 ** 2 * (alpha - alpha_c)
        ### 2. slightly non-linear
        # alpha_dd = -2*self.eta*self.omega0*alpha_dot - self.omega0**2*(alpha - alpha_c)*(1-self.ratio_non_linear*(alpha - alpha_c)**2)

        ### Compute forces
        ## Fy = -Ka''*alpha_dd - Ka'*alpha_d - Ka*alpha
        # Fx = (- self.Cd*x_dot * np.abs(x_dot) - self.Ct* alpha_dd * alpha)/self.mass
        # Fx = - (self.Ct* alpha_dd * alpha) # [F]=[ML][1/t*2]
        Fy = - self.Ka2 * alpha_dd - self.Ka1 * alpha_dot
        if self.FyBias is not None:
            Fy += self.FyBias

        Fx = Fy * np.sin(alpha)
        Fx2 = - self.Ct * alpha_dd * np.sin(alpha)
        Fx3 = - self.Ct * alpha_dd * alpha

        ### small angle beta between the sensor's axis and the fish's axis
        beta = np.deg2rad(self.sensorRot)
        trans = np.array([
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)]
        ])

        if self.targetFxtype == 1:
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx, Fy]))
        elif self.targetFxtype == 3:
            Fx_mesure, Fy_mesure = np.dot(trans, np.array([Fx3, Fy]))

        self.avgFx = (self.avgFx * (self.step_current) + Fx_mesure) / (self.step_current + 1)
        Fy_dot = (Fy_mesure - self.Fy_last) / self.tau
        ### Alpha update
        alpha_dot = alpha_dot + alpha_dd * self.tau
        alpha = alpha + self.tau * alpha_dot

        ### renew the state and memory
        self.state = np.array([Fy_mesure, Fy_dot, act], dtype=np.float32)  # QA: np.float16!?
        self.alpha = alpha
        self.alpha_dot = alpha_dot
        self.phi = phi
        self.Fy_last = Fy_mesure
        self.Fx = Fx_mesure

        done = bool(
            # self.target_counter >= self.target_number
            self.step_current > self._max_episode_steps
            # or abs(alpha) >= 2
        )
        reward = self.reward(done)
        self.step_current += 1
        self.timesteps_current = timeit.default_timer() - self.start

        dData = {
            'step': self.step_current,
            'time': self.timesteps_current,
            'Fx': Fx,
            'Fx2': Fx2,
            'Fx3': Fx3,
            'avgFx': self.avgFx,
            'Fy': Fy,
            'alpha': alpha,
            'alpha_dot': alpha_dot,
            'phi_c': phi_c,
            'phi': phi,
            'alpha_c': alpha_c,
            'act': act,
            'reward': reward,
            'action': action,
        }
        self.dList.append(dData)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.state[2] = 0
        if self.startCondition == 'random':
            self.alpha = self.np_random.uniform(low=-np.deg2rad(self.Gamma), high=np.deg2rad(self.Gamma))
        else:
            self.alpha = 0
        # print(self.state)
        self.alpha_dot = 0  # self.np_random.uniform(low=-0.1, high=0.1, size=(1,))
        self.phi = 0
        self.avgFx = 0
        self.Fy_last = 0
        self.steps_beyond_done = None
        self.step_current = 0
        self.start = timeit.default_timer()
        self.timesteps_current = timeit.default_timer()
        self.dList = [{
            'step': 0,
            'time': 0,
            'Fx': 0,
            'Fx2': 0,
            'Fx3': 0,
            'avgFx': 0,
            'Fy': self.state[0],
            'alpha': self.state[1],
            'alpha_dot': self.alpha_dot,
            'phi_c': 0,
            'phi': self.phi,
            'alpha_c': 0,
            'act': 0,
            'reward': 0,
            'action': 0}]
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, title="title"):
        self.df = pd.DataFrame.from_dict(self.dList)

        self.df.loc[self.df['action'] < 2, 'action_readable'] = self.df['action'] * np.deg2rad(self.Gamma)
        self.df.loc[self.df['action'] == 2, 'action_readable'] = -1 * np.deg2rad(self.Gamma)
        self.df.to_csv(title + '.csv')















