import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

#1-step env: Value function should converge towards 3
class ValFuncTest1(gym.Env):

   def __init__(self, continuous_actions: bool = True, num_actions: int = 3, num_obs: int = 3):
      if continuous_actions:
         self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
      else:
         self.action_space = spaces.Discrete(num_actions)
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
      self.num_obs = num_obs

   def step(self, action):
      return np.array([random.uniform(-5,5) for _ in range(self.num_obs)]), 3, True, False, {}

   def render(self):
      pass

   def close(self):
      pass

   def reset(self, seed=None,options=None):
      return np.array([random.uniform(-5,5) for _ in range(self.num_obs)]), {}


#1-step env: Value function should learn that the first entry in obs, will be the received reward in that step.
class ValFuncTest2(gym.Env):

   def __init__(self, continuous_actions: bool = True, num_actions: int = 3, num_obs:int = 3):
      if continuous_actions:
         self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
      else:
         self.action_space = spaces.Discrete(num_actions)
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
      self.num_obs = num_obs
      self.next_reward = None

   def step(self, action):
      return np.array([random.uniform(-5,5) for _ in range(self.num_obs)]), self.next_reward, True, False, {}

   def render(self):
      pass

   def close(self):
      pass

   def reset(self, seed=None,options=None):
      self.next_reward = random.randint(0,1)
      return np.array([self.next_reward] + [random.uniform(-5,5) for _ in range(self.num_obs-1)]), {}

#2-step env: The reward is not immediate. Value function should learn that the reward of the inital state is gamma * 4 (gamma is Agent dependent thus not specified here)
class ValFuncTest3(gym.Env):

   def __init__(self, continuous_actions: bool = True, num_actions: int = 3, num_obs: int = 3):
      if continuous_actions:
         self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
      else:
         self.action_space = spaces.Discrete(num_actions)
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
      self.num_obs = num_obs
      self.stepped = False

   def step(self, action):
      if not self.stepped:
         self.stepped = True
         return np.ones(self.num_obs), 0, False, False, {}
      else:
         return np.array([random.uniform(-5,5) for _ in range(self.num_obs)]), 4, True, False, {}

   def render(self):
      pass

   def close(self):
      pass

   def reset(self, seed=None,options=None):
      self.stepped = False
      return np.zeros(self.num_obs), {}

#1-step env: Agent should learn to always pick action (-1,2) (continuous case) or 0 (discrete case)
class ActionTest1(gym.Env):

   def __init__(self, continuous_actions: bool = True, num_obs:int = 3):
      self.continuous_actions = continuous_actions
      if continuous_actions:
         self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
      else:
         self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
      self.num_obs = num_obs

   def step(self, action):
      if self.continuous_actions:
         dist = abs(-1-action[0]) + abs(2 - action[1])
         reward = -dist
      else:
         reward = 1 if action == 0 else 0
      return np.zeros(self.num_obs), reward, True, False, {}

   def render(self):
      pass

   def close(self):
      pass

   def reset(self, seed=None,options=None):
      return np.zeros(self.num_obs), {}

#1-step env: Agent should learn to pick the current observation as the action.
class ActionTest2(gym.Env):

   def __init__(self, continuous_actions: bool = True, num_obs:int = 1):
      self.continuous_actions = continuous_actions
      if continuous_actions:
         self.action_space = spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
      else:
         self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
      self.num_obs = num_obs
      self.state = None

   def step(self, action):
      if self.continuous_actions:
         dist = abs(self.state-action[0])
         reward = -dist
      else:
         reward = 5 if action == self.state else -5
      return np.zeros(self.num_obs), reward, True, False, {}

   def render(self):
      pass

   def close(self):
      pass

   def reset(self, seed=None,options=None):
      self.state = 5*random.randint(0,1)
      return np.array([self.state]), {}