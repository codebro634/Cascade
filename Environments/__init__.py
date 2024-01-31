from copy import deepcopy
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete


@dataclass
class EnvSpaceDescription:

    action_space: gym.Space
    observation_space: gym.Space

    @staticmethod
    def get_descr(env: gym.Env):
        return EnvSpaceDescription(action_space=deepcopy(env.action_space), observation_space=deepcopy(env.observation_space))

    def flattened_input_size(self):
        return np.array(self.observation_space.shape).prod()

    def flattened_act_size(self):
        if self.is_discrete_action():
            if self.is_multi_discrete():
                return self.action_space.shape[0]
            else:
                return self.action_space.n
        else:
            return np.array(self.action_space.shape).prod()

    def is_discrete_action(self):
        return isinstance(self.action_space, Discrete) or isinstance(self.action_space,MultiDiscrete)

    def is_multi_discrete(self):
        return isinstance(self.action_space,MultiDiscrete)



