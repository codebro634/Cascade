import math
import random
import gymnasium as gym
import numpy as np
from Agents.Agent import Agent
from Environments.Utils import extend_flat_space


class CascadeEnv(gym.Wrapper):

    """
        Gym wrapper that extends the action space of the original environment by the so-called fallback action.

        When step is called with action x, then with a probability equal to sigmoid(x[-1]) (last index of x is the fallback-action) the action of 'fallback_agent'
        is executed on the original environment otherwise x[:-1] (x without the fallback action).

        fallback_agent: The agent whose action is taken when fallback is used
        fallback_punishment: Is added to the reward when the fallback_agent has been used
        propagate_value: The observations are extended by the value-function of fallback_agent at each state
        propagate_action: The observations are extended by the action fallback_agent would choose at the current state
    """

    def __init__(self, env: gym.Env, fallback_agent: Agent = None, fallback_punishment: float = 0.0, propagate_value: bool = False, propagate_action: bool = False):
        super().__init__(env)
        self.fallback_agent = fallback_agent

        #Adjust Action-Space / Observation
        low_ext, high_ext  = [], []
        if propagate_value:
            low_ext.append(-np.infty)
            high_ext.append(np.infty)
        if propagate_action:
            low_ext = low_ext + [-np.infty] * np.array(self.action_space.shape).prod()
            high_ext = high_ext +[np.infty] * np.array(self.action_space.shape).prod()
        self.observation_space = extend_flat_space(self.observation_space, low_ext = low_ext, high_ext = high_ext)
        self.action_space = extend_flat_space(self.action_space, low_ext = [-np.infty], high_ext = [np.infty])

        self.last_obs = None
        self.last_action = None
        self.last_val = None

        self.fallback_punishment = fallback_punishment
        self.propagate_value = propagate_value
        self.propagate_action = propagate_action

    def render(self):
        pass

    @staticmethod
    def cascade_step(action, fallback_action, hard_threshold = False, get_fallback: bool = False):
        if fallback_action is None:
            return action
        else:
            # sample if the fallback agent is to be used
            fall_back_prop = 1 / (1 + math.exp(-action[-1]))
            fallback = fall_back_prop >= 0.5 if hard_threshold else random.random() <= fall_back_prop
            if fallback:
                action = fallback_action
            else:
                action = action[0:-1]
            return (action, fallback) if get_fallback else action

    def step(self, action):
        assert self.last_obs is not None, "Must reset env before calling step."

        action,fallback = CascadeEnv.cascade_step(action,self.last_action, hard_threshold=False, get_fallback=True)

        x = list(self.env.step(action))
        if fallback:
            x[1] += self.fallback_punishment #Penalize the fallback action
        self.register_obs(x[0])
        x[0] = self.extend_obs(x[0])
        self.last_obs = x[0]
        return tuple(x)

    def register_obs(self, obs):
        #Calculate propagate values
        if self.propagate_value:
            x = self.fallback_agent.get_action_and_value(obs)
            self.last_val = x["value"]
            self.last_action = x["action"]
        else:
            self.last_action = self.fallback_agent.get_action(obs)

    #Extend observation by value or action of fallback_agent is necessary
    def extend_obs(self, obs):
        if self.last_val is not None and self.propagate_value:
            obs = np.concatenate([obs,[self.last_val]])
        if self.last_action is not None and self.propagate_action:
            obs = np.concatenate([obs,self.last_action])
        return obs


    def reset(self, seed=None,options=None):
        obs, info = self.env.reset()
        self.register_obs(obs)
        obs = self.extend_obs(obs)
        self.last_obs = obs
        return obs,  info