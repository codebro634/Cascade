from pathlib import Path
from typing import Union, Tuple
import gymnasium as gym
from Agents.Agent import Agent


class ChooserEnv(gym.Env):

    """
        Wrapper env with a discrete action-space which has a size equal to the number of 'agents'.
        If step is called with action i, then the action of the i-th agent in 'agents' with the observation of the original environment
        transformed with the i-ths obs_transform is executed on the original environment.
    """

    def __init__(self, hidden_env: gym.core.Env, agents: list[Union[Agent,Tuple["Class",Path]]], obs_transforms: list[callable] = None):
        self.hidden_env = hidden_env
        self.agents = [agent if isinstance(agent,Agent) else agent[0].load(agent[1]) for agent in agents]
        self.action_space = gym.spaces.Discrete(len(self.agents))
        self.observation_space = hidden_env.observation_space
        self.obs = None
        self.obs_transforms = obs_transforms
        assert obs_transforms is None or  len(obs_transforms) == len(agents), "obs_transforms must be None or have the same length as agents."

    def render(self):
        return self.hidden_env.render()

    def step(self, action: int):
        assert self.obs is not None, "Reset must have been called before first step."
        obs = self.obs if self.obs_transforms is None else self.obs_transforms[action](self.obs)
        x = self.hidden_env.step(self.agents[action].get_action(obs))
        self.obs = x[0]
        return x

    def reset(self, seed=None,options=None):
        x = self.hidden_env.reset()
        self.obs = x[0]
        return x


