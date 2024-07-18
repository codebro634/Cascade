import gymnasium as gym
import numpy as np


#Proudly written by ChatGPT in its entirety. Well done, little AI!
#This action space is fully discretized. If a continuous action space has n dimensions, then the discretized one has, n_bin^n dimensions.
class DiscretizeWrapper(gym.Wrapper):
    def __init__(self, env, n_bins=2):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = gym.spaces.Discrete(n_bins ** env.action_space.shape[0])

        # Define the action space bounds for each dimension
        self.action_bounds = []
        for dim in range(env.action_space.shape[0]):
            low, high = env.action_space.low[dim], env.action_space.high[dim]
            self.action_bounds.append(np.linspace(low, high, n_bins))

    def step(self, action: int):
        # Map the discrete action index to a continuous action value
        discrete_action = np.unravel_index(action, (self.n_bins,) * self.env.action_space.shape[0])
        continuous_action = [self.action_bounds[i][discrete_action[i]] for i in range(self.env.action_space.shape[0])]
        cont_action = np.array(continuous_action)
        return self.env.step(cont_action)

    def reverse_action(self, action):
        # Map the continuous action value to a discrete action index
        discrete_action = []
        for i in range(self.env.action_space.shape[0]):
            discrete_val = np.abs(self.action_bounds[i] - action[i]).argmin()
            discrete_action.append(discrete_val)
        return np.ravel_multi_index(discrete_action, (self.n_bins,) * self.env.action_space.shape[0])