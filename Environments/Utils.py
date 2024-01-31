import pickle
from os.path import exists
from pathlib import Path
from typing import Union

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RecordEpisodeStatistics
from copy import deepcopy

from Agents.Agent import Agent

def extend_flat_space(space: Box, low_ext: list, high_ext: list):

    """
        Returns an extension of the Box-space 'space'.

        low_ext: Lower bounds of values in the newly added dimensions
        high_ext: Upper bounds of values in the newly added dimensions
    """

    if len(space.shape) > 1:
        raise ValueError(f"The space must be flat and not of shape {space.shape}")

    if len(low_ext) != len(high_ext):
        raise ValueError(f"The sizes of of low_ext {len(low_ext)} and high_ext {len(high_ext)} do not match.")

    size = len(low_ext)
    new_shape = (space.shape[0] + size,)
    new_low = np.concatenate([space.low, low_ext])
    new_high = np.concatenate([space.high,high_ext])

    return Box(low=new_low, high=new_high, shape=new_shape)

def load_env(path: Path):
    """
        Loads an environment from a path and returns it.

        path: Path to the environment relative to the folder of the project's agent-models
        It is assumed that a file env_descr.pkl is present at that path.
    """

    abs_path = Agent.to_absolute_path(path)

    with open(abs_path.joinpath("env_descr.pkl"), 'rb') as file:
        env_descr = pickle.load(file)

    from Analysis import ExperimentSetup #To avoid circular imports
    env = ExperimentSetup.setup_env(env_descr)()
    #Load norm-state if it has been saved
    if exists(abs_path.joinpath("norm_state.pkl")):
        env = load_normalization_state(env, abs_path.joinpath("norm_state.pkl"))
    return env

def save_normalization_state(env: gym.Wrapper, path: Path):
    contents = {}
    if get_wrapper(env, NormalizeObservation) is not None:
        contents["obs"] = deepcopy(get_wrapper(env, NormalizeObservation).obs_rms)
    if get_wrapper(env, NormalizeReward) is not None:
        contents["rew"] = deepcopy(get_wrapper(env, NormalizeReward).return_rms)

    if contents:
        with open(path, "wb") as f:
            pickle.dump(contents,f)


# Returns the normalizations saved at the given path
def load_normalization_state_from_file(path: Path):
    if not exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# Apply the given observation-normalization to a given observation
def apply_observation_normalization(obs: np.ndarray, obs_rms):
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

#Returns mean and variance of the observation/reward-normalization wrapper if present
def get_normalization_state(env: gym.Wrapper):
    owrapper, rwrapper = get_wrapper(env, NormalizeObservation), get_wrapper(env,NormalizeReward)
    stats = {}
    if owrapper:
        stats.update({"obs mean": owrapper.obs_rms.mean, "obs var": owrapper.obs_rms.var, "obs": owrapper.obs_rms})
    if rwrapper:
        stats.update({"rew mean":rwrapper.return_rms.mean, "rew var": rwrapper.return_rms.var, "rew": rwrapper.return_rms})
    return stats

def load_normalization_state(env: gym.Wrapper, state: Union[Path,dict]) -> gym.Env:
    if (not get_wrapper(env, NormalizeObservation)) and (not get_wrapper(env,NormalizeReward)):
        print("No Normalization state loaded as env does not have one.")
        return env
    else:
        if isinstance(state,Path):
            assert exists(state), f"The folder '{state}' does not exist."
            with open(state, "rb") as f:
                contents = pickle.load(f)
                #load observation normalization
                obs_wrapper = get_wrapper(env, NormalizeObservation)
                if obs_wrapper is not None:
                    assert "obs" in contents, f"Observation normalization not found in {state}"
                    obs_wrapper.obs_rms = contents["obs"]
                #load return normalization
                return_wrapper = get_wrapper(env, NormalizeReward)
                if return_wrapper is not None:
                    assert "rew" in contents, f"Return normalization not found in {state}"
                    return_wrapper.return_rms = contents["rew"]
        else:
            if get_wrapper(env, NormalizeObservation) is not None:
                get_wrapper(env, NormalizeObservation).obs_rms = state["obs"]
            if get_wrapper(env, NormalizeReward) is not None:
                get_wrapper(env, NormalizeReward).return_rms = state["rew"]
        return env

#Updates the normalization state of 'to_sync' with the state from 'normed_env'
def sync_normalization_state(normed_env: gym.Wrapper, to_sync: gym.Wrapper) -> gym.Wrapper:
    wrapper,sync_wrapper = get_wrapper(normed_env,NormalizeObservation),get_wrapper(to_sync,NormalizeObservation)
    if wrapper:
        obs_rms = deepcopy(wrapper.obs_rms)
        assert obs_rms is not None
        sync_wrapper.obs_rms = obs_rms

    wrapper, sync_wrapper = get_wrapper(normed_env,NormalizeReward),get_wrapper(to_sync,NormalizeReward)
    if wrapper:
        return_rms = deepcopy(wrapper.return_rms)
        assert return_rms is not None
        sync_wrapper.return_rms = return_rms

    return to_sync

def wrappers_are_type_equiv(env1, env2):
    while env1:
        if not isinstance(env1, type(env2)):
            return False
        env1, env2 = getattr(env1, "env", None), getattr(env2, "env", None)
    return True

def get_wrapper(env: gym.Wrapper, _type: "Class"):
    current_env = env
    while current_env:
        if isinstance(current_env, _type):
            assert get_wrapper(getattr(current_env, "env", None),_type) is None, "The environment must not contain multiple wrappers of the same type."
            return current_env
        current_env = getattr(current_env, "env", None)
    return None

def wrap_env(env: gym.Env, flatten_obs: bool = False,
             clip_actions: bool = False,
             norm_obs: bool = False,
             clip_obs: float = None,
             gamma = None,
             clip_rew: float = None,
             record_stats: bool = False):

    assert not isinstance(clip_rew,bool) and not isinstance(clip_obs,bool)

    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    if flatten_obs:
        env = gym.wrappers.FlattenObservation(env)
    if isinstance(env.action_space, gym.spaces.Box) and clip_actions:
        env = gym.wrappers.ClipAction(env)
    if norm_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if clip_obs:
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -clip_obs, clip_obs))
    if gamma:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    if clip_rew:
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -clip_rew, clip_rew))

    return env
