from pathlib import Path
from typing import Callable, Union
import numpy as np
from torch import nn

from Agents.Agent import Agent
from Agents.SAC import SACConfig, SAC
from Analysis.AgentConfigs.General import gamma
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Environments.Utils import wrap_env
from copy import deepcopy
from Analysis.Parser import parse_tuple

"""
    Standard SAC-Configuration as in CleanRL (which the exception of weight's initialization where orthogonal init is used). Used for BaselineSAC experiments.
"""

def net_cfg(space_descr: EnvSpaceDescription, layer_sizes: tuple[int] = (256,256), critic_sizes: tuple = (256,256), low_std_critic: bool = False):
    obs_size, action_size = space_descr.flattened_input_size(), space_descr.flattened_act_size()

    q1_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": obs_size + action_size, "output_size": 1,"hidden_sizes": critic_sizes,"activation_function": nn.ReLU(),
                                                           "init_std": (np.sqrt(2), np.sqrt(2), 0.1 if low_std_critic else 1.0), "init_bias_const": (0.0, 0.0, 0.0)})

    shared_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": obs_size, "output_size": layer_sizes[-1], "hidden_sizes": layer_sizes[:-1],
                                                               "activation_function": nn.ReLU(),"ll_activation": nn.ReLU(), "init_std": np.sqrt(2),"init_bias_const": 0})

    mean_conf = NetworkConfig(class_name="FFNet",args_dict={"input_size": layer_sizes[-1], "output_size": action_size, "hidden_sizes": [], "init_std": 0.01, "init_bias_const": 0})

    log_conf = NetworkConfig(class_name="FFNet",args_dict={"input_size": layer_sizes[-1], "output_size": action_size, "hidden_sizes": []})

    actor_conf = NetworkConfig(class_name="CascadeActor", args_dict={"mean": mean_conf, "logstd": log_conf, "shared": shared_conf})


    return actor_conf, q1_conf, deepcopy(q1_conf)

def agent_cfg(space_descr: EnvSpaceDescription, layer_sizes: tuple[int] = (256,256), critic_hidden:tuple[int] = (256,256)):
    actor_conf, q1_conf, q2_conf = net_cfg(space_descr, layer_sizes=layer_sizes, critic_sizes=critic_hidden)

    return SACConfig(
        buffer_size=int(1e6),
        tau=0.005,
        batch_size=256,
        learning_starts=int(5e3),
        policy_lr=3e-4,
        q_lr=1e-3,
        policy_frequency=2,
        target_network_frequency=1,
        alpha=0.2,
        gamma=gamma,
        actor_conf=actor_conf,
        q1_conf=q1_conf,
        q2_conf=q2_conf,
    space_description = space_descr,
    name="SAC_Vanilla")


#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription, layer_sizes: Union[tuple[int],str] = (256,256), critic_hidden: Union[tuple[int],str] = (256,256), continuation: str = None):
    return (lambda: SAC(cfg = agent_cfg(space_descr, critic_hidden= parse_tuple(critic_hidden, lambda x: int(x)), layer_sizes=parse_tuple(layer_sizes, lambda x: int(x))))) if continuation is None else lambda: Agent.load(Path(continuation))

def env_wrapper(env: Callable):

    def env_maker():
        return wrap_env(env(), flatten_obs=True)

    return env_maker
