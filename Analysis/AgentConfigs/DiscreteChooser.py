from pathlib import Path
from typing import Callable

import numpy as np
import gymnasium as gym
from Agents.Agent import Agent
from Agents.PPO import PPO
from Analysis.AgentConfigs import VanillaPPO
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription


"""
   Discrete-Chooser configuration used in the Continuation experiments where this Discrete-Chooser has to learn to pick
   one policy. In principle, this Agent can be applied to any discrete action space environment.
"""

def net_cfg(space_descr: EnvSpaceDescription):
    input_size, output_size =  space_descr.flattened_input_size(), space_descr.flattened_act_size()

    chooser_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": input_size, "output_size": output_size, "hidden_sizes": (64,),
                                                             "activation_last_layer": False, "init_std": (np.sqrt(2), 0.01), "init_bias_const": (0.0, 0.0)})

    critic_conf = NetworkConfig(class_name="FFNet",
                                args_dict={"input_size": input_size, "output_size": 1, "hidden_sizes": (64, 64),
                                           "activation_last_layer": False,
                                           "init_std": (np.sqrt(2), np.sqrt(2), 1.0),
                                           "init_bias_const": (0.0, 0.0, 0.0)})

    actor_conf = NetworkConfig(class_name="ActorHead",args_dict={"mean": chooser_conf, "logstd": None})

    net_conf = NetworkConfig(class_name="ActorCritic",
                             args_dict={"shared": None, "actor": actor_conf, "critic": critic_conf})
    return net_conf


#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription, continuation: str = None):
    agent_cfg = VanillaPPO.agent_cfg(space_descr, continuous=False)
    agent_cfg.net_conf = net_cfg(space_descr)
    return (lambda: PPO(cfg = agent_cfg)) if continuation is None else lambda: Agent.load(Path(continuation))

def env_wrapper(env: Callable[[],gym.core.Env]):
    return VanillaPPO.env_wrapper(env)
