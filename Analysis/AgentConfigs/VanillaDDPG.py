from pathlib import Path
from typing import Callable, Union
import numpy as np
from Agents.Agent import Agent
from Agents.DDPG import DDPGConfig, DDPG
from Analysis.AgentConfigs.General import gamma
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Environments.Utils import wrap_env

from Analysis.Parser import parse_bool, parse_tuple

"""
    Standard DDPG configuration as in cleanRL. Used for BaselineDDPG experiments.
"""

def net_cfg(space_descr: EnvSpaceDescription, layer_sizes: tuple[int] = (64,64), critic_sizes: tuple = (64,64), low_critic_std: bool = False):
    obs_size, action_size =  space_descr.flattened_input_size(), space_descr.flattened_act_size()

    mean_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": obs_size, "output_size": action_size, "hidden_sizes": layer_sizes,
                                                             "init_std": [np.sqrt(2) for _ in range(len(layer_sizes))] + [0.01], "init_bias_const": [0.0 for _ in range(len(layer_sizes) + 1)]})

    critic_conf = NetworkConfig(class_name="FFNet",
                                args_dict={"input_size": obs_size + action_size, "output_size": 1, "hidden_sizes": critic_sizes,
                                           "init_std": (np.sqrt(2), np.sqrt(2), 0.1 if low_critic_std else 1.0),
                                           "init_bias_const": (0.0, 0.0, 0.0)})

    log_conf = NetworkConfig(class_name="FFNet",args_dict={"input_size": None, "output_size": action_size, "hidden_sizes": None})

    actor_conf = NetworkConfig(class_name="CascadeActor",args_dict={"mean": mean_conf, "logstd": log_conf, "shared": None})

    return actor_conf, critic_conf

def agent_cfg(space_descr: EnvSpaceDescription, learning_starts: int = 25e3, layer_sizes: tuple[int] = (64,64), critic_hidden:tuple[int] =(64,64), continuous:bool = True, anneal_lr: bool= True):
    actor_net_conf, critic_net_conf = net_cfg(space_descr, layer_sizes = layer_sizes, critic_sizes=critic_hidden)

    return DDPGConfig(
        learning_rate = 0.0003,
        buffer_size = int(1e6),
        tau=0.005,
        batch_size=256,
        exploration_noise =0.1,
        learning_starts=learning_starts,
        policy_frequency=2,
        noise_clip=0.5,
        cuda=False,
        anneal_lr = anneal_lr,
        actor_net_conf = actor_net_conf,
        critic_net_conf = critic_net_conf,
        gamma = gamma,
        tanh_in_net=False,
    space_description = space_descr,
    name="DDPG_Vanilla")


#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription, learning_starts: Union[int,str] = 25e3, layer_sizes: Union[tuple[int],str] = (64,64), critic_hidden: Union[tuple[int],str] = (64,64), anneal_lr: str|bool = True, continuation: str = None):
    return (lambda: DDPG(cfg = agent_cfg(space_descr, learning_starts=int(learning_starts), critic_hidden = parse_tuple(critic_hidden, lambda x: int(x)), layer_sizes = parse_tuple(layer_sizes, lambda x: int(x)), anneal_lr=parse_bool(anneal_lr)))) if continuation is None else lambda: Agent.load(Path(continuation))

def env_wrapper(env: Callable):

    def env_maker():
        return wrap_env(env(), flatten_obs=True)

    return env_maker
