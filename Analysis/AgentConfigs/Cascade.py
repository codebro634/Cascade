import math
from pathlib import Path
from typing import Callable, Union

import numpy as np
import gymnasium as gym
from Agents.Agent import Agent
from Agents.Cascade import Cascade, CascadeConfig

from Analysis.AgentConfigs import VanillaPPO
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Analysis.Parser import parse_bool, parse_tuple

from Analysis.AgentConfigs import VanillaDDPG

"""
    Cascade-Agent configuration used for the Cascade experiments.
"""


def net_cfg(space_description, init: bool, prop_val: bool, prop_action: bool, fallback_init: float, actor_hidden: tuple, alg: str):

    x,y = space_description.flattened_input_size(), space_description.flattened_act_size()

    #Extend input-size of either the value or the action of previous agents are propagated
    if prop_val and not init:
        x += 1
    if prop_action and not init:
        x += y

    fb_bias = math.log(fallback_init/(1-fallback_init)) #Set so that sigmoid(fb_bias) = fallback_init
    preset = [(2*len(actor_hidden),True,y,fb_bias)] if not init else [] #When passed to FF-net sets the bias of the neuron responsible for fallback-action to fb_bias

    mean_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": x, "output_size": y if init else y+1, "hidden_sizes": actor_hidden, "preset_params": preset, #y+1 to account for fallback-action
                                                                     "activation_last_layer": False,"init_std": tuple([np.sqrt(2) for _ in range(len(actor_hidden))] + [0.01]), "init_bias_const": tuple([0.0 for _ in range(len(actor_hidden)+1)]) })

    if alg == "PPO":
        critic_conf = NetworkConfig(class_name="FFNet",
                                    args_dict={"input_size": x, "output_size": 1, "hidden_sizes": (64,64),
                                               "activation_last_layer": False,
                                               "init_std": (np.sqrt(2), np.sqrt(2),  1.0),
                                               "init_bias_const": (0.0,0.0, 0.0)})
    elif alg == "DDPG":
        critic_conf = NetworkConfig(class_name="FFNet",
                                    args_dict={"input_size": x+y, "output_size": 1, "hidden_sizes": (64,64),
                                               "activation_last_layer": False,
                                               "init_std": (np.sqrt(2), np.sqrt(2),  1.0),
                                               "init_bias_const": (0.0,0.0, 0.0)})

    log_conf = NetworkConfig(class_name="FFNet",args_dict={"input_size": None, "output_size": y, "hidden_sizes": None})

    actor_conf = NetworkConfig(class_name="ActorHead", args_dict={ "mean": mean_conf, "logstd": log_conf,})

    net_conf = NetworkConfig(class_name="ActorCritic",args_dict={"shared": None, "actor": actor_conf, "critic": critic_conf})

    return net_conf

def agent_cfg(space_descr: EnvSpaceDescription, base_steps: int, propagate_value: bool, propagate_action: bool, fallback_coef: float, train_only_top: bool,
              fb_init: float, sequential: bool, stacks: int, actor_hidden: tuple[int], cyclical_lr: bool, continuous: bool = True, alg_name: str = "PPO") -> CascadeConfig:

    if alg_name == "PPO":
        training_alg_config = VanillaPPO.agent_cfg(space_descr, continuous=continuous)
    elif alg_name == "DDPG":
        assert continuous, "DDPG does not yet support discrete action spaces."
        training_alg_config = VanillaDDPG.agent_cfg(space_descr)
    else:
        raise ValueError(f"Unknown algorithm {alg_name}")

    training_alg_config.fallback_coef = fallback_coef
    init_net_conf = net_cfg(space_descr,True, prop_val=propagate_value, prop_action=propagate_action, fallback_init=fb_init, actor_hidden=actor_hidden, alg = alg_name)
    stacked_net_conf = net_cfg(space_descr, False, prop_val=propagate_value, prop_action=propagate_action, fallback_init=fb_init, actor_hidden=actor_hidden, alg = alg_name)

    cfg = CascadeConfig(space_description=space_descr, base_steps=base_steps,
                        propagate_value=propagate_value, propagate_action=propagate_action, train_only_top_net=train_only_top,
                        training_alg_cfg=training_alg_config,
                        stacked_net_cfg=stacked_net_conf, init_net_cfg=init_net_conf, sequential = sequential, stacks=stacks, cyclical_lr=cyclical_lr,
                        training_alg=alg_name)
    cfg.name = f"Cascade"
    return cfg

#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription,base_steps: Union[int,str] =1000000, propagate_value: Union[bool,str] = False, propagate_action: Union[bool,str] = False, fallback_coef: Union[float,str] = 0.0,
          train_only_top: Union[bool,str] = False, fb_init: Union[float,str] = 0.5, sequential: Union[bool,str] = True, stacks: Union[int,str] = -1,
          actor_hidden: Union[tuple[int],str] = (16,16), cyclical_lr: Union[bool,str] = True, continuous: bool = True, continuation: str = None, alg_name: str = "PPO"):
    return (lambda: Cascade(cfg = agent_cfg(space_descr, base_steps=int(base_steps), propagate_value=parse_bool(propagate_value), propagate_action=parse_bool(propagate_action),
                                            fallback_coef=float(fallback_coef), train_only_top=parse_bool(train_only_top), fb_init=float(fb_init),
                                            sequential=parse_bool(sequential), stacks=int(stacks), actor_hidden=parse_tuple(actor_hidden, lambda x: int(x)),cyclical_lr=parse_bool(cyclical_lr), continuous=parse_bool(continuous),alg_name=alg_name))) if continuation is None else lambda: Agent.load(Path(continuation))


def env_wrapper(env: Callable[[],gym.core.Env]):
    return VanillaPPO.env_wrapper(env)
