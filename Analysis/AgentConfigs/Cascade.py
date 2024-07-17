import math
from copy import deepcopy
from pathlib import Path
from typing import Callable, Union

import numpy as np
import gymnasium as gym
from torch import nn

from Agents.Agent import Agent
from Agents.Cascade import Cascade, CascadeConfig

from Analysis.AgentConfigs import VanillaPPO, VanillaSAC
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Analysis.Parser import parse_bool, parse_tuple

from Analysis.AgentConfigs import VanillaDDPG

"""
    Cascade-Agent configuration used for the Cascade experiments.
"""


def net_cfg(space_description, init: bool, fallback_init: float, actor_hidden: tuple, alg: str, use_tanh: bool, critic_stack: bool, critic_sizes=(64, 64)):

    x,y = space_description.flattened_input_size(), space_description.flattened_act_size()
    fb_bias = math.log(fallback_init/(1-fallback_init)) #Set so that sigmoid(fb_bias) = fallback_init

    if alg == "PPO":
        preset = [(2 * len(actor_hidden), True, y,fb_bias)] if not init else []  # When passed to FF-net sets the bias of the neuron responsible for fallback-action to fb_bias
        actor_conf, critic_conf = VanillaPPO.net_cfg(space_description, actor_hidden, critic_sizes=critic_sizes)
        critic_conf = [critic_conf]
        actor_conf.args_dict["mean"].args_dict["preset_params"] = preset
        actor_conf.args_dict["mean"].args_dict["output_size"] += 0 if init else 1
        critic_conf[0].args_dict["output_size"] += 1 if critic_stack and not init else 0
        if not init:
            actor_conf.args_dict["mean"] = NetworkConfig(class_name="CascadeFFNet",args_dict={"ff_net": actor_conf.args_dict["mean"]})
            critic_conf = [NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": critic_conf[0]})]

        return actor_conf, critic_conf

    elif alg == "DDPG":
        preset = [(2 * len(actor_hidden), True, y,fb_bias)] if not init else []  # When passed to FF-net sets the bias of the neuron responsible for fallback-action to fb_bias
        actor_conf, critic_conf = VanillaDDPG.net_cfg(space_description, actor_hidden, critic_sizes=critic_sizes)
        critic_conf = [critic_conf]
        actor_conf.args_dict["mean"].args_dict["ll_activation"] = nn.Tanh() if use_tanh else None
        actor_conf.args_dict["mean"].args_dict["ll_activation_range"] = [0, y] if use_tanh else None
        actor_conf.args_dict["mean"].args_dict["preset_params"] = preset
        actor_conf.args_dict["mean"].args_dict["output_size"] += 0 if init else 1
        critic_conf[0].args_dict["output_size"] += 1 if critic_stack and not init else 0
        if not init:
            actor_conf.args_dict["mean"] = NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": actor_conf.args_dict["mean"]})
            critic_conf = [NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": critic_conf[0]})]

        return actor_conf, critic_conf

    elif alg == "SAC":
        actor, q1, q2 = VanillaSAC.net_cfg(space_description, actor_hidden, critic_sizes=critic_sizes)
        preset = [(0, True, y,fb_bias)] if not init else []
        actor.args_dict["mean"].args_dict["preset_params"] = preset
        actor.args_dict["mean"].args_dict["output_size"] += 0 if init else 1
        q1.args_dict["output_size"] += 1 if critic_stack and not init else 0
        q2.args_dict["output_size"] += 1 if critic_stack and not init else 0
        if not init:
            actor.args_dict["mean"] = NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": actor.args_dict["mean"]})
            q1 = NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": q1})
            q2 = NetworkConfig(class_name="CascadeFFNet", args_dict={"ff_net": q2})

        return actor, [q1, q2]
    else:
        raise ValueError(f"Unknown algorithm {alg}")


def agent_cfg(space_descr: EnvSpaceDescription, base_steps: int, fallback_coef: float, train_only_top: bool,
              fb_init: float, sequential: bool, stacks: int, actor_hidden: tuple[int], critic_hidden: tuple[int], cyclical_lr: bool, anneal_lr: bool, tanh_in_net: bool = False, reset_rb:bool = False, stack_critic: bool = False, keep_critic: bool = False, continuous: bool = True, alg_name: str = "PPO") -> CascadeConfig:

    if alg_name == "PPO":
        training_alg_config = VanillaPPO.agent_cfg(space_descr, anneal_lr=anneal_lr, continuous=continuous)
    elif alg_name == "DDPG":
        assert continuous, "DDPG does not yet support discrete action spaces."
        training_alg_config = VanillaDDPG.agent_cfg(space_descr, anneal_lr=anneal_lr)
        training_alg_config.tanh_in_net = tanh_in_net
    elif alg_name == "SAC":
        training_alg_config = VanillaSAC.agent_cfg(space_descr)
    else:
        raise ValueError(f"Unknown algorithm {alg_name}")

    training_alg_config.fallback_coef = fallback_coef
    init_actor_net_conf, init_critic_net_conf = net_cfg(space_descr,True, fallback_init=fb_init, actor_hidden=actor_hidden, critic_sizes=critic_hidden, alg = alg_name, use_tanh=tanh_in_net, critic_stack=stack_critic)
    stacked_actor_net_conf, stacked_critic_net_conf = net_cfg(space_descr, False, fallback_init=fb_init, actor_hidden=actor_hidden, critic_sizes=critic_hidden, alg = alg_name, use_tanh=tanh_in_net, critic_stack=stack_critic)

    cfg = CascadeConfig(space_description=space_descr, base_steps=base_steps,
                        train_only_top_net=train_only_top,
                        training_alg_cfg=training_alg_config,
                        stack_critics=stack_critic,
                        keep_critic=keep_critic,
                        stacked_actor_cfg=stacked_actor_net_conf, init_actor_cfg=init_actor_net_conf,
                        stacked_critic_cfgs=stacked_critic_net_conf, init_critic_cfgs=init_critic_net_conf,
                        sequential = sequential, stacks=stacks, cyclical_lr=cyclical_lr,
                        training_alg=alg_name, reset_rb=reset_rb)
    cfg.name = f"Cascade"
    return cfg

#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription,base_steps: Union[int,str] =1000000, fallback_coef: Union[float,str] = 0.0,
          train_only_top: Union[bool,str] = False, fb_init: Union[float,str] = 0.5, sequential: Union[bool,str] = True, stacks: Union[int,str] = -1,
          actor_hidden: Union[tuple[int],str] = (16, 16), critic_hidden: Union[tuple[int],str] = (64, 64), tanh_in_net: Union[bool,str] = False, cyclical_lr: Union[bool,str] = True, anneal_lr: Union[bool,str] = True, reset_rb: Union[bool,str] = False, stack_critic: Union[bool,str] = False, keep_critic: bool = False, continuous: bool = True, continuation: str = None, alg_name: str = "PPO"):
    return (lambda: Cascade(cfg=agent_cfg(space_descr, base_steps=int(base_steps),
                                            fallback_coef=float(fallback_coef), train_only_top=parse_bool(train_only_top), fb_init=float(fb_init), keep_critic=parse_bool(keep_critic),
                                            sequential=parse_bool(sequential), stacks=int(stacks), actor_hidden=parse_tuple(actor_hidden, lambda x: int(x)),cyclical_lr=parse_bool(cyclical_lr), anneal_lr=parse_bool(anneal_lr),
                                            reset_rb=parse_bool(reset_rb), critic_hidden=parse_tuple(critic_hidden, lambda x: int(x)), stack_critic=parse_bool(stack_critic), tanh_in_net=parse_bool(tanh_in_net),
                                            continuous=parse_bool(continuous),alg_name=alg_name))) if continuation is None else lambda: Agent.load(Path(continuation))


def env_wrapper(env: Callable[[],gym.core.Env]):
    return VanillaPPO.env_wrapper(env)
