from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import gymnasium as gym
from Agents.Agent import Agent
from Agents.CombNHierarchical import HierComb, HCConfig

from Analysis.AgentConfigs import VanillaPPO

from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Analysis.Parser import parse_bool, parse_tuple

"""
    Hierarchical CombN configuration used for the the hierarchical CombN experiments. This includes the CombN experiments with pretraining
    as this is a special case of Hierarchical CombN (cycles=1).
"""

def chooser_net_cfg(space_descr, chooser_hidden: tuple[int], n: int):
    input_size, output_size = space_descr.flattened_input_size(), space_descr.flattened_act_size()

    c_std = tuple([np.sqrt(2) for _ in range(len(chooser_hidden))] + [0.01])

    c_bias = np.zeros(len(chooser_hidden) + 1)
    chooser_conf = NetworkConfig(class_name="FFNet",
                                 args_dict={"input_size": input_size, "output_size": n, "hidden_sizes": chooser_hidden,
                                            "init_std": c_std,
                                            "init_bias_const": c_bias})
    return chooser_conf

def agent_cfg(space_descr: EnvSpaceDescription ,cycle_steps: Union[int,tuple[int]], cycles: int, n: int, chooser_sizes: Tuple[int],
              action_net_sizes: Tuple[int], pretrain: bool, vent: float, freeze_bottom: bool):
    training_alg_config = VanillaPPO.agent_cfg(space_descr)
    training_alg_config.attention_vent_coef = vent
    training_alg_config.net_conf.args_dict["actor"].args_dict["mean"].args_dict["hidden_sizes"] = action_net_sizes
    training_alg_config.net_conf.args_dict["actor"].args_dict["mean"].args_dict["init_std"] = tuple([np.sqrt(2) for _ in range(len(action_net_sizes))] + [0.01])
    training_alg_config.net_conf.args_dict["actor"].args_dict["mean"].args_dict["init_bias_const"] = np.zeros(len(action_net_sizes)+1)

    cfg = HCConfig(space_description=space_descr, pretrain=pretrain, cycle_steps=cycle_steps, n=n,cycles=cycles, freeze_bottom=freeze_bottom, training_alg_cfg=training_alg_config, chooser_net_cfg=chooser_net_cfg(space_descr,chooser_sizes,n) )
    cfg.name = f"HierComb"
    return cfg

#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription,cycle_steps: Union[int,str] =1000000, cycles:Union[int,str]=1, n:Union[int,str]=2,
          chooser_sizes:Union[tuple[int],str] = (64,), action_net_sizes:Union[tuple[int],str] = (16,16), pretrain:Union[bool,str] = True,
          vent:Union[float,str] = 0.0, freeze_bottom: Union[bool,str] = False, continuation: str = None):
    cycle_steps = (int(cycle_steps) if cycle_steps.isdigit() else parse_tuple(cycle_steps, lambda x: int(x))) if isinstance(cycle_steps,str) else cycle_steps
    return (lambda: HierComb(cfg = agent_cfg(space_descr, cycle_steps=cycle_steps, cycles=int(cycles),n=int(n),chooser_sizes=parse_tuple(chooser_sizes, lambda x: int(x)),
                                             action_net_sizes = parse_tuple(action_net_sizes, lambda x: int(x)), pretrain = parse_bool(pretrain), vent =float(vent), freeze_bottom=parse_bool(freeze_bottom)))) if continuation is None else lambda: Agent.load(Path(continuation))


def env_wrapper(env: Callable[[],gym.core.Env]):
    return VanillaPPO.env_wrapper(env)
