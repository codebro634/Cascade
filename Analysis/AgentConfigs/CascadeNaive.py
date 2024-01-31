from pathlib import Path
from typing import Callable, Union

import numpy as np

from Agents.Agent import Agent
from Agents.CascadeNaive import CascadeNaiveConfig, CascadeNaive

from Analysis.AgentConfigs import VanillaPPO, Cascade
from Environments import EnvSpaceDescription
from Environments.Utils import extend_flat_space

from Analysis.Parser import parse_bool, parse_tuple

"""
    Naive Cascade-Agent configuration. This agent trains on a surrogate environment
    that extends the original environment by the fallback-action.
"""

#Action space dimension is increased by 1 to account for fallback-action.
#If value/action propagation the input-space dimension is also increased.
def extend_space_descr(space_descr: EnvSpaceDescription, prop_val: bool, prop_action: bool):
    low_ext, high_ext = [], []
    if prop_val:
        low_ext.append(-np.infty)
        high_ext.append(np.infty)
    if prop_action:
        low_ext = low_ext + [-np.infty] * space_descr.flattened_act_size()
        high_ext = high_ext + [np.infty] * space_descr.flattened_act_size()
    return EnvSpaceDescription(observation_space=extend_flat_space(space_descr.observation_space, low_ext=low_ext, high_ext=high_ext),
                        action_space=extend_flat_space(space_descr.action_space, low_ext=[-np.infty], high_ext=[np.infty]))

def agent_cfg(space_descr: EnvSpaceDescription ,base_steps:int, propagate_value:bool, propagate_action:bool, fallback_reward:float, fallback_init:float, actor_hidden:tuple[int]):
    init_config = VanillaPPO.agent_cfg(space_descr)
    init_config.net_conf = Cascade.net_cfg(space_descr,True, prop_val=propagate_value, prop_action=propagate_action, fallback_init=fallback_init, actor_hidden = actor_hidden)

    stacked_config = VanillaPPO.agent_cfg(extend_space_descr(space_descr, propagate_value, propagate_action))
    stacked_config.net_conf = Cascade.net_cfg(space_descr, False, prop_val=propagate_value, prop_action=propagate_action, fallback_init=fallback_init, actor_hidden=actor_hidden)
    #Cascade uses a weighted sum of the net-outputs. No sampling of the fallback-action needed.
    #Here, the fallback-action is applied on a surrogate environment, therefore logstd needs to be increased by 1.
    stacked_config.net_conf.args_dict["actor"].args_dict["logstd"].args_dict["output_size"] = 1 +  stacked_config.net_conf.args_dict["actor"].args_dict["logstd"].args_dict["output_size"]

    cfg = CascadeNaiveConfig(space_description=space_descr, base_steps=base_steps,
                             propagate_value=propagate_value, propagate_action=propagate_action, fallback_reward = fallback_reward,
                             init_training_alg_cfg=init_config, ensemble_size=99, stacked_training_alg_cfg=stacked_config)

    cfg.name = f"NaiveCasc"
    return cfg

#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription,base_steps: Union[int,str] =1000000, propagate_value: Union[bool,str] =False, propagate_action: Union[bool,str] =False,
          fallback_reward: Union[float,str] =0.0, fallback_init: Union[float,str] =0.5, actor_hidden: Union[tuple[int],str] = (16,16), continuation: str = None):
    return (lambda: CascadeNaive(cfg = agent_cfg(space_descr, base_steps=int(base_steps), propagate_value=parse_bool(propagate_value), propagate_action=parse_bool(propagate_action),
                                                 fallback_reward=float(fallback_reward), fallback_init=float(fallback_init), actor_hidden=parse_tuple(actor_hidden, lambda x: int(x))))) if continuation is None else lambda: Agent.load(Path(continuation))


def env_wrapper(env: Callable):
    return VanillaPPO.env_wrapper(env)
