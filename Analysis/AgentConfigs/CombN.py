from pathlib import Path
from typing import Callable, Union

import numpy as np
import gymnasium as gym
from Agents.Agent import Agent
from Agents.PPO import  PPO
from Analysis.AgentConfigs import VanillaPPO
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Analysis.Parser import parse_tuple, parse_list, parse_bool

"""
    CombN-Agent configuration used for the CombN experiments. This includes Entropy and Dropout experiments.
"""

def net_cfg(space_descr: EnvSpaceDescription, action_hidden: tuple[int], chooser_hidden: tuple[int], n:int, dropout_prob:float, temperature:float, freeze_action_nets: bool):
    input_size, output_size =  space_descr.flattened_input_size(), space_descr.flattened_act_size()

    action_std = tuple([np.sqrt(2) for _ in range(len(action_hidden))] + [0.01])
    action_bias = np.zeros(len(action_hidden)+1)
    action_mean_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": input_size, "output_size": output_size, "hidden_sizes": action_hidden,
                                                             "activation_last_layer": False, "init_std": action_std, "init_bias_const": action_bias})
    action_log_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": None, "output_size": output_size, "hidden_sizes": None})

    c_std = tuple([np.sqrt(2) for _ in range(len(chooser_hidden))] + [0.01])

    c_bias = np.zeros(len(chooser_hidden)+1)
    chooser_conf = NetworkConfig(class_name="FFNet",
                                args_dict={"input_size": input_size, "output_size": n, "hidden_sizes": chooser_hidden,
                                           "init_std": c_std,
                                           "init_bias_const": c_bias})

    actor_conf = NetworkConfig(class_name="ActorHead",
                               args_dict={ "mean": action_mean_conf, "logstd": action_log_conf,})

    critic_conf = NetworkConfig(class_name="FFNet",
                                args_dict={"input_size": input_size, "output_size": 1, "hidden_sizes": (64, 64),
                                           "init_std": (np.sqrt(2), np.sqrt(2), 1.0),
                                           "init_bias_const": (0.0, 0.0, 0.0)})


    comb_actor = NetworkConfig(class_name = "CombActorHead", args_dict={"chooser": chooser_conf, "heads": [actor_conf for _ in range(n)], "freeze_action_nets": freeze_action_nets, "dropout_prob": dropout_prob, "temperature": temperature})

    net_conf = NetworkConfig(class_name="ActorCritic",args_dict={"shared": None, "actor": comb_actor, "critic": critic_conf})

    return net_conf

def agent_cfg(space_descr: EnvSpaceDescription, action_hidden: tuple[int] =(16,16), vent_coef: float =0.0, hent_coef: float =0.0, 
              chooser_hidden: tuple[int] =(64,), n: int =4,  temperature: float =1.0, dropout_prob: float =0.0, chooser_epochs: int = 0, action_epochs: int = 0, freeze_action_nets: bool = False):
    
    net_conf = net_cfg(space_descr, action_hidden, chooser_hidden, n, dropout_prob,  temperature, freeze_action_nets)
    agent_config = VanillaPPO.agent_cfg(space_descr)
    agent_config.net_conf =net_conf
    agent_config.attention_vent_coef = vent_coef
    agent_config.attention_hent_coef = hent_coef
    agent_config.chooser_epochs = chooser_epochs
    agent_config.action_epochs = action_epochs
    agent_config.name = f"CombN"
    return agent_config


"""
    continuation: Load the CombN net saved at the continuation
    action_continuations: Loads only the action-nets, saved at action_continuations
"""
def agent(space_descr: EnvSpaceDescription, action_hidden: Union[tuple[int],str] =(16,16), vent_coef: Union[str,float] =0.0, hent_coef: Union[float,str] =0.0, 
          chooser_hidden: Union[tuple[int],str] =(64,), n: Union[int,str] =4, temperature: Union[float,str] =1.0, dropout_prob: Union[float,str]=0.0,
          chooser_epochs: Union[int,str] = 0, action_epochs: Union[int,str] = 0, freeze_action_nets: Union[bool,str] = False,
          action_continuations: Union[str,list[str]] = None, continuation: str = None):
    if action_continuations is None:
        return (lambda: PPO(cfg = agent_cfg(space_descr, action_hidden=parse_tuple(action_hidden, lambda x: int(x)), vent_coef = float(vent_coef),hent_coef = float(hent_coef), 
                                            chooser_hidden = parse_tuple(chooser_hidden, lambda x: int(x)), n = int(n), temperature = float(temperature), dropout_prob = float(dropout_prob),
                                            chooser_epochs=int(chooser_epochs), action_epochs=int(action_epochs), freeze_action_nets=parse_bool(freeze_action_nets)))) if continuation is None else lambda: Agent.load(Path(continuation))
    else:
        action_continuations = parse_list(action_continuations, lambda x: x)

        def thunk():
            agent = PPO(cfg=agent_cfg(space_descr, action_hidden = parse_tuple(action_hidden,lambda x: int(x)), vent_coef = float(vent_coef), hent_coef = float(hent_coef), 
                                      chooser_hidden = parse_tuple(chooser_hidden, lambda x: int(x)), n = int(n),  temperature = float(temperature), dropout_prob = float(dropout_prob),
                                      chooser_epochs=int(chooser_epochs), action_epochs=int(action_epochs), freeze_action_nets=parse_bool(freeze_action_nets)))

            for i, action in enumerate(action_continuations):
                agent.net.actor.reinit_action_net(i, action)

            return agent

        return thunk

def env_wrapper(env: Callable[[],gym.core.Env]):
    return VanillaPPO.env_wrapper(env)

