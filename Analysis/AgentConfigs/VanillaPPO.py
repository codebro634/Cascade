from pathlib import Path
from typing import Callable, Union
import numpy as np
from Agents.Agent import Agent
from Agents.PPO import PPOConfig, PPO
from Analysis.AgentConfigs.General import gamma
from Architectures.NetworkConfig import NetworkConfig
from Environments import EnvSpaceDescription
from Environments.Utils import wrap_env

from Analysis.Parser import parse_bool, parse_tuple

"""
    Standard PPO configuration as in cleanRL. Used for Baseline experiments.
"""

def net_cfg(space_descr: EnvSpaceDescription, layer_sizes: tuple[int] = (64,64), continuous: bool = True):
    input_size, output_size =  space_descr.flattened_input_size(), space_descr.flattened_act_size()

    mean_conf = NetworkConfig(class_name="FFNet", args_dict={"input_size": input_size, "output_size": output_size, "hidden_sizes": layer_sizes,
                                                             "init_std": [np.sqrt(2) for _ in range(len(layer_sizes))] + [0.01], "init_bias_const": [0.0 for _ in range(len(layer_sizes) + 1)]})

    critic_conf = NetworkConfig(class_name="FFNet",
                                args_dict={"input_size": input_size, "output_size": 1, "hidden_sizes": (64, 64),
                                           "init_std": (np.sqrt(2), np.sqrt(2), 1.0),
                                           "init_bias_const": (0.0, 0.0, 0.0)})

    log_conf = NetworkConfig(class_name="FFNet",args_dict={"input_size": None, "output_size": output_size, "hidden_sizes": None})

    actor_conf = NetworkConfig(class_name="CascadeActor",args_dict={"mean": mean_conf, "logstd": log_conf if continuous else None, "shared": None})

    return actor_conf, critic_conf

def agent_cfg(space_descr: EnvSpaceDescription, layer_sizes: tuple[int] = (64,64), continuous:bool = True, anneal_lr: bool= True):
    actor_net_conf, value_net_conf = net_cfg(space_descr, layer_sizes = layer_sizes, continuous = continuous)

    if continuous:
        return PPOConfig(
            learning_rate = 0.0003,
            num_envs = 1,
            num_steps = 2048,
            gae_lambda = 0.95,
            num_minibatches = 32,
            update_epochs = 10,
            clip_coef = 0.2,
            action_ent_coef = 0,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            norm_adv = True,
            clip_vloss = True,
            anneal_lr = anneal_lr,
            cuda = False,
            fallback_coef=0,
            actor_net_conf= actor_net_conf,
            value_net_conf = value_net_conf,
            gamma = gamma,
        space_description = space_descr,
        name="PPO_Vanilla")
    else:
        return PPOConfig(
            learning_rate=0.00025,
            num_envs=4,
            num_steps=128,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            clip_coef=0.2,
            action_ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            norm_adv=True,
            clip_vloss=True,
            anneal_lr=anneal_lr,
            cuda=False,
            fallback_coef=0,
            actor_net_conf=actor_net_conf,
            value_net_conf=value_net_conf,
            gamma=gamma,
            space_description=space_descr,
            name="PPO_Vanilla")


#continuation: Loads the Agent saved at continuation
def agent(space_descr: EnvSpaceDescription, layer_sizes: Union[tuple[int],str] = (64,64), anneal_lr: str|bool = True, continuation: str = None, continuous: bool = True):
    return (lambda: PPO(cfg = agent_cfg(space_descr, layer_sizes=parse_tuple(layer_sizes, lambda x: int(x)), continuous = parse_bool(continuous), anneal_lr=parse_bool(anneal_lr)))) if continuation is None else lambda: Agent.load(Path(continuation))

def env_wrapper(env: Callable):

    def env_maker():
        return wrap_env(env(), flatten_obs=True)

    return env_maker
