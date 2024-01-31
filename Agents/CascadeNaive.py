from dataclasses import dataclass
from os.path import exists
from pathlib import Path
from typing import Callable

import numpy as np
import gymnasium as gym
from Agents.Agent import AgentConfig, Agent
from Agents.PPO import PPO, PPOConfig
from Agents.SequentialEnsemble import SequentialEnsembleConfig, SequentialEnsemble
from Environments.CascadeEnv import CascadeEnv

@dataclass
class CascadeNaiveConfig(SequentialEnsembleConfig):

    init_training_alg_cfg: PPOConfig = None #Config of the initial Agent
    stacked_training_alg_cfg: PPOConfig = None #Config of an Agent that trains on the surrogate environment (env extended with fallback-action)

    fallback_reward: float = -1.0
    propagate_action: bool = False
    propagate_value: bool = False

    def validate(self):
        assert self.init_training_alg_cfg.gamma == self.gamma, "Gamma in init_training_alg differs from gamma in CascadeNaiveConfig."
        assert self.stacked_training_alg_cfg.gamma == self.gamma, "Gamma in stacked_training_alg differs from gamma in CascadeNaiveConfig."
        super().validate()

class CascadeNaive(SequentialEnsemble):

    def __init__(self, cfg: CascadeNaiveConfig):
        super().__init__(cfg, add_first = False)
        self.cfg.validate()

    """
        'ignore_last': If set, the last saved Agent of the Cascade is disgarded.
        Should be set to True if continuing training, so that the last potentially bad agent isn't stuck in the cascade.
    """
    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig, ignore_last: bool = False) -> "Agent":
        agent = CascadeNaive(cfg = cfg)

        #Determine size of the Cascade
        num_base = 1
        while exists(absolute_path.joinpath(f"base{num_base}")):
            num_base += 1
        num_base -= 1

        #Load the agents of the Cascade
        agent.base_agents = [PPO.load(relative_path.joinpath(f"base{i + 1}")) for i in range(num_base)]

        #Check if last base-agent has to be deleted
        if ignore_last and len(agent.base_agents) > 0:
            del agent.base_agents[-1]

        return agent

    """
        Calculates the input for the end_idx-th base-net of the Cascade and the action that the Cascade up to the end_idx-th base-net produces.
        Is used as a helper method for get_action_and_value and get_action
    """
    def cascade_up(self, obs, end_idx: int, deterministic: bool = False):
        #Checks the case that the ensemble trained for so lang that the ensemble reached its maximum capacity and the first Agent had to be replaced.
        if (self.cfg.propagate_value or self.cfg.propagate_action) and self.base_agents[0].cfg.space_description.flattened_input_size() > self.cfg.space_description.flattened_input_size():
            raise ValueError("No value or action-propagate is allowed when the ensemble cycles.")

        last_action, last_val = None, None

        for i, agent in enumerate(self.base_agents[:end_idx]):
            # Prepare input
            x = obs
            if last_val is not None:
                x = np.concatenate([x, [last_val]])
            if last_action is not None and self.cfg.propagate_action:
                x = np.concatenate([x, last_action])
            # Prepare function for actor critic
            fbm = agent.get_action_and_value if self.cfg.propagate_value else agent.get_action
            # Get output of the current agent
            y = fbm(x, deterministic=deterministic)
            last_val = y["value"] if isinstance(y,dict) and "value" in y else None
            last_action = CascadeEnv.cascade_step(y["action"] if isinstance(y,dict) else y, last_action, get_fallback=False, hard_threshold = deterministic)
            #For the case the sequential algorithm has already removed the original base agent
            last_action = last_action[0:self.cfg.space_description.flattened_act_size()]

        # Prepare for the end_idx-th agent
        next_input = obs
        if last_val is not None:
            next_input = np.concatenate([next_input, [last_val]])
        if last_action is not None and self.cfg.propagate_action:
            next_input = np.concatenate([next_input, last_action])

        vals = {"action": last_action, "next": next_input}
        if self.cfg.propagate_value:
            vals["value"] = last_val
        return vals

    def get_action_and_value(self, obs, deterministic: bool = False):
        if len(self.base_agents) > 1:
            casc_vals = self.cascade_up(obs, -1, deterministic=deterministic)
            top_out = self.base_agents[-1].get_action_and_value(casc_vals["next"], deterministic=deterministic)
            action = CascadeEnv.cascade_step(top_out["action"], casc_vals["action"], hard_threshold=deterministic, get_fallback=False)
            return {"action": action, "value": top_out["value"]}
        else:
            return self.base_agents[-1].get_action_and_value(obs, deterministic=deterministic)

    """
        Returns action of the entire Cascade. If in evaluation mode, the current base-agent that is being trained
        is considered as part of the Cascade.
    """
    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        if eval_mode and self.current_base_agent:
            self.base_agents.append(self.current_base_agent)
        if len(self.base_agents) > 1:
            casc_vals = self.cascade_up(obs, -1, deterministic=deterministic)
            top_action = self.base_agents[-1].get_action(casc_vals["next"])
            action = CascadeEnv.cascade_step(top_action, casc_vals["action"], hard_threshold=deterministic, get_fallback=False)
        else:
            action = self.base_agents[-1].get_action(obs, deterministic=deterministic)

        if eval_mode and self.current_base_agent:
            del self.base_agents[-1]
        return action

    """
        Returns surrogate environment that simply extends the original environment with the fallback-action.
    """
    def setup_base_training_env(self, env_maker: Callable[[],gym.core.Env]) -> Callable[[],gym.core.Env]:
        if self.base_agents:
            return lambda: CascadeEnv(env_maker(), self, fallback_punishment=self.cfg.fallback_reward, propagate_value=self.cfg.propagate_value, propagate_action=self.cfg.propagate_action)
        else:
            return env_maker

    def setup_base_agent(self) -> Agent:
        return PPO(cfg= self.cfg.stacked_training_alg_cfg if self.base_agents else self.cfg.init_training_alg_cfg)