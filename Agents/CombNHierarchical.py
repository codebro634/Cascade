from os.path import exists
from pathlib import Path
from typing import Callable, Union, Tuple

import torch

from Agents.Agent import Agent, AgentConfig
from Agents.PPO import PPO, PPOConfig
from dataclasses import dataclass
import gymnasium as gym
from Analysis.RunTracker import RunTracker, TrackMetric, TrackConfig
from Architectures.ActorCritic import ActorCritic
from Architectures.ActorHead import CombActorHead
import random

from Environments.Utils import sync_normalization_state


@dataclass
class HCConfig(AgentConfig):

    training_alg_cfg: PPOConfig = None #Training configuration with which all nets are trained.
    chooser_net_cfg: "NetworkConfig" = None #Network configuration of the chooser that combines action-nets or CombN-nets.
    cycle_steps: Union[int,Tuple[int]]  = 1000000 #The number of training steps for agents at each cycle. If int, this number is used for all cycles.
    cycles: int = 1 #Number of iterations of combining the current nets with a chooser in groups of n.
    n: int = 2 #Number of nets per layer. I.e. if total number of action-nets at bottom layer is n^cycles.
    pretrain: bool = True #Whether action-nets (or at higher levels CombN-nets) are trained before they are combined.
    freeze_bottom: bool = False #If set, only the uppermost chooser is trained.

    def validate(self):
        assert self.training_alg_cfg.gamma == self.gamma


class HierComb(Agent):

    def __init__(self,cfg: HCConfig):
        super().__init__(cfg)
        self.cycle = None #The number of cycles finished
        self.current_cycle_agents = None #List of all networks of the current cycle
        self.cfg.validate()


    def save_additionals(self, model_path: Path, absolute_path: Path):
        self.current_cycle_agents[0].save(model_path.joinpath(f"cycle{self.cycle}"))

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        agent = HierComb(cfg = cfg)

        for i in range(cfg.cycles+1):
            agent.combine_agents(cycle=i)
        assert len(agent.current_cycle_agents) == 1

        checkpoint = torch.load(absolute_path.joinpath(f"cycle{cfg.cycles}").joinpath("ac.nn"))
        agent.current_cycle_agents[0].net.load_state_dict(checkpoint)

        return agent

    """
        If deterministic, returns the action of the current agent at index 0, otherwise randomly
        takes the action of any of the current agents.
    """
    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        agent_idx = 0 if deterministic else random.randint(0,len(self.current_cycle_agents)-1)
        return self.current_cycle_agents[agent_idx].get_action(obs, eval_mode=eval_mode, deterministic=deterministic)

    def get_cycle_steps(self):
        return self.cfg.cycle_steps if isinstance(self.cfg.cycle_steps,int) else self.cfg.cycle_steps[self.cycle]


    """
        If cycle=0, then self.cfg.^self.cfg.cycles many base-agents are created
        If cycle>0, then the current agents are combined in groups of self.cfg.n into a CombN-net
    """
    def combine_agents(self, cycle: int):
        if cycle == 0:
            self.current_cycle_agents = [PPO(cfg=self.cfg.training_alg_cfg) for _ in range(self.cfg.n ** self.cfg.cycles)]
        else:
            new_cycle_agents = []
            for i in range(0, len(self.current_cycle_agents), self.cfg.n):
                agent = PPO(cfg=self.cfg.training_alg_cfg)
                comb_actor = CombActorHead(freeze_action_nets=self.cfg.freeze_bottom, chooser=self.cfg.chooser_net_cfg,heads=[self.current_cycle_agents[i + j].net.actor for j in range(0, self.cfg.n)])
                comb_ac = ActorCritic(actor=comb_actor, critic=agent.net.critic, shared=agent.net.shared)
                agent.replace_net(comb_ac)
                new_cycle_agents.append(agent)
            self.current_cycle_agents = new_cycle_agents

    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env: gym.core.Env = None):
        if norm_sync_env is None:
            norm_sync_env = env_maker()
        synced_env_maker = lambda: sync_normalization_state(norm_sync_env, env_maker())

        assert self.current_cycle_agents is None, "Training continuation not supported."

        self.cycle = 0
        while self.cycle < self.cfg.cycles+1:

            #Setup current cycle agents
            self.combine_agents(self.cycle)

            #Train cycle agents
            if self.cfg.pretrain or self.cycle == self.cfg.cycles:
                for agent in self.current_cycle_agents:
                    base_rt = RunTracker(cfg=TrackConfig(metric=TrackMetric.STEPS, total=self.get_cycle_steps() , eval_interval=0),eval_func=None, show_progress=False, nested=tracker, nested_progress=False)
                    agent.train(synced_env_maker, base_rt if self.cfg.pretrain else tracker, norm_sync_env=norm_sync_env)
                    if tracker.is_done():
                        break

            self.cycle += 1
            if tracker.is_done():
                break






