from os.path import exists
from pathlib import Path
from typing import Callable

import torch

from Agents.Agent import Agent, AgentConfig
from Agents.PPO import PPO, PPOConfig
from dataclasses import dataclass
import gymnasium as gym
from Analysis.RunTracker import RunTracker, TrackMetric, TrackConfig
from Architectures.CascadeAC import ActorCriticCascade

from Environments.Utils import sync_normalization_state

from Agents.DDPG import DDPG


@dataclass
class CascadeConfig(AgentConfig):

    training_alg: str = None  #The training algorithm used for training the Cascade-net
    training_alg_cfg: AgentConfig = None #The Agent config used for training the Cascade-net
    init_net_cfg: "NetworkConfig" = None #Network config for the first base-net
    stacked_net_cfg: "NetworkConfig" = None #Network config for all nets besides the first one. These must include the fallback-action

    base_steps: int = 100000

    propagate_action: bool = False
    propagate_value: bool = False
    train_only_top_net: bool = False #If set, only the last added base-net is trained.
    sequential: bool = True #If set, base-nets are sequentially added. If False, a Cascade-net with 'stacks' many base-nets is directly trained.
    stacks: int = 0 #Only used when not training sequentially
    cyclical_lr: bool = True #if set, the learning rate linearly decreases from its max value to 0 every iteration

    def validate(self):
        assert self.training_alg_cfg.gamma == self.gamma
        assert not (not self.sequential and self.stacks <= 0)
        assert not (not self.sequential and self.train_only_top_net)


class Cascade(Agent):

    def __init__(self, cfg: CascadeConfig):
        super().__init__(cfg)
        self.top= None #An Agent with an 'ActorCriticCascade'-network built from 'self.acs'
        self.acs = [] #List of Actor Critics that make up the Cascade
        self.cfg.validate()

    def model_size(self):
        actors = sum( sum([p.numel() for p in ac.actor.parameters()]) for ac in self.acs)
        critic = sum(p.numel() for p in self.acs[-1].critic.parameters())
        return actors+critic

    def save_additionals(self, model_path: Path, absolute_path: Path):
        for i,ac in enumerate(self.acs):
            torch.save(ac.state_dict(), absolute_path.joinpath(f"ac{i+1}.nn"))

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        agent = Cascade(cfg=cfg)

        #Determine out of how many base-nets the saved Cascade-net consists
        stacks = 1
        while exists(absolute_path.joinpath(f"ac{stacks}.nn")):
            stacks += 1
        stacks-=1

        #Parse the base-nets into a Cascade-net
        acs = []
        for i in range(stacks):
            conf = agent.cfg.stacked_net_cfg if i > 0 else agent.cfg.init_net_cfg
            net = conf.init_obj()
            if agent.cfg.train_only_top_net and i+1 < stacks:
                for param in net.parameters():
                    param.requires_grad = False

            net.load_state_dict(torch.load(absolute_path.joinpath(f"ac{i+1}.nn")))
            acs.append(net)
        wrapped_net = ActorCriticCascade(acs, propagate_action=agent.cfg.propagate_action,propagate_value=agent.cfg.propagate_value)

        #Create agent with the loaded Cascade-net
        if agent.cfg.training_alg == "PPO":
            top = PPO(cfg=agent.cfg.training_alg_cfg)
        elif agent.cfg.training_alg == "DDPG":
            raise NotImplementedError("DDPG Cascade load not implemented.")
            #top = DDPG(cfg=agent.cfg.training_alg_cfg)

        #top = top = PPO(cfg=agent.cfg.training_alg_cfg)
        top.replace_net(wrapped_net)

        agent.top = top
        agent.acs = acs
        return agent

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        return self.top.get_action(obs, eval_mode=eval_mode, deterministic=deterministic)

    #Trains the current Cascade-net represented by 'self.acs' for 'self.cfg.base_steps' steps.
    def train_current_cascade(self, tracker: RunTracker, env_maker: Callable, norm_sync_env: gym.Env = None):
        wrapped_net = ActorCriticCascade(self.acs, propagate_action=self.cfg.propagate_action,propagate_value=self.cfg.propagate_value)
        if self.cfg.training_alg == "DDPG":
            rb = self.top.rb if self.top is not None else None
            self.top = DDPG(cfg=self.cfg.training_alg_cfg)
            self.top.rb = rb
        elif self.cfg.training_alg == "PPO":
            self.top = PPO(cfg=self.cfg.training_alg_cfg)
        else:
            raise NotImplementedError()
        self.top.replace_net(wrapped_net)
        rt = RunTracker(cfg=TrackConfig(metric=TrackMetric.STEPS, total=self.cfg.base_steps, eval_interval=0),eval_func=None, show_progress=False, nested=tracker, nested_progress=not self.cfg.cyclical_lr)
        self.top.train(env_maker, rt, norm_sync_env=norm_sync_env)

    def train(self, env_maker: Callable, tracker: RunTracker, norm_sync_env: gym.Wrapper = None):
        #Makes sure each training cycle keeps the normalization state from the previous one.
        if norm_sync_env is None:
            norm_sync_env = env_maker()
        synced_env_maker = lambda: sync_normalization_state(norm_sync_env, env_maker())

        #assert self.top is None, "Training continuation not supported."

        if not self.cfg.sequential:
            self.acs = [(self.cfg.stacked_net_cfg if stack > 0 else self.cfg.init_net_cfg).init_obj() for stack in range(self.cfg.stacks)]
            while not tracker.is_done():
                self.train_current_cascade(tracker, synced_env_maker, norm_sync_env=norm_sync_env)
        else:
            init_cycle = self.top is None
            while not tracker.is_done():
                #Setup top-net of Cascade
                conf = self.cfg.init_net_cfg if init_cycle else self.cfg.stacked_net_cfg
                self.acs.append(conf.init_obj())

                #Train current Cascade-net
                self.train_current_cascade(tracker,synced_env_maker,norm_sync_env=norm_sync_env)

                #Freeze parameters of old-nets
                if self.cfg.train_only_top_net:
                    for param in self.top.net.parameters():
                        param.requires_grad = False

                init_cycle = False






