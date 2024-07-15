from os.path import exists
from pathlib import Path
from typing import Callable

import torch

from Agents.Agent import Agent, AgentConfig
from Agents.PPO import PPO, PPOConfig
from dataclasses import dataclass
import gymnasium as gym

from Agents.SAC import SAC
from Analysis.RunTracker import RunTracker, TrackMetric, TrackConfig
from Architectures.CascadeNet import CascadeNet

from Environments.Utils import sync_normalization_state

from Agents.DDPG import DDPG


@dataclass
class CascadeConfig(AgentConfig):

    training_alg: str = None  #The training algorithm used for training the Cascade-net
    training_alg_cfg: AgentConfig = None #The Agent config used for training the Cascade-net

    init_actor_cfg: "NetworkConfig" = None #Network config for the first base-net
    stacked_actor_cfg: "NetworkConfig" = None #Network config for all nets besides the first one. These must include the fallback-action

    init_critic_cfgs: list["NetworkConfig"] = None #Network config for the critic of all base-nets
    stacked_critic_cfgs: list["NetworkConfig"] = None #Network config for the critic of all base-nets besides the first one
    stack_critics: bool = False

    base_steps: int = 100000

    train_only_top_net: bool = False #If set, only the last added base-net is trained.
    sequential: bool = True #If set, base-nets are sequentially added. If False, a Cascade-net with 'stacks' many base-nets is directly trained.
    stacks: int = 0 #Only used when not training sequentially
    cyclical_lr: bool = True #if set, the learning rate linearly decreases from its max value to 0 every iteration

    """
        Alg specific params
    """
    reset_rb: bool = False #If set, the replay buffer is reset after each training cycle if the base algorithm uses a replay buffer.

    def validate(self):
        assert self.training_alg_cfg.gamma == self.gamma
        assert not (not self.sequential and self.stacks <= 0)
        assert not (not self.sequential and self.train_only_top_net)


class Cascade(Agent):

    def __init__(self, cfg: CascadeConfig):
        super().__init__(cfg)
        self.top = None
        self.critics, self.actors = [[] for _ in range(len(self.cfg.init_critic_cfgs))], []
        self.cfg.validate()

    def model_size(self):
        actors = sum(sum([p.numel() for p in ac.actor_net.parameters()]) for ac in self.actors)
        critics = sum (sum(sum([p.numel() for p in ac.actor_net.parameters()]) for ac in self.critics[i]) for i in range(len(self.critics)))
        return actors+critics

    def save_additionals(self, model_path: Path, absolute_path: Path):
        pass

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        raise NotImplementedError()

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        return self.top.get_action(obs, eval_mode=eval_mode, deterministic=deterministic)

    #Trains the current Cascade-net represented by 'self.acs' for 'self.cfg.base_steps' steps.
    def train_current_cascade(self, tracker: RunTracker, env_maker: Callable, norm_sync_env: gym.Env = None):

        if self.cfg.training_alg == "DDPG":
            rb = self.top.rb if self.top is not None else None
            self.top = DDPG(cfg=self.cfg.training_alg_cfg)
            if not self.cfg.reset_rb:
                self.top.rb = rb
            casc_actors, casc_q = CascadeNet(self.actors), CascadeNet(self.critics[0]) if self.cfg.stack_critics else self.critics[0][-1]
            self.top.replace_net(actor_net=casc_actors, q_net = casc_q)
        elif self.cfg.training_alg == "PPO":
            self.top = PPO(cfg=self.cfg.training_alg_cfg)
            casc_actors, casc_v = CascadeNet(self.actors), CascadeNet(self.critics[0]) if self.cfg.stack_critics else self.critics[0][-1]
            self.top.replace_net(actor_net=casc_actors, value_net=casc_v)
        elif self.cfg.training_alg == "SAC":
            self.top = SAC(cfg=self.cfg.training_alg_cfg)
            casc_actors, casc_q1, casc_q2 = CascadeNet(self.actors), CascadeNet(self.critics[0]) if self.cfg.stack_critics else self.critics[0][-1], CascadeNet(self.critics[1]) if self.cfg.stack_critics else self.critics[1][-1]
            self.top.replace_net(actor=casc_actors, q1=casc_q1, q2=casc_q2)
        else:
            raise NotImplementedError()

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
                if init_cycle:
                    actor_conf = self.cfg.init_actor_cfg
                    critic_confs = self.cfg.init_critic_cfgs
                else:
                    actor_conf = self.cfg.stacked_actor_cfg
                    critic_confs = self.cfg.stacked_critic_cfgs if self.cfg.stack_critics else self.cfg.init_critic_cfgs
                self.actors.append(actor_conf.init_obj())
                for i, conf in enumerate(critic_confs):
                    self.critics[i].append(conf.init_obj())

                #Train current Cascade-net
                self.train_current_cascade(tracker, synced_env_maker, norm_sync_env=norm_sync_env)

                #Freeze parameters of old-nets
                if self.cfg.train_only_top_net:
                    for param in self.top.actor_net.parameters():
                        param.requires_grad = False

                init_cycle = False






