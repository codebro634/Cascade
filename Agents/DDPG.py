# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from Architectures.NetworkConfig import NetworkConfig
from Cascade.Agents.Agent import AgentConfig, Agent
from pathlib import Path
from Analysis.RunTracker import TrackMetric as tm, RunTracker
import stable_baselines3 as sb3
from Analysis import RunTracker
from copy import deepcopy


@dataclass
class DDPGConfig(AgentConfig):
    cuda: bool = False #if toggled, cuda will be enabled by default
    learning_rate: float = 3e-4 #the learning rate of the optimizer
    buffer_size: int = int(1e6) #the replay memory buffer size
    tau: float = 0.005 #target smoothing coefficient (default: 0.005)
    batch_size: int = 256 #the batch size of sample from the reply memory
    exploration_noise: float = 0.1 #the scale of exploration noise
    learning_starts: int = 25e3 #timestep to start learning
    policy_frequency: int = 2 #the frequency of training policy (delayed)
    anneal_lr: bool = False #if toggled, the learning rate will be annealed linearly to 0 during training
    noise_clip: float = 0.5 #noise clip parameter of the Target Policy Smoothing Regularization
    net_conf: NetworkConfig = None #Configuration for Actor Critic network

    def validate(self):
        assert self.net_conf is not None

class DDPG(Agent):

    def __init__(self, cfg: DDPGConfig):
        super().__init__(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        self.net, self.target = None, None
        self.q_optim, self.actor_optim = None, None
        self.replace_net(self.cfg.net_conf.init_obj().to(self.device))

        self.action_scale = torch.tensor((self.cfg.space_description.action_space.high - self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)
        self.action_bias = torch.tensor((self.cfg.space_description.action_space.high + self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)

        # Check constraints
        self.cfg.validate()

    def model_size(self):
        return sum(p.numel() for p in self.net.parameters())

    def replace_net(self, net: nn.Module):
        self.net = net
        self.q_optim = optim.Adam(self.net.critic_params(), lr=self.cfg.learning_rate, eps=1e-5)
        self.actor_optim = optim.Adam(self.net.actor_params(), lr=self.cfg.learning_rate, eps=1e-5)

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        agent = DDPG(cfg=cfg)
        agent.net.load_state_dict(torch.load(absolute_path.joinpath("net.nn")))
        agent.target.load_state_dict(torch.load(absolute_path.joinpath("target.nn")))
        return agent

    def save_additionals(self, model_path: Path, absolute_path: Path):
        torch.save(self.net.state_dict(), absolute_path.joinpath("net.nn"))
        torch.save(self.target.state_dict(), absolute_path.joinpath("target.nn"))

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False, target: bool = False):
        squeeze = len(obs.shape) == 1
        if squeeze:
            obs = torch.Tensor(obs, device=self.device).unsqueeze(0)
        net = self.target if target else self.net
        actions = net.get_action(torch.Tensor(obs).to(self.device))["mean"]
        if not deterministic:
            actions += torch.normal(0, self.action_scale * self.cfg.exploration_noise)
        if eval_mode:
            actions = actions.cpu().detach().numpy().clip(self.cfg.space_description.action_space.low, self.cfg.space_description.action_space.high)

        if squeeze:
            return actions.squeeze(0)
        else:
            return actions


    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env:gym.core.Env = None):
        envs = gym.vector.SyncVectorEnv( [lambda: env_maker()])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        if self.target is None:
            self.target = deepcopy(self.net)
            self.target.load_state_dict(self.net.state_dict())

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            self.cfg.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        obs, _ = envs.reset()

        abort_training = False
        global_step = -1
        while not abort_training:
            if self.cfg.anneal_lr:
                frac = 1.0 - tracker.get_progress()
                lrnow = frac * self.cfg.learning_rate
                self.q_optim.param_groups[0]["lr"] = lrnow
                self.actor_optim.param_groups[0]["lr"] = lrnow

            global_step += 1
            if global_step < self.cfg.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.get_action(obs, eval_mode=True, deterministic=False, target=False)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            abort_training = abort_training or tracker.add_unit(tm.STEPS, 1)

            real_next_obs = next_obs.copy()
            rb.add(obs, real_next_obs, actions, rewards, terminations, [])
            obs = next_obs

            if global_step > self.cfg.learning_starts:
                data = rb.sample(self.cfg.batch_size)
                with torch.no_grad():
                    next_state_actions = self.get_action(data.next_observations, target=True)
                    qf1_next_target = self.target.q_value(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.gamma * (qf1_next_target).view(-1)

                qf1_a_values = self.net.q_value(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                self.q_optim.zero_grad()
                qf1_loss.backward()
                self.q_optim.step()

                if global_step % self.cfg.policy_frequency == 0:
                    actor_loss = -self.net.q_value(data.observations, self.get_action(data.observations,target=False)).mean()
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    # update the target network
                    for param, target_param in zip(self.net.actor_params(), self.target.actor_params()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
                    for param, target_param in zip(self.net.critic_params(), self.target.critic_params()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

                abort_training = abort_training or tracker.add_unit(tm.SAMPLES, self.cfg.batch_size)
                abort_training = abort_training or tracker.add_unit(tm.EPOCHS, 1)

        envs.close()