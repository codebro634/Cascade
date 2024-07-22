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
from Agents.Agent import AgentConfig, Agent
from pathlib import Path
from Analysis.RunTracker import TrackMetric as tm, RunTracker
import stable_baselines3 as sb3
from Analysis import RunTracker
from copy import deepcopy

from Environments.Utils import sync_normalization_state, get_normalization_state, apply_observation_normalization, apply_reward_normalization, reverse_observation_normalization, reverse_reward_normalization



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
    actor_net_conf: NetworkConfig = None #Configuration for Actor network
    critic_net_conf: NetworkConfig = None #Configuration for Critic network
    tanh_in_net: bool = False #If toggled, get_action does not apply tanh to the output of the actor network

    def validate(self):
        assert self.actor_net_conf is not None and self.critic_net_conf is not None, "actor_net_conf and critic_net_conf must be set"

class DDPG(Agent):

    def __init__(self, cfg: DDPGConfig):
        super().__init__(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        self.actor_net, self.target_actor = None, None
        self.q_net, self.target_q = None, None
        self.q_optim, self.actor_optim = None, None
        self.replace_net(self.cfg.actor_net_conf.init_obj().to(self.device), self.cfg.critic_net_conf.init_obj().to(self.device))
        self.rb = None

        self.action_scale = torch.tensor((self.cfg.space_description.action_space.high - self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)
        self.action_bias = torch.tensor((self.cfg.space_description.action_space.high + self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)

        # Check constraints
        self.cfg.validate()

    def model_size(self):
        return sum(p.numel() for p in ( list(self.actor_net.parameters()) + list(self.q_net.parameters())))

    def replace_net(self, actor_net, q_net):
        self.actor_net = actor_net
        self.q_net = q_net
        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        raise NotImplementedError()

    def save_additionals(self, model_path: Path, absolute_path: Path):
        pass

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False, target: bool = False):
        squeeze = len(obs.shape) == 1
        if squeeze:
            obs = torch.Tensor(obs, device=self.device).unsqueeze(0)
        net = self.target_actor if target else self.actor_net

        actions = net(torch.Tensor(obs).to(self.device))["mean"]
        actions = (actions if self.cfg.tanh_in_net else torch.tanh(actions)) * self.action_scale + self.action_bias

        if not deterministic:
            actions += torch.normal(0, self.action_scale * self.cfg.exploration_noise)
        if eval_mode:
            actions = actions.cpu().detach().numpy().clip(self.cfg.space_description.action_space.low, self.cfg.space_description.action_space.high)

        return actions.squeeze(0) if squeeze else actions



    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env:gym.core.Env = None):
        envs = gym.vector.SyncVectorEnv([lambda: env_maker()])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        if self.target_actor is None:
            self.target_actor = deepcopy(self.actor_net)
            self.target_actor.load_state_dict(self.actor_net.state_dict())
            self.target_q = deepcopy(self.q_net)
            self.target_q.load_state_dict(self.q_net.state_dict())

        envs.single_observation_space.dtype = np.float32

        if self.rb is None:
            self.rb = ReplayBuffer(
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

            #if "final_info" in infos:
            #    for info in infos["final_info"]:
            #        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

            abort_training = abort_training or tracker.add_unit(tm.STEPS, 1)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            norm_state = get_normalization_state(envs.envs[0])
            if len(norm_state) > 0:
                self.rb.add(reverse_observation_normalization(obs, norm_state["obs mean"], norm_state["obs var"]),
                            reverse_observation_normalization(real_next_obs, norm_state["obs mean"], norm_state["obs var"]),
                            actions, reverse_reward_normalization(rewards, norm_state["rew var"]), terminations, [])
            else:
                self.rb.add(obs, real_next_obs, actions, rewards, terminations, [])
            obs = next_obs

            if norm_sync_env:
                sync_normalization_state(envs.envs[0], norm_sync_env)

            if global_step > self.cfg.learning_starts:
                data = self.rb.sample(self.cfg.batch_size)

                if len(norm_state) > 0:
                    data_observations = apply_observation_normalization(data.observations, norm_state["obs mean"], norm_state["obs var"]).float()
                    data_next_observations = apply_observation_normalization(data.next_observations, norm_state["obs mean"], norm_state["obs var"]).float()
                    data_rewards = apply_reward_normalization(data.rewards, norm_state["rew var"]).float()
                else:
                    data_observations = data.observations
                    data_next_observations = data.next_observations
                    data_rewards = data.rewards

                with torch.no_grad():
                    next_state_actions = self.get_action(data_next_observations, target=True, deterministic=True)
                    qf1_next_target = self.target_q(torch.concat((data_next_observations,next_state_actions),dim=1))["y"]
                    next_q_value = data_rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.gamma * (qf1_next_target).view(-1)


                qf1_a_values = self.q_net(torch.concat((data_observations, data.actions), dim=1) )["y"].view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                self.q_optim.zero_grad()
                qf1_loss.backward()
                self.q_optim.step()

                if global_step % self.cfg.policy_frequency == 0:
                    actor_loss = -self.q_net(torch.concat((data_observations, self.get_action(data_observations, target=False, deterministic=True)),dim=1))["y"].mean()
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    # update the target network
                    for param, target_param in zip(self.actor_net.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
                    for param, target_param in zip(self.q_net.parameters(), self.target_q.parameters()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

                abort_training = abort_training or tracker.add_unit(tm.SAMPLES, self.cfg.batch_size)
                abort_training = abort_training or tracker.add_unit(tm.EPOCHS, 1)

        envs.close()