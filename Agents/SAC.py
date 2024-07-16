# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from Agents.Agent import Agent, AgentConfig
from Agents.DDPG import DDPGConfig
from Analysis.RunTracker import TrackMetric as tm, RunTracker

from Environments.Utils import get_normalization_state, reverse_observation_normalization, reverse_reward_normalization, \
    apply_observation_normalization, apply_reward_normalization, sync_normalization_state


@dataclass
class SACConfig(AgentConfig):

    cuda: bool = True
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    alpha: float = 0.2

    actor_conf: "NetworkConfig" = None
    q1_conf: "NetworkConfig" = None
    q2_conf: "NetworkConfig" = None

    def validate(self):
        assert self.actor_conf is not None and self.q1_conf is not None and self.q2_conf is not None, "actor_conf, q1_conf and q2_conf must be set"

class SAC(Agent):

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    def __init__(self, cfg: SACConfig):
        super().__init__(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        self.actor_net, self.q1, self.q2, self.q1_target, self.q2_target = None, None, None, None, None
        self.q_optim, self.actor_optim = None, None
        self.replace_net(self.cfg.actor_conf.init_obj(), self.cfg.q1_conf.init_obj(), self.cfg.q2_conf.init_obj())
        self.rb = None

        self.action_scale = torch.tensor((self.cfg.space_description.action_space.high - self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)
        self.action_bias = torch.tensor((self.cfg.space_description.action_space.high + self.cfg.space_description.action_space.low) / 2, dtype=torch.float32)

        # Check constraints
        self.cfg.validate()

    def save_additionals(self, model_path: Path, absolute_path: Path):
        pass

    def model_size(self):
        return sum(p.numel() for p in self.net.parameters())

    def replace_net(self, actor, q1, q2):
        self.actor_net = actor
        self.q1 = q1
        self.q2 = q2
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.cfg.q_lr, eps=1e-5)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=self.cfg.policy_lr, eps=1e-5)

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        squeeze = len(obs.shape) == 1
        if squeeze:
            obs = torch.Tensor(obs, device=self.device).unsqueeze(0)

        net_out = self.actor_net(obs)
        mean, log_std = net_out["mean"], net_out["logstd"]
        log_std = torch.tanh(log_std)
        log_std = SAC.LOG_STD_MIN + 0.5 * (SAC.LOG_STD_MAX - SAC.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = mean if deterministic else normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if eval_mode:
            action = action.cpu().detach().numpy().clip(self.cfg.space_description.action_space.low, self.cfg.space_description.action_space.high)
            return action.squeeze(0) if squeeze else action
        else:
            return action, log_prob, mean

    def train(self, env_maker: Callable[[], gym.core.Env], tracker: RunTracker, norm_sync_env: gym.core.Env = None):
        device="cpu"
        envs = gym.vector.SyncVectorEnv([lambda: env_maker()])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        actor = self.actor_net.to(device)
        qf1 = self.q1.to(device)
        qf2 = self.q2.to(device)

        if self.q1_target is None:
            self.q1_target = deepcopy(self.q1).to(device)
            self.q2_target = deepcopy(self.q2).to(device)
            self.q1_target.load_state_dict(qf1.state_dict())
            self.q2_target.load_state_dict(qf2.state_dict())

            qf1_target = self.q1_target
            qf2_target = self.q2_target

        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=self.cfg.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=self.cfg.policy_lr)


        alpha = self.cfg.alpha

        envs.single_observation_space.dtype = np.float32
        self.rb = ReplayBuffer(
            self.cfg.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset()

        global_step = -1
        abort_training = False
        while not abort_training:
            global_step += 1

            # ALGO LOGIC: put action logic here
            if global_step < self.cfg.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.get_action(torch.Tensor(obs).to(device), deterministic=False, eval_mode=True)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            abort_training = abort_training or tracker.add_unit(tm.STEPS, 1)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            #if "final_info" in infos:
            #    for info in infos["final_info"]:
            #        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #        break

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            norm_state = get_normalization_state(envs.envs[0])
            if len(norm_state) > 0:
                self.rb.add(reverse_observation_normalization(obs, norm_state["obs mean"], norm_state["obs var"]),
                            reverse_observation_normalization(real_next_obs, norm_state["obs mean"],
                                                              norm_state["obs var"]),
                            actions, reverse_reward_normalization(rewards, norm_state["rew var"]), terminations, [])
            else:
                self.rb.add(obs, real_next_obs, actions, rewards, terminations, [])

            if norm_sync_env:
                sync_normalization_state(envs.envs[0], norm_sync_env)

            obs = next_obs

            if global_step > self.cfg.learning_starts:
                data = self.rb.sample(self.cfg.batch_size)

                #normalize sample
                if len(norm_state) > 0:
                    data_observations = apply_observation_normalization(data.observations, norm_state["obs mean"], norm_state["obs var"]).float()
                    data_next_observations = apply_observation_normalization(data.next_observations, norm_state["obs mean"], norm_state["obs var"]).float()
                    data_rewards = apply_reward_normalization(data.rewards, norm_state["rew var"]).float()
                else:
                    data_observations = data.observations
                    data_next_observations = data.next_observations
                    data_rewards = data.rewards

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.get_action(data_next_observations)
                    qf1_next_target = qf1_target(torch.concat((data_next_observations,next_state_actions),dim=1))["y"]
                    qf2_next_target = qf2_target(torch.concat((data_next_observations,next_state_actions),dim=1))["y"]
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data_rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(torch.concat((data_observations, data.actions), dim=1))["y"].view(-1)
                qf2_a_values = qf2(torch.concat((data_observations, data.actions),dim=1))["y"].view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % self.cfg.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.cfg.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.get_action(data.observations)
                        qf1_pi = qf1(torch.concat((data.observations, pi),dim=1))["y"]
                        qf2_pi = qf2(torch.concat((data.observations, pi),dim=1))["y"]
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                # update the target networks
                if global_step % self.cfg.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        envs.close()