# Taken and modified from CleanRL
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete
from torch.distributions import Normal, Categorical

from Analysis.RunTracker import TrackMetric as tm, RunTracker
from pathlib import Path
from Agents.Agent import Agent, AgentConfig
from typing import Callable

from Architectures.Elementary import abs_difference
from Architectures.NetworkConfig import NetworkConfig
from Environments.Utils import sync_normalization_state


@dataclass
class PPOConfig(AgentConfig):
    learning_rate: float = 2.5e-4 #the learning rate of the optimizer
    num_envs: int = 4 #the number of parallel game environments
    num_steps: int = 128 #the number of steps to run in each environment per policy rollout
    gae_lambda: float = 0.95 #the lambda for the general advantage estimation
    num_minibatches: int = 4 #the number of mini-batches
    update_epochs: int = 4 #the K epochs to update the policy
    clip_coef: float = 0.2 #the surrogate clipping coefficient
    action_ent_coef: float = 0.01 #coefficient of the action-entropy
    vf_coef: float = 0.5 #coefficient of the value function
    max_grad_norm: float = 0.5 #the maximum norm for the gradient clipping
    norm_adv: bool = True #Toggles advantages normalization
    clip_vloss: bool = True #Toggles whether to use a clipped loss for the value function, as per the paper.
    anneal_lr: bool = False #Toggle learning rate annealing for policy and value networks
    cuda: bool = False #if toggled, cuda will be enabled by default
    actor_net_conf: NetworkConfig =None #Configuration for the used network
    value_net_conf: NetworkConfig = None #Configuration for the used value network

    #These parameters are only used when using the Cascade-Architecture
    fallback_coef: float = 0 #coefficient for fallback-weights. 0 if no Minimization

    def validate(self):
        assert self.actor_net_conf is not None and self.value_net_conf is not None, "Actor and Value Network Configurations must be set."

class PPO(Agent):

    def __init__(self, cfg: PPOConfig):
        super().__init__(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")

        self.batch_size = int(self.cfg.num_envs * self.cfg.num_steps)
        self.minibatch_size = int(self.batch_size // self.cfg.num_minibatches)

        self.actor_net, self.v_net = None, None
        self.optimizer = None
        self.replace_net(self.cfg.actor_net_conf.init_obj().to(self.device), self.cfg.value_net_conf.init_obj().to(self.device))

        #Check constraints
        self.cfg.validate()

    def model_size(self):
        return sum(p.numel() for p in self.actor_net.parameters() + self.v_net.parameters())

    """
        Replaces the current network. This updates the optimizer's parameters as well.
    """
    def replace_net(self, actor_net: nn.Module, value_net: nn.Module):
        self.actor_net = actor_net
        self.v_net = value_net
        self.optimizer = optim.Adam(list(self.actor_net.parameters()) + list(self.v_net.parameters()), lr=self.cfg.learning_rate, eps=1e-5)


    """
        Passes obs through the Actor Critic network and uses its outputs to obtain a probability distribution
        from which an action is sampled. If discrete, then the distribution is discrete and the output are the logits
        for Softmax and if continuous the output is interpreted as the logstd and mean of a normal-distribution.
        
        action: If not None, instead of sampling, this action is returned
        action_entropy: If set, returns entropy of the output distribution
        action_logprob: If set, returns log of probability of sampled action from the output distribution
        deterministic: If set and discrete, the most probable action is taken and if continuous the mean is returned
        get_value: If set, the state-value approximation of the network is returned
    """
    def get_action_and_value_net(self, obs, action = None, get_value: bool = True
                             ,action_entropy: bool = False, action_logprob: bool = False, deterministic: bool = False):
        #Pass obs through net
        x = self.actor_net(obs)

        discrete, mul_disc = self.cfg.space_description.is_discrete_action(), self.cfg.space_description.is_multi_discrete()
        #Continuous case
        if not discrete:
            assert "logstd" in x
            action_std = torch.exp(x["logstd"])
            assert x["mean"].shape == action_std.shape

            probs = Normal(x["mean"], action_std)
            if action is None:
                action = x["mean"] if deterministic else probs.sample()

            if action_logprob:
                x["action_logprob"] = probs.log_prob(action).sum(1)
            if action_entropy:
                x["action_entropy"] = probs.entropy().sum(1)
        else:
            #Differentiate between multi discrete and 1-dim discrete
            if mul_disc:
                if action is None:
                    probs = torch.sigmoid(x["mean"])
                    action = torch.where(probs < 0.5, torch.tensor(0), torch.tensor(1)) if deterministic else torch.bernoulli(probs)

                assert not action_entropy, "Action entropy penalty not implemented yet for MultiDiscrete Action spaces."
                if action_logprob:
                    #Exploit that log(p1 * ... * pn) = log(p1) + ... + log(pn)
                    x["action_logprob"] = torch.sum(-torch.nn.functional.binary_cross_entropy_with_logits(x["mean"], action.float(), reduction='none'),dim=1)

            else:
                probs = Categorical(logits=x["mean"])
                if action is None:
                    action = x["mean"].argmax(dim=-1) if deterministic else probs.sample()

                if action_entropy:
                    x["action_entropy"] = probs.entropy()
                if action_logprob:
                    x["action_logprob"] = probs.log_prob(action)

        x["action"] = action
        if get_value:
            x.update({"value": self.v_net(obs)["y"]})

        return x


    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        raise NotImplementedError()

    def save_additionals(self, model_path: Path, absolute_path: Path):
        raise NotImplementedError()

    def get_action_and_value(self, obs, deterministic: bool = False):
        obs = torch.Tensor(obs, device=self.device).unsqueeze(0)
        y = self.get_action_and_value_net(obs, get_value=True, deterministic=deterministic)
        return {"action": y["action"].detach().squeeze(0).numpy(), "value": y["value"].detach().squeeze(0).numpy().item()}

    def get_value(self, obs):
        assert not isinstance(obs,torch.Tensor)
        obs = torch.Tensor(obs, device=self.device).unsqueeze(0)
        return self.v_net(obs)["y"][0].detach().numpy()

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        obs = torch.Tensor(obs, device=self.device).unsqueeze(0)
        action = self.get_action_and_value_net(obs, get_value=False, deterministic=deterministic)["action"].squeeze(0).detach().numpy()
        return action.item() if (self.cfg.space_description.is_discrete_action() and not self.cfg.space_description.is_multi_discrete()) else action

    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env:gym.core.Env = None):

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [lambda: env_maker() for _ in range(self.cfg.num_envs)]
        )

        discrete = self.cfg.space_description.is_discrete_action()

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs) + envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.cfg.num_steps, self.cfg.num_envs) + envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        rewards = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        dones = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        values = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)

        # Start the game
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.cfg.num_envs).to(self.device)

        abort_training = False
        while not abort_training:

            # Annealing the rate if instructed to do so.
            if self.cfg.anneal_lr:
                frac = 1.0 - tracker.get_progress()
                lrnow = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.cfg.num_steps):
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    vals = self.get_action_and_value_net(next_obs, action_logprob=True)
                    action, logprob, value = vals["action"], vals["action_logprob"], vals["value"]
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Check for termination condition
                abort_training = abort_training or tracker.add_unit(tm.STEPS, self.cfg.num_envs)

                #Execute the game and log data
                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)
                if norm_sync_env:
                    sync_normalization_state(envs.envs[0], norm_sync_env)


            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.v_net(next_obs)["y"].reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.cfg.num_steps)):
                    if t == self.cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            for epoch in range(self.cfg.update_epochs):
                np.random.shuffle(b_inds)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    x = self.get_action_and_value_net(b_obs[mb_inds], b_actions.long()[mb_inds] if discrete else b_actions[mb_inds], action_logprob=True,action_entropy=self.cfg.action_ent_coef != 0)

                    newlogprob = x["action_logprob"]
                    action_entropy = x["action_entropy"] if self.cfg.action_ent_coef != 0 else None
                    newvalue = x["value"]
                    fallback_weights = x["weights"] if "weights" in x else None

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    if self.cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg.clip_coef,
                            self.cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    #Entropy losses
                    action_entropy_loss = action_entropy.mean() if self.cfg.action_ent_coef != 0 else 0

                    #Fallback loss
                    fallback_loss = fallback_weights.mean() * self.cfg.fallback_coef if fallback_weights is not None else 0

                    loss = pg_loss - self.cfg.action_ent_coef * action_entropy_loss + v_loss * self.cfg.vf_coef + fallback_loss


                    #Perform gradient descent
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor_net.parameters()) + list(self.v_net.parameters()), self.cfg.max_grad_norm)
                    self.optimizer.step()


                #Register used samples
                abort_training = abort_training or tracker.add_unit(tm.SAMPLES, self.batch_size)

            #Register updates
            abort_training = abort_training or tracker.add_unit(tm.EPOCHS, self.cfg.update_epochs)

        envs.close()
