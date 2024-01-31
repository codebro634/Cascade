from pathlib import Path
from typing import Callable, Type
from dataclasses import dataclass
from Agents.Agent import Agent, AgentConfig
from Agents.PPO import PPOConfig, PPO
from Analysis.RunTracker import RunTracker, TrackMetric, TrackConfig
import gymnasium as gym

from Environments.Utils import sync_normalization_state


@dataclass
class SequentialEnsembleConfig(AgentConfig):

    ensemble_size: int = 3 #Maximum size of the ensemble. At capacity when a new base-agent is added, the oldest one is removed
    base_steps: int = 1000

    def validate(self):
       pass

"""
    Abstract class for a sequential ensemble in which the training of an ensemble-member
    only depends on the previous ensemble members. I.e. the ensemble can be trained sequentially.
"""

class SequentialEnsemble(Agent):

    def __init__(self, cfg: SequentialEnsembleConfig, add_first: bool = False):
        super().__init__(cfg)
        self.add_first = add_first #If false, a base-agent is only added to the ensemble once it has finished training.
        self.base_agents = list() #All agents that together make up the ensemble
        self.current_base_agent = None
        self.total = 0
        self.cfg.validate()

    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, cfg: AgentConfig) -> "Agent":
        raise NotImplementedError()

    def save_additionals(self, model_path: Path, absolute_path: Path):
        for i,base in enumerate(self.base_agents + ([self.current_base_agent] if self.current_base_agent else [])):
            base.save(model_path.joinpath(f"base{i + 1}"))

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        raise NotImplementedError()

    def setup_base_training_env(self, env_maker: Callable[[],gym.core.Env]) -> Callable[[],gym.core.Env]:
        """
            Return the training environment, the latest ensemble-agent is supposed to train on
        """
        raise NotImplementedError()

    def setup_base_agent(self) -> Agent:
        """
            Return an agent that will be trained on the environment returned by setup_base_training_env.
            Once training finished or if 'add_first' is set, will be added to the ensemble.
        """
        raise NotImplementedError()

    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env: gym.core.Env = None):

        if norm_sync_env is None:
            norm_sync_env = env_maker()

        synced_env_maker = lambda: sync_normalization_state(norm_sync_env, env_maker())

        while not tracker.is_done():

            #If ensemble is at capacity, remove the oldest ensemble-member
            while len(self.base_agents) >= self.cfg.ensemble_size:
                del self.base_agents[0]

            #Setup base-agent
            self.current_base_agent = self.setup_base_agent()
            self.total += 1
            base_training_env = self.setup_base_training_env(synced_env_maker)

            if self.add_first:
                self.base_agents.append(self.current_base_agent)

            #Train base-agent
            wt = RunTracker(cfg=TrackConfig(metric=TrackMetric.STEPS, total=self.cfg.base_steps, eval_interval=0), eval_func=None, show_progress=False, nested=tracker, nested_progress=False)
            self.current_base_agent.train(base_training_env, wt, norm_sync_env=norm_sync_env)

            # Add latest base-agent to the ensemble
            if not self.add_first:
                self.base_agents.append(self.current_base_agent)

