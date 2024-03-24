import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable
import random
import string
import numpy as np
import gymnasium as gym
import wandb
import Agents
import Analysis
from Agents.Agent import Agent
from Analysis.RunTracker import RunTracker, TrackConfig
from Environments.Utils import save_normalization_state


class Experiment:

    def __init__(self, name: str, group: str, agent: Callable, env_descr: str, eval_func: Callable[[Agent,Agent,gym.core.Env],dict],
                 final_eval_func: Callable[[Agent,Agent,gym.core.Env],dict] =None, identifier: str = None):

        #If experiment's run saves models, then these are saved into the folder group/name/
        #Also, if wandb logging is used, the assigned wandb-group is 'group' and the run's name is 'name'
        self.name = name
        self.group = group
        self.agent = agent
        self.env_descr =env_descr
        self.env = Analysis.ExperimentSetup.setup_env(env_descr)
        self.eval_func = eval_func #Asserts that key 'average return' is always contained in the resulting dict
        self.final_eval_func = final_eval_func #Is only called once at the end of training. Its values are logged if wandb logging is set. Should return computationally expensive metrics, otherwise they can just be part of eval_func
        self.identifier = ''.join(random.choice(string.digits+string.ascii_letters ) for _ in range(5)) if identifier is None else identifier #Used to differentiate multiple instances of the same Experiment.

    @staticmethod
    def log_wandb_data(eval_data: dict):
        final_log = {}
        for key, value in eval_data.items():
            if hasattr(value,"__len__"):
                for i, entry in enumerate(value):
                    final_log[f"{key}{i+1}" if i > 0 else key] = entry
            else:
               final_log[key] = value

        wandb.log(final_log)

    #Uses wandb_logging if wandb_logging is not None. If set, it must contain the project name.
    def run(self, track_cfg: TrackConfig, runs=1, wandb_logging: str = None, show_progress: bool = True, save_latest: bool = True, save_best: bool = False):

        for run in range(1,runs+1):
            if show_progress:
                print(f"Starting run {run} of experiment {self.name}.")

            #Setup Agent and env. This is needed so they can be accessed from the inner class RunLogger
            agent = self.agent()
            initial_agent = deepcopy(agent)
            eval_env = self.env()
            name = self.name
            group = self.group
            identifier = self.identifier
            env_descr = self.env_descr
            eval_func = self.eval_func

            #print(f"Agent has {agent.model_size()} parameters")

            #Setup wandb
            if wandb_logging:
                cfg = agent.get_setup_descr()
                for key, value in cfg.items():
                    cfg[key] = str(value)

                wandb.init(
                    project=wandb_logging,
                    name = self.name,
                    group = self.group,
                    config= {**cfg, "env_description": self.env_descr, "identifier": identifier}
                )

            #Run and log experiment
            class RunLogger:

                def __init__(self):
                    self.highest_return = -np.inf

                def is_highest_return(self, return_):
                    if return_ > self.highest_return:
                        self.highest_return = return_
                        return True
                    return False

                def log(self):
                    eval_result = eval_func(initial_agent, agent, eval_env)

                    assert "average return" in eval_result
                    avg_return = eval_result["average return"]

                    print(f"Experiment.py: {eval_result['average return'][0]}")
                    if wandb_logging:
                        Experiment.log_wandb_data(eval_result)

                    """
                        Save agent
                    """
                    if save_latest or save_best:
                        #Setup paths
                        group_path = Path(group)
                        Agents.Agent.Agent.to_absolute_path(group_path).mkdir(parents=True,exist_ok=True)
                        model_path = group_path.joinpath(Path(f"{name}"))
                        Agents.Agent.Agent.to_absolute_path(model_path).mkdir(exist_ok=True)

                        #Save latest model
                        if save_latest:
                            latest_path = model_path.joinpath(f"run{run}_latest_{identifier}")
                            Agents.Agent.Agent.to_absolute_path(latest_path).mkdir(exist_ok=True)
                            agent.save(latest_path)
                            save_normalization_state(eval_env, Agents.Agent.Agent.to_absolute_path(latest_path).joinpath("norm_state.pkl"))
                            with open(Agents.Agent.Agent.to_absolute_path(latest_path).joinpath("env_descr.pkl"), "wb") as f:
                                pickle.dump(env_descr, f)
                        #If necessary, save best model
                        if save_best and self.is_highest_return(avg_return):
                            best_path = model_path.joinpath(f"run{run}_best_{identifier}")
                            Agents.Agent.Agent.to_absolute_path(best_path).mkdir(exist_ok=True)
                            agent.save(best_path)
                            save_normalization_state(eval_env, Agents.Agent.Agent.to_absolute_path(best_path).joinpath( "norm_state.pkl"))
                            with open(Agents.Agent.Agent.to_absolute_path(best_path).joinpath("env_descr.pkl"), "wb") as f:
                                pickle.dump(env_descr, f)


            rl = RunLogger()
            run_tracker = RunTracker(track_cfg, rl.log, show_progress=show_progress)
            agent.train(env_maker = self.env, tracker =run_tracker, norm_sync_env = eval_env)

            #Evaluate the final models
            if wandb_logging and self.final_eval_func is not None:
                Experiment.log_wandb_data(self.final_eval_func(initial_agent,agent,eval_env))

            #Close env and wandb
            eval_env.close()
            if wandb_logging:
                wandb.finish()


