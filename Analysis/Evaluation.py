import os
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.distributions import Categorical

import Agents
from Agents.Agent import Agent
import gymnasium as gym
from numpy import prod

from Agents.PPO import PPO
from Analysis.ExperimentSetup import experiment_final_eval_func
from Architectures.CascadeAC import ActorCriticCascade
from Environments import EnvSpaceDescription
from Environments.Utils import get_wrapper, load_env, get_normalization_state, \
    load_normalization_state


def evaluate_agent(agent: Agent,env: gym.core.Env, num_runs:int = 10, horizon_length:int = None, measure_return: bool = False, track_action_freqs: bool = False,
                   measure_fallback_stats: bool = False, cascade_net:ActorCriticCascade = None, measure_expected_state: bool = False,
                   verbose: bool = False):
    """
    :param agent:  Agent to be evaluted
    :param env: Environment the agent is evaluted on
    :param num_runs: Number of evaluation runs
    :param horizon_length: Maximum step number per runs
    :param measure_return: If set, returns the average return over all runs
    :param track_action_freqs: If set, returns the number each action has been chosen. Works only for discrete action-spaces
    :param measure_fallback_stats:  If set, returns the average fallback-weight per base-net in cascade_net. Only works if using a Cascade-net
    :param cascade_net: Mandatory if measure_fallback_stats is set. The Cascade-net of which the fallback-weights are to be measured
    :param measure_expected_state: If set, returns the average observation of all encountered observations
    """

    assert get_wrapper(env,RecordEpisodeStatistics) is not None, "Environment must record episode statistics."
    returns_sum = 0
    total_obs = 0

    if measure_expected_state:
        states_sum = np.zeros(shape=(EnvSpaceDescription.get_descr(env).flattened_input_size(),))

    if track_action_freqs:
        freqs = np.zeros(shape=(EnvSpaceDescription.get_descr(env).flattened_act_size()))

    if measure_fallback_stats:
        probs_sum = np.zeros(shape=(len(cascade_net.cascade)-1,))
        product_probs_sum = 0

    original_norm_state = get_normalization_state(env)

    for _ in range(num_runs):
        obs,_ = env.reset()
        if original_norm_state:
            load_normalization_state(env, deepcopy(original_norm_state))

        done = False
        steps = 0
        while not done and (not horizon_length or steps < horizon_length):
            total_obs += 1

            #Measure expected state
            if measure_expected_state:
                states_sum += obs

            #Measure fallback stats
            if measure_fallback_stats:
                probs = cascade_net.get_fallbacks(obs)
                probs_sum += probs
                product_probs_sum += prod(probs)

            #Get action
            action = agent.get_action(obs, eval_mode = True)

            #Track action frequency
            if track_action_freqs:
                freqs[action] += 1

            #Env step
            obs, _, term, trunc , final_info = env.step(action)
            done = term or trunc
            steps +=1

        if "episode" in final_info.keys():
            returns_sum += final_info['episode']['r']

    #Restore original normalization state
    if original_norm_state:
        load_normalization_state(env, original_norm_state)

    measurements = dict()
    #Add optional measurements
    if measure_return:
        measurements.update({"average return": returns_sum / num_runs})
    if measure_expected_state:
        measurements.update({"expected state": states_sum / total_obs})
    if track_action_freqs:
        measurements.update({"distribution": freqs / sum(freqs)})
        measurements.update({"max prob": max((freqs / sum(freqs)))})
    if measure_fallback_stats:
        measurements.update({f"fallbacks" : probs_sum / total_obs })
        measurements.update({f"fallbacks prod": product_probs_sum / total_obs})

    if verbose:
        print("--------- Statistics --------- ")
        for key, value in measurements.items():
            print(f"{key}: {value}")

    return measurements



# All paths in the following are relative to the directory where the project's agent-models are saved.
def get_num_runs(run_group: Path):
    return len(list(os.walk(Agent.to_absolute_path(run_group)))[0][1])


def get_nth_run(run_group: Path, num: int):
    relative_path = Path(run_group)
    run_path = Agent.to_absolute_path(relative_path)
    latest_path = list(os.walk(run_path))[0][1][num]
    return relative_path.joinpath(latest_path)


def iterate_runs(run_group: Path):
    for i in range(get_num_runs(run_group)):
        yield get_nth_run(run_group, i)


def evaluate_run_group(run_group: Path, eval_func_args: Callable[[Agent], dict], ignore: list[str] = list(), num_runs: int = 10, range: tuple[int,int] =None,
                       verbose=False) -> dict:
    """
        Evaluates all models in a given path by calling evaluate_agent on all of them and averaging the results.

        eval_func_args: Returns the optional parameters evaluate_agent is supposed to be called with dependent on the current agent.
        ignore: Discards all evaluation results with a name in ignore
    """

    results_sum = dict()
    for i,run in enumerate(iterate_runs(run_group)):
        if range is not None and (i < range[0] or i > range[1]):
            continue
        agent = Agent.load(run)
        env = load_env(run)
        eval_results = evaluate_agent(agent, env, num_runs=num_runs, **(eval_func_args(agent)))
        for metric, value in eval_results.items():
            if metric in ignore:
                continue
            if verbose:
                print(f"Run: {run.name}, Metric: {metric}, Value: {value}")
            results_sum[metric] = results_sum[metric] + value if metric in results_sum else value
        env.close()

    for metric, value in results_sum.items():
        results_sum[metric] = value / (get_num_runs(run_group) if range is None else range[1] - range[0] + 1)
        if verbose:
            print(f"Average {metric}: {results_sum[metric]}")

    return results_sum


def measure_base_nets(agent: Agent, env: gym.core.Env, num_runs: int = 10) -> dict:
    """Loads agents and environment from file then measures performances of an agents base nets. Assume the agent is Cascade

        The instances environments generated by calling env are not closed!
    """

    # Check if agent is either Cascade
    assert isinstance(agent,Agents.Cascade.Cascade)

    is_casc = isinstance(agent,Agents.Cascade.Cascade)

    n_base = len(agent.top.net.cascade)
    current_return = np.zeros(shape=(n_base,))
    for i in range(n_base):
        agent.top.net.regress_to_one_base(i)
        current_return[i] = evaluate_agent(agent, env, num_runs=num_runs, measure_return=True)["average return"]

    return {"average base returns": current_return.sum() / len(current_return), "max base return": current_return.max()}

def load_and_evaluate_agent_from_file(path: Path, num_runs=10):
    """Loads an Agent and environment from path and prints the average return of this agent."""

    agent = Agent.load(path)
    env = load_env(path)
    avg_return = evaluate_agent(agent, env, measure_return=True, num_runs=num_runs)["average return"]
    print(f"Average return of run {path.name}: {avg_return} of {num_runs} runs.")


"""
    Methods for examining and measuring metrics of final models. Should only be needed when wandb didn't log to required data.
"""

#Measures the average fallback weight of each base-net and the average product of all fallback weights for all models in run_group (This assumes that the models in run_group are Cascade models)
def measure_fallback_stats(run_group: Path, num_runs:int= 10):
    print(f"Fallback probabilities of group: {run_group}")
    args_dict = lambda agent: {"measure_fallback_stats": True, "measure_return": True, "cascade_net": agent.top.net}
    evaluate_run_group(run_group,args_dict,num_runs=num_runs,verbose=True)

#Measures the average performance (in terms of average return) of all models in run_group
def measure_performance(run_group: Path, num_runs:int = 10, max_of_groups_of: int = 1):
    print(f"Returns of group {run_group}:")
    args_dict = lambda agent: {"measure_return": True}
    n = get_num_runs(run_group)
    if max_of_groups_of > 1:
        assert n % max_of_groups_of == 0
        group_maxima_sum = {"average return": 0}
        for i in range(0,n,max_of_groups_of):
            data = [evaluate_run_group(run_group, args_dict, num_runs=num_runs, ignore=["distribution"], verbose=False, range=(i+j,i+j))["average return"] for j in range(max_of_groups_of)]
            group_maxima_sum["average return"] += max(data)
            print(f"Group {i}-{i+max_of_groups_of} maximum: {max(data)}")
        group_maxima_sum["average return"] /= n/max_of_groups_of
        print(f"Group maximum average: {group_maxima_sum}")
    else:
        evaluate_run_group(run_group, args_dict, num_runs=num_runs, verbose=True)

#Measures the average mean and variance of all observation and reward normalizations in run_group
def measure_norm_states(run_group: Path):
    stats_sum = dict()
    for run in iterate_runs(run_group):
        env = load_env(run)
        for key,value in get_normalization_state(env).items():
            stats_sum[key] = stats_sum[key] + value if key in stats_sum else value
        env.close()
    #Average
    for key in stats_sum:
        if key == "obs mean" or key == "obs var":
            stats_sum[key] = stats_sum[key].sum() / len(stats_sum[key])
        stats_sum[key] /= get_num_runs(run_group)

    print(f"Stats of group {run_group}:{stats_sum}")


#Measures the average best performance of all base-nets for all models in run_group (This assumes that these models in run_group are Cascade models)
#If average is set, then additionally the average performance of all base-nets is printed and if not set, then the average performance of each base-net is printed.
def measure_base_performance(run_group: Path, num_runs: int=10):
    print(f"Average base net returns of group {run_group}:")
    return_sum = None
    return_best_sum = 0
    for run in iterate_runs(run_group):
        env = load_env(run)
        eval_data = measure_base_nets(Agent.load(run),env,num_runs=num_runs)
        current_return = eval_data["average base returns"]
        print(f"Average return of base nets of run {run.name}: {current_return} | Max: {eval_data['max base return']}")
        return_sum = current_return if return_sum is None else return_sum + current_return
        return_best_sum += eval_data["max base return"]
        env.close()

    print(f"Average return: :{return_sum / get_num_runs(run_group)} | Average best return: {return_best_sum / get_num_runs(run_group)}")



