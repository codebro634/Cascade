from os.path import exists
from pathlib import Path
from typing import Callable, Tuple
import gymnasium as gym

import Agents.Cascade
from Agents.Agent import Agent
from Agents.PPO import PPO
from Analysis.AgentConfigs.General import gamma, obs_clip, rew_clip
from Analysis.Experiment import Experiment
from Analysis.AgentConfigs import VanillaPPO, Cascade, VanillaDDPG, VanillaSAC
from Architectures.ActorNet import CascadeActor
from Architectures.Elementary import abs_difference
from Environments import EnvSpaceDescription
from Environments.DiscreteWrapper import DiscretizeWrapper
from Environments.Utils import wrap_env, get_normalization_state
from Analysis.Parser import parse_bool

DEFAULT_EVAL_INTERVAL = 10000

def setup_env(env_param: str, recursive_call = False) -> Callable:
    """
        Converts a string into an environment-maker.

        Snytax: env_param starts with the name of the environment. If additional arguments
        are to be passed they are separated by ; and have the form argument:value.
        Example: Ant-v4; norm:False; clip:True

        Before wrappers are applied, the environment can be wrapped by an agent-specific wrapper
        by adding agent_config to the arguments. (e.g. agent_config:VanillaPPO)

        If the environment's argument include another env-description, then the arguments
        are separated by , and have the form argument=value.
        Example:  ChooserEnv; env:Walker2d-v4, norm=False; agents:[a,b,c,d]; norm:False; clip:True; agent_config:DiscreteChooser

        clip and norm are set to True per default.

        Parameters for all environment: norm,clip,agent_config
    """

    #Parse parameters and env name
    env_param = env_param.strip()
    split = [s.strip() for s in env_param.split("," if recursive_call else ";" )]
    env_name = split[0]
    env_params = split[1:]
    delim = "=" if recursive_call else ":"

    params_dict = {entry.split(delim)[0].strip(): entry.split(delim)[1].strip() for entry in env_params}

    #Assuming env can be created with gym.make if name is none of these special cases
    if env_name.startswith("discrete"):
        env_maker = lambda: DiscretizeWrapper(gym.make(env_name[len("discrete"):]))
    else:
        env_maker = lambda: gym.make(env_name)

    #Wrap env
    env_maker_stats = env_maker if recursive_call else lambda: wrap_env(env_maker(),record_stats=True)
    assert "agent_config" in params_dict
    env_maker_agent = globals()[params_dict["agent_config"]].env_wrapper(env_maker_stats)
    normalize = (not "norm" in params_dict) or parse_bool(params_dict["norm"])
    clip = (not "clip" in params_dict) or parse_bool(params_dict["clip"])
    env_maker_normed = lambda: wrap_env(env_maker_agent(), norm_obs=normalize, gamma=gamma if normalize else None,clip_obs=obs_clip if clip else None, clip_actions=clip, clip_rew=rew_clip if clip else None)

    return env_maker_normed

#Partially written by ChatGPT
def get_experiment_params_from_file(exp_group: str, exp_name: str) -> dict:

    """
        Returns the experiment params from the file Experiments/exp_group with name exp_name as a dict (if exp_name is an integer, then the exp-name-th experiment is returned)
        The syntax of an experiment is as follows:
            exp_name
            training_steps
            environment_description
            agent_configuration
            agent_params [Optional]
        Experiments are separated by blank lines.
    """

    with open(Path('Experiments').joinpath(exp_group) if exists(Path('Experiments')) else (Path('../Experiments').joinpath(exp_group) if
                                        exists(Path('../Experiments')) else Path('Cascade/Experiments').joinpath(exp_group)), 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if not line.strip().startswith("#")]

    variables = ['name', 'steps', 'env', 'agent_cfg', 'agent_params', 'group']

    exp_no = 1
    for i,line in enumerate(lines):
        if line == str(exp_name) or (exp_name.isdigit() and exp_no == int(exp_name) and line):
            if i + 4 < len(lines):
                values = [lines[i], int(lines[i+1]), lines[i+2], globals()[lines[i+3]], lines[i+4], exp_group]
                break
            else:
                raise ValueError(f"Error: Experiment {exp_name} has insufficient number of params.")
        if i >= len(lines) - 1:
            raise ValueError(f"Error: Experiment {exp_name} not found.")
        if line and not lines[i+1]:
            exp_no += 1

    values[variables.index("agent_params")] = { entry.split(":")[0].strip(): entry.split(":")[1].strip() for entry in values[variables.index("agent_params")].split(";")} if values[variables.index("agent_params")] else {}
    return dict(zip(variables, values))

#Sets up and returns an experiment with the given parameters
def setup_experiment(exp_group: str, exp_name: str, env_descr: str, agent_config: str, agent_params: str, exp_identifier: str = None) -> Experiment:
    if "agent_config:" not in env_descr:
        env_descr += f";agent_config:{agent_config}"
    tmp_env = setup_env(env_descr)()
    agent_params = {entry.split(":")[0].strip(): entry.split(":")[1].strip() for entry in agent_params.split(";")} if agent_params is not None else {}
    agent = globals()[agent_config].agent(EnvSpaceDescription.get_descr(tmp_env), **agent_params)
    tmp_env.close()
    experiment = Experiment(group=exp_group, name=exp_name, agent=agent, env_descr=env_descr,eval_func=experiment_eval_func,
                            final_eval_func=experiment_final_eval_func, identifier=exp_identifier)
    return experiment

#Sets up and returns an experiment with the parameters read from the experiment exp_name in Experiments/exp_group and the number of step the experiment is supposed to be run with
def setup_experiment_from_file(exp_group: str, exp_name: str, exp_identifier:str = None) -> Tuple[Experiment,int]:
    exp_params = get_experiment_params_from_file(exp_group, exp_name)

    agent_config = exp_params["agent_cfg"]
    env_setup = exp_params["env"]
    env_descr = f"{env_setup};agent_config:{str(agent_config.__name__).split('.')[-1]}"

    tmp_env = setup_env(env_descr)()
    agent = agent_config.agent(EnvSpaceDescription.get_descr(tmp_env), **exp_params["agent_params"])
    tmp_env.close()

    experiment = Experiment(name=exp_params["name"], group=exp_params["group"], agent=agent, env_descr=env_descr, eval_func=experiment_eval_func,
                            final_eval_func=experiment_final_eval_func,identifier=exp_identifier)
    return experiment, exp_params["steps"]

#Evaluation function for the experiment
def experiment_eval_func(initial_agent: Agent, agent: Agent, env: gym.core.Env) -> dict:
    eval_results = {}

    #Track weight differences to initial agent in L1 norm
    if isinstance(agent, PPO) and isinstance(agent.actor_net, CascadeActor):
        eval_results.update({"actor weights diff":abs_difference(agent.actor_net, initial_agent.actor_net)})

    #If normalizing, always add normalization stats
    if get_normalization_state(env):
        stats = get_normalization_state(env)
        relevant_stats = {}
        if "obs mean" in stats:
            relevant_stats["obs mean"] = stats["obs mean"].sum() / len(stats["obs mean"])
            relevant_stats["obs var"] = stats["obs var"].sum() / len(stats["obs var"])
        if "rew mean" in stats:
            relevant_stats["rew mean"] = stats["rew mean"]
            relevant_stats["rew var"] = stats["rew var"]
        eval_results.update(relevant_stats)

    eval_args = {"measure_return" :True} #Arguments with which to call evaluate_agent
    ignore = [] #Evaluation metrics which to discard

    if isinstance(agent,PPO):
        if isinstance(env.action_space,gym.spaces.Discrete):
            eval_args.update({"track_action_freqs": True})
            ignore.append("distribution")

    if isinstance(agent,Agents.Cascade.Cascade):
        eval_args.update({"measure_fallback_stats":True, "cascade_net": agent.top.actor_net})


    from Analysis.Evaluation import evaluate_agent #Avoid circular import
    eval_results.update(evaluate_agent(agent,env, **eval_args))

    for ignore_metric in ignore:
        del eval_results[ignore_metric]

    return eval_results

#Evaluation function for final models of the experiment. Should be used for computationally heavy measurements
def experiment_final_eval_func(initial_agent: Agent, agent: Agent, eval_env: gym.core.Env, num_runs: int = 10) -> dict:
    evals = {}

    #Measure individual base nets' performances
    if isinstance(agent,Agents.Cascade.Cascade):
        from Analysis.Evaluation import measure_base_nets
        evals.update(measure_base_nets(agent, eval_env, num_runs=num_runs))

    return evals



