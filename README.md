
# Purpose
Code for the Paper: Cascade - a Sequential Ensemble Method for Continuous Control Tasks

# Acknowledgements
The code has been developed with the assistance of GitHub Copilot.

# Run Experiments

To run experiments, use the following commands:

## Experiment from the console:
```
python mainExperiment.py --exp_group <experiment_group> --exp_name <experiment_name> --steps <total_number_of_env_steps> --agent_config <agent_configuration> --env_descr <environment_description> [--agent_params <agent_parameters>]
```

- `experiment_group`: Specify the experiment group.
- `experiment_name`: Specify the name of the experiment.
- `total_number_of_env_steps`: Specify the total number of environment steps.
- `agent_configuration`: Selects the agent for the experiment. Must be one of the classes in Analysis/AgentConfigs.
- `environment_description`: Specify the environment to run the experiment on. The syntax for the description is `env_name;parameter1:value1;parameter2:value2;...`. If any parameter requires another `environment_description`, use `,` and `=` as delimiters. All environments share the following parameters: `norm` (True or False, default is True), `clip` (True or False, default is True), and `agent_config` (Agent configuration from Analysis/AgentConfigs, which wraps the environment in the `agent_config`'s wrapper before the potential normalization wrapper). If the prefix `discrete` is added to the `env_name` (i.e. discreteAnt-v4), then the action space is discretized if it was continuous.  

- `agent_params` (optional): Use this parameter to override the default parameters used in `agent_config`. The syntax is `parameter1:value1;parameter2:value2;...`.

## Experiment from file:
```
python mainExperiment.py --exp_group <experiment_group> --exp_name <experiment_name> --load
```

The experiment `experiment_name` must be located in the file `Experiments/exp_group` and have the format:

1. line: `experiment_name`
2. line: `total_number_of_env_steps`
3. line: `environment_description`
4. line: `agent_configuration`
5. line: `agent_params` (optional, leave blank if not needed)

`experiment_group` is automatically determined by the name of the file the experiment is in.

## Optional Parameters:
- `wandb_logging`: Logs the experiment with Weights and Biases. Value must be the name of the wandb project. `experiment_group` is used as the group and `experiment_name` as the name of Wandb run.
- `save_latest`: Saves the latest model to `nobackup/Models/exp_group/exp_name/run<number_of_run>_latest_<exp_identifier>`.
- `save_best`: Saves the model that performed best to `nobackup/Models/exp_group/exp_name/run<number_of_run>_best_<exp_identifier>`.
- `num_runs`: Number of times the experiment should be repeated.
- `eval_interval`: Number of environment steps between each model save and data log. The default value is `10,000`.
- `show_progress`: Displays a progress bar in the console.
- `exp_identifier`: The identifier used the differentiate between experiments of the same name and group. Default is a 5 letter random string

# Load and evaluate saved models

The folder in which all models are saved is defined by `MODEL_SAVE_DIR` in `Agents/Agent`. The default value is `nobackup/Models`

An agent saved in `MODEL_SAVE_DIR/path` can be loaded and evaluated on the environment it has been trained on by executing the following code:
```
  from pathlib import Path
  from Agents.Agent import Agent
  from Analysis.Evaluation import evaluate_agent
  from Environments.Utils import load_env

  agent = Agent.load(Path(path))
  env = load_env(Path(path))
  evaluate_agent(agent,env, measure_return=True, verbose=True)
```
# Run experiments from the paper

The experiments used for the thesis are located in the folder `Experiments` and can be loaded from there. Alternatively, they can be manually configured. The following lists the necessary commands.

`<env_name>` is left blank. Possible values are `Ant-v4`, `Walker2d-v4`, `Hopper-v4`, etc. Also, every parameter that denotes a training duration is left blank. Just insert the training durations used for the experiment
you are trying to replicate. For all other parameters, a concrete value used for the respective experiment suite was inserted to exemplify the usage.


## Standard Cascade

```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> 
```

## Baseline experiments 

Used in Fig. 2 and 9

``` 
python mainExperiment.py --wandb_logging --exp_group BaselinePPO --exp_name <name>  --steps <steps>  --agent_config VanillaPPO --env_descr <env_name>
```

To run the baselines for SAC and DDPG, replace `BaselinePPO` and `VanillaPPO` with `BaselineSAC` and `VanillaSAC` or `BaselineDDPG` and `VanillaDDPG`

## Cascade single base agent with cylical learning rate

Used in Fig. 3
```
python mainExperiment.py --wandb_logging --exp_group BaselinePPO --exp_name <name>  --steps <steps>  --agent_config Cascade --agent_params sequential:False;stacks:1;base_steps:<base_steps> --env_descr <env_name>
```

## Keeping base nets frozen

Used in Fig. 4 (with normalization)
```
 python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name> --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params train_only_top:True 
```  

Used in Fig. 13 (without normalization)
```
 python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name> --steps <steps>  --agent_config Cascade --env_descr <env_name>;norm:False --agent_params train_only_top:True 
```  

## Starting with final ensemble

Used in Fig. 5
```  
Not sequential: python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name TestCascadeNotSequential  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params sequential:False;stacks:6 
```

## Cascade with tiny base nets

Used in Fig. 8
```
python mainExperiment.py --wandb_logging --exp_group CascadeTiny --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params actor_hidden:(1,);base_steps:<base_steps>
```

## Different training durations per iteration:

Used in Fig. 10
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params base_steps:<base_steps>
```

## Different fallback initializations:

Used in Fig. 11
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params fb_init:0.9 
```

All cascade experiments, including this one, automatically track the fallback weights of all base nets.

## Cascade with no cyclical learning rate

Used in Fig. 12
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params cyclical_lr:False 
```

## Cascade/PPO on discrete action spaces

Used in Fig. 13

Discrete PPO:
``` 
python mainExperiment.py --wandb_logging --exp_group BaselinePPO --exp_name <name>  --steps <steps>  --agent_config VanillaPPO --env_descr discrete<env_name> --agent_params continuous:False
```

Discrete Cascade:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr discrete<env_name> --agent_params continuous:False
```

## Comparison to naive approach
Described in Method section where a surrogate environment is used.
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config CascadeNaive --env_descr <env_name> 
```

## Cascade with different base training algorithms
Used in Fig. X and Fig. Y

Cascade using SAC or DDPG:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name>;norm:False --agent_params reset_rb:True;actor_hidden:(64,64);stack_critic:True;low_critic_std:True;alg_name:<DDPG or SAC>
```

