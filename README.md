# Disclaimer
Next to the code for all experiments, this repo also contains code that is not directly related to the Cascade paper. However, after review this will be shortened to contain only the paper related parts.

# Purpose
Code for Master's thesis: Combining Weak Reinforcement Learners for Enhanced Performance.

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
- `environment_description`: Specify the environment to run the experiment on. The syntax for the description is `env_name;parameter1:value1;parameter2:value2;...`. If any parameter requires another `environment_description`, use `,` and `=` as delimiters. For example: `ChooserEnv; env=Walker2d-v4, norm=False; agents:[a,b,c,d]; norm:True; clip:True; agent_config:DiscreteChooser`. All environments share the following parameters: `norm` (True or False), `clip` (True or False), and `agent_config` (Agent configuration from Analysis/AgentConfigs, which wraps the environment in the `agent_config`'s wrapper before the potential normalization wrapper).

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
- `wandb_logging`: Logs the experiment with Weights and Biases. `experiment_group` is used as the group and `experiment_name` as the name of Wandb run.
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
# Run experiments from the thesis

The experiments used for the thesis are located in the folder `Experiments` and can be loaded from there. Alternatively, they can be manually configured. The following lists the necessary commands.

`<env_name>` is left blank. Possible values are `Ant-v4`, `Walker2d-v4`, `Hopper-v4`, etc. Also, every parameter that denotes a training duration is left blank. Just insert the training durations used for the experiment
you are trying to replicate. For all other parameters, a concrete value used for the respective experiment suite was inserted to exemplify the usage.


## Baseline experiments 

``` 
python mainExperiment.py --wandb_logging --exp_group Baseline --exp_name <name>  --steps <steps>  --agent_config VanillaPPO --env_descr <env_name>
```

## CombN experiments

### Standard CombN:

```
python mainExperiment.py --wandb_logging --exp_group CombN --exp_name <name>  --steps <steps>  --agent_config CombN --env_descr <env_name>
```

### Varying hidden sizes:  

```
python mainExperiment.py --wandb_logging --exp_group CombN --exp_name <name>  --steps <steps>  --agent_config CombN --agent_params action_hidden:(16,16);chooser_hidden:(64,) --env_descr <env_name>
```

### Varying number of action nets:  
```
python mainExperiment.py --wandb_logging --exp_group CombN --exp_name <name> --steps <steps>  --agent_config CombN --agent_params n:3 --env_descr <env_name>
```

### Hierarchical CombN

```
python mainExperiment.py --wandb_logging --exp_group CombNHierarchical --exp_name <name>  --steps <steps> --agent_config CombNHierarchical --env_descr <env_name> --agent_params pretrain:False;n:3;cycles:2 
```

### Entropy experiments

Minimizing vertical entropy:
```
python mainExperiment.py --wandb_logging --exp_group CombNEntropy --exp_name <name>  --steps <steps>  --agent_config CombN --env_descr <env_name> --agent_params vent_coef:0.1 
```

Using pure averager for maximal vertical entropy:
```
python mainExperiment.py --wandb_logging --exp_group CombNEntropy --exp_name <name> --steps <steps>  --agent_config CombN --env_descr <env_name> --agent_params temperature:inf 
```

Maximizing horizontal entropy:
```
python mainExperiment.py --wandb_logging --exp_group CombNEntropy --exp_name <name>  --steps <steps>  --agent_config CombN --env_descr <env_name> --agent_params hent_coef:-0.1 
```
        
### Using dropout probabilities

```
python mainExperiment.py --wandb_logging --exp_group CombNEntropy --exp_name <name>  --steps <steps>  --agent_config CombN --env_descr <env_name> --agent_params dropout_prob:0.05 
```
   
### Alternating training
      
```
python mainExperiment.py --wandb_logging --exp_group CombNAlternating --exp_name <name>  --steps <steps>  --agent_config CombN --env_descr <env_name> --agent_params chooser_epochs:20;action_epochs:10 
```

Weights differences are automatically tracked and logged.
         

### Pretraining

```
Pretraining python mainExperiment.py --wandb_logging --exp_group CombNHierarchical --exp_name <name>  --steps <steps> --agent_config CombNHierarchical --agent_params n:4;cycles:1;cycle_steps:(<steps_action_nets>,<steps_combination>) --env_descr <env_name> 
```

And if additionally, the action nets' parameters should stay frozen after they are combined, use:

```
python mainExperiment.py --wandb_logging --exp_group CombNHierarchical --exp_name <name> --steps <steps> --agent_config CombNHierarchical --agent_params cycles:1;n:4;freeze_bottom:True;cycle_steps:(<steps_action_nets>,<steps_combination>) --env_descr <env_name>
```   

### CombN comparison with a naive chooser        

First, pretrain single actions nets:
```
python mainExperiment.py --wandb_logging --exp_group Baseline --exp_name <name>  --steps <steps>  --agent_config VanillaPPO --agent_params layer_sizes:(16,16) --env_descr <env_name>
```

Then, train Discrete Chooser on surrogate environment with pretrained agents. Insert the paths of the pretrained action nets:

```
python mainExperiment.py --wandb_logging --exp_group DiscreteChooser --exp_name <name>  --steps <steps>  --agent_config DiscreteChooser --env_descr ChooserEnv;env:<env_name>,norm=False,agent_config=VanillaPPO;agents:[agent_path_1,agent_path_2,agent_path_3,agent_path_4] 
```				
  
        
## Pretraining + hierarchical CombN     

```
python mainExperiment.py --wandb_logging --exp_group CombNHierarchical --exp_name <name>  --steps <steps> --agent_config CombNHierarchical --env_descr <env_name> --agent_params n:3;cycles:2;cycle_steps:(<steps_layer_0>,<steps_layer_1>,<steps_layer_2>) 
```

In this case `cycles=2`, therefore `cycle_steps` must have 3 entries to specify the training duration for each base agent at every layer. Therefore, the total training duration for the bottom layer is `3^2 * steps_layer_0` and for the last layer simply `1 * steps_layer_2`.

## Cascade

### Standard Cascade
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> 
```
        
### Different fallback initializations:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params fb_init:0.9 
```

All cascade experiments, including this one, automatically track the fallback weights of all base nets.


### Different training durations per iteration:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params base_steps:<base_steps>
```

### Keeping base nets frozen
```
 python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name> --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params train_only_top:True 
```

### Cascade with tiny base nets
```
python mainExperiment.py --wandb_logging --exp_group CascadeTiny --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params actor_hidden:(1,);base_steps:<base_steps>
```

### Starting with final ensemble
```  
Not sequential: python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name TestCascadeNotSequential  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params sequential:False;stacks:6 
```

### Cascade with no cyclical learning rate
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params cyclical_lr:False 
```

### Comparison to naive approach
Normed:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config CascadeNaive --env_descr <env_name> 
```
Not normed:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config CascadeNaive --env_descr <env_name>;norm:False 
```

### Comparison to a single base agent

```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name> --steps <steps> --agent_config Cascade --env_descr <env_name> --agent_params sequential:False;stacks:1
```

### Propagation

Value function propagation:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params propagate_value:True 
```
Action function propagation:
```
python mainExperiment.py --wandb_logging --exp_group Cascade --exp_name <name>  --steps <steps>  --agent_config Cascade --env_descr <env_name> --agent_params propagate_action:True 
```

# Miscellaneous

Comparison with a base agent had it continued training is just standard PPO with the learning rate annealing restarting at `base_steps` steps.
```
python mainExperiment.py --wandb_logging --exp_group Baseline --exp_name <name>  --steps <steps>  --agent_config Cascade --agent_params sequential:False;stacks:1;base_steps:<base_steps> --env_descr <env_name>
```

Standard PPO with smaller networks:
```
python mainExperiment.py --wandb_logging --exp_group Baseline --exp_name <name>  --steps <steps>  --agent_config VanillaPPO --agent_params layer_sizes:(16,) --env_descr <env_name>
```
        
        
        
        

