from Analysis.ExperimentSetup import setup_experiment, setup_experiment_from_file
import argparse

from Analysis.RunTracker import TrackConfig, TrackMetric

parser = argparse.ArgumentParser()

#Experiment params
parser.add_argument('--exp_group', help='The group of the experiment.')
parser.add_argument('--exp_name', help='The experiment to be conducted.')
parser.add_argument('--exp_identifier', help='Identifier of the experiment. Needed when multiple instances of the same experiment have to be saved.')
parser.add_argument('--load', action='store_true', help='If set loads the experiment setup from the file exp_group/exp_name')
parser.add_argument('--steps', type=int,help='Training duration in environment steps. Overwrites the steps loaded from exp_group/exp_name is load is set.')
parser.add_argument('--agent_config', help='Which config is used to build the agent.')
parser.add_argument('--env_descr', help='Environment description.')
parser.add_argument('--agent_params', help='Overwrites the agent params of the experiment with these values.')
parser.add_argument('--num_runs', type=int, default=1, help='Number of times the experiment is repeated.')

#Logging params
parser.add_argument('--wandb_logging', help='Tracks experiment with wandb. The argument expects the wandb project name.')
parser.add_argument('--save_latest', action='store_true', help='Saves the latest agent.')
parser.add_argument('--save_best', action='store_true', help='Saves the best agent.')
parser.add_argument('--show_progress', action='store_true', help='Show progress bar.')
parser.add_argument('--eval_interval', default=10000, type=int, help='Evaluates after every eval_interval steps.')

args = parser.parse_args()

if args.exp_group is None:
    raise ValueError("--exp_group is not set. Please specify the experiment group.")
if args.exp_name is None:
    raise ValueError("--exp_name is not set. Please specify the experiment name.")
if args.agent_config is None and not args.load:
    raise ValueError("--agent_config is not set. Please specify the agent config. Example Values: Cascade,VanillaPPO.")
if args.env_descr is None and not args.load:
    raise ValueError("--env_descr is not set. Please specify the environment description. Example: Ant-v4")
if args.steps is None and not args.load:
    raise ValueError("--steps is not set. Please specify the number of training steps.")

if args.load:
    experiment, steps = setup_experiment_from_file(args.exp_group,args.exp_name, exp_identifier=args.exp_identifier)
else:
    experiment = setup_experiment(args.exp_group, args.exp_name, args.env_descr, args.agent_config, args.agent_params, exp_identifier=args.exp_identifier)

if args.steps:
    steps = args.steps

track_cfg = TrackConfig(TrackMetric.STEPS, steps, args.eval_interval)
experiment.run(track_cfg=track_cfg,runs=args.num_runs, show_progress=args.show_progress, wandb_logging=args.wandb_logging, save_latest=args.save_latest, save_best=args.save_best)


