from pathlib import Path

from Analysis.ExperimentSetup import setup_experiment, setup_experiment_from_file
import argparse
from Analysis.RunTracker import TrackConfig, TrackMetric

#TODO:
# 1. Tests with removed lr decay
# 2. Test if 0.95 is improvements
# 3. Test smaller val function + test whats happens when valfunction is not newly initialized
# 3. Implement A2C and SAC
# 3. Test network plasticity


from Analysis.Evaluation import evaluate_agent
from Analysis.PlotMaker import make_plot
from Agents.Agent import Agent
from Environments.Utils import load_env

# paths = ["Cascade/Cascade_Ant-v4/run1_latest_D46f8",
# "Cascade/Cascade_Ant-v4/run1_latest_P7HCX",
# "Cascade/Cascade_Ant-v4/run1_latest_tCAsp",
# "Cascade/Cascade_Ant-v4/run1_latest_w779Z",
# "Cascade/Cascade_Walker2d-v4/run1_latest_bu7F3",
# "Cascade/Cascade_Walker2d-v4/run1_latest_hAQJl",
# "Cascade/Cascade_Walker2d-v4/run1_latest_OowaZ",
# "Cascade/Cascade_Walker2d-v4/run1_latest_SjGOi"]
#
# for i,path in enumerate(paths):
#     agent = Agent.load(path)
#     env = load_env(path)
#     y = evaluate_agent(agent, env, get_fallback_distr=True, cascade_net=agent.top.net, num_runs=1)
#
#     make_plot(experiments=[([x[0] for x in y], "Base net 2", "blue"), ([x[4] for x in y], "Base net 6", "red")],
#               save_dir=Path("../nobackup/Plots/fb_distr"+str(i)), legend_position='lower right', x_step_size=1,
#               title="Fallbacks for one Episode", ylabel="Fallback Value", show=True)
#     # for j in range(5):
#     #     make_plot(experiments=[([x[i] for x in y], "Ant", "blue")], title="Fallbacks for one Episode",
#     #               ylabel="Fallback value", show=True)





parser = argparse.ArgumentParser()

#Experiment params
parser.add_argument('--exp_group',help='The group of the experiment.')
parser.add_argument('--exp_name', type=str,  help='The experiment to be conducted.')
parser.add_argument('--exp_identifier', help='Identifier of the experiment. Needed when multiple instances of the same experiment have to be saved.')
parser.add_argument('--load', action='store_true', help='If set loads the experiment setup from the file exp_group/exp_name')
parser.add_argument('--steps', type=int, help='Training duration in environment steps. Overwrites the steps loaded from exp_group/exp_name is load is set.')
parser.add_argument('--agent_config', default='Cascade', help='Which config is used to build the agent.')
parser.add_argument('--env_descr', help='Environment description.')
parser.add_argument('--agent_params', help='Overwrites the agent params of the experiment with these values.')
parser.add_argument('--num_runs', type=int, default=1, help='Number of times the experiment is repeated.')

#Logging params
parser.add_argument('--save_latest', action='store_true', help='Saves the latest agent.')
parser.add_argument('--save_best', action='store_true', help='Saves the best agent.')
parser.add_argument('--show_progress', action='store_true', help='Show progress bar.')
parser.add_argument('--eval_interval', default=10000, type=int, help='Evaluates after every eval_interval steps.')
parser.add_argument('--wandb_logging', help='Tracks experiment with wandb. The argument expects the wandb project name.')

args = parser.parse_args()

if args.exp_group is None:
    raise ValueError("--exp_group is not set. Please specify the experiment group.")
if args.exp_name is None:
    raise ValueError("--exp_name is not set. Please specify the experiment name.")
if args.agent_config is None and not args.load:
    raise ValueError("--agent_config is not set. Please specify the agent config. Example Values: Cascade,VanillaPPO,VanillaDDPG.")
if args.env_descr is None and not args.load:
    raise ValueError("--env_descr is not set. Please specify the environment description. Example: Ant-v4")
if args.steps is None and not args.load:
    raise ValueError("--steps is not set. Please specify the number of training steps.")

if args.load:
    experiment, steps = setup_experiment_from_file(args.exp_group,args.exp_name, exp_identifier=args.exp_identifier)
else:
    experiment = setup_experiment(args.exp_group, args.exp_name, args.env_descr, args.agent_config, args.agent_params, exp_identifier=args.exp_identifier)

if args.steps and not args.load:
    steps = args.steps

track_cfg = TrackConfig(TrackMetric.STEPS, steps, args.eval_interval)
experiment.run(track_cfg=track_cfg,runs=args.num_runs, show_progress=args.show_progress, wandb_logging=args.wandb_logging, save_latest=args.save_latest, save_best=args.save_best)


