import os
import pickle
from pathlib import Path

import numpy as np
import wandb
import math
from scipy.stats import sem
import seaborn as sns
import matplotlib.pyplot as plt

wandb_path = None # '<team>/<project>'
if wandb_path is None:
    raise ValueError("Please specify the path to wandb project.")

def wandb_runs_to_dict(group: str, name: str | set[str], metrics: list[str] = ["average return"], merge_metrics: dict[str,str] = {"average return0": "average return"}, verbose: bool = True) -> tuple[dict,dict]:
    """
        Retrieves the runs with the given name (within a set of names) and group from wandb and returns the mean and standard error of the
        given metrics at each timestep. The mean and standard error are calculated over all runs with the given name and group.

        merge_metrics: In case some runs were accidentally trained with different metric names, this dict can be used to merge them.
    """

    #Retrieve runs from wandb
    api = wandb.Api()
    runs_data = {metric: [] for metric in metrics} # Key: metric name, Value: list of lists. Each list contains the values of the metric at a certain timestep for all runs.
    num_runs = 0
    for run in api.runs(path=wandb_path, filters={"group": group}):
        if (isinstance(name,str) and run.name == name) or (isinstance(name,set) and run.name in name):
            num_runs += 1
            if verbose:
                print(f"Run{num_runs}: {run.name}. State: {run.state}")
            #Get entire history of single run
            keys = [key for key in merge_metrics if (merge_metrics[key] not in run.summary and key in run.summary and merge_metrics[key] in metrics)] if merge_metrics is not None else []
            keys += [metric for metric in metrics if metric in run.summary]
            keys  = [ key+"."+list(run.summary[key].keys())[0] if isinstance(run.summary[key],wandb.old.summary.SummarySubDict) else key for key in keys]  #correct for mistakes that sometimes a dictionary was logged

            data = run.scan_history(keys=keys)

            #Replace all metrics logged as dicts with the value inside dict
            if any([ (key in metrics or key in merge_metrics) and isinstance(value,wandb.old.summary.SummarySubDict) for key,value in run.summary.items()]):
                data = [
                    { key.split(".")[0]: value for key,value in row.items() } for row in data
                ]

            #Replace all metrics in data that occur in 'merge_metrics' with their true name
            if merge_metrics is not None:
                logged_merge_metrics = [metric for metric in merge_metrics if metric in run.summary and merge_metrics[metric] in metrics]
                new_data = []
                for row in data:
                    for metric in logged_merge_metrics:
                        row[merge_metrics[metric]] = row[metric]
                    new_data.append(row)
                data = new_data

            #Parse data into 'runs_data' and discard NaN values
            metrics_data = {metric: [(row[metric][0] if row[metric] else math.nan) if isinstance(row[metric],list) else row[metric] for row in data] for metric in metrics} #Conditional to account for some bugs made during the wandb loggin phase
            for metric in metrics:
                for i, val in enumerate(metrics_data[metric][i] for i in range(len(metrics_data[metric])) if not math.isnan(metrics_data[metric][i])):
                    if i >= len(runs_data[metric]):
                        runs_data[metric].append([])
                    runs_data[metric][i].append(val)


    if not runs_data:
        raise ValueError(f"Run {name} in group {group} not found.")

    #Calculate mean and standard error for each metric
    runs_stderr = {}
    runs_mean = {}
    for col in runs_data:
        runs_stderr[col], runs_mean[col] = [], []
        for i in range(len(runs_data[col])):
            runs_mean[col].append(sum(runs_data[col][i])/len(runs_data[col][i]))
            runs_stderr[col].append(sem(runs_data[col][i]) if len(runs_data[col][i]) > 1 else 0)

    return runs_mean, runs_stderr


def make_plot(experiments: list[tuple[str,str,str,str] | tuple[tuple[list,list],str,str]], step_range_to_plot: tuple[int,int] | list[tuple[int,int]] = None, title: str = None,
              metric: str = "average return", ylabel: str = None, legend_position:str = 'upper left', save_dir: Path = None, show: bool = True):
    """
        Plots experiment results in a single figure.

        experiments: list of tuples. Each tuple contains the group name, the run name, the label for the legend and the color for the plot of one experiment.
                    Alternatively, the group name and run name and be replaced by the run data directly given as a list of values and standard errors.
        step_range_to_plot: tuple of ints. The first int is the first timestep to plot, the second int is the last timestep to plot. Can be a list for different values for different experiments.
        title: str. Title of the plot.
        metric: str. Name of the logged metric to plot.
        save_dir: Path. Directory to save the plot to.
        ylabel: str. Label of the y-axis. If None, the metric name is used.
        show: bool. Whether to display the plot
    """

    sns.set_theme(style="darkgrid")
    plt.clf()
    max_x = 0

    #Add plot of each experiment to figure
    for i,experiment in enumerate(experiments):
        #Get run data
        if isinstance(experiment[0], str):
            data  = wandb_runs_to_dict(experiment[0],experiment[1], metrics=[metric])
            y = data[0][metric]
            y_stderr = data[1][metric]
        else:
            y = experiment[0][0]
            y_stderr = experiment[0][1]
        if step_range_to_plot is not None:
            step_range = step_range_to_plot[i] if isinstance(step_range_to_plot, list) else step_range_to_plot
            y = y[step_range[0]:step_range[1]]
            y_stderr = y_stderr[step_range[0]:step_range[1]]
        x = [i for i in range(len(y))]
        max_x = max(max_x, len(x))
        plt.plot(x, y, color=experiment[-1], label=experiment[-2])
        plt.fill_between(x, np.array(y) - np.array(y_stderr), np.array(y) + np.array(y_stderr), color = experiment[-1], alpha=0.3)

    #Beautify plot
    plt.xlabel("Million Steps",fontweight='bold', fontsize=20)
    plt.ylabel(metric.title() if ylabel is None else ylabel, fontweight='bold', fontsize=20)
    if title is not None:
        plt.title(title)
    legend = plt.legend(loc=legend_position, fontsize='15')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)

    step_size = 50
    num_ticks = max_x/step_size
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks([i for i in range(0, max_x+step_size, step_size if num_ticks <= 6 else 2 * step_size)], [i/100 for i in range(0, max_x+step_size, step_size if num_ticks <= 6 else 2 * step_size)]) #i/100 because data has been logged in 1e^4 steps
    plt.gcf().set_size_inches(10 if max_x <= 800 else 20, 5)

    #Save plot
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        svg_path = save_dir.joinpath(f"{os.path.basename(save_dir)}.svg")
        png_path = save_dir.joinpath(f"{os.path.basename(save_dir)}.png")

        # Check if plot already exists to replace the corresponding files
        for filename in os.listdir(save_dir):
            if str(filename).endswith(".svg"):
                svg_path = save_dir.joinpath(filename)
            if str(filename).endswith(".png"):
                png_path = save_dir.joinpath(filename)

        plt.savefig(svg_path, format='svg')
        plt.savefig(png_path, dpi=400, bbox_inches = 'tight')
        # Save the plot description to a file, so it could be recreated later
        with open(save_dir.joinpath("plot_descr.pkl"), "wb") as f:
            to_save = {"experiments": experiments, "step_range_to_plot": step_range_to_plot, "title": title, "metric": metric, "ylabel":ylabel, "legend_position": legend_position}
            pickle.dump(to_save, f)

    if show:
        plt.show()


def redo_plot(plot_path: Path):
    """
        Redo a plot that has been saved. This assumes the existence of 'plot_descr.pkl' in the directory at plot_path.
    """
    with open(plot_path.joinpath("plot_descr.pkl"), "rb") as f:
        data = pickle.load(f)
    make_plot(data["experiments"], step_range_to_plot=data["step_range_to_plot"], title=data["title"], legend_position=data["legend_position"], metric=data["metric"], ylabel=data["ylabel"], save_dir=plot_path, show=False)


def redo_entire_plot_directory(plot_dir: Path):
    for plot_path in plot_dir.iterdir():
        redo_plot(plot_path)


ant_2mil_baseline = ["Baseline","PPO_Ant-v4","2m",'red']
walker_2mil_baseline = ["Baseline","PPO_Walker2d-v4","2m",'red']
humanoid_2mil_baseline = ["Baseline","PPO_Humanoid-v4","2m",'red']
hopper_2mil_baseline = ["Baseline","PPO_Hopper-v4","2m",'red']
cheetah_2mil_baseline = ["Baseline","PPO_HalfCheetah-v4","2m",'red']

ant_3mil_baseline = ("Baseline","PPO_Ant-v4_3mil","3m",'blue')
walker_3mil_baseline = ("Baseline","PPO_Walker2d-v4_3mil","3m",'blue')
humanoid_3mil_baseline = ("Baseline","PPO_Humanoid-v4_3mil","3m",'blue')
hopper_3mil_baseline = ("Baseline","PPO_Hopper-v4_3mil","3m",'blue')
cheetah_3mil_baseline = ("Baseline","PPO_HalfCheetah-v4_3mil","3m",'blue')

ant_4mil_baseline = ("Baseline","PPO_Ant-v4_4mil","4m",'green')
walker_4mil_baseline = ("Baseline","PPO_Walker2d-v4_4mil","4m",'green')
humanoid_4mil_baseline = ("Baseline","PPO_Humanoid-v4_4mil","4m",'green')
hopper_4mil_baseline = ("Baseline","PPO_Hopper-v4_4mil","4m",'green')
cheetah_4mil_baseline = ("Baseline","PPO_HalfCheetah-v4_4mil","4m",'green')

ant_5mil_baseline = ("Baseline","PPO_Ant-v4_5mil","5m",'yellow')
walker_5mil_baseline = ("Baseline","PPO_Walker2d-v4_5mil","5m",'yellow')
humanoid_5mil_baseline = ("Baseline","PPO_Humanoid-v4_5mil","5m",'yellow')
hopper_5mil_baseline = ("Baseline","PPO_Hopper-v4_5mil","5m",'yellow')
cheetah_5mil_baseline = ("Baseline","PPO_HalfCheetah-v4_5mil","5m",'yellow')

ant_6mil_baseline = ["Baseline","PPO_Ant-v4_6mil","6m",'pink']
walker_6mil_baseline = ["Baseline","PPO_Walker2d-v4_6mil","6m",'pink']
humanoid_6mil_baseline = ["Baseline","PPO_Humanoid-v4_6mil","6m",'pink']
hopper_6mil_baseline = ["Baseline","PPO_Hopper-v4_6mil","6m",'pink']
cheetah_6mil_baseline = ["Baseline","PPO_HalfCheetah-v4_6mil","6m",'pink']


"""
    Baseline plots
"""

# make_plot(experiments=[ant_2mil_baseline,ant_3mil_baseline,ant_4mil_baseline,ant_5mil_baseline,ant_6mil_baseline], legend_position='lower right', metric="average return" , save_dir=Path("../nobackup/Plots/ant_baseline"), show=True)
# make_plot(experiments=[walker_2mil_baseline,walker_3mil_baseline,walker_4mil_baseline,walker_5mil_baseline,walker_6mil_baseline], legend_position='lower right', metric="average return" , save_dir=Path("../nobackup/Plots/walker_baseline"), show=True)
# make_plot(experiments=[humanoid_2mil_baseline,humanoid_3mil_baseline,humanoid_4mil_baseline,humanoid_5mil_baseline,humanoid_6mil_baseline], legend_position='lower right', metric="average return" , save_dir=Path("../nobackup/Plots/humanoid_baseline"), show=True)
# make_plot(experiments=[hopper_2mil_baseline,hopper_3mil_baseline,hopper_4mil_baseline,hopper_5mil_baseline,hopper_6mil_baseline], legend_position='lower right', metric="average return" , save_dir=Path("../nobackup/Plots/hopper_baseline"), show=True)
# make_plot(experiments=[cheetah_2mil_baseline,cheetah_3mil_baseline,cheetah_4mil_baseline,cheetah_5mil_baseline,cheetah_6mil_baseline], legend_position='lower right', metric="average return" , save_dir=Path("../nobackup/Plots/cheetah_baseline"), show=True)