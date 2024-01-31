import time
from enum import Enum
from collections import defaultdict
from typing import Callable
from tqdm import tqdm


class TrackMetric(Enum):
    STEPS = "steps"
    EPOCHS = "epochs"
    SAMPLES = "samples"
    TIME = "time"

class TrackConfig:
    def __init__(self, metric: TrackMetric, total: int, eval_interval: int):
        self.metric = metric
        self.total = total
        self.eval_interval = eval_interval

"""
    Class to control the duration and evaluation of an Agent's training.
"""
class RunTracker:

    def __init__(self, cfg: TrackConfig, eval_func: Callable[[],None], show_progress: bool = True, nested: "RunTracker" = None, nested_progress: bool = False):
        self.show_progress = show_progress
        self.progress_bar = tqdm(total=100) if show_progress else None
        self.cfg = cfg
        self.eval_func = eval_func
        self.values, self.delta_values = defaultdict(int), defaultdict(int) #delta_values contain how much steps in each metric have been accumulated since last evaluation
        self.start_time = time.time()
        self.nested = nested #Updates of this tracker are also tracked in this nested tracker
        self.nested_progress = nested_progress #If set, get_progress then returns the progress from the nested tracker

    def get_progress(self):
        return self.nested.get_progress() if self.nested_progress else self.values[self.cfg.metric] / self.cfg.total

    def is_done(self):
        return (self.values[self.cfg.metric] >= self.cfg.total) or (self.nested and self.nested.is_done())

    def update_time(self):
        current_time = time.time() - self.start_time
        self.delta_values[TrackMetric.TIME] += current_time - self.values[TrackMetric.TIME]
        self.values[TrackMetric.TIME] = current_time

    def add_unit(self, metric: TrackMetric, count: int):
        #Update units for the nested tracker
        if self.nested:
            self.nested.add_unit(metric, count)

        #treat time as special case
        self.update_time()

        #Update the value for the metric
        self.values[metric] += count
        self.delta_values[metric] += count

        #Check if it is time for evaluation
        if self.delta_values[self.cfg.metric] >= self.cfg.eval_interval:
            self.delta_values[self.cfg.metric] -= self.cfg.eval_interval
            if self.eval_func:
                self.eval_func()

        #Update progress bar
        if self.progress_bar:
            self.progress_bar.update(int(100*self.get_progress()) - self.progress_bar.n)

        return self.is_done()


