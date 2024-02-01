from pathlib import Path
from typing import Union

import torch
from torch import nn

from Agents.Agent import Agent


class ActorHead(nn.Module):

    #Set logstd None for a discrete Actor Head
    def __init__(self, mean: Union[nn.Module,"NetworkConfig"], logstd: Union[nn.Module,"NetworkConfig"] = None):
        super().__init__()

        self.mean = mean if isinstance(mean, nn.Module) else mean.init_obj()
        self.logstd = logstd if logstd is None or isinstance(logstd,nn.Module) else logstd.init_obj()


    def get_action(self, x):
        output = {"mean": self.mean(x)["y"]}
        if self.logstd: #Only neede in continuous case
            output["logstd"] = self.logstd(x)["y"]
        return output

