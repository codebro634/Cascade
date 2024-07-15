from pathlib import Path
from typing import Union
from torch import nn

class CascadeActor(nn.Module):

    #Set logstd None for a discrete Actor Head
    def __init__(self, shared: Union[nn.Module, "NetworkConfig"], mean: Union[nn.Module,"NetworkConfig"], logstd: Union[nn.Module,"NetworkConfig"] = None):
        super().__init__()
        self.shared = shared if shared is None or isinstance(shared, nn.Module) else shared.init_obj()
        self.mean = mean if isinstance(mean, nn.Module) else mean.init_obj()
        self.logstd = logstd if logstd is None or isinstance(logstd,nn.Module) else logstd.init_obj()


    def forward(self, x):
        shared_out = self.shared(x)["y"] if self.shared is not None else x
        mean_out = self.mean(shared_out)
        logstd_out = self.logstd(shared_out) if self.logstd is not None else {}

        out = {"mean": mean_out["y"]}
        if "y" in logstd_out:
            out["logstd"] = logstd_out["y"]
        if "fallback" in mean_out:
            out["fallback"] = mean_out["fallback"]

        return out

