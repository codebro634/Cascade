from dataclasses import dataclass
from torch import nn
from Architectures.Elementary import FFNet
from Architectures.CascadeNet import CascadeNet
from Architectures.ActorNet import CascadeActor
from Architectures.CascadeNet import CascadeFFNet

@dataclass
class NetworkConfig:
    class_name: str
    args_dict: dict

    def init_obj(self) -> nn.Module:
        return globals()[self.class_name](**self.args_dict)
