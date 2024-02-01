from dataclasses import dataclass
from torch import nn
from Architectures.Elementary import FFNet
from Architectures.ActorCritic import ActorCritic
from Architectures.CascadeAC import ActorCriticCascade
from Architectures.ActorHead import ActorHead

@dataclass
class NetworkConfig:
    class_name: str
    args_dict: dict

    def init_obj(self) -> nn.Module:
        return globals()[self.class_name](**self.args_dict)
