from dataclasses import dataclass
from torch import nn
from Architectures.Elementary import FFNet
from Architectures.ActorHead import ActorHead, CombActorHead
from Architectures.ActorCritic import ActorCritic
from Architectures.AttentionOverN import AttentionOverNNet
from Architectures.CascadeAC import ActorCriticCascade

@dataclass
class NetworkConfig:
    class_name: str
    args_dict: dict

    def init_obj(self) -> nn.Module:
        return globals()[self.class_name](**self.args_dict)
