from pathlib import Path
from typing import Union

import torch
from torch import nn

from Agents.Agent import Agent
from Architectures.AttentionOverN import AttentionOverNNet


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


class CombActorHead(nn.Module):

    #Set logstd for 0-th head to None for discrete CombActorHead
    def __init__(self, chooser: Union[nn.Module,"NetworkConfig"], heads: list[Union[nn.Module,"NetworkConfig"]], dropout_prob: float = 0.0, temperature: float = 1.0, freeze_action_nets: bool = False):
        super().__init__()

        self.head_cfgs = [cfg for cfg in heads if not isinstance(cfg,nn.Module)]
        self.heads = [head if isinstance(head,nn.Module) else head.init_obj() for head in heads]
        if freeze_action_nets:
            for head in self.heads:
                for param in head.parameters():
                    param.requires_grad = False
        self.chooser = chooser if isinstance(chooser, nn.Module) else chooser.init_obj()
        self.mean = AttentionOverNNet(self.chooser,[head.mean for head in self.heads], dropout_prob=dropout_prob, temperature = temperature)
        self.logstd = AttentionOverNNet(self.chooser,[head.logstd for head in self.heads], dropout_prob=dropout_prob, temperature = temperature) if self.heads[0].logstd else None

        self.n = len(heads)
        self.action_idx = None

    #Only uses the action_idx-th action-net for output after call
    def regress_to_one_action_net(self, action_idx: int):
        self.action_idx = action_idx

    def get_action(self, x):
        if self.action_idx is not None:
            return self.heads[self.action_idx].get_action(x)
        else:
            attention_data = self.mean.get_attention(x)
            weighted_means = self.mean(x, attention = attention_data["p"])
            return_vals = dict( {"mean": weighted_means["y"]}.items() | attention_data.items() )
            if self.logstd: #Only needed in continuous case
                weighted_logstds = self.logstd(x, attention = attention_data["p"], chosen_net = weighted_means["chosen"] if "chosen" in weighted_means else None)
                return_vals["logstd"] = weighted_logstds["y"]
            return return_vals

    #Resets the parameters of the idx-th action-net if action_path is None, otherwise the parameters from the net saved in action_path are loaded
    def reinit_action_net(self, idx, action_path:Path=None):
        if action_path is None:
            self.heads[idx].load_state_dict(self.head_cfgs[idx].init_obj().state_dict())
        else:
            checkpoint = torch.load(Agent.to_absolute_path(action_path))
            self.heads[idx].load_state_dict(checkpoint)



