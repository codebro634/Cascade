import math

import torch
from torch import nn
from torch.distributions import Categorical
import random

class AttentionOverNNet(nn.Module):

    """
        Combines base-nets with a chooser such that the output for input x is:
            softmax(chooser(x))_1 * base_1(x) + ... + softmax(chooser(x))_n * base_n(x)

        dropout_prob: With this probability the output of a random base-net is chosen as the output
        temperature: If != 1 then softmax(chooser(x)) is now softmax(chooser(x) / temperature)
    """

    def __init__(self, chooser_cfg: "NetworkConfig", base_cfgs: list["NetworkConfig"], dropout_prob: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.chooser = chooser_cfg if isinstance(chooser_cfg,nn.Module) else chooser_cfg.init_obj()
        self.base_cfgs = base_cfgs
        self.base_nets = [cfg if isinstance(cfg, nn.Module) else cfg.init_obj() for cfg in base_cfgs]
        #Register parameters of the base nets
        for i, net in enumerate(self.base_nets):
            self.add_module(f'base{i}', net)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout_prob = dropout_prob
        self.temperature = temperature


    def get_attention(self, x):
        attention = self.chooser(x)["y"]
        probs = self.softmax(attention / (self.temperature if self.temperature else 1.0))
        if self.temperature == math.inf:
            probs = probs.detach()
        return {"p": probs, "vent": Categorical(probs=probs).entropy(), "hent": Categorical(probs.mean(dim=0)).entropy()}

    def forward(self, x, attention = None, chosen_net: int = None):

        output = {}

        #Get attention and entropy of attention
        if attention is None:
            attention_data = self.get_attention(x)
            attention = attention_data["p"]
            output["vent"] = attention_data["vent"]
            output["hent"] = attention_data["hent"]

        #Calculate output of base-nets
        base_outputs = torch.stack([base_net(x)["y"] for base_net in self.base_nets], dim=1)

        #Check for random dropout
        if chosen_net is not None:
            weighted_output = base_outputs[:,chosen_net,:]
        elif self.dropout_prob and random.random() < self.dropout_prob:
            chosen_net = random.randint(0, len(self.base_nets) - 1)
            output["chosen"]= chosen_net
            weighted_output = base_outputs[:,chosen_net,:]
        else:
            #Calculate weighted sum
            attention = attention.unsqueeze(-1)
            weighted_output = torch.sum(base_outputs * attention, dim = 1)


        output["y"] = weighted_output

        return output
