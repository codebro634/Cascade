from typing import Union, List

import numpy as np
import torch
from torch import nn


class CascadeFFNet(nn.Module):

    def __init__(self, ff_net: Union[nn.Module, "NetworkConfig"]):
        super().__init__()
        self.net = ff_net if isinstance(ff_net, nn.Module) else ff_net.init_obj()

    def forward(self, obs):
        batch = len(obs.shape) > 1
        y = self.net(obs)["y"]
        fallback_val = y[:, -1] if batch else y[-1]
        y = y[:, :-1] if batch else y[:-1]
        return {"y": y, "fallback": fallback_val}


class CascadeNet(nn.Module):

    def __init__(self, nets: List[Union[nn.Module, "NetworkConfig"]]):
        super().__init__()

        self.cascade = []
        for i, ac in enumerate(nets):
            ac_inited = ac if isinstance(ac,nn.Module) else ac.init_obj()
            self.cascade.append(ac_inited)
            self.add_module(f'seq{i}', ac_inited) #register parameters of net

        self.sigm = torch.nn.Sigmoid()

        self.base_idx = None #If not None, only the Actor Critic at the base_idx-th position is used for action selection

    def params(self):
        if self.prop_val or self.prop_action:
            raise NotImplementedError()
        params = []
        for ac in self.cascade:
            params += list(ac.actor_params())
        return params

    #Once called only uses action of the base_idx-th Actor Critic
    def regress_to_one_base(self, base_idx: int):
        assert not self.prop_val and not self.prop_action
        self.base_idx = base_idx

    def forward(self, obs):
        if self.base_idx is None:
            weighted_outputs = None
            for net in self.cascade:
                net_out = net(obs)
                weighted_outputs = self.weighted_sum(net_out, weighted_outputs)
            return weighted_outputs
        else:
            return self.cascade[self.base_idx].forward(obs)


    #List of fallback probabilities for each Net
    def get_fallbacks(self, obs):
        obs = torch.Tensor(obs)
        probs = np.zeros(shape=(len(self.cascade)-1))
        with torch.no_grad():
            for i, ac in enumerate(self.cascade[1:]):
                probs[i] = self.sigm(ac.forward(obs)["fallback"])
        return probs


    def weighted_sum(self, net_out, last_net_out):
        if "fallback" in net_out:
            weights = self.sigm(net_out["fallback"]).unsqueeze(dim=-1)
            for k in net_out.keys():
                if k != "fallback":
                    net_out[k] = weights * last_net_out[k] + (1 - weights) * net_out[k]
            net_out.update({"weights": weights.squeeze(dim=-1)})
        else:
            return net_out


