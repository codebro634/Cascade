from typing import List, Union
import numpy as np
import torch
import torch.nn as nn


#From CleanRL
def layer_init(layer, std=None, bias_const=None):
    if std is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if bias_const is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FFNet(nn.Module):

    """
        Simple Feed-Forward-Net

        hidden_sizes: If None, then the net degenerates to a learnable parameter (i.e. not dependent on the input)
        preset_params: List of parameters to set manually. A list entry is a tuple of the form (layer_idx,is_bias,layer_param_idx,value)
    """

    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple[int] = (64,64), activation_function: nn.Module = nn.Tanh(),
                 ll_activation: nn.Module = None, ll_activation_range: tuple[int] = None, init_std: list[float] = None, init_bias_const: list[float] = None, preset_params: list[tuple[int,bool,int,float]] = []):
        super().__init__()

        self.ll_activation, self.ll_activation_range = ll_activation, ll_activation_range
        self.hidden_sizes = hidden_sizes
        if hidden_sizes is not None:

            if init_std is not None:
                if not isinstance(init_std, list) and not isinstance(init_std, tuple):
                    init_std = [init_std] * (len(hidden_sizes) + 1)
                    init_bias_const = [init_bias_const] * (len(hidden_sizes) + 1)
                assert len(init_std) == len(hidden_sizes) + 1 and len(init_bias_const) == len(hidden_sizes) + 1
            else:
                init_std = init_bias_const = [None for _ in range(len(hidden_sizes) + 1)]

            layers = []
            for hidden_size, std, bias in zip(hidden_sizes,init_std[:-1],init_bias_const[:-1]):
                layers.append(layer_init(nn.Linear(input_size,hidden_size), std, bias))
                layers.append(activation_function)
                input_size = hidden_size
            layers.append(layer_init(nn.Linear(input_size,output_size), init_std[-1], init_bias_const[-1]))
            self.layers = nn.Sequential(*layers)
        else:
            self.output = nn.Parameter(torch.zeros(output_size))

        for layer_idx, is_bias, layer_param_idx, value in preset_params:
            if is_bias:
                self.layers[layer_idx].bias.data[layer_param_idx] = value
            else:
                self.layers[layer_idx].weight.data[layer_param_idx] = value


    def forward(self, x):
        batch = len(x.shape) > 1

        if self.hidden_sizes is not None:
            out = self.layers(x)
        else: #For the case the net is just an input-independent parameter
            out = self.output.expand((x.shape[0],self.output.shape[0])) if batch else self.output

        if self.ll_activation is not None:
            if batch:
                slice = out[:,self.ll_activation_range[0]:self.ll_activation_range[1]] if self.ll_activation_range is not None else out
                if self.ll_activation_range is not None:
                    out[:,self.ll_activation_range[0]:self.ll_activation_range[1]] = self.ll_activation(slice)
                else:
                    out = self.ll_activation(out)
            else:
                slice = out[self.ll_activation_range[0]:self.ll_activation_range[1]] if self.ll_activation_range is not None else out
                if self.ll_activation_range is not None:
                    out[self.ll_activation_range[0]:self.ll_activation_range[1]] = self.ll_activation(slice)
                else:
                    out = self.ll_activation(out)

        return {"y": out}

#Average absolute distance between all parameters of net1 and net2
def abs_difference(net1: nn.Module, net2: nn.Module):
    with torch.no_grad():
        params1, params2 = np.array([]), np.array([])
        for p1,p2 in zip(net1.parameters(),net2.parameters()):
            x, y = np.array(p1).flatten(), np.array(p2).flatten()
            params1 = np.concatenate((params1,x))
            params2 = np.concatenate((params2,y))
    absd = np.mean(np.abs(params1 - params2))
    return absd.item()

#a = FFNet(10,10)
# b = FFNet(10,10)
# print(abs_difference(a,b))