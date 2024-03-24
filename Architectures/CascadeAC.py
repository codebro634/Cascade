from typing import Union, List

import numpy as np
import torch
from torch import nn


class ActorCriticCascade(nn.Module):

    """
        Represents a Cascade-net made up of a list of Actor Critics 'acs'
    """

    def __init__(self, acs: List[Union[nn.Module,"NetworkConfig"]], propagate_value: bool = False, propagate_action: bool = False):
        super().__init__()

        self.cascade = []
        for i,ac in enumerate(acs):
            ac_inited = ac if isinstance(ac,nn.Module) else ac.init_obj()
            self.cascade.append(ac_inited)
            self.add_module(f'seq{i}', ac_inited) #register parameters of net

        self.prop_val = propagate_value
        self.prop_action = propagate_action
        self.sigm = torch.nn.Sigmoid()

        self.base_idx = None #If not None, only the Actor Critic at the base_idx-th position is used for action selection

    def critic_params(self):
        if self.prop_val or self.prop_action:
            raise NotImplementedError()
        return self.cascade[-1].critic_params()

    def actor_params(self):
        if self.prop_val or self.prop_action:
            raise NotImplementedError()
        params = []
        for ac in self.cascade:
            params += list(ac.actor_params())
        return params

    def q_value(self, obs, action):
        if self.prop_val or self.prop_action:
            raise NotImplementedError()
        return self.cascade[-1].q_value(obs,action)

    def get_value(self, obs):
        if len(self.cascade) == 1 or ((not self.prop_val) and (not self.prop_action)):
            #Since no propagation, value is just value of uppermost value function
            return self.cascade[-1].get_value(obs)
        else:
            #Case discrimination only for efficiency reasons
            if (self.prop_val and self.prop_action) or (self.prop_action and not self.prop_val):
                casc_vals = self.cascade_up(obs, -1)
                return self.cascade[-1].get_value(casc_vals["next"])
            else:
                #Since no action-propagation, only values functions need to be used
                last_val = None
                for i, ac in enumerate(self.cascade):
                    x = obs
                    if last_val:
                        x = torch.cat([obs, last_val], dim=1)
                    last_val = ac.get_value(x)
                return last_val

    #Once called only uses action of the base_idx-th Actor Critic
    def regress_to_one_base(self, base_idx: int):
        assert not self.prop_val and not self.prop_action
        self.base_idx = base_idx

    def get_action(self, obs):
        if self.base_idx is None:
            if len(self.cascade) > 1:
                #Get inputs for top Actor Critic
                casc_vals = self.cascade_up(obs,-1)
                top_action = self.cascade[-1].get_action(casc_vals["next"])
                return self.weighted_sum(top_action, casc_vals["mean"], casc_vals["logstd"])
            else:
                return self.cascade[-1].get_action(obs)
        else:
            x = self.cascade[self.base_idx].get_action(obs)
            mean = x["mean"][:, 0:-1] if self.base_idx > 0 else x["mean"]
            return {"mean": mean, "logstd": x["logstd"]}


    def get_action_and_value(self, obs):

        if len(self.cascade) > 1:
            #Get inputs for top Actor Critic
            casc_vals = self.cascade_up(obs,-1)
            top_out =  self.cascade[-1].get_action_and_value(casc_vals["next"])
            return {"action": self.weighted_sum(top_out["action"], casc_vals["mean"], casc_vals["logstd"]), "value": top_out["value"]}
        else:
            return self.cascade[-1].get_action_and_value(obs)

    #List of fallback probabilities for each Actor Critic
    def get_fallbacks(self, obs):
        assert not self.prop_val and not self.prop_action

        obs = torch.Tensor(obs)
        probs = np.zeros(shape=(len(self.cascade)-1))
        with torch.no_grad():
            for i,ac in enumerate(self.cascade[1:]):
                probs[i] = self.sigm(ac.get_action(obs)["mean"][-1])
        return probs


    def weighted_sum(self, actor_output, fb_mean, fb_std):
        if fb_mean is not None:
            assert fb_std is not None
            mean = actor_output["mean"][:, 0:-1]
            weights = self.sigm(actor_output["mean"][:, -1]).unsqueeze(dim=-1)
            weighted_mean = weights * fb_mean + (1 - weights) * mean
            weighted_logstd = weights * fb_std + (1 - weights) * actor_output["logstd"]
            return {"mean": weighted_mean, "logstd": weighted_logstd, "weights": weights.squeeze(dim=-1)}
        else:
            return {"mean": actor_output["mean"], "logstd": actor_output["logstd"]}

    """
        Calculates the input for the end_idx-th Actor Critic of the Cascade and the action that the Cascade up the end_idx-th Actor Critic produces.
        Is used as a helper method for get_action_and_value, get_action and get_value
    """
    def cascade_up(self, obs, end_idx: int):
        last_mean, last_std, last_val = None, None, None

        for ac in self.cascade[:end_idx]:
            # Prepare input
            x = obs
            if last_val is not None:
                x = torch.cat([x, last_val], dim=1)
            if last_mean is not None and self.prop_action:
                x = torch.cat([x, last_mean], dim=1)
            # Prepare function for actor critic
            fbm = ac.get_action_and_value if self.prop_val else ac.get_action
            # Get output of the current Actor Critic
            y = fbm(x)
            last_val = y["value"] if "value" in y else None
            temp = self.weighted_sum(y["action"] if "action" in y else y, last_mean, last_std)
            last_mean, last_std = temp["mean"], temp["logstd"]

        #Prepare for the end_idx-th Actor Critic
        next_input = obs
        if last_val is not None:
            next_input = torch.cat([next_input, last_val], dim=1)
        if last_mean is not None and self.prop_action:
            next_input = torch.cat([next_input, last_mean], dim=1)

        vals = {"mean": last_mean, "logstd": last_std, "next": next_input}
        if self.prop_val:
            vals["value"] = last_val
        return vals

