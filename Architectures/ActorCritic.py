from typing import Union
from torch import nn


class ActorCritic(nn.Module):

    def __init__(self, actor: Union["NetworkConfig", nn.Module], critic: Union[nn.Module, "NetworkConfig"],shared: Union[nn.Module, "NetworkConfig"] = None):
        super().__init__()

        # Init Networks
        self.critic = critic if isinstance(critic, nn.Module) else critic.init_obj()
        self.actor = actor if isinstance(actor, nn.Module) else actor.init_obj()
        self.shared = None if shared is None else (shared if isinstance(shared, nn.Module) else shared.init_obj())


    def get_value(self, x, x_after_shared=False):
        x = self.shared(x)["y"] if (self.shared and not x_after_shared) else x
        return self.critic(x)["y"]

    def get_action(self, x, x_after_shared = False):
        x = self.shared(x)["y"] if (self.shared and not x_after_shared) else x
        return self.actor.get_action(x)

    def get_action_and_value(self, x):
        x = self.shared(x)["y"] if self.shared else x
        return {"action": self.get_action(x, x_after_shared= True), "value": self.get_value(x, x_after_shared=True) }











