from os.path import exists
from pathlib import Path
import pickle
from typing import Callable, Union

from Analysis.RunTracker import RunTracker
from dataclasses import dataclass
import gymnasium as gym
from Environments import EnvSpaceDescription


@dataclass
class AgentConfig:

    space_description: EnvSpaceDescription #Description of the action/observation-space the agent will be trained on
    gamma: float = 0.99
    name: str = None

    def validate(self):
        raise NotImplementedError()

    @staticmethod
    def load_from_absolute_path(path :Path) -> "AgentConfig":
        with open(path, 'rb') as file:
            cfg = pickle.load(file)
        return cfg

    def save_to_absolute_path(self, path: Path):
        with open(path, 'wb') as file:
            pickle.dump(self,file)


class Agent:

    MODEL_SAVE_DIR = Path("nobackup/Models") if exists(Path("nobackup/Models")) or not exists(Path("../nobackup/Models")) else Path("../nobackup/Models")

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg


    def get_setup_descr(self) -> dict:
        """Returns a dictionary with the agent's configuration as entries.
          This will be passed to Wandb for logging purposes."""

        params = dict()
        for field in self.cfg.__dataclass_fields__:
            params[field] = getattr(self.cfg, field)
        return params

    @staticmethod
    def to_absolute_path(path: Union[Path,str]):
        """  path: A path relative to the directory where agent-models are saved. Is converted to the absolute path within the project-folder. """
        return Agent.MODEL_SAVE_DIR.joinpath(path)

    @staticmethod
    def load(path: Path) -> "Agent":
        """ path: A path relative to the directory where agent-models are saved. """

        abs_path = Agent.to_absolute_path(path)
        assert exists(abs_path), f"The folder '{abs_path}' does not exist."
        assert exists(abs_path.joinpath("cfg.pkl")), f"cfg.pkl does not exist in {abs_path}"
        assert exists(abs_path.joinpath("alg.pkl")), f"alg.pkl does not exist in {abs_path}"
        with open(abs_path.joinpath("alg.pkl"), 'rb') as file:
            alg = pickle.load(file)
        cfg = AgentConfig.load_from_absolute_path(path=abs_path.joinpath("cfg.pkl"))
        return alg.load_with_no_checks(path, abs_path, cfg)


    @staticmethod
    def load_with_no_checks(relative_path: Path, absolute_path: Path, config: AgentConfig) -> "Agent":
        """ Loads the Agent. It is assumed that the existence of all necessary files is checked beforehand and the Agent's config is already loaded.
        relativ_path: A path to an Agent relative to the directory where agent-models are saved
        absolute_path: An absolute-path to an Agent within the project-folder """

        raise NotImplementedError()


    def save(self, path: Path):
        """ path: A path relative to the directory where agent-models are saved  """

        abs_path = Agent.to_absolute_path(path)
        abs_path.mkdir(exist_ok=True)
        with open(abs_path.joinpath("alg.pkl"), 'wb') as file:
            pickle.dump(self.__class__,file)
        self.cfg.save_to_absolute_path(abs_path.joinpath("cfg.pkl"))
        self.save_additionals(path,abs_path)

    def save_additionals(self, model_path: Path, absolute_path: Path):
        """Save Agent specific data that is not yet saved with 'save'. For example, a neural-network. """

        raise NotImplementedError()

    def get_action(self, obs, eval_mode: bool = False, deterministic: bool = False):
        """ obs: An observation returned from any gym-environment
            eval_mode: Whether this method is called for evaluation purposes """
        raise NotImplementedError()

    def train(self, env_maker: Callable[[],gym.core.Env], tracker: RunTracker, norm_sync_env: gym.core.Env = None):
        """
            norm_sync_env: An environment whose normalization state should constantly be updated with the current norm-state used in training.
        """
        raise NotImplementedError()