import gymnasium as gym
import torch
import torch.nn as nn
from typing import Optional

from rainbow_demorl.agents.actors import NormalizedActor
from rainbow_demorl.utils.common import Args, Logger
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer


class BaseAgent:
    def __init__(self, envs, device, args):
        self.envs: gym.Env = envs
        self.device: torch.device = device
        self.args: Args = args
        self.logging_tracker = {}

        self.train_episode_timestep = 0
        self.eval_episode_timestep = 0
        self.actor: NormalizedActor = None

        self.action_h = torch.tensor(self.envs.single_action_space.high, device=self.device)
        self.action_l = torch.tensor(self.envs.single_action_space.low, device=self.device)

    def update(self, rb_online: TrajReplayBuffer, rb_offline: TrajReplayBuffer, global_update: int, global_step: int) -> int:
        """
        Update the agent, and return the global update count.
        """
        raise NotImplementedError

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action from the actor. This action will be used to collect experience during training.
        obs: [batch_size, obs_dim]
        Returns: 
            action: [batch_size, action_dim]
        """
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the best action from the actor. This action will be used to evaluate the policy.
        obs: [batch_size, obs_dim]
        Returns: 
            action: [batch_size, action_dim]
        """
        raise NotImplementedError
    
    def save_model(self, model_path: str):
        raise NotImplementedError
    
    def load_model(self, model_path: str):
        raise NotImplementedError
    
    def log_losses(self, logger: Logger, global_step: int):
        for k, v in self.logging_tracker.items():
            logger.add_scalar(k, v, global_step)

    def __repr__(self):
        repr = self.__class__.__name__ + ":\n"
        seen_modules = set()
        for name, module in self.__dict__.items():
            if isinstance(module, nn.Module) and id(module) not in seen_modules:
                repr += f"{name}: {module.__repr__()}\n"
                seen_modules.add(id(module))
        return repr
