import torch
import torch.nn as nn
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.utils.common import Args


class NormalizedActor(nn.Module):
    """
    Base class for all actors. 
    The network outputs should be between -1 and 1. They should be normalized to the action
    space limits in the forward pass using the `action_scale` and `action_bias` buffers.
    """

    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        self.args = args

        h, l = envs.single_action_space.high, envs.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        self.exclude_from_copy = ["action_scale", "action_bias"]
        # These buffers are persistent and will be saved in state_dict()

    def forward(self, x):
        """
        This should return an (unnormalized) action sampled from the policy distribution.
        """
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        This should return the best (unnormalized) action as per the current policy.
        """
        raise NotImplementedError

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
