import numpy as np
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents.actors import NormalizedActor
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.layers import mlp


class DeterministicActor(NormalizedActor):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        self.net = mlp(
            np.prod(envs.single_observation_space.shape), 
            [args.mlp_dim] * args.num_layers_actor,
            np.prod(envs.single_action_space.shape),
        )

    def forward(self, x) -> torch.Tensor:
        action = self.net(x)
        action = torch.tanh(action)
        return action * self.action_scale + self.action_bias

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)