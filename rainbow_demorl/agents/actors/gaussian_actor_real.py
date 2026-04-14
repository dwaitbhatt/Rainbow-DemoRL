from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.layers import mlp


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class GaussianActorReal(nn.Module):
    def __init__(self, args: Args, obs_space, act_space, device):
        super().__init__()
        self.args = args

        h, l = act_space.high, act_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

        self.action_scale_real = torch.tensor((h - l) / 2.0, dtype=torch.float32).to(device)
        self.action_bias_real = torch.tensor((h + l) / 2.0, dtype=torch.float32).to(device)

        self.backbone = mlp(
            np.prod(obs_space.shape),
            [args.mlp_dim] * (args.num_layers_actor - 1),
            args.mlp_dim
        ).to(device)
        self.fc_mean = nn.Linear(args.mlp_dim, np.prod(act_space.shape)).to(device)
        self.fc_logstd = nn.Linear(args.mlp_dim, np.prod(act_space.shape)).to(device)
        self.device = device

    def get_gaussian_params(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the Gaussian policy distribution parameters.

        Returns:
            A tuple containing:
            
            - (unnormalized) mean: The mean of the action distribution before normalization
            - log_std: The log standard deviation of the action distribution, bounded between -5 and 2 for
            numerical stability
        """
        x = x.to(self.device)
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        mean, log_std = self.get_gaussian_params(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale_real + self.action_bias_real
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale_real * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale_real + self.action_bias_real
        return action, log_prob, mean

    def get_eval_action(self, x) -> torch.Tensor:
        x = x.to(self.device)
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale_real + self.action_bias_real
        return action

    def get_log_probs(self, input, output) -> torch.Tensor:
        """
        Get log probabilities of actions given input observations.
        """
        input = input.to(self.device)
        output = output.to(self.device)
        mean, log_std = self.get_gaussian_params(input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(output)
        return log_prob