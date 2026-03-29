import random
from typing import Dict, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import BaseAgent
from rainbow_demorl.agents.actors import NormalizedActor
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.layers import Ensemble, mlp, weight_init
from rainbow_demorl.utils.math import soft_ce, two_hot_inv
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer, TrajReplayBufferSample


class SoftQNetwork(nn.Module):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__()
        '''
        Using parallelized ensemble of MLPs for the Q-value network.
        '''
        if args.use_ce_loss:
            self.net = Ensemble([mlp(
                                    np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape), 
                                    [args.mlp_dim] * args.num_layers_critic, 
                                    args.num_bins,
                                    dropout=args.q_dropout
                                ) for _ in range(args.num_critics)])
        else:
            self.net = Ensemble([mlp(
                                    np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape),
                                    [args.mlp_dim] * args.num_layers_critic,
                                    1,
                                ) for _ in range(args.num_critics)])

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], -1)
        return self.net(x)
    
    def __len__(self):
        return len(self.net)
    
    def __getitem__(self, idx):
        return self.net[idx]


class ActorCriticAgent(BaseAgent):
    """
    Base class for all actor-critic agents. 
    For training, inherited agents must implement the `update_actor` and `update_critic` methods.
    For evaluation, inherited agents must implement the `get_eval_action` method.
    For exploration, agents must implement and use the `sample_action` method.
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, 
                 actor_class: Type[NormalizedActor],
                 qf_class: Type[SoftQNetwork]):
        super().__init__(envs, device, args)

        if actor_class is not None:
            self.actor = actor_class(envs, args).to(device)
        else:
            self.actor: NormalizedActor = None

        self.qfs = qf_class(envs, args).to(device)
        self.qfs_target = qf_class(envs, args).to(device)
        
        if args.checkpoint is not None:
            self.load_model(args.checkpoint)
        else:
            self.initialize_networks()

        self.qfs_target.load_state_dict(self.qfs.state_dict())

        self.q_optimizer = optim.Adam(self.qfs.parameters(), lr=args.lr)
        if self.actor is not None:
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.lr)

        self.all_modules: list[nn.Module] = [self.actor, self.qfs, self.qfs_target]

    def initialize_networks(self):
        if self.actor is not None:
            self.actor.apply(weight_init)
        self.qfs.apply(weight_init)

    def update_target_networks(self):
        for param, target_param in zip(self.qfs.parameters(), self.qfs_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    def Q(self, obs: torch.Tensor, action: torch.Tensor, target: bool = False, logits: bool = False, 
          random_sample_two_qf: bool = False, 
          random_close_qf: bool = False
          ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        obs: [..., obs_dim]
        action: [..., action_dim]
        target: whether to use the target network
        logits: whether to use the logits of the Q-values (for CE loss)

        Returns:
            qvals: [num_critics, ..., 1] or [num_critics, ..., num_bins]
            min_qval: [..., 1]
        """
        qfs = self.qfs_target if target else self.qfs
        if random_sample_two_qf:
            idx = random.sample(range(len(qfs)), 2)
        elif random_close_qf:
            k = random.randint(2, len(qfs)) # minimum qf is 2 from CHEQ paper
            idx = random.sample(range(len(qfs)), k)
        else:
            idx = None
            
        qvals_all = qfs(obs, action)
        qvals = qvals_all if idx is None else qvals_all[idx] # shape: [num_sampled_critics, ..., num_bins] or [num_sampled_critics, ..., 1]
        if logits:
            min_qval = None
        else:
            if self.args.use_ce_loss:
                qvals = torch.stack([two_hot_inv(qval, self.args) for qval in qvals], dim=0)
            min_qval = torch.min(qvals, dim=0).values
        return qvals, min_qval

    def update(self, rb_online: TrajReplayBuffer, rb_offline: TrajReplayBuffer, global_update: int, global_step: int) -> int:
        """
        Update the actor and critic networks as per the algorithm.
        """
        for local_update in range(self.args.grad_steps_per_iteration):
            global_update += 1

            if rb_online is not None:
                online_data = rb_online.sample()
            if rb_offline is not None:
                offline_data = rb_offline.sample()
                # since CHEQ adds one dimension, code below is to give the trajectories a lam_buffer
                if self.args.is_cheq:
                    H, B, _ = offline_data.obs.shape
                    lam_offline = torch.full((H, B, 1), self.args.lam_buffer, device=offline_data.obs.device, dtype=offline_data.obs.dtype)
                    offline_data.obs = torch.cat([offline_data.obs, lam_offline], dim=-1)
                    offline_data.next_obs = torch.cat([offline_data.next_obs, lam_offline], dim=-1)
            else: 
                offline_data = None

            if self.args.learning_mode == "online":
                data = online_data
            elif self.args.learning_mode == "offline":
                data = offline_data

            if self.args.use_offline_data_for_rl:
                if offline_data is not None:
                    data = online_data.cat(offline_data, dim=1)
                else:
                    raise ValueError("Filled offline replay buffer is required when use_offline_data_for_rl is True")

            self.update_critic(data, global_step)

            # Delayed policy updates
            if global_update % self.args.policy_frequency == 0:
                self.update_actor(data, offline_data, global_step)
            if global_update % self.args.target_network_frequency == 0:
                self.update_target_networks()
        return global_update

    def estimate_nstep_return(self, data: TrajReplayBufferSample) -> torch.Tensor:
        """
        Estimate the return of a trajectory chunk, to be used as the target for the critic update.
        Must be implemented by inherited agents.
        """
        raise NotImplementedError
    
    def update_critic(self, data: TrajReplayBufferSample, global_step: int):
        """
        Update the critic networks with n-step returns and CE loss if specified.
        Values are estimated for all steps in the trajectory chunk with k<=n-step returns.
        """
        q_target = self.estimate_nstep_return(data)

        if self.args.use_ce_loss:
            qfs_a_values = self.Q(data.obs, data.actions, logits=True)[0]
            qfs_total_loss = [soft_ce(qf_a_values, q_target, self.args).mean() for qf_a_values in qfs_a_values]
        else:
            qfs_a_values = self.Q(data.obs, data.actions)[0]
            qfs_total_loss = [F.mse_loss(qf_a_values, q_target) for qf_a_values in qfs_a_values]
        qf_loss = torch.stack(qfs_total_loss, dim=0).mean()

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_a_values, qf_total_loss) in enumerate(zip(qfs_a_values, qfs_total_loss), start=1):
                self.logging_tracker[f"losses/qf{i}_values"] = qf_a_values.mean().item()
                self.logging_tracker[f"losses/qf{i}_loss"] = qf_total_loss.item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item()
    
    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        raise NotImplementedError

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save_model(self, model_path: str):
        raise NotImplementedError
    
    def load_model(self, model_path: str):
        raise NotImplementedError

    def load_pretrained(self, policy_path: str, value_path: str):
        if self.args.finetune_offline_policy and policy_path is not None:
            policy_ckpt = torch.load(policy_path)
            self.safe_load(self.actor, policy_ckpt['actor'])
            print(f"Loaded pretrained policy from {policy_path}")
        if self.args.finetune_offline_value and value_path is not None:
            value_ckpt = torch.load(value_path)
            if "qfs" in value_ckpt:
                self.safe_load(self.qfs, value_ckpt['qfs'])
                self.safe_load(self.qfs_target, value_ckpt['qfs_target'])
            else:
                raise ValueError(f"Q-value networks not found in model at {value_path}")
            print(f"Loaded pretrained value function from {value_path}")

    def train_mode(self):
        for module in self.all_modules:
            module.train()

    def eval_mode(self):
        for module in self.all_modules:
            module.eval()

    def safe_load(self, target: nn.Module, source_dict: Dict[str, torch.Tensor]):
        """
        Safely copy pretrained weights onto online agents while avoiding shape mismatch problems, especially for CHEQ.
        """
        target_dict = target.state_dict()

        new_dict = {}
        for k, v in target_dict.items():
            if k in source_dict:
                if target_dict[k].shape == source_dict[k].shape:
                    new_dict[k] = source_dict[k]
                else:
                    # align dimensions (e.g., CHEQ adds one more output neuron)
                    min_shape = tuple(min(a, b) for a, b in zip(v.shape, source_dict[k].shape))
                    aligned_tensor = v.clone()
                    aligned_tensor[..., :min_shape[-1]] = source_dict[k][..., :min_shape[-1]]
                    new_dict[k] = aligned_tensor
                    print(f"[WARN] Shape mismatch on {k}: {source_dict[k].shape} -> {v.shape}, partially copied")
            else:
                new_dict[k] = v  # keep target default

        target.load_state_dict(new_dict)
