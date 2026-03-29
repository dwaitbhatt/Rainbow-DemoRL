from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import SACAgent, TD3Agent
from rainbow_demorl.agents.actors import (DeterministicActor, GaussianActor,
                                          NormalizedActor)
from rainbow_demorl.agents.bc import make_bc_control_prior
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.layers import mlp
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class DeterministicResidualActor(DeterministicActor):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        self.net = mlp(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape), 
            [args.mlp_dim] * args.num_layers_actor,
            np.prod(envs.single_action_space.shape),
        )

class GaussianResidualActor(GaussianActor):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        self.backbone = mlp(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape), 
            [args.mlp_dim] * (args.num_layers_actor - 1), 
            args.mlp_dim
        )
        self.fc_mean = nn.Linear(args.mlp_dim, np.prod(envs.single_action_space.shape))
        self.fc_logstd = nn.Linear(args.mlp_dim, np.prod(envs.single_action_space.shape))

class ResidualRLMixin:
    """
    Mixin class for Residual RL functionality.
    from paper: https://arxiv.org/pdf/1812.06298
    """
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, actor_class: Type[NormalizedActor]):
        super().__init__(envs, device, args, actor_class=actor_class)

        self.control_prior = make_bc_control_prior(envs, device, args)

        # The residual action is added to the control prior action before getting scaled and biased as per action space.
        self.actor.action_bias = torch.zeros_like(self.actor.action_bias)
        self.actor.action_scale = torch.ones_like(self.actor.action_scale)
        self.prior_action_bias = self.control_prior.actor.action_bias.detach()
        self.prior_action_scale = self.control_prior.actor.action_scale.detach()
        print(f"prior_action_bias: {self.prior_action_bias}, prior_action_scale: {self.prior_action_scale}")
        
        self.args.is_residual_rl = True
        self.critic_warming_up = (self.args.resrl_critic_burn_in_steps > 0)
        print(f"\n[Residual RL] Will warmup critic for first {self.args.resrl_critic_burn_in_steps} steps:")

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a_il  = self.control_prior.get_eval_action(obs)
            a_il_raw = (a_il - self.prior_action_bias) / self.prior_action_scale
            a_rl_raw = super().sample_action(torch.cat([obs, a_il_raw], dim=-1))
            mixed_action = (a_il_raw + a_rl_raw) * self.prior_action_scale + self.prior_action_bias
            return mixed_action

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        il_action = self.control_prior.get_eval_action(obs).detach()
        il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
        rl_action_raw = self.actor.get_eval_action(torch.cat([obs, il_action_raw], dim=-1))
        mixed_action = (il_action_raw + rl_action_raw) * self.prior_action_scale + self.prior_action_bias
        return mixed_action

    def load_pretrained(self, policy_path: str, value_path: str):
        # Skip loading pretrained policy for residual RL
        super().load_pretrained(None, value_path)


class TD3BaseResidualRLAgent(ResidualRLMixin, TD3Agent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=DeterministicResidualActor)

        # Ensure that the RL actor's initial outputs are zero by setting the last layer to be zeros
        self.actor: DeterministicResidualActor
        self.actor.net[-1].weight.data.zero_()
        self.actor.net[-1].bias.data.zero_()
        self.actor_target.net[-1].weight.data.zero_()
        self.actor_target.net[-1].bias.data.zero_()

    @torch.no_grad()
    def estimate_nstep_return(self, data: TrajReplayBufferSample) -> torch.Tensor:
        """
        Estimate the return of a trajectory chunk assuming data with shapes:
        data.next_obs: [horizon, batch_size, obs_dim]
        data.rewards: [horizon, batch_size, 1]

        Returns:
            R: [horizon, batch_size, 1]
        """
        # At final step, R = r + gamma * Q(next_obs, pi(next_obs))
        # At previous steps, R[t] = r + gamma * R[t+1]
        R = torch.zeros_like(data.rewards)

        next_state_actions_il = self.control_prior.get_eval_action(data.next_obs[-1]).detach()
        next_state_actions_il_raw = (next_state_actions_il - self.prior_action_bias) / self.prior_action_scale

        # target policy smoothing
        noise = (torch.randn_like(data.actions[-1]) * self.args.target_policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip
        )
        next_state_actions_rl_raw = (
            self.actor_target(
                torch.cat([data.next_obs[-1], next_state_actions_il_raw], dim=-1)
            )
            + noise
        ).clamp(
            self.action_l, self.action_h
        )

        next_state_mixed_actions = (next_state_actions_il_raw + next_state_actions_rl_raw) * self.prior_action_scale + self.prior_action_bias

        # double Q-learning
        min_q_next_target = self.Q(data.next_obs[-1], next_state_mixed_actions, target=True)[1]

        # data.dones is always assumed to be 0, according to args.bootstrap_at_done = "always"
        R[-1] = data.rewards[-1] + self.args.gamma * (min_q_next_target)

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        """
        Update the actor networks only if the critic has been trained for at least resrl_critic_burn_in_steps.
        This allows the critic to learn the value of the control prior policy while the RL actor's outputs are zero.
        We skip burn_in if the critic is pretrained.
        """
        if global_step < self.args.resrl_critic_burn_in_steps:
            return
        if self.critic_warming_up:
            self.critic_warming_up = False
            print(f"Critic warmup finished, now training actor also")

        il_action = self.control_prior.get_eval_action(data.obs).detach()
        il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
        rl_action_raw = self.actor(torch.cat([data.obs, il_action_raw], dim=-1))
        mixed_action = (il_action_raw + rl_action_raw) * self.prior_action_scale + self.prior_action_bias

        min_qf_pi = self.Q(data.obs, mixed_action)[1]
        actor_loss = -min_qf_pi.mean()

        if self.args.use_auxiliary_bc_loss:
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()

            il_action = self.control_prior.get_eval_action(offline_data.obs).detach()
            il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
            offline_action_raw = (offline_data.actions - self.prior_action_bias) / self.prior_action_scale
            desired_residual_action = offline_action_raw - il_action_raw

            bc_loss = F.mse_loss(self.actor(torch.cat([offline_data.obs, il_action_raw], dim=-1)), desired_residual_action)
            actor_loss = bc_loss_lam * actor_loss + bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()    

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            if self.args.use_auxiliary_bc_loss:
                self.logging_tracker["losses/bc_loss"] = bc_loss.item()
                self.logging_tracker["losses/bc_loss_lam"] = bc_loss_lam.item()

class SACBaseResidualRLAgent(ResidualRLMixin, SACAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=GaussianResidualActor)

        # Ensure that the RL actor's initial outputs are zero by setting the mean layer to be zeros
        self.actor: GaussianResidualActor
        self.actor.fc_mean.weight.data.zero_()
        self.actor.fc_mean.bias.data.zero_()
        
    @torch.no_grad()
    def estimate_nstep_return(self, data: TrajReplayBufferSample) -> torch.Tensor:
        """
        Estimate the return of a trajectory chunk assuming data with shapes:
        data.next_obs: [horizon, batch_size, obs_dim]
        data.rewards: [horizon, batch_size, 1]

        Returns:
            R: [horizon, batch_size, 1]
        """
        R = torch.zeros_like(data.rewards)
        next_state_actions_il = self.control_prior.get_eval_action(data.next_obs[-1]).detach()
        next_state_actions_il_raw = (next_state_actions_il - self.prior_action_bias) / self.prior_action_scale
        next_state_actions_rl_raw, next_state_log_pi, _ = self.actor(torch.cat([data.next_obs[-1], next_state_actions_il_raw], dim=-1))
        next_state_mixed_actions = (next_state_actions_il_raw + next_state_actions_rl_raw) * self.prior_action_scale + self.prior_action_bias

        # data.dones is always assumed to be 0, according to args.bootstrap_at_done = "always"
        R[-1] = data.rewards[-1] + self.args.gamma * self.Q(data.next_obs[-1], next_state_mixed_actions, target=True)[1]
        if self.args.critic_entropy:
            R[-1] -= self.args.gamma * self.alpha * next_state_log_pi

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        """
        Update the actor networks only if the critic has been trained for at least resrl_critic_burn_in_steps.
        This allows the critic to learn the value of the control prior policy while the RL actor's outputs are zero.
        We skip burn_in if the critic is pretrained.
        """
        if global_step < self.args.resrl_critic_burn_in_steps:
            return
        if self.critic_warming_up:
            self.critic_warming_up = False
            print(f"Critic warmup finished, now training actor also")

        il_action = self.control_prior.get_eval_action(data.obs).detach()
        il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
        rl_action_raw, log_pi, _ = self.actor(torch.cat([data.obs, il_action_raw], dim=-1))
        mixed_action = (il_action_raw + rl_action_raw) * self.prior_action_scale + self.prior_action_bias

        min_qf_pi = self.Q(data.obs, mixed_action)[1]
        actor_loss = -min_qf_pi.mean()
        
        if self.args.use_auxiliary_bc_loss:
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()

            il_action = self.control_prior.get_eval_action(offline_data.obs).detach()
            il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
            offline_action_raw = (offline_data.actions - self.prior_action_bias) / self.prior_action_scale
            desired_residual_action = offline_action_raw - il_action_raw

            bc_loss = F.mse_loss(self.actor.get_eval_action(torch.cat([offline_data.obs, il_action_raw], dim=-1)), desired_residual_action)
            actor_loss = bc_loss_lam * actor_loss + bc_loss

        if self.args.actor_entropy:
            actor_loss += self.alpha * log_pi.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.autotune:
            with torch.no_grad():
                il_action = self.control_prior.get_eval_action(data.obs).detach()
                il_action_raw = (il_action - self.prior_action_bias) / self.prior_action_scale
                _, log_pi, _ = self.actor(torch.cat([data.obs, il_action_raw], dim=-1))
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            self.logging_tracker["losses/alpha"] = self.alpha
            if self.args.autotune:
                self.logging_tracker["losses/alpha_loss"] = alpha_loss.item() 
            if self.args.use_auxiliary_bc_loss:
                self.logging_tracker["losses/bc_loss"] = bc_loss.item()
                self.logging_tracker["losses/bc_loss_lam"] = bc_loss_lam.item()  
    