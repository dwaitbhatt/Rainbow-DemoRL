from typing import Type

import torch
import torch.nn.functional as F
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents.actor_critic import ActorCriticAgent
from rainbow_demorl.agents.actor_critic import SoftQNetwork
from rainbow_demorl.agents.actors import DeterministicActor
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class TD3Agent(ActorCriticAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    from paper: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, actor_class: Type[DeterministicActor] = DeterministicActor, qf_class: Type[SoftQNetwork] = SoftQNetwork):
        super().__init__(envs, device, args, actor_class=actor_class, qf_class=qf_class)
        if actor_class is not None:
            self.actor_target = actor_class(envs, args).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            self.actor_target: DeterministicActor = None
        self.args.policy_frequency = 2

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.actor(obs)
            # Add exploration noise
            noise = torch.randn_like(actions) * self.args.exploration_noise
            actions = (actions + noise).clamp(self.action_l, self.action_h)
        return actions
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.get_eval_action(obs)
    
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
        # target policy smoothing
        noise = (torch.randn_like(data.actions[-1]) * self.args.target_policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip
        )
        next_state_actions = (self.actor_target(data.next_obs[-1]) + noise).clamp(
            self.action_l, self.action_h
        )

        #double Q-learning
        min_q_next_target = self.Q(data.next_obs[-1], next_state_actions, target=True)[1]

        # data.dones is always assumed to be 0, according to args.bootstrap_at_done = "always"
        R[-1] = data.rewards[-1] + self.args.gamma * (min_q_next_target)

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        min_qf_pi = self.Q(data.obs, self.actor(data.obs))[1]
        actor_loss = -min_qf_pi.mean()
        
        if self.args.use_auxiliary_bc_loss: 
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()
            bc_loss = F.mse_loss(self.actor(offline_data.obs), offline_data.actions)
            actor_loss = bc_loss_lam * actor_loss + bc_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()    

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            if self.args.use_auxiliary_bc_loss:
                self.logging_tracker["losses/bc_loss"] = bc_loss.item()
                self.logging_tracker["losses/bc_loss_lam"] = bc_loss_lam.item()

    def update_target_networks(self):
        super().update_target_networks()
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def load_pretrained(self, policy_path: str, value_path: str):
        """
        Load a pretrained policy and value function from checkpoint files.
        """
        super().load_pretrained(policy_path, value_path)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'qfs': self.qfs.state_dict(),
            'qfs_target': self.qfs_target.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path)

        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.qfs.load_state_dict(ckpt['qfs'])
        self.qfs_target.load_state_dict(ckpt['qfs_target'])

        print(f"{self.__class__.__name__} model loaded from {model_path}")
