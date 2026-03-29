from typing import Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import ActorCriticAgent
from rainbow_demorl.agents.actor_critic import SoftQNetwork
from rainbow_demorl.agents.actors import GaussianActor
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class SACAgent(ActorCriticAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, actor_class: Type[GaussianActor] = GaussianActor, qf_class: Type[SoftQNetwork] = SoftQNetwork):
        super().__init__(envs, device, args, actor_class=actor_class, qf_class=qf_class)

        # Automatic entropy tuning
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        else:
            self.alpha = args.alpha

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)[0]
    
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
        next_state_actions, next_state_log_pi, _ = self.actor(data.next_obs[-1])

        # data.dones is always assumed to be 0, according to args.bootstrap_at_done = "always"
        R[-1] = data.rewards[-1] + self.args.gamma * self.Q(data.next_obs[-1], next_state_actions, target=True)[1]
        if self.args.critic_entropy:
            R[-1] -= self.args.gamma * self.alpha * next_state_log_pi

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R
    
    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        pi, log_pi, _ = self.actor(data.obs)
        min_qf_pi = self.Q(data.obs, pi)[1]
        actor_loss = -min_qf_pi.mean()
        
        if self.args.use_auxiliary_bc_loss:
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()
            bc_loss = F.mse_loss(self.actor.get_eval_action(offline_data.obs), offline_data.actions)
            actor_loss = bc_loss_lam * actor_loss + bc_loss

        if self.args.actor_entropy:
            actor_loss += self.alpha * log_pi.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor(data.obs)
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
    
    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'log_alpha': self.log_alpha,
            'qfs': self.qfs.state_dict(),
            'qfs_target': self.qfs_target.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path)

        self.actor.load_state_dict(ckpt['actor'])
        self.qfs.load_state_dict(ckpt['qfs'])
        self.qfs_target.load_state_dict(ckpt['qfs_target'])
        self.log_alpha = ckpt['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        
        print(f"{self.__class__.__name__} model loaded from {model_path}")
    