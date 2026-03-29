import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import SACAgent, TD3Agent
from rainbow_demorl.agents.bc import make_bc_control_prior
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class TD3BaseIBRLAgent(TD3Agent):
    """
    The TD3 base Imitation Bootstrapped Reinforcement Learning (IBRL) agent.
    from paper: https://arxiv.org/pdf/2311.02198
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)
        self.control_prior = make_bc_control_prior(envs, device, args)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a_il  = self.control_prior.get_eval_action(obs)
            a_rl = super().sample_action(obs)

            q_val_il = self.Q(obs, a_il, random_sample_two_qf=True)[1]
            q_val_rl = self.Q(obs, a_rl, random_sample_two_qf=True)[1]

            return torch.where((q_val_il > q_val_rl), a_il, a_rl)
    
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

        next_actions_il = self.control_prior.get_eval_action(data.next_obs[-1])

        noise = (torch.randn_like(data.actions[-1]) * self.args.target_policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip
        )
        next_actions_rl = (self.actor_target(data.next_obs[-1]) + noise).clamp(
            self.action_l, self.action_h
        )

        min_q_il_next_target = self.Q(data.next_obs[-1], next_actions_il, target=True, random_sample_two_qf=True)[1]
        min_q_rl_next_target = self.Q(data.next_obs[-1], next_actions_rl, target=True, random_sample_two_qf=True)[1]
        
        R[-1] = data.rewards[-1] + self.args.gamma * torch.max(min_q_il_next_target, min_q_rl_next_target)

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R
    
class SACBaseIBRLAgent(SACAgent):
    """
    The SAC base Imintation Bootstrapping Reinforcement Learning (IBRL) agent.
    from paper: https://arxiv.org/pdf/2311.02198
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)
        self.control_prior = make_bc_control_prior(envs, device, args)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a_il = self.control_prior.get_eval_action(obs)
            a_rl = super().sample_action(obs)

            q_val_il = self.Q(obs, a_il, random_sample_two_qf=True)[1]
            q_val_rl = self.Q(obs, a_rl, random_sample_two_qf=True)[1] 

            return torch.where((q_val_il > q_val_rl), a_il, a_rl)

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
        next_actions_il = self.control_prior.get_eval_action(data.next_obs[-1])
        next_actions_rl, next_log_pi_rl, _ = self.actor(data.next_obs[-1])

        min_q_il_next_target = self.Q(data.next_obs[-1], next_actions_il, target=True, random_sample_two_qf=True)[1]
        min_q_rl_next_target = self.Q(data.next_obs[-1], next_actions_rl, target=True, random_sample_two_qf=True)[1]

        R[-1] = data.rewards[-1] + self.args.gamma * torch.max(min_q_il_next_target, min_q_rl_next_target)
        if self.args.critic_entropy:
            R[-1] -= self.args.gamma * self.alpha * next_log_pi_rl

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R
