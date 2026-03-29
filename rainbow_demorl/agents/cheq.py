import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import SACAgent, TD3Agent
from rainbow_demorl.agents.actor_critic import SoftQNetwork
from rainbow_demorl.agents.bc import make_bc_control_prior
from rainbow_demorl.agents.actors import DeterministicActor, GaussianActor
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.layers import Ensemble, mlp
from rainbow_demorl.utils.math import soft_ce
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class DeterministicActorWithLambda(DeterministicActor):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        self.net = mlp(
            np.prod(envs.single_observation_space.shape) + 1, 
            [args.mlp_dim] * args.num_layers_actor,
            np.prod(envs.single_action_space.shape),
        )

class GaussianActorWithLambda(GaussianActor):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        self.backbone = mlp(
            np.prod(envs.single_observation_space.shape) + 1, 
            [args.mlp_dim] * (args.num_layers_actor - 1), 
            args.mlp_dim
        )
        self.fc_mean = nn.Linear(args.mlp_dim, np.prod(envs.single_action_space.shape))
        self.fc_logstd = nn.Linear(args.mlp_dim, np.prod(envs.single_action_space.shape))

class SoftQNetworkWithLambda(SoftQNetwork):
    def __init__(self, envs: ManiSkillVectorEnv, args: Args):
        super().__init__(envs, args)
        if args.use_ce_loss:
            self.net = Ensemble([mlp(
                                    np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape) + 1, 
                                    [args.mlp_dim] * args.num_layers_critic, 
                                    args.num_bins,
                                    dropout=args.q_dropout
                                ) for _ in range(args.num_critics)])
        else:
            self.net = Ensemble([mlp(
                                    np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape) + 1,
                                    [args.mlp_dim] * args.num_layers_critic,
                                    1,
                                ) for _ in range(args.num_critics)])

class CHEQTool:
    def __init__(self, args: Args):
        self.ulow = args.ulow
        self.uhigh = args.uhigh
        self.lam_low = args.lam_low
        self.lam_high = args.lam_high

    def inject_lambda_into_obs(self, obs: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, lam], dim=-1)
    
    def compute_u(self, qfs, obs_plus_lam: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_diff = qfs(obs_plus_lam, action)
        return torch.std(q_diff, dim=0, unbiased=False)
    
    def get_lam(self, u_val: float)-> torch.Tensor:
        lam_vals = torch.empty_like(u_val)
        lower_mask = (u_val <= self.ulow)
        upper_mask = (u_val >= self.uhigh)
        mid_mask   = ~ (lower_mask | upper_mask)

        lam_vals[lower_mask] = self.lam_high
        lam_vals[upper_mask] = self.lam_low

        mid_u = u_val[mid_mask]
        frac = (mid_u - self.uhigh)/(self.ulow - self.uhigh)  # in [0,1]
        lam_vals[mid_mask] = self.lam_low + frac*(self.lam_high - self.lam_low)
        return lam_vals  # shape [N,1]   
    
class TD3BaseCHEQAgent(TD3Agent):
    """
    The TD3 base Contextualized Hybrid Ensemble Q-learning (CHEQ) agent.
    from paper: https://arxiv.org/pdf/2406.19768
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=DeterministicActorWithLambda, qf_class=SoftQNetworkWithLambda)
        # Only the CHEQ agent's args should have is_cheq set to True, not the global args
        self.args = copy.deepcopy(args)
        self.args.is_cheq = True
        
        self.cheq_tool = CHEQTool(args)
        self.control_prior = make_bc_control_prior(envs, device, args)

        # initialize the lambda value with lambda start
        self.lam = torch.full((args.num_envs, 1), args.lam_start, device=device)

    def warmup_trajectory_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, self.lam)
            a_il = self.control_prior.get_eval_action(obs)
            a_rl = torch.tensor(self.envs.action_space.sample(), dtype=torch.float32, device=self.device)
            mixed_action = self.lam * a_rl + (1 - self.lam) * a_il

            return mixed_action, a_rl,  obs_plus_lam, self.lam
        
    def lambda_reset(self):
        self.lam = torch.full((self.args.num_envs, 1), self.args.lam_low, device=self.device)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, self.lam)
            a_rl = super().sample_action(obs_plus_lam)
            a_il = self.control_prior.get_eval_action(obs)

            mixed_action = self.lam * a_rl + (1 - self.lam) * a_il

            u_val = self.cheq_tool.compute_u(self.qfs, obs_plus_lam, a_rl)
            self.lam = self.cheq_tool.get_lam(u_val)

            return mixed_action, a_rl, obs_plus_lam, u_val, self.lam
        
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        if self.args.num_eval_envs != self.args.num_envs:
            eval_lam = torch.full((self.args.num_eval_envs, 1), float(self.lam.mean()), device=self.device)
        else: 
            eval_lam = self.lam
        obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, eval_lam)
        a_rl = super().get_eval_action(obs_plus_lam)
        return a_rl

    @torch.no_grad()
    def estimate_nstep_return(self, data: TrajReplayBufferSample) -> torch.Tensor:
        R = torch.zeros_like(data.rewards)
        
        noise = (torch.randn_like(data.actions[-1]) * self.args.target_policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip
        )
        next_a_rl = (self.actor_target(data.next_obs[-1]) + noise).clamp(
            self.action_l, self.action_h
        )

        min_q_next_target = self.Q(data.next_obs[-1], next_a_rl, target=True, random_close_qf=True)[1]

        R[-1] = data.rewards[-1] + self.args.gamma * (min_q_next_target)

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_critic(self, data: TrajReplayBufferSample, global_step: int):
        q_target = self.estimate_nstep_return(data)
        mask = torch.bernoulli(torch.full((len(self.qfs),), self.args.bernoulli_masking, device=self.device))

        if self.args.use_ce_loss:
            qfs_a_values = self.Q(data.obs, data.actions, logits=True)[0]
            qfs_total_loss = [m * soft_ce(qf_a_values, q_target, self.args).mean() for m, qf_a_values in zip(mask, qfs_a_values)]
        else:
            qfs_a_values = self.Q(data.obs, data.actions)[0]
            qfs_total_loss = [m * F.mse_loss(qf_a_values, q_target) for m, qf_a_values in zip(mask, qfs_a_values)]
        qf_loss = torch.stack(qfs_total_loss, dim=0).mean()

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_a_values, qf_total_loss) in enumerate(zip(qfs_a_values, qfs_total_loss), start=1):
                self.logging_tracker[f"losses/qf{i}_values"] = qf_a_values.mean().item()
                self.logging_tracker[f"losses/qf{i}_loss"] = qf_total_loss.item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item()

class SACBaseCHEQAgent(SACAgent):
    """
    The SAC base Contextualized Hybrid Ensemble Q-learning (CHEQ) agent.
    from paper: https://arxiv.org/pdf/2406.19768
    """

    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, actor_class=GaussianActorWithLambda, qf_class=SoftQNetworkWithLambda)
        self.args = copy.deepcopy(args)
        self.args.is_cheq = True
        
        self.cheq_tool = CHEQTool(args)
        self.control_prior = make_bc_control_prior(envs, device, args)

        # initialize the lambda value with lambda start
        self.lam = torch.full((args.num_envs, 1), args.lam_start, device=device)

    def warmup_trajectory_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, self.lam)
            a_il = self.control_prior.get_eval_action(obs)
            a_rl = torch.tensor(self.envs.action_space.sample(), dtype=torch.float32, device=self.device)
            mixed_action = self.lam * a_rl + (1 - self.lam) * a_il

            return mixed_action, a_rl, obs_plus_lam, self.lam
        
    def lambda_reset(self):
        self.lam = torch.full((self.args.num_envs, 1), self.args.lam_low, device=self.device)

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, self.lam)
            a_rl = super().sample_action(obs_plus_lam)
            a_il = self.control_prior.get_eval_action(obs)

            mixed_action = self.lam * a_rl + (1 - self.lam) * a_il

            u_val = self.cheq_tool.compute_u(self.qfs, obs_plus_lam, a_rl)
            self.lam = self.cheq_tool.get_lam(u_val)

            return mixed_action, a_rl, obs_plus_lam, u_val, self.lam
        
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        if self.args.num_eval_envs != self.args.num_envs:
            eval_lam = torch.full((self.args.num_eval_envs, 1), float(self.lam.mean()), device=self.device)
        else: 
            eval_lam = self.lam
        obs_plus_lam = self.cheq_tool.inject_lambda_into_obs(obs, eval_lam)
        a_rl = super().get_eval_action(obs_plus_lam)
        return a_rl
        
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
        next_a_rl, next_log_pi_rl, _ = self.actor(data.next_obs[-1])

        min_q_next_target = self.Q(data.next_obs[-1], next_a_rl, target=True, random_close_qf=True)[1]
        R[-1] = data.rewards[-1] + self.args.gamma * (min_q_next_target)

        if self.args.critic_entropy:
            R[-1] -= self.args.gamma * self.alpha * next_log_pi_rl

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_critic(self, data: TrajReplayBufferSample, global_step: int):
        q_target = self.estimate_nstep_return(data)
        mask = torch.bernoulli(torch.full((len(self.qfs),), self.args.bernoulli_masking, device=self.device))

        if self.args.use_ce_loss:
            qfs_a_values = self.Q(data.obs, data.actions, logits=True)[0]
            qfs_total_loss = [m * soft_ce(qf_a_values, q_target, self.args).mean() for m, qf_a_values in zip(mask, qfs_a_values)]
        else:
            qfs_a_values = self.Q(data.obs, data.actions)[0]
            qfs_total_loss = [m * F.mse_loss(qf_a_values, q_target) for m, qf_a_values in zip(mask, qfs_a_values)]
        qf_loss = torch.stack(qfs_total_loss, dim=0).mean()

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_a_values, qf_total_loss) in enumerate(zip(qfs_a_values, qfs_total_loss), start=1):
                self.logging_tracker[f"losses/qf{i}_values"] = qf_a_values.mean().item()
                self.logging_tracker[f"losses/qf{i}_loss"] = qf_total_loss.item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item()