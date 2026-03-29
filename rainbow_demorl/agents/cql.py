from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import SACAgent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.math import soft_ce
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class CQLAgent(SACAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

        self.args.critic_entropy = False
        self.args.actor_entropy = True
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.cql_actor_lr)
        
        # Automatic CQL regularization weight tuning
        if args.cql_autotune:
            self.log_cql_alpha = torch.tensor(np.log(args.cql_alpha_init), requires_grad=True, device=device)
            self.cql_alpha = self.log_cql_alpha.exp()
            self.cql_alpha_optimizer = optim.Adam([self.log_cql_alpha], lr=args.lr)
        else:
            self.cql_alpha = args.cql_alpha
            self.log_cql_alpha = torch.tensor(np.log(args.cql_alpha), requires_grad=False, device=device)
        self.cql_lagrange_tau = args.cql_lagrange_tau

        self.log_action_prob_inv = torch.log(torch.prod(self.action_h - self.action_l))
    
    def cql_penalty(self, data: TrajReplayBufferSample, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the importance-weighted, autotuned CQL penalty term: eq (30) in https://arxiv.org/pdf/2006.04779
        """
        obs = data.obs.reshape(-1, 1, self.args.obs_dim)    # [n_obs, 1, obs_dim]
        n_obs = obs.shape[0]                                # n_obs = horizon * batch_size
        n_cql_act = self.args.cql_num_actions
        obs = obs.repeat(1, n_cql_act, 1)                   # [n_obs, n_cql_act, obs_dim]

        if self.args.cql_variant in ["cql-h", "cql_h"]:
            unif_actions = torch.rand(n_obs, n_cql_act, self.args.action_dim, device=self.device)
            unif_actions = unif_actions * (self.action_h - self.action_l) + self.action_l
            
            qfs_qvals_unif = self.Q(obs, unif_actions)[0]                                                     # [num_critics, n_obs, n_cql_act, 1]
            qfs_qvals_unif_imp_sampled = qfs_qvals_unif + self.log_action_prob_inv                            # [num_critics, n_obs, n_cql_act, 1]

            pol_curr_actions, pol_curr_log_pi, _ = self.actor(obs)                                            # [n_obs, n_cql_act, action_dim]
            qfs_qvals_pol_curr = self.Q(obs, pol_curr_actions)[0]                                             # [num_critics, n_obs, n_cql_act, 1]
            qfs_qvals_pol_curr_imp_sampled = qfs_qvals_pol_curr - pol_curr_log_pi                             # [num_critics, n_obs, n_cql_act, 1]
            
            qfs_qvals_all_imp_sampled = torch.cat([qfs_qvals_unif_imp_sampled, 
                                                   qfs_qvals_pol_curr_imp_sampled], dim=-2).squeeze(-1)       # [num_critics, n_obs, 2*n_cql_act]
            qfs_logsumexp_q = torch.logsumexp(qfs_qvals_all_imp_sampled, dim=-1) - float(np.log(2*n_cql_act)) # [num_critics, n_obs]
            qfs_qvals_mu = qfs_logsumexp_q                                                                    # [num_critics, n_obs]

        elif self.args.cql_variant in ["cql-rho", "cql_rho"]:
            pol_curr_actions = self.actor(obs)[0]                                                           # [n_obs, n_cql_act, action_dim]
            qfs_qvals_pol_curr = self.Q(obs, pol_curr_actions)[0]                                           # [num_critics, n_obs, n_cql_act, 1]
            qfs_pol_weighted = qfs_qvals_pol_curr * torch.softmax(qfs_qvals_pol_curr, dim=-2)               # [num_critics, n_obs, n_cql_act, 1]
            qfs_qvals_mu = qfs_pol_weighted.mean(dim=-2).squeeze(-1)                                        # [num_critics, n_obs]

        qfs_qvals_beh = self.Q(data.obs, data.actions)[0]                                                   # [num_critics, horizon, batch_size, 1]
        qfs_qvals_beh = qfs_qvals_beh.reshape(qfs_qvals_beh.shape[0], -1)                                   # [num_critics, n_obs]
        
        cql_penalties = (qfs_qvals_mu - qfs_qvals_beh).mean(dim=-1)                                         # [num_critics]
        if self.args.cql_autotune:
            cql_penalties = cql_penalties - self.cql_lagrange_tau

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_mu, qf_beh, qf_cql_penalty) in enumerate(zip(qfs_qvals_mu, qfs_qvals_beh, cql_penalties), start=1):
                self.logging_tracker[f"losses/cql_mu_q_{i}"] = qf_mu.mean().item()
                self.logging_tracker[f"losses/cql_data_q_{i}"] = qf_beh.mean().item()
                self.logging_tracker[f"losses/cql_penalty_{i}"] = qf_cql_penalty.item()
            self.logging_tracker["losses/cql_mu_q_mean"] = qfs_qvals_mu.mean().item()
            self.logging_tracker["losses/cql_data_q_mean"] = qfs_qvals_beh.mean().item()
            self.logging_tracker["losses/cql_penalty_mean"] = cql_penalties.mean().item()

        return cql_penalties

    def update_critic(self, data: TrajReplayBufferSample, global_step: int):
        """
        Update the critic networks with n-step returns and CE loss if specified.
        Values are estimated for all steps in the trajectory chunk with k<=n-step returns.
        """

        q_target = self.estimate_nstep_return(data)

        if self.args.use_ce_loss:
            qfs_a_values = self.Q(data.obs, data.actions, logits=True)[0]
            qfs_td_loss = [soft_ce(qf_a_values, q_target, self.args).mean() for qf_a_values in qfs_a_values]
        else:
            qfs_a_values = self.Q(data.obs, data.actions)[0]
            qfs_td_loss = [F.mse_loss(qf_a_values, q_target) for qf_a_values in qfs_a_values]

        cql_penalties = self.cql_penalty(data, global_step)
        qfs_total_loss = [self.cql_alpha * cql_penalty + qf_loss for cql_penalty, qf_loss in zip(cql_penalties, qfs_td_loss)]
        qf_loss = torch.stack(qfs_total_loss, dim=0).sum() / len(self.qfs)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.args.cql_autotune:
            self.cql_alpha = torch.clip(self.log_cql_alpha.exp(), min=0.0, max=1000000.0)
            qfs_cql_alpha_loss = [-self.cql_alpha * q_penalty.detach() for q_penalty in cql_penalties]
            cql_alpha_loss = torch.stack(qfs_cql_alpha_loss, dim=0).sum() / len(self.qfs)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
            self.cql_alpha = self.log_cql_alpha.exp().clamp(min=0.0, max=1000000.0).item()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_a_values, qf_loss) in enumerate(zip(qfs_a_values, qfs_total_loss), start=1):
                self.logging_tracker[f"losses/qf{i}_values"] = qf_a_values.mean().item()
                self.logging_tracker[f"losses/qf{i}_loss"] = qf_loss.item()
            self.logging_tracker["losses/qf_values_mean"] = qf_a_values.mean().item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item()
            self.logging_tracker["losses/cql_alpha"] = self.cql_alpha
            if self.args.cql_autotune:
                self.logging_tracker["losses/cql_alpha_loss"] = cql_alpha_loss.item()
    
    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'qfs': self.qfs.state_dict(),
            'qfs_target': self.qfs_target.state_dict(),
            'log_alpha': self.log_alpha,
            'log_cql_alpha': self.log_cql_alpha,
            'cql_lagrange_tau': self.cql_lagrange_tau,
        }, model_path)
        print(f"model saved to {model_path}")

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.qfs.load_state_dict(checkpoint['qfs'])
        self.qfs_target.load_state_dict(checkpoint['qfs_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.log_cql_alpha = checkpoint['log_cql_alpha']
        self.cql_lagrange_tau = checkpoint['cql_lagrange_tau']
        self.alpha = self.log_alpha.exp().item()
        self.cql_alpha = self.log_cql_alpha.exp().item()
        print(f"{self.__class__.__name__} model loaded from {model_path}")
    

class CalQLAgent(CQLAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

    def cql_penalty(self, data: TrajReplayBufferSample, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the CalQL penalty term following official implementation at 
        https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac.py#L205
        """
        obs = data.obs.reshape(-1, 1, self.args.obs_dim)              # [n_obs, 1, obs_dim]
        n_obs = obs.shape[0]                                          # n_obs = horizon * batch_size
        n_cql_act = self.args.cql_num_actions
        obs = obs.repeat(1, n_cql_act, 1)                             # [n_obs, n_cql_act, obs_dim]

        next_obs = data.next_obs.reshape(-1, 1, self.args.obs_dim)    # [n_obs, 1, obs_dim]
        next_obs = next_obs.repeat(1, n_cql_act, 1)                   # [n_obs, n_cql_act, obs_dim]

        lower_bounds = data.mc_return.reshape(-1, 1, 1)               # [n_obs, 1, 1]
        lower_bounds = lower_bounds.repeat(1, n_cql_act, 1)           # [n_obs, n_cql_act, 1]

        # CalQL penalty based on CQL-H
        # [n_obs, n_cql_act, action_dim]
        unif_actions = torch.rand(n_obs, n_cql_act, self.args.action_dim, device=self.device)
        unif_actions = unif_actions * (self.action_h - self.action_l) + self.action_l
        pol_curr_actions, pol_curr_log_pi, _ = self.actor(obs)
        pol_next_actions, pol_next_log_pi, _ = self.actor(next_obs)

        # [num_critics, n_obs, n_cql_act, 1]
        qfs_qvals_unif = self.Q(obs, unif_actions)[0]
        qfs_qvals_pol_curr = self.Q(obs, pol_curr_actions)[0]
        qfs_qvals_pol_next = self.Q(obs, pol_next_actions)[0]

        # Calculate what percentage of the q-values are below the lower bounds
        num_qvals = qfs_qvals_pol_curr[0].numel()
        qfs_bound_rate_pol_curr = [torch.sum(qf_qvals_pol_curr < lower_bounds) / num_qvals for qf_qvals_pol_curr in qfs_qvals_pol_curr]
        qfs_bound_rate_pol_next = [torch.sum(qf_qvals_pol_next < lower_bounds) / num_qvals for qf_qvals_pol_next in qfs_qvals_pol_next]
        qfs_bound_rate_all = [(qf_bound_rate_pol_curr + qf_bound_rate_pol_next) / 2.0
                              for qf_bound_rate_pol_curr, qf_bound_rate_pol_next in zip(qfs_bound_rate_pol_curr, qfs_bound_rate_pol_next)]
        
        # Replace the q-values below the lower bounds with the lower bounds
        qfs_qvals_pol_curr = torch.max(qfs_qvals_pol_curr, lower_bounds)
        qfs_qvals_pol_next = torch.max(qfs_qvals_pol_next, lower_bounds)

        # [num_critics, n_obs, n_cql_act, 1]
        qfs_qvals_unif_imp_sampled = qfs_qvals_unif + self.log_action_prob_inv
        qfs_qvals_curr_imp_sampled = qfs_qvals_pol_curr - pol_curr_log_pi
        qfs_qvals_next_imp_sampled = qfs_qvals_pol_next - pol_next_log_pi

        qfs_qvals_all_imp_sampled = torch.cat([qfs_qvals_unif_imp_sampled, 
                                               qfs_qvals_curr_imp_sampled, 
                                               qfs_qvals_next_imp_sampled], dim=-2).squeeze(-1)                # [num_critics, n_obs, 3*n_cql_act]
        qfs_logsumexp_q = torch.logsumexp(qfs_qvals_all_imp_sampled, dim=-1) - float(np.log(3*n_cql_act))      # [num_critics, n_obs]
        qfs_qvals_mu = qfs_logsumexp_q

        qfs_qvals_beh = self.Q(data.obs, data.actions)[0]                                                      # [num_critics, horizon, batch_size, 1]
        qfs_qvals_beh = qfs_qvals_beh.reshape(qfs_qvals_beh.shape[0], -1)                                      # [num_critics, n_obs]
        
        cql_penalties = (qfs_qvals_mu - qfs_qvals_beh).mean(dim=-1)                                            # [num_critics]
        if self.args.cql_autotune:
            cql_penalties = cql_penalties - self.cql_lagrange_tau

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_mu, qf_beh, cql_penalty, qf_bound_rate) in enumerate(zip(qfs_qvals_mu, qfs_qvals_beh, cql_penalties, qfs_bound_rate_all)):
                self.logging_tracker[f"losses/cql_mu_q_{i}"] = qf_mu.mean().item()
                self.logging_tracker[f"losses/cql_data_q_{i}"] = qf_beh.mean().item()
                self.logging_tracker[f"losses/cql_penalty_{i}"] = cql_penalty.item()
                self.logging_tracker[f"losses/cql_bound_rate_all_{i}"] = qf_bound_rate.item()
            self.logging_tracker["losses/cql_mu_q_mean"] = qfs_qvals_mu.mean().item()
            self.logging_tracker["losses/cql_data_q_mean"] = qfs_qvals_beh.mean().item()
            self.logging_tracker["losses/cql_bound_rate_all_mean"] = sum(br.item() for br in qfs_bound_rate_all) / len(qfs_bound_rate_all)
            self.logging_tracker["losses/cql_penalty_mean"] = cql_penalties.mean().item()

        return cql_penalties