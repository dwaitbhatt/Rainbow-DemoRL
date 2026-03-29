from random import random

import torch
import torch.nn.functional as F
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import SACAgent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.math import soft_ce
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class MonteCarloQAgent(SACAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)
        self.args.critic_entropy = False
        self.args.actor_entropy = False

    def update_critic(self, data: TrajReplayBufferSample, global_step: int):
        """
        Update the critic networks with n-step returns and CE loss if specified.
        Values are estimated for all steps in the trajectory chunk with k<=n-step returns.
        """ 
        if random() < self.args.mcq_bootstrap_epsilon:
            q_target = self.estimate_nstep_return(data)
        else:
            q_target = data.mc_return

        if self.args.use_ce_loss:
            qfs_a_values = self.Q(data.obs, data.actions, logits=True)[0]
            qfs_total_loss = [soft_ce(qf_a_values, q_target, self.args).mean() for qf_a_values in qfs_a_values]
        else:
            qfs_a_values = self.Q(data.obs, data.actions)[0]
            qfs_total_loss = [F.mse_loss(qf_a_values, q_target) for qf_a_values in qfs_a_values]
        qf_loss = torch.stack(qfs_total_loss, dim=0).sum() / len(self.qfs)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            for i, (qf_a_values, qf_loss) in enumerate(zip(qfs_a_values, qfs_total_loss), start=1):
                self.logging_tracker[f"losses/qf{i}_values"] = qf_a_values.mean().item()
                self.logging_tracker[f"losses/qf{i}_loss"] = qf_loss.item()
            self.logging_tracker["losses/qf_values_mean"] = qf_a_values.mean().item()
            self.logging_tracker["losses/qf_loss"] = qf_loss.item()
