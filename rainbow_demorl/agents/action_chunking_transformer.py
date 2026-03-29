from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents.td3 import TD3Agent
from rainbow_demorl.agents.actors import ACTActor
from rainbow_demorl.agents.actors.action_chunking_transformer_network import (build_encoder,
                                                      build_transformer)
from rainbow_demorl.agents.bc import BCAgent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer, TrajReplayBufferSample


class ACTMixin:
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, None)
        assert len(envs.single_observation_space.shape) == 1 # (obs_dim,)
        assert len(envs.single_action_space.shape) == 1 # (act_dim,)

        self.kl_weight = args.act_kl_weight
        self.state_dim = envs.single_observation_space.shape[0]
        self.act_dim = envs.single_action_space.shape[0]
        self.num_queries = args.act_num_queries

        # CNN backbone
        backbones = None

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.actor = ACTActor(
            envs,
            args,
            backbones,
            transformer,
            encoder,
            self.state_dim,
            self.act_dim,
            self.num_queries,
        ).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.act_lr)

        # Only used during evaluation
        if args.act_temporal_agg:
            self.query_frequency = 1
        else:
            self.query_frequency = self.num_queries
        self.norm_stats = None

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

    def compute_bc_loss(self, obs: torch.Tensor, action_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
            obs: [batch_size, obs_dim]
            action_seq: [batch_size, num_queries, action_dim] or [batch_size, action_dim]
            Returns: 
                loss_dict: dict(str: torch.Tensor)
        '''
        action_seq_shape = action_seq.shape
        if len(obs.shape) == len(action_seq_shape):
            # If action sequence length is 1, expand tensor dim for compatibility
            assert (
                self.num_queries == 1
            ), f"Only 1 action was predicted per obs for ACT loss computation, but num_queries is {self.num_queries}"

            action_seq = action_seq[:, None, :]             # [batch_size, 1, action_dim]
        # forward pass
        a_hat, (mu, logvar) = self.actor(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        total_loss = l1 + total_kld[0] * self.kl_weight

        loss_dict = {
            "bc_loss": total_loss,
            "l1": l1,
            "kl": total_kld[0]
        }

        return loss_dict

    def _pre_process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.norm_stats['state_mean']) / self.norm_stats['state_std']

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        '''
            obs: [batch_size, obs_dim]
            Returns: 
                action: [batch_size, action_dim]
        '''
        if obs.shape[0] == self.args.num_envs:
            # Called during online interaction (via sample_action)
            num_envs = self.args.num_envs
            max_steps = self.args.env_horizon
            ts = self.train_episode_timestep
            buffer_attr = '_train'
        elif obs.shape[0] == self.args.num_eval_envs:
            # Called during evaluation
            num_envs = self.args.num_eval_envs
            max_steps = self.args.num_eval_steps
            ts = self.eval_episode_timestep
            buffer_attr = '_eval'
        elif len(obs.shape) == 3:
            raise ValueError(f"Unsupported observation shape for ACT inference: {obs.shape}, must be [batch_size, obs_dim]")
        
        if ts == 0:
            if self.norm_stats is None:
                assert self.args.norm_stats is not None, "Norm stats must be provided for ACT inference"
                self.norm_stats = self.args.norm_stats
            all_time_actions_buffer = torch.zeros([num_envs, max_steps, max_steps+self.num_queries, self.act_dim], device=self.device)
            setattr(self, f'{buffer_attr}_all_time_actions', all_time_actions_buffer)
            actions_to_take_buffer = torch.zeros([num_envs, self.num_queries, self.act_dim], device=self.device)
            setattr(self, f'{buffer_attr}_actions_to_take', actions_to_take_buffer)

        all_time_actions = getattr(self, f'{buffer_attr}_all_time_actions')
        actions_to_take = getattr(self, f'{buffer_attr}_actions_to_take')

        obs = self._pre_process_obs(obs)
        if ts % self.query_frequency == 0:
            action_seq = self.actor.get_eval_action(obs)  # (num_envs, num_queries, action_dim)

        if self.args.act_temporal_agg:
            assert self.query_frequency == 1, "query_frequency != 1 has not been implemented for temporal_agg==True."
            all_time_actions[:, ts, ts:ts+self.num_queries] = action_seq # (num_envs, num_queries, act_dim)
            actions_for_curr_step = all_time_actions[:, :, ts] # (num_envs, max_timesteps, act_dim)
            actions_populated = torch.zeros(max_steps, dtype=torch.bool, device=self.device) # (max_timesteps,)
            actions_populated[max(0, ts + 1 - self.num_queries):ts+1] = True
            actions_for_curr_step = actions_for_curr_step[:, actions_populated] # (num_envs, num_populated, act_dim)
            k = 0.01
            if ts < self.num_queries:
                exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step[0]), device=self.device)) # (num_populated,)
                exp_weights = exp_weights / exp_weights.sum() # (num_populated,)
                exp_weights = torch.tile(exp_weights, (num_envs, 1)) # (num_envs, num_populated)
                exp_weights = torch.unsqueeze(exp_weights, -1) # (num_envs, num_populated, 1)
                setattr(self, f'{buffer_attr}_exp_weights', exp_weights)
            exp_weights = getattr(self, f'{buffer_attr}_exp_weights')
            action = (actions_for_curr_step * exp_weights).sum(dim=1)  # (num_envs, act_dim)
        else:
            if ts % self.query_frequency == 0:
                actions_to_take[:] = action_seq
            action = actions_to_take[:, ts % self.query_frequency]

        return action

    def sample_first_action(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        '''
            Sample first action from ACT's prediction (no temporal aggregation)
            Useful for updating the actor/critic during RL finetuning
            obs: [batch_size, obs_dim] or [horizon, batch_size, obs_dim]
            Returns: 
                first_action: [batch_size, action_dim] or [horizon, batch_size, action_dim]
        '''
        if target:
            actor = self.actor_target
        else:
            actor = self.actor

        obs_shape = obs.shape
        if len(obs_shape) == 3:
            obs = obs.reshape(-1, obs_shape[-1])
        obs = self._pre_process_obs(obs)
        action_seq = actor.get_eval_action(obs)  # [..., num_queries, action_dim]
        if len(obs_shape) == 3:
            action_seq = action_seq.reshape(obs_shape[0], obs_shape[1], self.num_queries, self.act_dim)
            first_action = action_seq[:, :, 0, :]
        else:
            first_action = action_seq[:, 0, :]
        return first_action                      # [..., action_dim]


class ACT_BCAgent(ACTMixin, BCAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        if not args.is_online:
            # Skip check when used as control prior during online training
            assert (
                args.offline_horizon == args.act_num_queries
            ), "While training offline, args.offline_horizon must be equal to args.act_num_queries to sample action_seq of correct length from the buffer"
        super().__init__(envs, device, args)

    def update_actor(self, offline_data: TrajReplayBufferSample, global_step: int):
        obs = offline_data.obs[0, :, :]                    # First observation from trajectory: (batch_size, obs_dim)
        obs = self._pre_process_obs(obs)
        action_seq = offline_data.actions.transpose(0, 1)  # Action sequence from trajectory: (batch_size, offline_horizon, act_dim)
        loss_dict = self.compute_bc_loss(obs, action_seq)
        total_loss = loss_dict["bc_loss"]
        
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        
        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = total_loss.item()
            self.logging_tracker["losses/l1"] = loss_dict["l1"].item()
            self.logging_tracker["losses/kl"] = loss_dict["kl"].item()

    def save_model(self, model_path: str):
        torch.save({'actor': self.actor.state_dict(), 'norm_stats': self.norm_stats}, model_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.norm_stats = checkpoint['norm_stats']
        print(f"{self.__class__.__name__} model loaded from {model_path}")


class ACT_TD3Agent(ACTMixin, TD3Agent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

        self.actor_target = deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # This ensures that the offline buffer is filled from demos, and norm_stats are calculated.
        self.args.offline_buffer_type = "demos"

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ 
            Used for collecting experience during online training
            obs: [batch_size, obs_dim]
            Returns: action: [batch_size, action_dim]
        """
        # Ensure norm_stats is initialized (same logic as get_eval_action)
        if self.norm_stats is None:
            assert self.args.norm_stats is not None, "Norm stats must be provided for ACT inference"
            self.norm_stats = self.args.norm_stats
            
        with torch.no_grad():
            actions = self.get_eval_action(obs)
            # Add exploration noise
            noise = torch.randn_like(actions) * self.args.exploration_noise
            actions = (actions + noise).clamp(self.action_l, self.action_h)
        return actions

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
        next_state_actions = (self.sample_first_action(data.next_obs[-1], target=True) + noise).clamp(
            self.action_l, self.action_h
        )

        # double Q-learning
        min_q_next_target = self.Q(data.next_obs[-1], next_state_actions, target=True)[1]

        # data.dones is always assumed to be 0, according to args.bootstrap_at_done = "always"
        R[-1] = data.rewards[-1] + self.args.gamma * (min_q_next_target)

        for t in reversed(range(self.args.horizon - 1)):
            R[t] = data.rewards[t] + self.args.gamma * R[t+1]

        return R

    def update_actor(self, data: TrajReplayBufferSample, offline_data: Optional[TrajReplayBufferSample], global_step: int):
        min_qf_pi = self.Q(data.obs, self.sample_first_action(data.obs))[1]
        actor_loss = -min_qf_pi.mean()

        if self.args.use_auxiliary_bc_loss: 
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            obs = offline_data.obs[0, :, :]                    # First observation from trajectory: (batch_size, obs_dim)
            obs = self._pre_process_obs(obs)
            action_seq = offline_data.actions.transpose(0, 1)  # Action sequence from trajectory: (batch_size, horizon, act_dim)
            aux_loss_dict = self.compute_bc_loss(obs, action_seq)
            bc_loss = aux_loss_dict["bc_loss"]

            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()
            actor_loss = bc_loss_lam * actor_loss + bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()    

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            if self.args.use_auxiliary_bc_loss:
                self.logging_tracker["losses/bc_loss"] = bc_loss.item()
                self.logging_tracker["losses/bc_loss_lam"] = bc_loss_lam.item()
                self.logging_tracker["losses/bc_loss_l1"] = aux_loss_dict["l1"].item()
                self.logging_tracker["losses/bc_loss_kl"] = aux_loss_dict["kl"].item()


    def update(self, rb_online: TrajReplayBuffer, rb_offline: TrajReplayBuffer, global_update: int, global_step: int) -> int:
        """
        Update the actor and critic networks as per the algorithm.
        Difference from parent (actor_critic.py):
        - Sampling offline_data_for_auxbc with args.act_num_queries as horizon for ACT auxiliary BC loss.
        """
        for local_update in range(self.args.grad_steps_per_iteration):
            global_update += 1

            if rb_online is not None:
                online_data = rb_online.sample()
            if rb_offline is not None:
                # Sampled with args.horizon to allow prefill (use_offline_data_for_rl)
                offline_data = rb_offline.sample()
                # Sampled with args.act_num_queries for auxiliary BC
                offline_data_for_auxbc = rb_offline.sample(custom_horizon=self.args.act_num_queries)
                if self.args.is_cheq:
                    H, B, _ = offline_data.obs.shape
                    lam_offline = torch.full((H, B, 1), self.args.lam_buffer, device=offline_data.obs.device, dtype=offline_data.obs.dtype)
                    offline_data.obs = torch.cat([offline_data.obs, lam_offline], dim=-1)
                    offline_data.next_obs = torch.cat([offline_data.next_obs, lam_offline], dim=-1)
            else: 
                offline_data = None
                offline_data_for_auxbc = None

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
                self.update_actor(data, offline_data_for_auxbc, global_step)
            if global_update % self.args.target_network_frequency == 0:
                self.update_target_networks()
        return global_update

    def load_pretrained(self, policy_path: str, value_path: str):
        """
        Load a pretrained policy and value function from checkpoint files.
        """
        if policy_path is not None:
            ckpt = torch.load(policy_path)
            actor_ckpt = ckpt['actor']

            actor_ckpt.pop('pos_table')
            query_embed_weight_ckpt = actor_ckpt.pop('query_embed.weight')
            self.actor.load_state_dict(actor_ckpt, strict=False)

            m = min(self.num_queries, query_embed_weight_ckpt.shape[0])
            self.actor.query_embed.weight.data[:m] = query_embed_weight_ckpt[:m]
            self.norm_stats = ckpt['norm_stats']
        super().load_pretrained(None, value_path)

    def save_model(self, model_path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'qfs': self.qfs.state_dict(),
            'qfs_target': self.qfs_target.state_dict(),
            'norm_stats': self.norm_stats,
        }, model_path)
        print(f"model saved to {model_path}")

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path)

        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.qfs.load_state_dict(ckpt['qfs'])
        self.qfs_target.load_state_dict(ckpt['qfs_target'])
        self.norm_stats = ckpt['norm_stats']

        print(f"{self.__class__.__name__} model loaded from {model_path}")


class ACT_ControlPrior(ACT_BCAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sample_first_action(obs)