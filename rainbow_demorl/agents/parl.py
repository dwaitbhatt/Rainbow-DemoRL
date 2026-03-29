import torch
import torch.nn.functional as F
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents import ACT_TD3Agent, SACAgent, TD3Agent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBufferSample


class PARLMixin:
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

        self.num_base_policy_actions = args.num_base_policy_actions
        self.num_actions_to_keep = args.num_actions_to_keep
        self.num_local_optimization_steps = args.num_local_optimization_steps
        self.local_optimization_step_size = args.local_optimization_step_size

    def _perform_local_optimization(self, obs: torch.Tensor, actions: torch.Tensor):
        '''
                obs: [batch_size, num_actions_to_keep, obs_dim]
            actions: [batch_size, num_actions_to_keep, action_dim]
            Returns: [batch_size, num_actions_to_keep, action_dim]
        '''
        actions_shape = actions.shape
        obs = obs.reshape(-1, self.args.obs_dim)             # [batch_size * num_actions_to_keep, obs_dim]
        actions = actions.reshape(-1, self.args.action_dim)  # [batch_size * num_actions_to_keep, action_dim]
        for _ in range(self.num_local_optimization_steps):
            actions = actions.detach().requires_grad_(True)
            qfs_a_values = self.Q(obs, actions)[0].mean(dim=0)             # [batch_size * num_actions_to_keep, 1]
            qfs_a_grads = torch.autograd.grad(qfs_a_values, actions,
                              grad_outputs=torch.ones_like(qfs_a_values)
                          )[0]                                             # [batch_size * num_actions_to_keep, action_dim]
            
            actions = actions + self.local_optimization_step_size * qfs_a_grads
            actions = actions.clamp(self.action_l, self.action_h)
        
        optimized_actions = actions.detach().requires_grad_(True)
        optimized_actions = optimized_actions.reshape(actions_shape[0], actions_shape[1], self.args.action_dim)
        return optimized_actions

    def _perform_global_optimization(self, obs: torch.Tensor, actions: torch.Tensor):
        '''
                obs: [batch_size, num_base_policy_actions, obs_dim]
            actions: [batch_size, num_base_policy_actions, action_dim]
            Returns: [batch_size, num_actions_to_keep, action_dim]
        '''
        qfs_a_values = self.Q(obs, actions)[0].mean(dim=0)                                         # [batch_size, num_base_policy_actions, 1]
        top_actions_indices = torch.topk(qfs_a_values, self.num_actions_to_keep, dim=1).indices    # [batch_size, num_actions_to_keep, 1]
        top_indices_expanded = top_actions_indices.expand(-1, -1, self.args.action_dim)            # [batch_size, num_actions_to_keep, action_dim]
        top_actions = actions.gather(1, top_indices_expanded)                                      # [batch_size, num_actions_to_keep, action_dim]
        return top_actions

    def get_improved_actions(self, obs: torch.Tensor):
        '''
            obs: [batch_size, obs_dim]
            Returns: expanded_obs: [batch_size, num_actions_to_keep, obs_dim]
                          actions: [batch_size, num_actions_to_keep, action_dim]
        '''
        expanded_obs = obs.unsqueeze(1).repeat(1, self.num_base_policy_actions, 1)        # [batch_size, num_base_policy_actions, obs_dim]

        expanded_obs_query = expanded_obs.reshape(-1, self.args.obs_dim)                  # [batch_size * num_base_policy_actions, obs_dim]
        actions = self.sample_different_actions(expanded_obs_query)                          # [batch_size * num_base_policy_actions, ?, action_dim]
        if len(actions.shape) == 3:
            # If actor predicts action chunk, we choose the first action
            actions = actions[:, 0, :]
        actions = actions.reshape(-1, self.args.num_base_policy_actions, self.args.action_dim)    # [batch_size, num_base_policy_actions, action_dim]

        global_optimized_actions = self._perform_global_optimization(expanded_obs, actions)       # [batch_size, num_actions_to_keep, action_dim]
        expanded_obs = expanded_obs[:, :self.num_actions_to_keep, :]                              # [batch_size, num_actions_to_keep, obs_dim]
        local_optimized_actions = self._perform_local_optimization(
            expanded_obs, 
            global_optimized_actions
        )                                                                                         # [batch_size, num_actions_to_keep, action_dim]

        return expanded_obs, local_optimized_actions

    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        ''' 
            Update the actor with the PARL objective: Predict several actions, filter by q-values, improve with gradient 
            ascent on q-function to get optimized action distribution. Finally perform BC to move actor to predict optimized actions.
            Assume the class using PARL Mixin has a compute_bc_loss() that returns a loss_dict with a "bc_loss" key
        ''' 
        obs = data.obs                                                              # [horizon, batch_size, obs_dim]
        obs = obs.reshape(-1, self.args.obs_dim)                                    # [horizon * batch_size, obs_dim]
        
        expanded_obs, improved_actions = self.get_improved_actions(obs)             # [horizon * batch_size, num_actions_to_keep, obs_dim/act_dim]
        q_vals = self.Q(expanded_obs, improved_actions)[0].mean(dim=0).squeeze()    # [horizon * batch_size, num_actions_to_keep]
        pi_opt = torch.softmax(q_vals, dim=1)                                       # [horizon * batch_size, num_actions_to_keep]
        if self.args.sample_from_pi_opt:
            inds = torch.multinomial(pi_opt, 1)                                     # [horizon * batch_size, 1]
        else:
            inds = torch.argmax(pi_opt, dim=1, keepdim=True)                        # [horizon * batch_size, 1]
        actions_for_distillation = improved_actions.gather(
            1, 
            inds.unsqueeze(1).expand(-1, -1, self.args.action_dim)
        ).squeeze(1)                                                                # [horizon * batch_size, action_dim]
        
        parl_loss_dict = self.compute_bc_loss(obs, actions_for_distillation)
        actor_loss = parl_loss_dict["bc_loss"]      

        if self.args.use_auxiliary_bc_loss:
            if offline_data is None:
                raise ValueError("Offline data is required when use_auxiliary_bc_loss is True")
            with torch.no_grad():
                bc_loss_lam = self.args.bc_loss_alpha / self.Q(offline_data.obs, offline_data.actions)[0].mean()
            offline_obs = offline_data.obs.reshape(-1, self.args.obs_dim)
            offline_actions = offline_data.actions.reshape(-1, self.args.action_dim)
            bc_loss = self.compute_bc_loss(offline_obs, offline_actions)["bc_loss"]
            actor_loss = bc_loss_lam * actor_loss + bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()
            for k, v in parl_loss_dict.items():
                if k != "bc_loss":
                    self.logging_tracker[f"losses/actor_loss_{k}"] = v.item()
            if self.args.use_auxiliary_bc_loss:
                self.logging_tracker["losses/bc_loss"] = bc_loss.item()
                self.logging_tracker["losses/bc_loss_lam"] = bc_loss_lam.item()


class PARL_TD3Agent(PARLMixin, TD3Agent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

    def update_actor(self, data: TrajReplayBufferSample, offline_data: TrajReplayBufferSample, global_step: int):
        super().update_actor(data, offline_data, global_step)

    def compute_bc_loss(self, obs: torch.Tensor, actions: torch.Tensor):
        bc_loss = F.mse_loss(self.actor(obs), actions)
        return {"bc_loss": bc_loss}

    def sample_different_actions(self, obs: torch.Tensor):
        return super().sample_action(obs)

class PARL_SACAgent(PARLMixin, SACAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

    def compute_bc_loss(self, obs: torch.Tensor, actions: torch.Tensor):
        bc_loss = F.mse_loss(self.actor.get_eval_action(obs), actions)
        return {"bc_loss": bc_loss}

    def sample_different_actions(self, obs: torch.Tensor):
        return super().sample_action(obs)


class PARL_ACTAgent(PARLMixin, ACT_TD3Agent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args)

    def sample_different_actions(self, obs: torch.Tensor):
        actions, (_, _) = self.actor.forward(obs, sample_different_actions=True)
        return actions