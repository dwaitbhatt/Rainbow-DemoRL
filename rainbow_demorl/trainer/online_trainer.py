import os
import time
from collections import defaultdict
from typing import Optional, Union

import gymnasium as gym
import torch
import tqdm
from tensordict import TensorDict

from rainbow_demorl.agents import BaseAgent, MonteCarloQAgentReal
from rainbow_demorl.utils.common import Args, experiment_run_dir
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer

from .base_trainer import BaseTrainer


class OnlineTrainer(BaseTrainer):
    def __init__(self, 
                 args: Args, 
                 agent: Union[BaseAgent, MonteCarloQAgentReal],
                 envs: gym.Env,
                 eval_envs: gym.Env,
                 env_kwargs: dict,
                 device: torch.device, 
                 existing_writer = None,
                 initial_global_step = 0,
                 ):
        super().__init__(args, agent, envs, eval_envs, env_kwargs, device, existing_writer=existing_writer)

        # Set the initial global step to continue from where offline training left off
        self.initial_global_step = initial_global_step
        self.global_step = self.initial_global_step
        
        self.done = True
        self.learning_has_started = False

        # initial the lam_val with default lam_low
        self.final_lam_val = torch.full((self.args.num_envs, 1), self.args.lam_start, device=self.device)

        ## Setup replay buffer
        self.rb_online = TrajReplayBuffer(self.args, self.args.online_buffer_size)

        # Enable on-disk saving of the replay buffer if requested
        if self.args.save_buffer:
            buffer_save_dir = os.path.join("demos", self.args.robot, self.args.env_id, "rl_buffer", self.args.exp_name)
            self.rb_online.enable_saving(
                save_dir=buffer_save_dir,
                env_id=self.args.env_id,
                env_kwargs=self.env_kwargs,
                exp_name=self.args.exp_name,
            )

        if self.args.evaluate:
            self.rb_offline = None
            return

        if args.offline_buffer_type.lower() == "demos":
            assert self.args.demo_path is not None, "Offline buffer type 'demos' is only allowed when a demo type/path is provided"
            self.rb_offline = TrajReplayBuffer(self.args, self.args.offline_buffer_size)
            pad_repeat = self.args.act_num_queries if self.args.algorithm.startswith("ACT") else 0
            self.rb_offline.fill_from_demos(self.args.demo_path, pad_repeat=pad_repeat)
        elif args.offline_buffer_type.lower() == "rollout":
            assert self.args.pretrained_offline_policy_path is not None, "Offline buffer type 'rollout' is only allowed when finetuning a pretrained offline policy"
            self.rb_offline = TrajReplayBuffer(self.args, self.args.offline_buffer_size)
            # Temporarily disable CHEQ steps to rollout the offline policy
            is_cheq = self.agent.args.is_cheq
            self.agent.args.is_cheq = False
            # agent has actor already loaded from pretrained policy, so we can rollout the offline policy
            self.rollout_and_fill_traj_buffer(None, fill_offline_buffer=True)
            self.agent.args.is_cheq = is_cheq
        elif args.offline_buffer_type.lower() == "none":
            self.rb_offline = None
        else:
            raise ValueError(f"Offline buffer type {args.offline_buffer_type} not supported")

    def to_td(self, obs, num_envs, action=None, reward=None, terminated=None, truncated=None):
        device = "cpu"
        if isinstance(obs, dict): 
            obs = {k: v.unsqueeze(1).to(device) for k,v in obs.items()}
            obs = TensorDict(obs, batch_size=(), device=device)
        else:
            obs = obs.unsqueeze(1).to(device).float()
        if action is None:
            action = torch.full((num_envs, self.args.action_dim), float('nan'), device=device)
        else:
            action = action.to(device).float()
        if reward is None:
            reward = torch.full((num_envs,), float('nan'), device=device)
        else:
            reward = reward.to(device).float()
        if terminated is None:
            terminated = torch.full((num_envs,), float('nan'), device=device)
        else:
            terminated = terminated.to(device).float()
        if truncated is None:
            truncated = torch.full((num_envs,), float('nan'), device=device)
        else:
            truncated = torch.tensor([truncated]).to(device).float()
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(1),
            reward=reward.unsqueeze(1),
            terminated=terminated.unsqueeze(1),
            truncated=truncated.unsqueeze(1),
            ), batch_size=(num_envs, 1), device=device)
        return td

    def set_sim_cube_to_real_pos(self):
        import mani_skill.envs.utils.randomization as randomization
        from mani_skill.utils.structs.pose import Pose

        real_cube_rel_pos = self.envs.curr_obs["tcp_to_obj_pos"]
        sim_robot_eef_pos = self.envs.base_sim_env.agent.tcp_pose.p
        sim_cube_pos = sim_robot_eef_pos + real_cube_rel_pos

        qs = randomization.random_quaternions(1, lock_x=True, lock_y=True, lock_z=True)
        self.envs.base_sim_env.cube.set_pose(Pose.create_from_pq(sim_cube_pos, qs))

    def rollout_and_fill_traj_buffer(self, starting_obs: torch.Tensor, 
                                     fill_offline_buffer: bool = False):
        if fill_offline_buffer:
            target_buffer = self.rb_offline
            rollout_steps = self.args.offline_rollout_steps
        else:
            target_buffer = self.rb_online
            rollout_steps = self.args.steps_per_env
        obs = starting_obs

        if fill_offline_buffer:
            iterable = tqdm.tqdm(range(0, rollout_steps, self.args.num_envs), desc="Rollout and fill offline buffer")
        else:
            iterable = range(rollout_steps)
        for local_step in iterable:
            # print(f"[Rollout] local_step: {local_step}/{rollout_steps}")
            if self.done:
                if self.global_step > 0 or (fill_offline_buffer and local_step > 0):
                    if self.agent.args.is_cheq:
                        self.episode_tds.append(self.to_td(torch.cat([obs, self.final_lam_val], dim=1), self.args.num_envs))
                        if self.learning_has_started:
                            self.agent.lambda_reset()
                    else:
                        self.episode_tds.append(self.to_td(obs, self.args.num_envs))
                    tds = torch.cat(self.episode_tds, dim=1) # [num_envs, episode_len + 1, ..]
                    target_buffer.add(tds)
                obs, _ = self.envs.reset()
                self.set_sim_cube_to_real_pos()
                self.episode_tds = []
                self.agent.train_episode_timestep = -1

            if fill_offline_buffer:
                iterable.update(1)
            else:
                self.global_step += 1 * self.args.num_envs
                self.agent.train_episode_timestep += 1

            # ALGO LOGIC: put action logic here
            if self.agent.args.is_cheq:
                if not self.learning_has_started and not self.args.finetune_offline_policy:
                    mixed_actions, a_rl, obs_plus_lam, lam_val = self.agent.warmup_trajectory_action(obs)
                else:
                    mixed_actions, a_rl, obs_plus_lam, u_val, lam_val = self.agent.sample_action(obs)
                    a_rl = a_rl.detach()
                    mixed_actions = mixed_actions.detach()
                    self.logger.add_scalar("train/u", u_val.mean().item(), self.global_step)
                    self.logger.add_scalar("train/lam", lam_val.mean().item(), self.global_step)
                self.final_lam_val = lam_val
                actions = mixed_actions
            else:
                if not self.learning_has_started and not self.args.finetune_offline_policy:
                    actions = torch.tensor(self.envs.action_space.sample(), dtype=torch.float32, device=self.device)
                else:
                    actions = self.agent.sample_action(obs)
                    actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions.cpu())
            real_next_obs = next_obs.clone()

            if self.args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if self.args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    self.logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), self.global_step)

            dones = terminations | truncations
            self.done = dones[0]

            if self.agent.args.is_cheq:
                self.episode_tds.append(self.to_td(obs_plus_lam, self.args.num_envs, a_rl, rewards, terminations, truncations))
            else:
                self.episode_tds.append(self.to_td(obs, self.args.num_envs, actions, rewards, terminations, truncations))

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = real_next_obs

        return self.global_step, obs

    def train(self):
        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset(seed=self.args.seed) # in Gymnasium, seed is given to reset() instead of seed()
        
        global_steps_per_iteration = self.args.num_envs * (self.args.steps_per_env)
        
        total_steps = self.global_step + self.args.online_learning_timesteps
        print(f"Online training: starting from global_step={self.global_step}, total_steps={total_steps}")
        print(f"Online learning timesteps={self.args.online_learning_timesteps}")
        # Create progress bar that shows the actual global step progression
        self.pbar = tqdm.tqdm(
            range(self.global_step, total_steps),
            initial=self.global_step,
            total=total_steps,
            desc="Online Training"
        )

        self.episode_tds = []
        while self.global_step < total_steps:
            if self.args.eval_freq > 0 and (self.global_step - self.args.training_freq) // self.args.eval_freq < self.global_step // self.args.eval_freq:
                # evaluate
                self.evaluate()
                if self.args.evaluate:
                    break

            ### Collect samples from environments
            rollout_time = time.perf_counter()
            self.global_step, obs = self.rollout_and_fill_traj_buffer(obs)
            rollout_time = time.perf_counter() - rollout_time

            self.cumulative_times["rollout_time"] += rollout_time
            self.pbar.update(self.args.num_envs * self.args.steps_per_env)

            # ALGO LOGIC: training.
            if self.global_step < self.args.learning_starts + self.initial_global_step:
                continue

            # Ensure buffer has data before training
            if not self.rb_online.is_ready:
                print(f"Buffer not ready yet. Episodes: {self.rb_online.num_eps}, Global step: {self.global_step}")
                continue

            update_time = time.perf_counter()
            self.learning_has_started = True
            
            self.global_update = self.agent.update(self.rb_online, self.rb_offline, self.global_update, self.global_step)
            
            update_time = time.perf_counter() - update_time
            self.cumulative_times["update_time"] += update_time

            # Log training-related data
            if (self.global_step - self.args.training_freq) // self.args.log_freq < self.global_step // self.args.log_freq:
                self.agent.log_losses(self.logger, self.global_step)
                self.logger.add_scalar("time/update_time", update_time, self.global_step)
                self.logger.add_scalar("time/rollout_time", rollout_time, self.global_step)
                self.logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, self.global_step)
                for k, v in self.cumulative_times.items():
                    self.logger.add_scalar(f"time/total_{k}", v, self.global_step)
                self.logger.add_scalar("time/total_rollout+update_time", self.cumulative_times["rollout_time"] + self.cumulative_times["update_time"], self.global_step)

        if not self.args.evaluate and self.args.save_model:
            run_dir = experiment_run_dir(self.args)
            model_path = os.path.join(run_dir, "final_ckpt.pt")
            self.agent.save_model(model_path)
        
        print("Online training completed. Performing final cleanup...")
        self.cleanup()
