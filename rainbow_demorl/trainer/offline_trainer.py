import copy
import time

import gymnasium as gym
import torch
import tqdm

from rainbow_demorl.agents import BaseAgent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer

from .base_trainer import BaseTrainer


class OfflineTrainer(BaseTrainer):
    def __init__(self, 
                 args: Args, 
                 agent: BaseAgent,
                 envs: gym.Env,
                 eval_envs: gym.Env,
                 env_kwargs: dict,
                 device: torch.device, 
                 ):
        super().__init__(args, agent, envs, eval_envs, env_kwargs, device)

        ## Set offline learning parameters
        offline_args = copy.deepcopy(self.args)
        offline_args.horizon = self.args.offline_horizon
        offline_args.training_freq = 1
        offline_args.policy_frequency = 1
        offline_args.target_network_frequency = 1
        offline_args.utd = 1
        offline_args.grad_steps_per_iteration = 1
        offline_args.eval_freq = min(10000, self.args.eval_freq)
        self.args = offline_args

        ## Load replay buffer from demonstrations
        self.rb_offline = TrajReplayBuffer(self.args, self.args.offline_buffer_size, self.args.offline_horizon, device=self.device)
        if not self.args.evaluate:
            pad_repeat = 0
            if self.args.algorithm == "ACT":
                pad_repeat = self.args.act_num_queries
            self.rb_offline.fill_from_demos(self.args.demo_path, pad_repeat=pad_repeat)

        # Update agent args after Replay Buffer updates norm stats
        self.agent.args = self.args 

    def train(self):
        self.pbar = tqdm.tqdm(
            range(self.args.offline_learning_grad_steps),
            desc="Offline Training"
        )
        
        while self.global_step < self.args.offline_learning_grad_steps:
            if self.args.eval_freq > 0 and (self.global_step - self.args.training_freq) // self.args.eval_freq < self.global_step // self.args.eval_freq:
                # evaluate
                self.evaluate()
                if self.args.evaluate:
                    break

            update_time = time.perf_counter()
            self.global_update = self.agent.update(None, self.rb_offline, self.global_update, self.global_step)
            update_time = time.perf_counter() - update_time

            self.cumulative_times["offline_update_time"] += update_time
            
            self.global_step += 1
            self.pbar.update(1)

            # Log training-related data
            if (self.global_step - self.args.training_freq) // self.args.log_freq < self.global_step // self.args.log_freq:
                self.agent.log_losses(self.logger, self.global_step)
                self.logger.add_scalar("time/offline_update_time", update_time, self.global_step)
                for k, v in self.cumulative_times.items():
                    self.logger.add_scalar(f"time/total_{k}", v, self.global_step)

        if not self.args.evaluate and self.args.save_model:
            model_path = f"{self.args.save_model_dir}/{self.args.env_id}/{self.args.robot}/{self.args.exp_name}/final_ckpt.pt"
            self.agent.save_model(model_path)
        
        # Use selective cleanup based on learning mode
        if self.args.learning_mode == "offline_to_online":
            print("Offline training completed. Preserving resources for online training phase...")
            self.cleanup(cleanup_envs=False, cleanup_buffers=False, cleanup_writer=False, cleanup_wandb=False)
        else:
            # For pure offline mode, clean up everything
            self.cleanup()
