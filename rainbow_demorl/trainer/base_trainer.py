import os
import time
from collections import defaultdict

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from rainbow_demorl.agents import BaseAgent
from rainbow_demorl.utils.common import Args, Logger, experiment_run_dir


class BaseTrainer:
    def __init__(self, 
                 args: Args, 
                 agent: BaseAgent,
                 envs: gym.Env,
                 eval_envs: gym.Env,
                 env_kwargs: dict,
                 device: torch.device, 
                 existing_writer=None,
                 ):
        self.args = args
        self.device = device
        self.agent = agent
        self.envs = envs
        self.eval_envs = eval_envs
        self.env_kwargs = env_kwargs

        self.global_step = 0
        self.global_update = 0

        self.cumulative_times = defaultdict(float)
        self.best_return = float('-inf')  # Track best return for model saving
        self.setup_logger(existing_writer=existing_writer)

    def setup_logger(self, existing_writer=None):
        self.logger = None
        if not self.args.evaluate:
            print("\n\nRunning training")
            if self.args.track:
                import wandb

                # Check if wandb is already initialized
                if wandb.run is None:
                    config = vars(self.args)
                    config["env_cfg"] = dict(**self.env_kwargs, num_envs=self.args.num_envs, env_id=self.args.env_id, reward_mode="normalized_dense", env_horizon=self.args.env_horizon, partial_reset=self.args.partial_reset)
                    config["eval_env_cfg"] = dict(**self.env_kwargs, num_envs=self.args.num_eval_envs, env_id=self.args.env_id, reward_mode="normalized_dense", env_horizon=self.args.env_horizon, partial_reset=False)
                    wandb.init(
                        project=self.args.wandb_project_name,
                        entity=self.args.wandb_entity,
                        sync_tensorboard=False,
                        config=config,
                        name=self.args.exp_name,
                        save_code=True,
                        group=self.args.wandb_group,
                        tags=[self.args.algorithm.lower(), "walltime_efficient"]
                    )
                    print("Initialized new wandb run")
                else:
                    print("Reusing existing wandb run")
            if existing_writer is None:
                run_dir = experiment_run_dir(self.args)
                os.makedirs(run_dir, exist_ok=True)
                self.writer = SummaryWriter(run_dir)
                self.writer.add_text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
                )
            else:
                self.writer = existing_writer
            self.logger = Logger(log_wandb=self.args.track, tensorboard=self.writer)
        else:
            print("Running evaluation")

        if self.args.save_model and not self.args.evaluate:
            os.makedirs(experiment_run_dir(self.args), exist_ok=True)

    def cleanup(self, cleanup_envs=True, cleanup_buffers=True, cleanup_writer=True, cleanup_wandb=True):
        """Properly cleanup resources to prevent segmentation faults.
        
        Args:
            cleanup_envs: Whether to close environments
            cleanup_buffers: Whether to clear replay buffers
            cleanup_writer: Whether to close TensorBoard writer
            cleanup_wandb: Whether to finish wandb run
        """
        print("Cleaning up resources...")
        
        # 1. Close TensorBoard writer first (if requested)
        if cleanup_writer and hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                print(f"Warning: Error closing writer: {e}")
        
        # 2. Clear replay buffers if they exist (if requested)
        if cleanup_buffers:
            if hasattr(self, 'rb_online'):
                try:
                    del self.rb_online
                except Exception as e:
                    print(f"Warning: Error clearing online buffer: {e}")
                    
            if hasattr(self, 'rb_offline'):
                try:
                    del self.rb_offline
                except Exception as e:
                    print(f"Warning: Error clearing offline buffer: {e}")
        
        # 3. Force CUDA synchronization
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Error synchronizing CUDA: {e}")
        
        # 4. Close environments last (they contain GPU resources) - if requested
        if cleanup_envs:
            if hasattr(self, 'eval_envs') and self.eval_envs is not None:
                try:
                    self.eval_envs.close()
                except Exception as e:
                    print(f"Warning: Error closing eval envs: {e}")
                    
            if hasattr(self, 'envs') and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    print(f"Warning: Error closing training envs: {e}")
        
        # 5. Finish wandb run (if requested)
        if cleanup_wandb and self.args.track:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Error finishing wandb run: {e}")
        
        # 6. Force garbage collection
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error during garbage collection: {e}")
        
        # 7. Small delay to ensure CUDA operations complete
        try:
            import time
            time.sleep(0.5)  # Give CUDA operations time to complete
        except Exception as e:
            print(f"Warning: Error during cleanup delay: {e}")
        

    def evaluate(self):
        self.agent.actor.eval()
        stime = time.perf_counter()
        eval_obs, _ = self.eval_envs.reset()
        self.agent.eval_episode_timestep = 0
        eval_metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(self.args.num_eval_steps):
            with torch.no_grad():
                eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = self.eval_envs.step(self.agent.get_eval_action(eval_obs))
                self.agent.eval_episode_timestep += 1
                if "episode" in eval_infos:
                    for k, v in eval_infos["episode"].items():
                        eval_metrics[k].append(v)
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
        
        eval_metrics_mean = {}
        for k, v in eval_metrics.items():
            if len(v) > 0:
                # Take the last recorded value for each environment (most recent state)
                last_values = v[-1]
                mean = last_values.float().mean()
                eval_metrics_mean[k] = mean
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/{k}", mean, self.global_step)
        
        # Handle case where no metrics were collected
        if 'success_once' not in eval_metrics_mean:
            eval_metrics_mean['success_once'] = torch.tensor(0.0)
        if 'return' not in eval_metrics_mean:
            eval_metrics_mean['return'] = torch.tensor(0.0)
            
        self.pbar.set_description(
            f"success_once: {eval_metrics_mean['success_once']:.2f}, "
            f"return: {eval_metrics_mean['return']:.2f}"
        )
        if self.logger is not None:
            eval_time = time.perf_counter() - stime
            self.cumulative_times["eval_time"] += eval_time
            self.logger.add_scalar("time/eval_time", eval_time, self.global_step)
        self.agent.actor.train()

        # Check if current return is better than best return so far
        current_return = eval_metrics_mean.get('return', float('-inf'))
        if current_return > self.best_return:
            previous_best = self.best_return
            self.best_return = current_return
            if self.args.save_model and not self.args.evaluate:
                # Delete previous best model
                previous_best_model_path = f"{self.args.save_model_dir}/{self.args.env_id}/{self.args.robot}/{self.args.exp_name}/best_model_ret_{previous_best:.2f}.pt"
                if os.path.exists(previous_best_model_path):
                    os.remove(previous_best_model_path)
                best_model_path = f"{self.args.save_model_dir}/{self.args.env_id}/{self.args.robot}/{self.args.exp_name}/best_model_ret_{self.best_return:.2f}.pt"
                self.agent.save_model(best_model_path)
                print(f"[Step {self.global_step}] New best model saved! Return: {current_return:.2f} (previous best: {previous_best:.2f})")

        if self.args.save_model and not self.args.evaluate:
            model_path = f"{self.args.save_model_dir}/{self.args.env_id}/{self.args.robot}/{self.args.exp_name}/ckpt_{self.global_step}.pt"
            self.agent.save_model(model_path)

        return eval_metrics_mean

