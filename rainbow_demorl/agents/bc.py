from typing import Type, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rainbow_demorl.agents.actors import DeterministicActor, GaussianActor, NormalizedActor
from rainbow_demorl.agents.base_agent import BaseAgent
from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.replay_buffer_traj import TrajReplayBuffer, TrajReplayBufferSample


class BCAgent(BaseAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args, actor_class: Optional[Type[NormalizedActor]]):
        super().__init__(envs, device, args)
        if actor_class is not None:
            self.actor = actor_class(envs, args).to(device)
            if args.checkpoint is not None:
                self.load_model(args.checkpoint)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        else:
            self.actor: NormalizedActor = None

    def update(self, rb_online: TrajReplayBuffer, rb_offline: TrajReplayBuffer, global_update: int, global_step: int) -> int:
        global_update += 1
        offline_data = rb_offline.sample()
        self.update_actor(offline_data, global_step)
        return global_update

    def update_actor(self, offline_data: TrajReplayBufferSample, global_step: int):
        raise NotImplementedError
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save_model(self, model_path: str):
        torch.save({'actor': self.actor.state_dict()}, model_path)
        print(f"model saved to {model_path}")
    
    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor'])
        print(f"{self.__class__.__name__} model loaded from {model_path}")


class BCDeterministicAgent(BCAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, DeterministicActor)
        self.actor: DeterministicActor

    def update_actor(self, offline_data: TrajReplayBufferSample, global_step: int):
        pred_actions = self.actor(offline_data.obs)
        actor_loss = F.mse_loss(pred_actions, offline_data.actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    
class BCGaussianAgent(BCAgent):
    def __init__(self, envs: ManiSkillVectorEnv, device: torch.device, args: Args):
        super().__init__(envs, device, args, GaussianActor)
        self.actor: GaussianActor

    def update_actor(self, offline_data: TrajReplayBufferSample, global_step: int):
        # action_log_probs = self.actor.get_log_probs(data.obs, data.actions)
        # actor_loss = -action_log_probs.mean()
        actor_loss = F.mse_loss(self.actor.get_eval_action(offline_data.obs), offline_data.actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
            self.logging_tracker["losses/actor_loss"] = actor_loss.item()

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.get_eval_action(obs)


def make_bc_control_prior(envs: ManiSkillVectorEnv, device: torch.device, args: Args):
    if args.control_prior_type == "BC_DET":
        cp = BCDeterministicAgent(envs, device, args)
    elif args.control_prior_type == "BC_GAUSS":
        cp = BCGaussianAgent(envs, device, args)
    elif args.control_prior_type == "ACT":
        from rainbow_demorl.agents.action_chunking_transformer import ACT_ControlPrior

        cp = ACT_ControlPrior(envs, device, args)
    else:
        raise ValueError(f"Invalid control_prior_type: {args.control_prior_type}")
    cp.load_model(args.control_prior_path)
    return cp
