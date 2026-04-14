import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

from rainbow_demorl.utils.common import Args
from rainbow_demorl.utils.dataset import ManiSkillTrajectoryDataset
from rainbow_demorl.utils.math import nanstd


@dataclass
class TrajReplayBufferSample:
    """
    Sample of trajectory slices of length horizon.
    Contains obs, next_obs, action, reward, mc_return.
    Shapes: [horizon, batch_size, ...]
    """
    obs: torch.Tensor       # [horizon, batch_size, obs_dim]
    next_obs: torch.Tensor  # [horizon, batch_size, obs_dim]
    actions: torch.Tensor   # [horizon, batch_size, action_dim]
    rewards: torch.Tensor   # [horizon, batch_size, 1]
    mc_return: torch.Tensor # [horizon, batch_size, 1]

    def cat(self, other: 'TrajReplayBufferSample', dim: int = 1):
        """
        Concatenate two samples along the given dimension.
        """
        return TrajReplayBufferSample(
            obs=torch.cat([self.obs, other.obs], dim=dim),
            next_obs=torch.cat([self.next_obs, other.next_obs], dim=dim),
            actions=torch.cat([self.actions, other.actions], dim=dim),
            rewards=torch.cat([self.rewards, other.rewards], dim=dim),
            mc_return=torch.cat([self.mc_return, other.mc_return], dim=dim),
        )

class TrajReplayBuffer:
    """
    Replay buffer based on torchrl adapted from TD-MPC2.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, args: Args, buffer_size: int, horizon: Optional[int] = None, device: Optional[torch.device] = None):
        self.args = args
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device
        self._capacity = buffer_size
        self._sampler = SliceSampler(
            num_slices=self.args.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self.horizon = horizon if horizon is not None else args.horizon
        self._batch_size = args.batch_size * (self.horizon+1)
        self._num_eps = 0
        self._discount_matrix_cache = {}
        self._buffer = None  # Initialize to None, will be created when first episode is added
        # Saving support
        self._save_dir: Optional[str] = None
        self._h5_path: Optional[str] = None
        self._json_path: Optional[str] = None
        self._next_episode_id: int = 0

        self.norm_stats = self.args.norm_stats  # Normalization statistics for offline dataset

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity
    
    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps
    
    @property
    def is_ready(self):
        """Return whether the buffer is ready for sampling."""
        return self._buffer is not None and self._num_eps > 0

    def _reserve_buffer(self, storage):
        """Reserve a buffer with the given storage."""
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            # prefetch=int(self.args.num_envs / self.args.grad_steps_per_iteration),
            # batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        
        # Get available GPU memory
        if torch.cuda.is_available():
            mem_free, mem_total = torch.cuda.mem_get_info()
            print(f'Available GPU memory: {mem_free/1e9:.2f} GB / {mem_total/1e9:.2f} GB')
        else:
            mem_free = float('inf')
            print('CUDA not available, using CPU memory')
        
        # Calculate memory requirements more conservatively
        bytes_per_step = sum([
                (v.numel()*v.element_size() if not isinstance(v, TensorDict) \
                else sum([x.numel()*x.element_size() for x in v.values()])) \
            for v in tds.values()
        ]) / len(tds)
        total_bytes = bytes_per_step*self._capacity
        print(f'Storage required: {total_bytes/1e9:.2f} GB')
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = 'cuda' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (arg.to(device, non_blocking=True) \
            if arg is not None else None for arg in args)
    
    def clean_sample(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor):
        nan_mask = torch.isnan(obs).any(dim=-1) | torch.isnan(next_obs).any(dim=-1) | torch.isnan(action).any(dim=-1) | torch.isnan(reward).any(dim=-1)
        inf_mask = torch.isinf(reward).any(dim=-1)
        large_mask = (obs > 1e4).any(dim=-1) | (next_obs > 1e4).any(dim=-1) | (action > 1e4).any(dim=-1)
        nan_or_large_indices = torch.where(nan_mask | large_mask | inf_mask)[0] 
        if len(nan_or_large_indices) > 0:
            problems = []
            if nan_mask.any():
                problems.append(f"{nan_mask.sum().item()} NaN")
            if inf_mask.any():
                problems.append(f"{inf_mask.sum().item()} Inf")
            if large_mask.any():
                problems.append(f"{large_mask.sum().item()} Large")

            problem_str = ", ".join(problems)
            print(f"[CleanSamples] Detected {problem_str} → converted to zero")

            obs[nan_or_large_indices] = 0
            next_obs[nan_or_large_indices] = 0
            action[nan_or_large_indices] = 0
            reward[nan_or_large_indices] = 0
        return obs, next_obs, action, reward

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size (H+1)xB.
        """
        # NOTE: Ensure that while adding to the buffer, we're adding the obs (and not next_obs), action, reward
        obs_all = td['obs']
        obs = obs_all[:-1]
        next_obs = obs_all[1:]
        action = td['action'][:-1]
        reward = td['reward'][:-1].unsqueeze(-1)
        mc_return = td['mc_return'][:-1].unsqueeze(-1)
        obs, next_obs, action, reward, mc_return = self._to_device(obs, next_obs, action, reward, mc_return)
        obs, next_obs, action, reward = self.clean_sample(obs, next_obs, action, reward)

        return TrajReplayBufferSample(obs=obs, next_obs=next_obs, actions=action, rewards=reward, mc_return=mc_return)

    def sample(self, custom_horizon: Optional[int] = None):
        """Sample a batch of subsequences from the buffer."""
        if not self.is_ready:
            raise RuntimeError(f"Cannot sample from empty buffer. Buffer has {self._num_eps} episodes. Add some episodes first.")
        horizon = self.horizon
        total_batch_size = self._batch_size
        if custom_horizon is not None:
            horizon = custom_horizon
            total_batch_size = self.args.batch_size * (custom_horizon + 1)
        td = self._buffer.sample(total_batch_size).view(-1, horizon+1).permute(1, 0)
        return self._prepare_batch(td)

    def _get_discount_matrix(self, horizon, gamma):
        """Get or create the discount matrix for given horizon and gamma."""
        cache_key = (horizon, gamma)
        if cache_key not in self._discount_matrix_cache:
            discount_matrix = torch.zeros(horizon, horizon, device='cpu')
            
            i_indices, j_indices = torch.meshgrid(
                torch.arange(horizon, device='cpu'),
                torch.arange(horizon, device='cpu'),
                indexing='ij'
            )
            
            mask = j_indices >= i_indices
            discount_matrix[mask] = gamma ** (j_indices[mask] - i_indices[mask])
            
            self._discount_matrix_cache[cache_key] = discount_matrix
        
        return self._discount_matrix_cache[cache_key]

    def calculate_mc_return(self, rewards):
        """
        Calculate the Monte Carlo return for each transition.
        rewards: [batch_size, horizon+1]
        Returns: [batch_size, horizon+1]
        """
        gamma = self.args.gamma
        horizon = rewards.shape[1] - 1
        
        discount_matrix = self._get_discount_matrix(horizon, gamma)
        rewards_reshaped = rewards[:,:-1]
        # [batch_size, horizon] @ [horizon, horizon] = [batch_size, horizon]
        returns = rewards_reshaped @ discount_matrix.T
        
        # Return corrresponding to the last transition (final_obs, nan, nan) is nan
        returns = torch.cat([returns, torch.full_like(rewards[:,:1], float('nan'))], dim=1)
        
        return returns

    def add(self, td: TensorDict):
        """
        Add an episode to the buffer. Ensure that each transition is (obs, action, reward),
        with the last transition of an episode being (final_obs, nan, nan).
        td[num_env, episode_len+1, ...], where ... = act_dim, obs_dim, None
        """
        # Check that the last transition of an episode is (final_obs, nan, nan)
        assert len(td['reward'].shape) == 2, f"Expected 2D tensor for reward, got {td['reward'].shape}"
        assert torch.isnan(td['reward'][:,-1]).all(), f"Expected nan at last transition, got {td['reward'][:,-1]}"
        assert torch.isnan(td['action'][:,-1]).all(), f"Expected nan at last transition, got {td['action'][:,-1]}"
        
        td['mc_return'] = self.calculate_mc_return(td['reward']) # [num_env, episode_len+1]
        
        for _td in td:
            _td['episode'] = torch.ones_like(_td['reward'], dtype=torch.int64) * self._num_eps
            if self._num_eps == 0:
                self._buffer = self._init(_td)
            self._buffer.extend(_td)
            # Save to disk if enabled (one file per episode)
            if self._save_dir is not None:
                self._save_episode_to_disk(_td, self._next_episode_id)
                self._next_episode_id += 1
            self._num_eps += 1
        return self._num_eps

    def _estimate_norm_stats(self):
        obs_tensor = self._buffer['obs']
        action_tensor = self._buffer['action']

        state_mean = torch.nanmean(obs_tensor, dim=0, keepdim=True).detach()
        state_std = nanstd(obs_tensor, dim=0, keepdim=True).clamp_min(1e-6).detach()
        action_mean = torch.nanmean(action_tensor, dim=0).detach()
        action_std = nanstd(action_tensor, dim=0).clamp_min(1e-6).detach()

        self._norm_stats = {
            'state_mean': state_mean,
            'state_std': state_std,
            'action_mean': action_mean,
            'action_std': action_std,
        }
        self.args.norm_stats = self._norm_stats

    def fill_from_demos(self, demos_path: str, pad_repeat: int = 0):
        ds = ManiSkillTrajectoryDataset(demos_path, device='cpu', load_count=self._capacity)

        _tds = []
        curr_eps_id = 0
        for i in range(len(ds) - 1):
            out = ds[i]
            if out["eps_id"] != curr_eps_id:
                curr_eps_id = out["eps_id"]
                if pad_repeat > 0:
                    obs_pad = _tds[-1]["obs"].repeat(1, pad_repeat, 1)

                    nan_action = _tds[-1]["action"].clone()
                    _tds[-1]["action"] = _tds[-2]["action"].clone()
                    # action_pad = _tds[-1]["action"].repeat(1, pad_repeat-1, 1)
                    action_pad = torch.zeros_like(_tds[-1]["action"]).repeat(1, pad_repeat-1, 1)
                    action_pad = torch.cat([action_pad, nan_action], dim=1)

                    nan_reward = _tds[-1]["reward"].clone()
                    _tds[-1]["reward"] = _tds[-2]["reward"].clone()
                    reward_pad = _tds[-1]["reward"].repeat(1, pad_repeat-1)
                    reward_pad = torch.cat([reward_pad, nan_reward], dim=1)
                    _tds.append(TensorDict(
                        {
                            "obs": obs_pad,
                            "action": action_pad,
                            "reward": reward_pad,
                        }, batch_size=(1, pad_repeat)
                    ))
                tds = torch.cat(_tds, dim=1)
                self.add(tds)
                _tds = []
            _tds.append(TensorDict(
                {
                    "obs": out["obs"].unsqueeze(0).unsqueeze(0),
                    "action": out["action"].unsqueeze(0).unsqueeze(0),
                    "reward": out["reward"].unsqueeze(0).unsqueeze(0),
                }, batch_size=(1, 1)
            ))
        self._estimate_norm_stats()
        
    # -----------------------------
    # Saving utilities
    # -----------------------------
    def enable_saving(self, save_dir: str, env_id: str, env_kwargs: Dict[str, Any], exp_name: str):
        """
        Enable on-the-fly saving of each added episode into a single HDF5 file
        with a side JSON index matching ManiSkill's demonstration format.
        """
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir

        # Decide file names to mirror the demo format
        sim_backend = env_kwargs.get("sim_backend", "gpu")
        h5_name = f"trajectory.state.{self.args.control_mode}.physx_{sim_backend}.h5"
        json_name = h5_name.replace(".h5", ".json")
        self._h5_path = os.path.join(save_dir, h5_name)
        self._json_path = os.path.join(save_dir, json_name)

        # Initialize H5 file if it doesn't exist
        if not os.path.exists(self._h5_path):
            with h5py.File(self._h5_path, 'w') as f:
                f.attrs['created_at'] = int(time.time())
                f.attrs['exp_name'] = exp_name

        # Initialize JSON index with env_info and empty episodes
        json_payload = {
            "env_info": {
                "env_id": env_id,
                "env_kwargs": env_kwargs,
            },
            "episodes": []
        }
        if not os.path.exists(self._json_path):
            with open(self._json_path, 'w') as jf:
                json.dump(json_payload, jf, indent=2)
        else:
            try:
                with open(self._json_path, 'r') as jf:
                    existing = json.load(jf)
                json_payload = existing
                # Set next episode id based on current count
                self._next_episode_id = len(json_payload.get("episodes", []))
            except Exception:
                pass

    def _write_obs(self, grp: h5py.Group, obs_any: Any):
        """Write observations as either a single dataset 'obs' (tensor/state) or a group 'obs' (dict-like)."""
        if torch.is_tensor(obs_any):
            # Plain state tensor -> store directly as dataset 'obs'
            data = obs_any.detach().to('cpu').numpy()
            grp.create_dataset('obs', data=data, compression="gzip")
            return
        if isinstance(obs_any, TensorDict) or isinstance(obs_any, dict):
            # Dict-like -> create group and write each key
            obs_grp = grp.create_group('obs')
            items = obs_any.items() if isinstance(obs_any, dict) else obs_any.items()
            for k, v in items:
                arr = v.detach().to('cpu').numpy() if torch.is_tensor(v) else np.asarray(v)
                obs_grp.create_dataset(str(k), data=arr, compression="gzip")
            return
        # Fallback: try numpy into single dataset
        grp.create_dataset('obs', data=np.asarray(obs_any), compression="gzip")

    def _save_episode_to_disk(self, ep_td: TensorDict, eps_id: int):
        """Append a single episode to HDF5 and update the JSON episodes list."""
        assert self._h5_path is not None and self._json_path is not None

        # Extract sequences; drop last timestep for action/reward/terminated/truncated
        obs_seq = ep_td['obs']              # [T+1, ...]
        act_seq = ep_td['action'][:-1]      # [T, act_dim]
        rew_seq = ep_td['reward'][:-1]      # [T]
        term_seq = ep_td.get('terminated', torch.full_like(ep_td['reward'], float('nan')))[:-1]
        trunc_seq = ep_td.get('truncated', torch.full_like(ep_td['reward'], float('nan')))[:-1]

        # Write to HDF5 under group traj_<episode_id>
        with h5py.File(self._h5_path, 'a') as f:
            gname = f"traj_{eps_id}"
            if gname in f:
                del f[gname]
            g = f.create_group(gname)
            # obs
            self._write_obs(g, obs_seq)
            # actions
            g.create_dataset('actions', data=act_seq.detach().to('cpu').numpy(), compression="gzip")
            # rewards
            g.create_dataset('rewards', data=rew_seq.detach().to('cpu').numpy(), compression="gzip")
            # terminated / truncated as uint8
            g.create_dataset('terminated', data=term_seq.detach().to('cpu').to(torch.uint8).numpy(), compression="gzip")
            g.create_dataset('truncated', data=trunc_seq.detach().to('cpu').to(torch.uint8).numpy(), compression="gzip")

        # Prepare episode metadata similar to ManiSkill's demo format
        T = int(act_seq.shape[0])
        term_seq = torch.nan_to_num(term_seq, nan=0.0)
        trunc_seq = torch.nan_to_num(trunc_seq, nan=0.0)
        # Check success at end of episode
        success_val = bool(term_seq[-1].item() > 0.5) if term_seq.numel() > 0 else False
        episode_meta = {
            "episode_id": int(eps_id),
            "episode_seed": int(self.args.seed) if hasattr(self.args, 'seed') else None,
            "control_mode": self.args.control_mode,
            "elapsed_steps": T,
            "reset_kwargs": {
                "options": {},
                "seed": int(self.args.seed) if hasattr(self.args, 'seed') else None,
            },
            "success": success_val,
        }

        # Update JSON index
        try:
            with open(self._json_path, 'r') as jf:
                index = json.load(jf)
        except Exception:
            index = {"env_info": {}, "episodes": []}
        episodes = index.get("episodes", [])
        episodes.append(episode_meta)
        index["episodes"] = episodes
        with open(self._json_path, 'w') as jf:
            json.dump(index, jf, indent=2)
