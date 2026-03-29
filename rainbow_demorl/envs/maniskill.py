import os
from typing import Dict, Optional

import gymnasium as gym
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.gym_utils import find_max_episode_steps_value
from mani_skill.utils.registration import register_env
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import wandb
from rainbow_demorl.utils.common import Args, experiment_run_dir


@register_env("PickCubeCustom-v1", max_episode_steps=100)
class PickCubeCustomEnv(PickCubeEnv):
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp_pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs


@register_env("PickCubeCustomNoGrasp-v1", max_episode_steps=100)
class PickCubeCustomNoGraspEnv(PickCubeEnv):
    def _get_obs_extra(self, info: Dict):
        # obs = dict(
        #     # is_grasped=info["is_grasped"],
        #     # tcp_pose=self.agent.tcp_pose.raw_pose,
        #     # goal_pos=self.goal_site.pose.p,
        # )
        obs = dict()
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs


@register_env("PickCubeCustomNoisy-v1", max_episode_steps=100)
class PickCubeCustomNoisyEnv(PickCubeEnv):
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp_pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        cube_pos_noisy = self.cube.pose.p + torch.randn_like(self.cube.pose.p) * 0.01
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=cube_pos_noisy - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - cube_pos_noisy,
            )
        return obs


@register_agent(asset_download_ids=["xarm6"])
class XArm6RobotiqCustom(XArm6Robotiq):
    uid = "xarm6_robotiq_custom"
    def get_proprioception(self):
        obs = super().get_proprioception()
        obs["qpos"] = obs["qpos"][..., :6]
        obs["qvel"] = obs["qvel"][..., :6]
        return obs


class RecordEpisodeWandb(RecordEpisode):
    def __init__(
        self,
        env: BaseEnv,
        output_dir: str,
        wandb_video_freq: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super().__init__(env, output_dir, **kwargs)
        self.wandb_video_freq = wandb_video_freq


    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        super().flush_video(name, suffix, verbose, ignore_empty_transition, save)
        if save:
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
            else:
                video_name = name
            if self.wandb_video_freq != 0 and self._video_id % self.wandb_video_freq == 0:
                # print(f"Logging video {video_name} to wandb")
                video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
                wandb.log({"video": wandb.Video(f"{self.output_dir}/{video_name}", fps=self.video_fps)})


def make_envs(args: Args, run_name: str):
    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=args.robot)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        run_root = experiment_run_dir(args)
        eval_output_dir = os.path.join(run_root, "videos")
        if args.evaluate:
            if args.checkpoint is not None:
                eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
            else:
                eval_output_dir = os.path.join(run_root, "test_videos")
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisodeWandb(envs, output_dir=os.path.join(run_root, "train_videos"), save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisodeWandb(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30, 
                                       wandb_video_freq=(args.wandb_video_freq if args.track else 0))
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, auto_reset=args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    return envs, eval_envs, env_kwargs