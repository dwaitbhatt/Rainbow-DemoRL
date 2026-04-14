import os
from pathlib import Path
from typing import Dict, Optional, Any, List

import gymnasium as gym
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs import PickCubeEnv
from mani_skill.utils.gym_utils import find_max_episode_steps_value
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
import mani_skill.envs.utils.randomization as randomization

# Define custom asset directory
PACKAGE_DIR = Path(__file__).parent.parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"

import sapien
import numpy as np
import wandb
from rainbow_demorl.utils.common import Args


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

@register_env("PickCubeAllegroXarm", max_episode_steps=80)
class PickCubeAllegroXarmEnv(PickCubeEnv):

    SUPPORTED_ROBOTS = ["xarm6_allegro_left", "xarm6_allegro_left_custom"]
    
    cube_half_size_allegro = 0.03
    cube_half_size = 0.03

    def __init__(self, *args, robot_uids="xarm6_allegro_left_custom", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # PickCubeEnv._load_agent only accepts (options); it passes the initial pose to BaseEnv.
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # xarm6_allegro_left_custom hits TableSceneBuilder's generic "allegro" branch (`pass`); mirror
            # initialize() for xarm6_allegro_left (keyframes rest + noise, base pose).
            if self.robot_uids == "xarm6_allegro_left_custom":
                qpos = self.agent.keyframes["rest"].qpos
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.1 - 0.05
            # xyz[:,0] = -0.038031
            # xyz[:,1] = -0.159084
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.1 - 0.05
            goal_xyz[:, 2] = torch.rand((b)) * 0.2 + 2*self.cube_half_size
            # goal_xyz = xyz.clone()
            # goal_xyz[:, 2] = xyz[:, 2] + 0.2
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        obs.update(
            # obj_pose=self.cube.pose.raw_pose,
            obj_to_tcp_pos=self.agent.tcp.pose.p - self.cube.pose.p,
            obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
        )
        # if "state" in self.obs_mode:
        #     obs.update(
        #         # obj_pose=self.cube.pose.raw_pose,
        #         obj_to_tcp_pos=self.agent.tcp.pose.p - self.cube.pose.p,
        #         obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
        #     )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        # XArm6AllegroLeftCustom uses a different is_grasping signature than e.g. Panda
        if self.robot_uids == "xarm6_allegro_left_custom":
            is_grasped = self.agent.is_grasping(self.cube_half_size, self.cube)
        else:
            is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2) # threshold is 0.2 here

        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def staged_rewards(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        place_reward *= is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward *= info["is_obj_placed"]

        return reaching_reward.mean(), is_grasped.mean(), place_reward.mean(), static_reward.mean()

      
    def compute_modified_reward(self, obs: Any, action: torch.Tensor, info: Dict): # staging the previous reward designed for pickcube without grasping info

        joint_pos = torch.tensor(self.agent.robot.get_qpos(), dtype=torch.float32)
        joint_5_pos = joint_pos[..., 4]
        # reward += torch.where(joint_5_pos < -0.75, 0.5, -0.5)
        reward = 1 / (1 + torch.exp(5.8 * (joint_5_pos + 1))) - 1/(1 + torch.exp(5.8 * (-joint_5_pos + 1))) # 0.947 at -1.5 joint value, and 0.19 at -0.75 value. Check desmos for its graph
        reward = 1 / (1 + torch.exp(5.8 * (joint_5_pos + 1))) # 0.947 at -1.5 joint value, and 0.19 at -0.75 value. Check desmos for its graph
        
        mask_joint_pos = joint_pos[..., 4] < -1.25
        cube_position_z_offseted = self.cube.pose.p.clone()
        cube_position_z_offseted[:, 2] += self.cube_half_size+0.01
        tcp_to_obj_dist = torch.linalg.norm(
            cube_position_z_offseted - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 + 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward[mask_joint_pos] = reaching_reward[mask_joint_pos]

        mask_reached = tcp_to_obj_dist < (self.cube_half_size * np.sqrt(2) + 0.01)
        object_grabbing_closeness = self.agent.object_reward(self.cube)
        
        # mask_thumb_close = object_grabbing_closeness[..., 0] < self.cube_half_size * np.sqrt(1.25)+ 0.013 
        thumb_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 0]))/2
        finger1_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 1]))/2
        finger2_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 2]))/2
        finger3_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 3]))/2

        # reward[mask_reached] = (2 + (1 - torch.tanh(5 * object_grabbing_closeness[..., 0])))[mask_reached]
        # reward[mask_thumb_close] = (3 + (finger1_reward + finger2_reward + finger3_reward))[mask_thumb_close]
        reward[mask_reached] = (2 + (thumb_reward + finger1_reward + finger2_reward + finger3_reward))[mask_reached]
        
        is_grasped = info["is_grasped"]/2
        mask_grasp = is_grasped >= 1
        
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 4*(1 - torch.tanh(10 * obj_to_goal_dist))
        
        reward[mask_grasp] = (4 + place_reward)[mask_grasp]


        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        elif self.robot_uids in ("xarm6_allegro_left", "xarm6_allegro_right"):
            qvel_without_gripper = qvel_without_gripper[..., :6]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )

        reward[info["is_obj_placed"]] = (static_reward + 8)[info["is_obj_placed"]]

        # if tcp_to_obj_dist < self.cube_half_size*np.sqrt(2) + 0.01:
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,0])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,1])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,2])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,3])
        
        # the below reward encourages pressing the cube with the gripper
        
                
        reward[info["success"]] = (9+5) # 2 for success bonus


        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_modified_reward(obs=obs, action=action, info=info) / 14

    def debug(self):
        self.agent.robot.get_qpos()
        print(self.cube.pose.p)
        self.agent.debug()


from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPosControllerConfig, PDEEPoseControllerConfig, PDJointVelControllerConfig, PDJointPosVelControllerConfig
from mani_skill.agents.controllers import *
from copy import deepcopy
from mani_skill.utils.structs.pose import vectorize_pose
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["xarm6"])
class XArm6AllegroLeftCustom(BaseAgent):
    uid = "xarm6_allegro_left_custom"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_allegro_left.urdf"
    
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [1.56280772e-03, 0.37, -0.61, 1.52969832e-04,  -1.31606723e+00, 1.66234924e-03,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j1=Keyframe(
            qpos=np.array([np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j2=Keyframe(
            qpos=np.array([0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j3=Keyframe(
            qpos=np.array([0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j4=Keyframe(
            qpos=np.array([0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j5=Keyframe(
            qpos=np.array([0, 0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j6=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        palm_up=Keyframe(
            qpos=np.array(
                [0, 0, 0, 0, 0, 0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                ]
            ),
            pose=sapien.Pose([0, 0, 0.5], q=[-0.707, 0, 0.707, 0]),
        )
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    gripper_joint_names = [
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ]

    arm_stiffness = 1000
    arm_damping = 50 # [50, 50, 50, 50, 50, 50]
    arm_friction = 0.1 # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100

    gripper_stiffness = 4e2
    gripper_damping = 1e1
    gripper_force_limit = 5e1

    # Order for left hand: thumb finger, index finger, middle finger, ring finger
    # Order for right hand: thumb finger, ring finger, middle finger, index finger 
    tip_link_names = [
        "link_15.0_tip",
        "link_11.0_tip",
        "link_7.0_tip",
        "link_3.0_tip",
    ]

    palm_link_name = "palm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_target_delta_pos = deepcopy(gripper_pd_joint_delta_pos)
        gripper_pd_joint_target_delta_pos.use_target = True
        
        # PD joint velocity
        gripper_pd_joint_vel = PDJointVelControllerConfig(
            self.gripper_joint_names,
            -1.0,
            1.0,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # PD joint position and velocity
        gripper_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Arm + Gripper Controller Configs
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_vel),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos_vel
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_delta_pos_vel
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)


    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        # obs.update(
        #     {
        #         "palm_pose": self.palm_pose,
        #         "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
        #     }
        # )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)


    def _after_init(self):
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )
        self.tcp = self.palm_link


    def is_grasping(self, cube_half_size, object: Actor, min_force=0.5, max_angle=85):
        thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[0], object
        )
        finger1_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[1], object
        )
        finger2_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[2], object
        )
        finger3_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[3], object
        )

        thumb_force = torch.linalg.norm(thumb_contact_forces, axis=1)
        finger1_force = torch.linalg.norm(finger1_contact_forces, axis=1)
        finger2_force = torch.linalg.norm(finger2_contact_forces, axis=1)
        finger3_force = torch.linalg.norm(finger3_contact_forces, axis=1)

        thumb_direction = self.tip_links[0].pose.to_transformation_matrix()[..., :3, 0] # its the x axis which is pointing outwards, checked on urdf visualizer
        finger1_direction = self.tip_links[1].pose.to_transformation_matrix()[..., :3, 0]
        finger2_direction = self.tip_links[2].pose.to_transformation_matrix()[..., :3, 0]
        finger3_direction = self.tip_links[3].pose.to_transformation_matrix()[..., :3, 0]
        
        thumb_angle = common.compute_angle_between(thumb_direction, thumb_contact_forces)
        finger1_angle = common.compute_angle_between(finger1_direction, finger1_contact_forces)
        finger2_angle = common.compute_angle_between(finger2_direction, finger2_contact_forces)
        finger3_angle = common.compute_angle_between(finger3_direction, finger3_contact_forces)

        # compute dot product between thumb_contact_forces and finger_contact_forces #TODO
        # thumb_flag = torch.logical_and(
        #     thumb_force >= min_force, torch.logical_and(torch.abs(torch.cos(thumb_angle)) <= torch.cos(torch.tensor(max_angle, dtype=torch.float)), )
        # )

        # Angles less than (180 - max_angle) degrees indicate contact from the back side (not grasping)
        # Only consider angles from (180 - max_angle) to 180 degrees as valid grasping angles
        min_grasp_angle_rad = torch.deg2rad(torch.tensor(180.0 - max_angle, device=thumb_angle.device, dtype=thumb_angle.dtype))
        max_grasp_angle_rad = torch.deg2rad(torch.tensor(180.0, device=thumb_angle.device, dtype=thumb_angle.dtype))
        
        thumb_flag = torch.logical_and(
            thumb_force >= min_force, 
            torch.logical_and(
                thumb_angle >= min_grasp_angle_rad,
                thumb_angle <= max_grasp_angle_rad
            )
        )
        finger1_flag = torch.logical_and(
            finger1_force >= min_force, 
            torch.logical_and(
                finger1_angle >= min_grasp_angle_rad,
                finger1_angle <= max_grasp_angle_rad
            )
        )
        finger2_flag = torch.logical_and(
            finger2_force >= min_force, 
            torch.logical_and(
                finger2_angle >= min_grasp_angle_rad,
                finger2_angle <= max_grasp_angle_rad
            )
        )
        finger3_flag = torch.logical_and(
            finger3_force >= min_force, 
            torch.logical_and(
                finger3_angle >= min_grasp_angle_rad,
                finger3_angle <= max_grasp_angle_rad
            )
        )

        confidence = torch.where(thumb_flag, (thumb_flag).int() + (finger1_flag).int() + (finger2_flag).int() + (finger3_flag).int(), torch.tensor(0, device=thumb_force.device))
        confidence_val = confidence.item() if confidence.numel() == 1 else confidence[0].item()
        return confidence
    
    def object_reward(self, object: Actor, min_force=0.5, max_angle=85):
        thumb_position = self.tip_links[0].pose.p
        finger1_position = self.tip_links[1].pose.p
        finger2_position = self.tip_links[2].pose.p
        finger3_position = self.tip_links[3].pose.p

        object_position = object.pose.p

        thumb_distance = torch.linalg.norm(thumb_position - object_position, axis=1)
        finger1_distance = torch.linalg.norm(finger1_position - object_position, axis=1)
        finger2_distance = torch.linalg.norm(finger2_position - object_position, axis=1)
        finger3_distance = torch.linalg.norm(finger3_position - object_position, axis=1)

        return torch.stack([thumb_distance, finger1_distance, finger2_distance, finger3_distance], dim=1)
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

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
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            if args.checkpoint is not None:
                eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
            else:
                eval_output_dir = f"runs/{run_name}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisodeWandb(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisodeWandb(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30, 
                                       wandb_video_freq=(args.wandb_video_freq if args.track else 0))
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, auto_reset=args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    return envs, eval_envs, env_kwargs


def make_envs_simple(args: Args, run_name: str):
    """Lightweight env factory for sim2real -- no video recording or RecordEpisode wrappers."""
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=args.robot)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, auto_reset=args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    return envs, eval_envs, env_kwargs