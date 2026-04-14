from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.sensors.base_sensor import BaseSensorConfig
from mani_skill.utils.structs.types import Array

from devices.xarm6 import XArmControl
from devices.camera import Camera, load_extrinsics
from rainbow_demorl.utils.common import Args


class RealXarm6Agent(BaseRealAgent):
    """
    Base agent class for representing real robots, sensors, and controlling them in a real environment. This generally should be used with the :py:class:`mani_skill.envs.sim2real_env.Sim2RealEnv` class for deploying policies learned in simulation
    to the real world.

    Args:
        sensor_configs (Dict[str, BaseSensorConfig]): the sensor configs to create the agent with.
    """

    def __init__(self, args: Args, simulated: bool = False, sensor_configs: Dict[str, BaseSensorConfig] = dict(), use_sim_gripper_joints: bool = True):
        super().__init__(sensor_configs)
        
        if args.control_mode == "pd_joint_vel":
            self.mode = 4
        elif args.control_mode == "pd_joint_pos":
            self.mode = 1
        else:
            raise ValueError(f"Unsupported control mode: {args.control_mode}")

        self.simulated = simulated
        self.use_sim_gripper_joints = use_sim_gripper_joints

        if sensor_configs:
            base_camera_config = sensor_configs['base_camera']
            self.cam = Camera(debug=True, save_video=True, imsize=(base_camera_config.width, base_camera_config.height), 
                              use_sam2=True, objects_to_track=['r'])
        else:
            self.cam = Camera(debug=True, save_video=True, use_sam2=True, objects_to_track=['r'])
        self.cam.flush()
        self.cam.cam2arm, self.cam.arm2cam = load_extrinsics()
        print("Camera initialized")
        
        self.xarm = XArmControl(
            ip="192.168.1.242", 
            mode=self.mode, 
            simulated=self.simulated,
            tcp_z_offset=145,
            object_to_grip="none"
        )

    def start(self):
        """
        Start the agent, which include turning on the motors/robot, setting up cameras/sensors etc.

        For sensors you have access to self.sensor_configs which is the requested sensor setup. For e.g. cameras these sensor configs will define the camera resolution.

        For sim2real/real2sim alignment when defining real environment interfaces we instantiate the real agent with the simulation environment's sensor configs.
        """
        pass

    def stop(self):
        """
        Stop the agent, which include turning off the motors/robot, stopping cameras/sensors etc.
        """
        print("Resetting xArm, please wait")
        self.xarm.close()
        self.cam.close()
        print("xArm and camera closed")

    # ---------------------------------------------------------------------------- #
    # functions for controlling the agent
    # ---------------------------------------------------------------------------- #
    def set_target_qpos(self, qpos: Array):
        """
        Set the target joint positions of the agent.
        Args:
            qpos: the joint positions in radians to set the agent to.
        """
        if len(qpos.shape) > 1:
            qpos = qpos.squeeze(0)

        # Maniskill sets gripper with set_target_qpos. Handling that special case here.
        if all(qpos[:6] == 0):
            # self._sim_agent.set_target_qpos(qpos.unsqueeze(0))
            gripper_action = qpos[6]
            # print(f"[set_target_qpos] Setting gripper action to {gripper_action}, qpos = {qpos}")
            # self.xarm.set_gripper_pos(gripper_action)
            if gripper_action > 0.5:
                self.xarm.close_gripper()
            elif gripper_action < -0.5:
                self.xarm.open_gripper()
            return
            
        # equivalent to set_drive_targets in simulation
        if self.mode != 1:
            print(f"Switching from mode {self.mode} to mode 1 for setting joint positions")
            self.xarm.switch_mode(1)

        joint_angles = qpos[:6]
        self.xarm.set_joint_angles(joint_angles)

        if self.mode != 1:
            print(f"Switching back from mode 1 to original mode {self.mode}")
            self.xarm.switch_mode(self.mode)

    def set_target_qvel(self, qvel: Array):
        """
        Set the target joint velocities of the agent.
        Args:
            qvel: the joint velocities in radians/s to set the agent to.
        """
        # equivalent to set_drive_velocity_targets in simulation
        assert self.mode == 4, "Real xarm must operate in mode 4 to set joint velocities"

        if len(qvel.shape) > 1:
            qvel = qvel.squeeze(0)
        
        # 6 DOF for joint velocity and 1 for gripper
        # assert len(qvel) == 7, f"Must pass 7 dim qvel, got {len(qvel)}"
        joint_vel = qvel[:6]

        self.xarm.set_joint_velocity(joint_vel, duration=0.05)
        # NOTE: Never set gripper position with qvel, it is always set with set_target_qpos by maniskill
        # Doing this might lead to "undoing" the gripper set_target_qpos 
        # self.xarm.set_gripper_pos(gripper_action)


    def reset(self, qpos: Array):
        """
        Reset the agent to a given qpos. For real robots this function should move the robot at a safe and controlled speed to the given qpos and aim to reach it accurately.
        Args:
            qpos: the qpos in radians to reset the agent to.
        """
        self.xarm.reset()
        self.set_target_qpos(qpos)

    # ---------------------------------------------------------------------------- #
    # data access for e.g. joint position values, sensor observations etc.
    # All of the def get_x() functions should return numpy arrays and be implemented
    # ---------------------------------------------------------------------------- #
    def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
        """
        Capture the sensor data asynchronously from the agent based on the given sensor names. If sensor_names is None then all sensor data should be captured. This should not return anything and should be async if possible.
        """
        pass

    def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
        """
        Get the desired sensor observations from the agent based on the given sensor names. If sensor_names is None then all sensor data should be returned. The expected format for cameras is in line with the simulation's
        format for cameras.

        .. code-block:: python

            {
                "sensor_name": {
                    "rgb": torch.uint8 (1, H, W, 3), # red green blue image colors
                    "depth": torch.int16 (1, H, W, 1), # depth in millimeters
                }
            }

        whether rgb or depth is included depends on the real camera and can be omitted if not supported or not used. Note that a batch dimension is expected in the data.

        For more details see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html in order to ensure
        the real data aligns with simulation formats.
        """
        sensor_data = {}
        if 'base_camera' in sensor_names:
            rgb, depth = self.cam.fetch_image()
            sensor_data['base_camera'] = {
                'rgb': torch.from_numpy(rgb).unsqueeze(0),
                'depth': torch.from_numpy(depth).unsqueeze(0)
            }
        else:
            print(f"Sensor {sensor_names} not found, skipping")
        return sensor_data
        
    def get_sensor_params(self, sensor_names: List[str] = None):
        """
        Get the parameters of the desired sensors based on the given sensor names. If sensor_names is None then all sensor parameters should be returned. The expected format for cameras is in line with the simulation's
        format is:

        .. code-block:: python

            {
                "sensor_name": {
                    "cam2world_gl": [4, 4], # transformation from the camera frame to the world frame (OpenGL/Blender convention)
                    "extrinsic_cv": [4, 4], # camera extrinsic (OpenCV convention)
                        "intrinsic_cv": [3, 3], # camera intrinsic (OpenCV convention)
                }
            }


        If these numbers are not needed/unavailable it is okay to leave the fields blank. Some observation processing modes may need these fields however such as point clouds in the world frame.
        """
        sensor_params = {}
        # TODO: This is actually cam2arm, we need to estimate and compose with arm2world if we actually use this
        if 'base_camera' in sensor_names:
            sensor_params['base_camera'] = {
                'cam2world_gl': self.cam.cam2arm
            }
        return sensor_params

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.

        Note that if qpos or qvel functions are not implemented they will return None.
        """
        qpos, qvel = self.xarm.get_qpos_qvel()
        qpos = torch.from_numpy(qpos).unsqueeze(0).to(torch.float32)
        qvel = torch.from_numpy(qvel).unsqueeze(0).to(torch.float32)

        # Copy gripper qpos and qvel from simulation
        if self.use_sim_gripper_joints:
            sim_proprio = self._sim_agent.get_proprioception()
            new_qpos = sim_proprio["qpos"].to(torch.float32)
            new_qpos[..., :6] = qpos
            new_qvel = sim_proprio["qvel"].to(torch.float32)
            new_qvel[..., :6] = qvel
            
            obs = dict(qpos=new_qpos, qvel=new_qvel)
        else:
            obs = dict(qpos=qpos, qvel=qvel)
        controller_state = self._sim_agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_qpos(self):
        """
        Get the current joint positions in radians of the agent as a torch tensor. Data should have a batch dimension, the shape should be (1, N) for N joint positions.
        """
        return self.get_proprioception()["qpos"]

    def get_qvel(self):
        """
        Get the current joint velocities in radians/s of the agent as a torch tensor. Data should have a batch dimension, the shape should be (1, N) for N joint velocities.
        """
        return self.get_proprioception()["qvel"]

    def is_static(self, threshold: float = 0.2):
        qvel = self.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


    def __getattr__(self, name):
        """
        Delegate attribute access to self._sim_agent if the attribute doesn't exist in self.
        This allows accessing sim_agent properties and methods directly from the real agent. Some simulation agent include convenience functions to access e.g. end-effector poses
        or various properties of the robot.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if hasattr(self, "_sim_agent") and hasattr(self._sim_agent, name):
                return getattr(self._sim_agent, name)
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
