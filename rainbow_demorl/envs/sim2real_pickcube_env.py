import mani_skill.envs.utils.randomization as randomization
import numpy as np
import torch
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.structs.pose import Pose

from rainbow_demorl.agents.real_agent import RealXarm6Agent
from devices.xarm6 import CollisionError, SafetyBoundaryError


class PickCubeSim2RealEnv(Sim2RealEnv):
    def __init__(
        self, 
        sim_env: ManiSkillVectorEnv, 
        agent: RealXarm6Agent, 
        device: torch.device,
        has_grasp_info: bool = True,
    ):
        self.sim_env = sim_env
        self.agent = agent
        self.device = device

        self.curr_obs = {}
        self.goal_pos = None
        self.goal_thresh = sim_env.goal_thresh
        self.has_grasp_info = has_grasp_info

        real_robot_eef_pos = self.agent.xarm.get_eef_position()
        sim_robot_eef_pos = sim_env.base_env.agent.tcp_pose.p.cpu()
        sim_cube_pos = sim_env.base_env.cube.pose.p.cpu()
        init_real_cube_pos_estimate = sim_cube_pos - sim_robot_eef_pos + real_robot_eef_pos
        self.prev_cube_pos = init_real_cube_pos_estimate.cpu().numpy()

        def real_reset_function(self, seed=None, options=None):
            self.sim_env.reset(seed=seed, options=options)
            self.agent.reset(qpos=self.sim_env.base_env.agent.robot.qpos.cpu().flatten())

            # TODO: Check if this works, or if it is required. OnlineTrainer handles this already.
            # real_cube_pos = self.agent.cam.detect(arm_frame=True, color='r')
            # real_cube_pos = torch.from_numpy(real_cube_pos).unsqueeze(0).to(torch.float32)

            # real_robot_eef_pos = self.agent.xarm.get_eef_position()
            # real_robot_eef_pos = torch.from_numpy(real_robot_eef_pos).unsqueeze(0).to(torch.float32)
            
            # sim_eef_pos = self.sim_env.base_env.agent.tcp_pose.p.cpu()
            # sim_cube_pos = sim_eef_pos + real_cube_pos - real_robot_eef_pos
            # with torch.device(self.device):
            #     qs = randomization.random_quaternions(1, lock_x=True, lock_y=True)
            #     self.sim_env.cube.set_pose(Pose.create_from_pq(sim_cube_pos, qs))

            #     goal_xyz = torch.zeros((1, 3))
            #     goal_xyz[:, :2] = (
            #         torch.rand((b, 2)) * self.cube_spawn_half_size * 2
            #         - self.cube_spawn_half_size
            #     )
            #     goal_xyz[:, 0] += self.cube_spawn_center[0]
            #     goal_xyz[:, 1] += self.cube_spawn_center[1]
            #     goal_xyz[:, 2] = torch.rand((1)) * self.max_goal_height + sim_cube_pos[:, 2]
            #     self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


        super().__init__(
            sim_env, 
            agent, 
            # obs_mode="state",
            # reward_mode="dense", 
            render_mode=None,
            real_reset_function=real_reset_function
        )

    def _get_obs_agent(self):
        # using the original user implemented sim env's _get_obs_agent function in case they modify it e.g. to remove qvel values as they might be too noisy
        return self.agent.get_proprioception()

    def  _get_obs_extra(self, info):
        real_cube_pos = self.agent.cam.detect(arm_frame=True, color='r', use_sam2=True)
        if real_cube_pos is not None:
            self.prev_cube_pos = real_cube_pos
        else:
            real_cube_pos = self.prev_cube_pos

        self.curr_real_cube_pos = torch.from_numpy(real_cube_pos).unsqueeze(0).to(torch.float32)

        if self.goal_pos is None:
            self.sim_robot_base_pos = self.base_sim_env.agent.robot.pose.p[0].numpy()
            self.goal_pos = real_cube_pos + np.array([0, 0, 0.02]) + self.sim_robot_base_pos
            self.goal_pos = torch.from_numpy(self.goal_pos).unsqueeze(0).to(torch.float32)

        real_robot_eef_pos = self.agent.xarm.get_eef_position()
        self.real_robot_eef_pos = torch.from_numpy(real_robot_eef_pos).unsqueeze(0).to(torch.float32)
        
        obs = dict()
        if self.has_grasp_info:
            is_grasping = self.agent.xarm.is_grasping()
            is_grasping = torch.from_numpy(is_grasping).unsqueeze(0).to(torch.float32)
            obs["is_grasped"] = is_grasping

        tcp_to_obj_pos = self.curr_real_cube_pos - self.real_robot_eef_pos
        obj_to_goal_pos = self.goal_pos - self.sim_robot_base_pos - self.curr_real_cube_pos

        if self.agent.xarm.is_grasping() and tcp_to_obj_pos.norm(dim=1) < 0.05:
            tcp_to_obj_pos = torch.zeros_like(tcp_to_obj_pos)

        obs.update({
            "tcp_to_obj_pos": tcp_to_obj_pos,
            "obj_to_goal_pos": obj_to_goal_pos,
        })
        print(f"[obs] eef to cube pos: {obs['tcp_to_obj_pos']}, cube to goal pos: {obs['obj_to_goal_pos']}")
        print(f"[obs] (real) goal pos: {self.goal_pos - self.sim_robot_base_pos}, cube pos: {self.curr_real_cube_pos}")
        self.curr_obs = obs
        return obs

    def get_obs(self, info=None, unflattened=False):
        if unflattened:
            return dict(
                agent=self._get_obs_agent(),
                extra=self._get_obs_extra(info),
            )

        states = []
        for values in self._get_obs_agent().values():
            states.append(torch.tensor(values))
        for values in self._get_obs_extra(info).values():
            states.append(torch.tensor(values))
        try:
            return torch.hstack(states).to(self.device)
        except Exception as e:
            # print(f"[error] {e}")
            print(f"[error] {e}\nstates: {states}")
            if states[0].shape[0] == 1:
                for i in range(len(states)):
                    states[i] = states[i].squeeze().unsqueeze(0)
                return torch.hstack(states).to(self.device)
            else:
                raise e

    def compute_dense_reward(self, obs, action, info):
        tcp_to_obj_dist = torch.linalg.norm(
            self.curr_obs["tcp_to_obj_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # is_grasped = info["is_grasped"]
        # is_grasped = self.agent._sim_agent.is_grasping(self.base_sim_env.cube)
        is_grasped = self.agent.xarm.is_grasping()
        print(f"[reward] is_grasped: {is_grasped}")
        
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.curr_obs["obj_to_goal_pos"], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.get_qvel()
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        is_obj_placed = (
            torch.linalg.norm(self.curr_obs["obj_to_goal_pos"], axis=1)
            <= self.goal_thresh
        )
        reward += static_reward * is_obj_placed

        is_robot_static = self.agent.is_static(0.2)
        success = is_obj_placed & is_robot_static

        reward[success] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def reset(self, seed=None, options=None):
        self.real_reset_function(self, seed, options)
        if self._handle_wrappers:
            orig_env = self._last_wrapper.env
            self._last_wrapper.env = self._env_with_real_step_reset
            ret = self._first_wrapper.reset(seed=seed, options=options)
            self._last_wrapper.env = orig_env
        else:
            ret = self._env_with_real_step_reset.reset(seed, options)
        # sets sim to whatever the real agent reset to in order to sync them. Some controllers use the agent's
        # current qpos and as this is the sim controller we copy the real world agent qpos so it behaves the same
        # moreover some properties of the robot like forward kinematic computed poses are done through the simulated robot and so qpos has to be up to date
        real_agent_qpos = torch.tensor(self.agent.robot.qpos)
        sim_agent_qpos = self.base_sim_env.agent.robot.qpos
        new_sim_agent_qpos = sim_agent_qpos
        new_sim_agent_qpos[..., :6] = real_agent_qpos[..., :6]
        self._elapsed_steps = 0
        
        self.base_sim_env.agent.robot.set_qpos(new_sim_agent_qpos)
        self.agent.controller.reset()
        return ret

    def _create_error_return(self):
        """
        Create a return structure for collision scenario when step execution fails.
        Creates minimal structure with terminations=True and collision info added.

        Returns:
            Tuple of (obs, rewards, terminations, truncations, infos) with terminations=True and collision info
        """
        try:
            obs = self.get_obs()
            num_envs = obs.shape[0] if hasattr(obs, "shape") else 1
        except Exception:
            num_envs = 1
            obs = torch.zeros((num_envs, 30), dtype=torch.float32, device=self.device)

        rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        terminations = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        truncations = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        infos = {
            "collision": True,
            "episode": {
                "collision": torch.ones(num_envs, dtype=torch.float32, device=self.device)
            },
        }

        return (obs, rewards, terminations, truncations, infos)

    def step(self, action):
        """
        In order to make users able to use most gym environment wrappers without having to write extra code for the real environment
        we temporarily swap the last wrapper's .env property with the RealEnvStepReset environment that has the real step/reset functions
        """
        ret = None

        try:
            if self._handle_wrappers:
                orig_env = self._last_wrapper.env
                self._last_wrapper.env = self._env_with_real_step_reset
                ret = self._first_wrapper.step(action)
                self._last_wrapper.env = orig_env
            else:
                ret = self._env_with_real_step_reset.step(action)
        except (CollisionError, SafetyBoundaryError) as e:
            print(f"Critical error detected via exception: {e}")
            self.agent.xarm.recover_from_error()
            ret = self._create_error_return()

        # ensure sim agent qpos is synced
        if hasattr(self.base_sim_env.agent, "set_qpos"):
            self.base_sim_env.agent.set_qpos(self.agent.robot.qpos)
        else:
            self.base_sim_env.agent.robot.set_qpos(self.agent.robot.qpos)
        return ret

    def __del__(self):
        self.agent.stop()