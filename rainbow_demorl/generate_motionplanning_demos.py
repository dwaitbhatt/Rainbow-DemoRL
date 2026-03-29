from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS as MP_SOLUTIONS_PANDA
from mani_skill.examples.motionplanning.panda.run import _main as run_motionplanning_panda
from mani_skill.examples.motionplanning.panda.solutions import solvePickCube as solvePickCubePanda
from mani_skill.examples.motionplanning.xarm6.run import MP_SOLUTIONS as MP_SOLUTIONS_XARM6
from mani_skill.examples.motionplanning.xarm6.run import _main as run_motionplanning_xarm6
from mani_skill.examples.motionplanning.xarm6.solutions import solvePickCube as solvePickCubeXArm6

from mani_skill.trajectory.replay_trajectory import main as replay_trajectory
import multiprocessing as mp

# Import to register custom environments
import rainbow_demorl.envs

MP_SOLUTIONS_XARM6.update({
    "PickCubeCustom-v1": solvePickCubeXArm6,
})

MP_SOLUTIONS_PANDA.update({
    "PickCubeCustom-v1": solvePickCubePanda,
})


@dataclass
class MotionPlanningArgs:
    env_id: str = "PickCubeCustom-v1"
    """Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}"""
    obs_mode: str = "none"
    """Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script."""
    num_traj: int = 10
    """Number of trajectories to generate."""
    only_count_success: bool = False
    """If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos"""
    reward_mode: Optional[str] = None
    """Reward mode to use"""
    sim_backend: str = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""
    render_mode: str = "rgb_array"
    """can be 'sensors' or 'rgb_array' which only affect what is saved to videos"""
    vis: bool = False
    """whether or not to open a GUI to visualize the solution live"""
    save_video: bool = False
    """whether or not to save videos locally"""
    traj_name: Optional[str] = None
    """The name of the trajectory .h5 file that will be created."""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    record_dir: str = "demos"
    """where to save the recorded trajectories"""
    num_procs: int = 1
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment."""


@dataclass
class ReplayTrajectoryArgs:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Optional[str] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Optional[str] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Optional[str] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""
    num_envs: int = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""


@dataclass
class GenerateDemosArgs:
    num_traj: Annotated[int, tyro.conf.arg(aliases=["-nt"])] = 1000
    """Number of trajectories to generate."""
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCubeCustom-v1"
    f"""Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS_XARM6.keys())} and {list(MP_SOLUTIONS_PANDA.keys())}"""
    robot: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "xarm6_robotiq"
    """Robot to use."""
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    """Observation mode to use in final trajectory."""
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_vel"
    """Control mode to use in final trajectory."""
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-ne"])] = 10
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""
    skip_video: bool = False
    """Whether to skip the sample video generation."""
    traj_name: str = "trajectory"
    """Name of the trajectory file to save."""
    sim_backend: str = "cpu"
    """Sim backend for ManiSkill during motion planning (`cpu`, `gpu`, or `auto`). Default is `cpu` (often more stable than `auto`/GPU for PhysX)."""


def main():
    args = tyro.cli(GenerateDemosArgs)
    print(f"Generating demos with args: {args}")

    if not args.skip_video:
        # Generate sample video with motionplanning
        mp_args = MotionPlanningArgs()
        mp_args.traj_name = args.traj_name
        mp_args.only_count_success = True
        mp_args.save_video = True
        mp_args.num_traj = 1
        # mp_args.vis = True
        mp_args.shader = "rt"
    
    mp_args = MotionPlanningArgs(
        env_id=args.env_id,
        num_traj=args.num_traj,
        only_count_success=True,
        traj_name=args.traj_name,
        sim_backend=args.sim_backend,
        record_dir=f"demos/{args.robot}",
    )
    print(f"Generating motion planning trajectories for {args.robot} with args: {mp_args}")
    if args.robot == "xarm6_robotiq":
        run_motionplanning_xarm6(mp_args)
    elif args.robot == "panda":
        run_motionplanning_panda(mp_args)
    else:
        raise ValueError(f"Invalid robot: {args.robot}")

    # Replay trajectories
    rt_args = ReplayTrajectoryArgs(
        traj_path=f"demos/{args.robot}/{args.env_id}/motionplanning/{args.traj_name}.h5",
        obs_mode=args.obs_mode,
        target_control_mode=args.control_mode,
        num_envs=args.num_envs,
        sim_backend="cpu",
        use_first_env_state=True,
        save_traj=True,
        record_rewards=True,
    )
    print(f"Replaying trajectories with args: {rt_args}")
    mp.set_start_method("spawn", force=True)
    replay_trajectory(rt_args)


if __name__ == "__main__":
    main()
