import os
import sys
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from torch.utils.tensorboard import SummaryWriter

import wandb

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


@dataclass
class Args:
    #########################################################
    # Core experiment parameters
    #########################################################
    exp_name: Optional[str] = None
    """the name of this experiment"""
    algorithm: Annotated[str, 
                tyro.conf.arg(
                    name="algorithm", 
                    help="Algorithm to use (TD3, SAC, CQL, CALQL, BC_DET, BC_GAUSS, ACT, ACT_TD3, IBRL_TD3, IBRL_SAC, CHEQ_TD3, CHEQ_SAC, RESRL_TD3, RESRL_SAC)",
                    aliases=["-a"]
                    )] = "SAC"
    robot: Annotated[str, 
                tyro.conf.arg(
                    name="robot", 
                    help="Robot to use (e.g. xarm6_robotiq, panda)",
                    aliases=["-r"]
                    )] = "xarm6_robotiq"
    control_mode: str = "pd_joint_vel"
    """which control mode to use for the experiment"""
    learning_mode: str = "default"
    """the learning mode to use for the experiment. Can be 'default', 'online, 'offline'"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained model checkpoint file to start evaluation/training from"""
    print_info: bool = False
    """if toggled, prints the experiment information"""
    
    #########################################################
    # Logging and tracking parameters
    #########################################################
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rainbow-demorl-final"
    """the wandb's project name"""
    wandb_entity: str = "ucsd_erl"
    """the entity (team) of wandb's project"""
    wandb_group: str = "rainbow-demorl"
    """the group of the run for wandb"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    #########################################################
    # Video and trajectory saving parameters
    #########################################################
    capture_video: bool = True
    """whether to capture videos of the agent performances (saved in `videos` directory)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    wandb_video_freq: int = 2
    """frequency to upload saved videos to wandb (every nth saved video will be uploaded)"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of timesteps"""

    #########################################################
    # Model saving parameters
    #########################################################
    save_model: bool = True
    """whether to save the model checkpoints"""
    save_model_dir: Optional[str] = "runs"
    """the directory to save the model"""

    #########################################################
    # Environment parameters
    #########################################################
    env_id: Annotated[str, 
                tyro.conf.arg(
                    name="env_id", 
                    help="Environment ID (e.g., PickCube-v1)",
                    aliases=["-e"]
                    )] = "PickCube-v1"
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 12
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 0
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 5000
    """evaluation frequency in terms of timesteps"""

    #########################################################
    # Online trainer parameters
    #########################################################
    online_learning_timesteps: Annotated[int, 
                tyro.conf.arg(
                    name="online_learning_timesteps", 
                    help="total online learning timesteps",
                    aliases=["-ton"]
                    )] = 1_000_000
    online_buffer_size: int = 500_000
    """the online replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    horizon: int = 3
    """the horizon of the trajectory buffer (for n-step return)"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory (effective batch size is batch_size*horizozn)"""
    learning_starts: int = 5_000
    """timesteps for warmup phase to fill the replay buffer before learning"""
    lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed for TD3)"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 1
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    save_buffer: bool = False
    """whether to save the replay buffer"""

    #########################################################
    # Offline trainer parameters
    #########################################################
    offline_learning_grad_steps: Annotated[int, 
                tyro.conf.arg(
                    name="offline_learning_grad_steps", 
                    help="the number of timesteps to learn from the offline dataset",
                    aliases=["-toff"]
                    )] = 50_000
    """the number of timesteps to learn from the offline dataset"""
    offline_buffer_size: int = 10_000
    """the offline replay memory buffer size, or number of trajectories to load from the dataset"""
    offline_horizon: int = 1
    """the horizon for offline learning"""
 
    #########################################################
    # Finetuning pretrained offline models parameters
    #########################################################
    pretrained_offline_policy_type: Optional[str] = None
    """the type of pretrained offline policy. Can be 'BC_GAUSS' or 'BC_DET' or 'CQL_H' or 'CQL_RHO' or 'CALQL' or 'ACT'"""
    pretrained_offline_policy_path: Optional[str] = None
    """the path to the pretrained offline policy. If None, will use the default path as per pretrained_offline_policy_type."""
    pretrained_offline_value_type: Optional[str] = None
    """the type of pretrained offline value function. Can be 'CQL_H' or 'CQL_RHO' or 'CALQL' or 'MCQ'"""
    pretrained_offline_value_path: Optional[str] = None
    """the path to the pretrained offline value function. If None, will use the default path as per pretrained_offline_value_type."""
    finetune_offline_policy: bool = True
    """whether to finetune the offline policy during online training or start from scratch"""
    finetune_offline_value: bool = True
    """whether to finetune the offline value function during online training or start from scratch"""

    #########################################################
    # Demonstration dataset and replay buffer parameters
    #########################################################
    demo_type: Optional[str] = None
    """the type of demonstration data to use. Can be 'motionplanning' or 'rlbuffer' or 'rlexpert-gauss' or 'rlexpert-det'. If None, will use the default type if necessary for the algorithm."""
    demo_path: Optional[str] = None
    """the path to the demonstration data. If set, will use this path instead of the default path as per demo_type."""
    offline_buffer_type: str = "none"
    """the fixed dataset to be sampled during online training. Can be 'demos' or 'rollout' or 'none'"""
    offline_rollout_steps: Optional[int] = 50000
    """the number of steps to rollout for the offline buffer type 'rollout'"""

    #########################################################
    # Offline buffer usage parameters
    #########################################################
    # Use offline data for RL (RLPD)
    use_offline_data_for_rl: bool = False
    """whether to use offline data for RL"""

    # Auxiliary BC loss parameters
    use_auxiliary_bc_loss: bool = False
    """whether to use auxiliary BC loss for update actor"""
    bc_loss_alpha: float = 2.5
    """scaling factor associated with auxiliary BC loss"""

    #########################################################
    # Control prior parameters
    #########################################################
    control_prior_type: Optional[str] = None
    """the type of control prior to use. Can be 'BC_GAUSS' or 'BC_DET' or 'ACT'"""
    control_prior_path: Optional[str] = None
    """the path to the control prior model checkpoint. If None, will use the default path as per control_prior_type."""
    
    #########################################################
    # Algorithm specific parameters
    #########################################################
    # TD3 parameters
    noise_clip: float = 0.5
    """clip parameter of the target policy noise"""
    target_policy_noise: float = 0.1
    """noise added to target policy during critic update"""
    exploration_noise: float = 0.1
    """standard deviation of exploration noise"""

    # SAC parameters
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    critic_entropy: bool = True
    """whether to use entropy regularization for the critic"""
    actor_entropy: bool = True
    """whether to use entropy regularization for the actor"""

    # CQL parameters
    cql_variant: str = "cql-h"
    """the variant of CQL to use (cql-h, cql-rho)"""
    cql_actor_lr: float = 3e-5
    """the learning rate for the actor network in CQL"""
    cql_autotune: bool = False
    """whether to automatically tune the regularization weight"""
    cql_lagrange_tau: float = 10.0
    """the Lagrange threshold for CQL autotuning"""
    cql_alpha_init: float = 5.0
    """the initial regularization weight for CQL, when autotuning"""
    cql_alpha: float = 5.0
    """the regularization weight for CQL, if not autotuning"""
    cql_num_actions: int = 10
    """the number of actions to sample for CQL regularization terms"""

    # Monte Carlo Q parameters
    mcq_bootstrap_epsilon: float = 0.1
    """the probability of using bootstrapped Q-target instead of Monte Carlo return"""

    # CHEQ parameters
    ulow: float = 0.15
    """lower-bound uncertainty for CHEQ"""
    uhigh: float = 0.275
    """upper-bound uncertainty for CHEQ"""
    lam_buffer: float = 0
    """set the lambda value for the expert buffer or rollout"""
    lam_start : float = 0.5
    """set the starting lam for warmup rollout"""
    lam_low: float = 0.2
    """lower-bound lambda value for CHEQ"""
    lam_high:float = 1.0
    """upper-bound lambda value for CHEQ"""
    bernoulli_masking: float = 0.8
    """to randomly mask out some qfs during training, to avoid overfitting"""

    # Residual RL parameters
    resrl_critic_burn_in_steps: int = 20000
    """the number of steps to burn in the critic"""

    # Policy Agnostic RL (PARL) parameters
    num_base_policy_actions: int = 32
    """the number of actions to sample from the base policy"""
    num_actions_to_keep: int = 10
    """the number of top actions to keep from the base policy actions"""
    num_local_optimization_steps: int = 10
    """the number of local optimization steps on the top actions"""
    local_optimization_step_size: float = 1e-4
    """the step size for the local optimization"""
    sample_from_pi_opt: bool = False
    """whether to sample from or argmax over pi_opt"""

    #########################################################
    # Neural network architecture hyperparameters
    #########################################################
    mlp_dim: int = 256
    """the hidden dimension of the networks"""
    num_layers_actor: int = 3  # NOTE: num_layers_actor = 2 works better than 3 for BC_DET. 3 layers tend to overfit.
    """the number of hidden layers in the actor network"""
    num_layers_critic: int = 3
    """the number of hidden layers in the critic network"""

    # Q modeling parameters
    use_ce_loss: bool = False
    """whether to use cross entropy loss for the Q network"""
    num_bins: int = 101
    """the number of bins for the Q network"""
    vmin: float = -10
    """the minimum value of the Q network"""
    vmax: float = 10
    """the maximum value of the Q network"""
    q_dropout: float = 0.01
    """the dropout rate for the Q network in case of CE loss"""
    num_critics: int = 2
    """the number of critics to use in the ensemble"""

    # ACT specific arguments
    act_lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    act_kl_weight: float = 10
    """weight for the kl loss term"""
    act_temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""
    # Backbone
    act_position_embedding: str = 'sine'
    act_backbone: str = 'resnet18'
    act_lr_backbone: float = 1e-5
    act_masks: bool = False
    act_dilation: bool = False
    # Transformer
    act_enc_layers: int = 2
    act_dec_layers: int = 4
    act_dim_feedforward: int = 512
    act_hidden_dim: int = 256
    act_dropout: float = 0.1
    act_nheads: int = 4
    act_num_queries: int = 30
    act_pre_norm: bool = False

    #### To be filled during runtime
    obs_dim: int = 0
    """the dimension of the observation space"""
    action_dim: int = 0
    """the dimension of the action space"""
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""
    bin_size: float = 0
    """the size of the bins for the Q network"""
    env_horizon: int = 0
    """the horizon of the environment, max_episode_steps"""
    is_cheq: bool = False
    """utility variable for trainer to know if the rollout agent is a CHEQ agent"""
    act_query_freq: int = 0
    """the frequency of querying the action from the actor, depends on whether temporal aggregation is used"""
    norm_stats: Optional[dict] = None
    """the normalization statistics of the observations and actions in the offline dataset. Used by ACT"""
    is_online: Optional[bool] = None
    """whether the training is online or offline"""


def experiment_run_dir(args: Args) -> str:
    """Root directory for checkpoints, TensorBoard, and run-scoped videos (env/robot/exp layout)."""
    root = args.save_model_dir or "runs"
    return os.path.join(root, args.env_id, args.robot, args.exp_name)

def is_pipeline_execution():
    """Detect if running in a shell pipeline (which can cause interpreter shutdown segfaults)"""
    try:
        # Check if stdin is a pipe (indicates pipeline execution like 'echo y | python')
        return not os.isatty(0)
    except:
        return False

def safe_exit_for_environment():
    """Exit safely based on execution environment to prevent interpreter shutdown segfaults"""
    
    if is_pipeline_execution():
        # Use os._exit(0) to bypass all Python cleanup which can segfault in pipelines
        os._exit(0)
    else:
        # Use sys.exit(0) for cleaner exit in direct execution
        sys.exit(0)
