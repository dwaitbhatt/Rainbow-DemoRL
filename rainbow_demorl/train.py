import random
import time
from pprint import pprint

import numpy as np
import torch
import tyro

from rainbow_demorl.agents import *
from rainbow_demorl.envs.maniskill import find_max_episode_steps_value, make_envs
from rainbow_demorl.trainer.offline_trainer import OfflineTrainer
from rainbow_demorl.trainer.online_trainer import OnlineTrainer
from rainbow_demorl.utils.common import Args, safe_exit_for_environment
from rainbow_demorl.utils.defaults import (
    ALGO_DEFAULTS,
    DEMO_PATH_DEFAULTS,
    DEMO_PATH_REQUIRED_MSG,
    MODEL_PATH_REQUIRED_MSG,
    PRETRAINED_MODEL_PATH_DEFAULTS,
    resolve_default_path,
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        run_name = f"{args.env_id}__{args.algorithm.lower()}__{args.robot}__{args.seed}__{int(time.time())}"
        args.exp_name = run_name
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    ## Setup envs
    envs, eval_envs, env_kwargs = make_envs(args, args.exp_name)
    envs.single_observation_space.dtype = np.float32

    ## Convenience variables
    args.obs_dim = envs.single_observation_space.shape[0]
    args.action_dim = envs.single_action_space.shape[0]
    args.env_horizon = find_max_episode_steps_value(envs._env)
    args.bin_size = (args.vmax - args.vmin) / (args.num_bins - 1)

    if args.demo_type is None:
        args.demo_type = ALGO_DEFAULTS[args.algorithm]["default_demo_type"]
    if args.demo_path is None and args.demo_type is not None:
        args.demo_path = resolve_default_path(DEMO_PATH_DEFAULTS.get(args.demo_type), args.env_id, args.robot)
        if args.demo_path is None:
            raise ValueError(DEMO_PATH_REQUIRED_MSG)

    if args.control_prior_type is None:
        args.control_prior_type = ALGO_DEFAULTS[args.algorithm]["default_control_prior_type"]
    if args.control_prior_path is None and args.control_prior_type is not None:
        args.control_prior_path = resolve_default_path(
            PRETRAINED_MODEL_PATH_DEFAULTS.get(args.control_prior_type), args.env_id, args.robot
        )

    ## Setup learning mode and pretrained model paths
    if args.algorithm in ALGO_DEFAULTS:
        default_learning_mode = ALGO_DEFAULTS[args.algorithm]["default_learning_mode"]
    else:
        raise ValueError(f"Algorithm {args.algorithm} not supported, choose from {list(ALGO_DEFAULTS.keys())}")

    if args.learning_mode == "default":
        learning_mode = default_learning_mode
        args.learning_mode = learning_mode
    else:
        learning_mode = args.learning_mode

    args.is_online = learning_mode == "online"

    if args.pretrained_offline_policy_path is None and args.pretrained_offline_policy_type is not None:
        args.pretrained_offline_policy_path = resolve_default_path(
            PRETRAINED_MODEL_PATH_DEFAULTS.get(args.pretrained_offline_policy_type), args.env_id, args.robot
        )
    if args.pretrained_offline_value_path is None and args.pretrained_offline_value_type is not None:
        args.pretrained_offline_value_path = resolve_default_path(
            PRETRAINED_MODEL_PATH_DEFAULTS.get(args.pretrained_offline_value_type), args.env_id, args.robot
        )

    if args.control_prior_type is not None and args.control_prior_path is None:
        raise ValueError(MODEL_PATH_REQUIRED_MSG)
    if args.pretrained_offline_policy_type is not None and args.pretrained_offline_policy_path is None:
        raise ValueError(MODEL_PATH_REQUIRED_MSG)
    if args.pretrained_offline_value_type is not None and args.pretrained_offline_value_path is None:
        raise ValueError(MODEL_PATH_REQUIRED_MSG)

    ## Setup agent
    agent = ALGO_DEFAULTS[args.algorithm]["agent_class"](envs, device, args)
    if args.pretrained_offline_policy_path is not None or args.pretrained_offline_value_path is not None:
        assert isinstance(agent, ActorCriticAgent), "Using pretrained offline policy and value functions is only supported for ActorCritic agents"
        agent.load_pretrained(args.pretrained_offline_policy_path, args.pretrained_offline_value_path)

    if args.print_info:
        print(f"\n\n\n############# Network architecture: ##############\n{agent}\n\n\n")
        print(f"############# Environment: ##############\n")
        print(f"Observation space: {envs.single_observation_space}")
        print(f"Action space: {envs.single_action_space}")
        print(f"Max episode steps: {args.env_horizon}")
        print(f"\n\n\n############# Learning mode: ##############")
        print(f"Training with {args.algorithm}, learning mode: {learning_mode}\n\n\n")

    if not args.evaluate:
        if args.offline_buffer_type.lower() == "rollout" and args.pretrained_offline_policy_path is None:
            raise ValueError(MODEL_PATH_REQUIRED_MSG)
        needs_demo_h5 = learning_mode == "offline" or (
            learning_mode == "online" and args.offline_buffer_type.lower() == "demos"
        )
        if needs_demo_h5 and args.demo_path is None:
            raise ValueError(DEMO_PATH_REQUIRED_MSG)

    ## Setup trainer and train
    if learning_mode == "offline":
        trainer = OfflineTrainer(args, agent, envs, eval_envs, env_kwargs, device)
    elif learning_mode == "online":
        trainer = OnlineTrainer(args, agent, envs, eval_envs, env_kwargs, device)
    else:
        raise ValueError(f"Learning mode {learning_mode} not supported")

    if args.print_info:
        print("Trainer args:")
        pprint(vars(trainer.args))

    trainer.train()

    safe_exit_for_environment()
