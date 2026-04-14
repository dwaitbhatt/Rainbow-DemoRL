import os
from typing import Optional

from rainbow_demorl.agents import *
from rainbow_demorl.utils.common import Args

ALGO_DEFAULTS = {
    "TD3": {"default_learning_mode": "online", "agent_class": TD3Agent, "default_demo_type": None, "default_control_prior_type": None},
    "SAC": {"default_learning_mode": "online", "agent_class": SACAgent, "default_demo_type": None, "default_control_prior_type": None},
    "CQL": {"default_learning_mode": "offline", "agent_class": CQLAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "CALQL": {"default_learning_mode": "offline", "agent_class": CalQLAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "MCQ": {"default_learning_mode": "offline", "agent_class": MonteCarloQAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "MCQ_REAL": {"default_learning_mode": "offline", "agent_class": MonteCarloQAgentReal, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "BC_DET": {"default_learning_mode": "offline", "agent_class": BCDeterministicAgent, "default_demo_type": "rlexpert-det", "default_control_prior_type": None},
    "BC_GAUSS": {"default_learning_mode": "offline", "agent_class": BCGaussianAgent, "default_demo_type": "rlexpert-gauss", "default_control_prior_type": None},
    "ACT": {"default_learning_mode": "offline", "agent_class": ACT_BCAgent, "default_demo_type": "rlexpert-det", "default_control_prior_type": None},
    "ACT_TD3": {"default_learning_mode": "online", "agent_class": ACT_TD3Agent, "default_demo_type": "rlexpert-det", "default_control_prior_type": None},
    "IBRL_TD3": {"default_learning_mode": "online", "agent_class": TD3BaseIBRLAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "IBRL_SAC": {"default_learning_mode": "online", "agent_class": SACBaseIBRLAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "CHEQ_TD3": {"default_learning_mode": "online", "agent_class": TD3BaseCHEQAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "CHEQ_SAC": {"default_learning_mode": "online", "agent_class": SACBaseCHEQAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "RESRL_TD3": {"default_learning_mode": "online", "agent_class": TD3BaseResidualRLAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "RESRL_SAC": {"default_learning_mode": "online", "agent_class": SACBaseResidualRLAgent, "default_demo_type": None, "default_control_prior_type": "BC_DET"},
    "PARL_TD3": {"default_learning_mode": "online", "agent_class": PARL_TD3Agent, "default_demo_type": None, "default_control_prior_type": None},
    "PARL_SAC": {"default_learning_mode": "online", "agent_class": PARL_SACAgent, "default_demo_type": None, "default_control_prior_type": None},
    "PARL_ACT": {"default_learning_mode": "online", "agent_class": PARL_ACTAgent, "default_demo_type": "rlexpert-det", "default_control_prior_type": None},
}

DEMO_PATH_REQUIRED_MSG = (
    "A demonstration path must be provided for the current experiment. You can avoid passing paths every time as CLI args by setting default data paths in defaults.py"
)

MODEL_PATH_REQUIRED_MSG = (
    "A model path must be provided for the current experiment. You can avoid passing paths every time as CLI args by setting default model paths in defaults.py"
)

DEMO_PATH_DEFAULTS = {
    "motionplanning": "/pers_vol/dwait/saved_demos/demos/PickCubeCustom-v1/motionplanning/trajectory.state.pd_joint_vel.physx_cpu.h5",
    "rlbuffer": "/pers_vol/dwait/saved_buffers/rl_buffer/{}_SAC_{}_save_buffer/trajectory.state.pd_joint_vel.physx_gpu.h5",
    "rlexpert-gauss": "/pers_vol/dwait/saved_buffers/rl_buffer/{}_SAC_{}_save_buffer/trajectory.state.pd_joint_vel.physx_gpu.top10.h5",
    "rlexpert-det": "/pers_vol/dwait/saved_buffers/rl_buffer/{}_TD3_{}_save_buffer/trajectory.state.pd_joint_vel.physx_gpu.top15.h5",
}

PRETRAINED_MODEL_PATH_DEFAULTS = {
    "BC_DET": "runs/{env_id}/{robot}/bc_det/best_model.pt",
    "BC_GAUSS": "runs/{env_id}/{robot}/bc_gauss/best_model.pt",
    "ACT": "runs/{env_id}/{robot}/act/best_model.pt",
    "CQL_H": "runs/{env_id}/{robot}/cql_h/best_model.pt",
    "CQL_RHO": "runs/{env_id}/{robot}/cql_rho/best_model.pt",
    "CALQL": "runs/{env_id}/{robot}/calql/best_model.pt",
    "MCQ": "runs/{env_id}/{robot}/mcq/final_ckpt.pt",
}


def resolve_default_path(template: Optional[str], env_id: str, robot: str) -> Optional[str]:
    if template is None:
        return None
    return template.format(env_id=env_id, robot=robot)


def get_pretrained_model_path_default(args: Args, pretrained_model_type: str):
    default_model_dir = os.environ.get(
        "RAINBOW_DEMORL_MODEL_DIR",
        os.path.join("runs", args.env_id, args.robot),
    )
    
    algo = pretrained_model_type.replace("-", "_")
    if pretrained_model_type in ["CQL_H", "CQL_RHO"]:
        algo = "CQL"
    env_short = args.env_id.split('-')[0].lower()

    is_offline_rl = algo in ["CQL", "CALQL", "MCQ"]
    
    starts_with = f"{env_short}_{args.robot}_{algo}_save"
    if is_offline_rl and args.num_critics == 5:
        starts_with += '5'
    if pretrained_model_type == "CQL_H":
        starts_with += '_container2'
    elif pretrained_model_type == "CQL_RHO":
        starts_with += '_container3'
    
    model_dir = None
    for dir_name in os.listdir(default_model_dir):
        if dir_name.startswith(starts_with):
            model_dir = os.path.join(default_model_dir, dir_name)
            break
    if model_dir is None:
        raise ValueError(f"Model directory not found for {starts_with}")
    
    for file_name in os.listdir(model_dir):
        if file_name.startswith("best_model") and file_name.endswith(".pt"):
            return os.path.join(model_dir, file_name)
    raise ValueError(f"Best model file not found in {model_dir}")
