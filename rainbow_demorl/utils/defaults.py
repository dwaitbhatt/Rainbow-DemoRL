from typing import Optional

from rainbow_demorl.agents import *

ALGO_DEFAULTS = {
    "TD3": {"default_learning_mode": "online", "agent_class": TD3Agent, "default_demo_type": None, "default_control_prior_type": None},
    "SAC": {"default_learning_mode": "online", "agent_class": SACAgent, "default_demo_type": None, "default_control_prior_type": None},
    "CQL": {"default_learning_mode": "offline", "agent_class": CQLAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "CALQL": {"default_learning_mode": "offline", "agent_class": CalQLAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
    "MCQ": {"default_learning_mode": "offline", "agent_class": MonteCarloQAgent, "default_demo_type": "rlbuffer", "default_control_prior_type": None},
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

# Default demo paths per demo_type. Supports {env_id} and {robot} template variables.
# Modify these to match your local data layout; set to None to always require --demo-path.
DEMO_PATH_DEFAULTS = {
    "motionplanning": "demos/{robot}/{env_id}/motionplanning/trajectory.state.pd_joint_vel.physx_cpu.h5",
    "rlbuffer": "demos/{robot}/{env_id}/rl_buffer/trajectory.state.pd_joint_vel.h5",
    "rlexpert-gauss": "demos/{robot}/{env_id}/rl_buffer/trajectory.state.pd_joint_vel.top10.h5",
    "rlexpert-det": "demos/{robot}/{env_id}/rl_buffer/trajectory.state.pd_joint_vel.top15.h5",
}

# Default model checkpoint paths per model type. Supports {env_id} and {robot} template variables.
# Modify these to match your local checkpoint layout; set to None to always require a CLI path.
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
