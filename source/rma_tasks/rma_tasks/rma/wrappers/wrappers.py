from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)

@configclass
class BasePolicyCfg(RslRlPpoActorCriticCfg):
    encoder_hidden_dims: list = MISSING
    prev_step_size: int = MISSING,
    z_size: int = MISSING

@configclass
class AdaptionModuleCfg(RslRlPpoActorCriticCfg):
    encoder_hidden_dims: list = MISSING
    z_size: int = MISSING