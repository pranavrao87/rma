
from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)

@configclass
class BasePolicyConfig(RslRlPpoActorCriticsCfg):
    encoder_hidden_dims = list[int] = MISSING
    z_size: int = MISSING