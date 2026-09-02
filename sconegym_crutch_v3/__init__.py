from gym.envs.registration import register

ENV_ID_A0 = "sconewalk_rajagopal_crutch_v3_A0_stand_005-v1"

try:
    register(
        id=ENV_ID_A0,
        entry_point="sconegym_crutch_v3.a0_walk_gaitgym:RajagopalCrutchA0StandingGym",
        max_episode_steps=1000,
    )
    print("Registered:", ENV_ID_A0)
except Exception:
    pass

from gym.envs.registration import register


ENV_ID_STAGE1_STAND = "sconewalk_rajagopal_crutch_v3_stage1_stand-v1"

try:
    register(
        id=ENV_ID_STAGE1_STAND,
        entry_point=(
            "sconegym_crutch_v3.a0_stand_gaitgym:"
            "RajagopalCrutchA0StandStageGym"
        ),
        max_episode_steps=1000,
    )
    print("Registered:", ENV_ID_STAGE1_STAND)
except Exception as e:
    print(f"Registration issue for {ENV_ID_STAGE1_STAND} "
          f"(may be harmless if already registered): {e!r}")

from gym.envs.registration import register


ENV_ID_STAGE1B = "sconewalk_rajagopal_crutch_v3_stage1b_stand_crutch-v1"

try:
    register(
        id=ENV_ID_STAGE1B,
        entry_point=(
            "sconegym_crutch_v3.stage1b_stand_crutch_gaitgym:"
            "RajagopalCrutchStage1bStandGym"
        ),
        max_episode_steps=1000,
    )
    print("Registered:", ENV_ID_STAGE1B)
except Exception as e:
    print(f"Registration issue for {ENV_ID_STAGE1B} "
          f"(may be harmless if already registered): {e!r}")
from gym.envs.registration import register

ENV_ID_STAGE2 = "sconewalk_rajagopal_crutch_v3_stage2_tiny_forward_crutch-v1"

try:
    register(
        id=ENV_ID_STAGE2,
        entry_point=(
            "sconegym_crutch_v3.stage2_tiny_forward_crutch_gaitgym:"
            "RajagopalCrutchStage2TinyForwardGym"
        ),
        max_episode_steps=1000,
    )
    print("Registered:", ENV_ID_STAGE2)
except Exception as e:
    print(f"Registration issue: {e!r}")

from gym.envs.registration import register

from .stage2d_posture_fix_gaitgym import (
    RajagopalCrutchStage2DPostureFixGym
)
ENV_ID_STAGE2D = "sconewalk_rajagopal_crutch_v3_stage2d_posture_fix-v1"

try:
    register(
        id=ENV_ID_STAGE2D,
        entry_point=(
            "sconegym_crutch_v3.stage2d_posture_fix_gaitgym:"
            "RajagopalCrutchStage2DPostureFixGym"
        ),
        max_episode_steps=1000,
    )
    print("Registered:", ENV_ID_STAGE2D)
except Exception as e:
    print("Registration issue:", repr(e))
