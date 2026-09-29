"""Gymnasium registration for the muscle-locomotion curriculum.

As in v4, registration failures are raised rather than swallowed: an entry
point pointing at a module that no longer exists should fail here, not later
as a confusing make() error.
"""
from __future__ import annotations

from typing import Dict

from gymnasium.envs.registration import register, registry

from .rewards import RewardSpec, gaussian, smoothstep
from .stages import (
    STAGE_ORDER,
    STAGES,
    StageSpec,
    TermParams,
    describe_stages,
    get_stage,
)

__all__ = [
    "ENV_IDS",
    "RewardSpec",
    "STAGES",
    "STAGE_ORDER",
    "StageSpec",
    "TermParams",
    "describe_stages",
    "env_id_for",
    "gaussian",
    "get_stage",
    "make",
    "register_variant",
    "smoothstep",
]

ENTRY_POINT = "myo_curriculum.env:MyoLocomotionEnv"
ID_TEMPLATE = "MyoLocomotion{key}-v0"

ENV_IDS: Dict[str, str] = {key: ID_TEMPLATE.format(key=key) for key in STAGE_ORDER}


def env_id_for(stage: str) -> str:
    key = str(stage).upper()
    if key not in ENV_IDS:
        raise KeyError(
            "unknown stage %r; expected one of %s" % (stage, ", ".join(STAGE_ORDER))
        )
    return ENV_IDS[key]


def _register_all() -> None:
    for key in STAGE_ORDER:
        env_id = ENV_IDS[key]
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point=ENTRY_POINT,
            kwargs={"stage": key},
            max_episode_steps=int(STAGES[key].episode_steps),
        )


_register_all()


def register_variant(stage: str, env_id: str, **overrides) -> str:
    """Register an id that is `stage` with `overrides` baked in.

    Overrides are validated here, before the registry is touched, so a typo
    raises immediately rather than training for millions of steps against a
    default nobody chose.
    """
    key = str(stage).upper()
    get_stage(key).with_overrides(**overrides)
    if env_id not in registry:
        kwargs = {"stage": key}
        kwargs.update(overrides)
        register(
            id=env_id,
            entry_point=ENTRY_POINT,
            kwargs=kwargs,
            max_episode_steps=int(
                overrides.get("episode_steps", STAGES[key].episode_steps)
            ),
        )
    return env_id


def make(stage: str, **kwargs):
    import gymnasium

    return gymnasium.make(env_id_for(stage), **kwargs)
