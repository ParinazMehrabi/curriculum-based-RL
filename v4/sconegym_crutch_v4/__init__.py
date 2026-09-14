"""Gym registration for the v4 crutch curriculum.

Registration failures are raised, not swallowed. v3 wrapped every register()
call in a bare "except Exception: pass", so an entry point pointing at a module
that had been deleted surfaced much later as a confusing gym.make error.
"""
from __future__ import annotations

from typing import Dict

from gym.envs.registration import register

from .rewards import RewardSpec, gaussian, smoothstep
from .stages import (
    GAIT_KEYFRAMES,
    INIT_FRAME,
    KEYFRAME_WINDOWS,
    NEUTRAL,
    REFERENCE_SPEED,
    RSIConfig,
    STAGE_ORDER,
    STAGES,
    StageSpec,
    TermParams,
    get_stage,
)
from .trajectory import Trajectory, load_sto, summarise

__all__ = [
    "ENV_IDS",
    "GAIT_KEYFRAMES",
    "KEYFRAME_WINDOWS",
    "REFERENCE_SPEED",
    "RSIConfig",
    "RewardSpec",
    "Trajectory",
    "STAGES",
    "STAGE_ORDER",
    "StageSpec",
    "TermParams",
    "describe_stages",
    "env_id_for",
    "gaussian",
    "get_stage",
    "load_sto",
    "make",
    "register_variant",
    "smoothstep",
]

ENTRY_POINT = "sconegym_crutch_v4.env:CrutchCurriculumGym"
ID_TEMPLATE = "sconewalk_crutch_v4_stage{key}-v1"

ENV_IDS: Dict[str, str] = {
    key: ID_TEMPLATE.format(key=key.lower()) for key in STAGE_ORDER
}


def env_id_for(stage: str) -> str:
    key = str(stage).upper()
    if key not in ENV_IDS:
        raise KeyError(
            "unknown stage %r; expected one of %s" % (stage, ", ".join(STAGE_ORDER))
        )
    return ENV_IDS[key]


def _is_duplicate(exc: Exception) -> bool:
    text = str(exc).lower()
    return "re-register" in text or "already registered" in text or "cannot re-register" in text


def _register_all() -> None:
    for key in STAGE_ORDER:
        env_id = ENV_IDS[key]
        try:
            register(
                id=env_id,
                entry_point=ENTRY_POINT,
                kwargs={"stage": key},
                max_episode_steps=int(STAGES[key].episode_steps),
            )
        except Exception as exc:
            if _is_duplicate(exc):
                continue
            raise RuntimeError(
                "failed to register %s (entry point %s): %r" % (env_id, ENTRY_POINT, exc)
            ) from exc


_register_all()


def register_variant(stage: str, env_id: str, **overrides) -> str:
    """Register a new env id that is `stage` with `overrides` baked in.

    Overrides travel through gym's registration kwargs, which reach the
    constructor directly. That sidesteps the open question in v3 about whether
    a tonic env_args block is forwarded to the environment at all -- nothing in
    this package reads env_args, so the values in stages.py plus whatever is
    passed here are the whole story.

    Intended for the header line of a tonic config:

        header: >
          import deprl, gym, sconegym, sconegym_crutch_v4;
          sconegym_crutch_v4.register_variant('D', 'my_d-v1', alive=0.10)

    Raises immediately if an override name is unknown, so a typo stops the run
    instead of training for 10M steps against a default nobody chose.
    """
    key = str(stage).upper()
    # Validate before touching the registry.
    get_stage(key).with_overrides(**overrides)
    kwargs = {"stage": key}
    kwargs.update(overrides)
    try:
        register(
            id=env_id,
            entry_point=ENTRY_POINT,
            kwargs=kwargs,
            max_episode_steps=int(
                overrides.get("episode_steps", STAGES[key].episode_steps)
            ),
        )
    except Exception as exc:
        if not _is_duplicate(exc):
            raise RuntimeError(
                "failed to register variant %s: %r" % (env_id, exc)
            ) from exc
    return env_id


def make(stage: str, **overrides):
    """gym.make the given stage, applying constructor overrides."""
    import gym

    return gym.make(env_id_for(stage), **overrides)


def describe_stages() -> str:
    return "\n".join(
        "%s\n     %s" % (ENV_IDS[k], STAGES[k].describe()) for k in STAGE_ORDER
    )
