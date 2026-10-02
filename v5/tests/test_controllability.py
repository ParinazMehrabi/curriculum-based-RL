"""Every dof the policy owns can be driven both ways, and nothing is a cliff.

Two properties, tested together because they are the same question from two
sides: can the policy act on this model, and can it afford to try. A joint with
no muscle authority in one direction, and an outcome priced as a wall rather
than as a cost, each remove part of the search space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

V5 = Path(__file__).resolve().parents[1]
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from myo_curriculum.env import (  # noqa: E402
    INDEPENDENT_JOINTS,
    ROOT_JOINTS,
    MyoLocomotionEnv,
)
from myo_curriculum.stages import STAGE_W  # noqa: E402

QUADRICEPS = ("vasint", "vaslat", "vasmed", "recfem")
KNEE_FLEXORS = ("bflh", "bfsh", "semimem", "semiten", "gasmed", "gaslat")


@pytest.fixture(scope="module")
def env():
    e = MyoLocomotionEnv(stage="W", seed=0)
    try:
        yield e
    finally:
        e.close()


def dense_moment(model, data) -> np.ndarray:
    """`data.actuator_moment` as (nu, nv), sparse or dense.

    MuJoCo stores it sparse from 3.2 -- `moment_rownnz`, `moment_rowadr` and
    `moment_colind` -- and dense before that. Both are supported because this
    project runs on both: a 2014 CPU cannot load any MuJoCo newer than 3.1.
    """
    if not hasattr(data, "moment_rownnz"):
        return np.asarray(data.actuator_moment).reshape(model.nu, model.nv)
    out = np.zeros((model.nu, model.nv))
    flat = np.asarray(data.actuator_moment).ravel()
    for i in range(model.nu):
        n, adr = int(data.moment_rownnz[i]), int(data.moment_rowadr[i])
        cols = np.asarray(data.moment_colind[adr:adr + n], dtype=int)
        out[i, cols] = flat[adr:adr + n]
    return out


def actuator_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def coupled_dofs(model, joint: str):
    """The dofs an equality constraint drives from `joint`, with polynomials."""
    driver = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return [
        (int(model.eq_obj1id[k]), model.eq_data[k][:5].copy())
        for k in range(model.neq)
        if model.eq_type[k] == mujoco.mjtEq.mjEQ_JOINT
        and model.eq_obj2id[k] == driver
    ]


def moment_arms_through_coupling(env, joint: str, theta: float, h: float = 1e-4):
    """d(muscle length)/d(joint angle), moving the coupled dofs as well.

    This is the measurement that matters for a coupled joint, and it is not
    `actuator_moment`. MyoSuite's knee drives seven other dofs -- two
    translations, two rotations and the three-dof patella chain -- through
    equality constraints, and the quadriceps insert on the *patella*. Their
    moment arm on `knee_angle` alone reads -0.003 m, the same sign as the
    hamstrings, which would make the knee inextensible. Moving the coupled dofs
    too gives the real arm, +0.038 m/rad of extension, as anatomy requires.
    """
    model, data = env.model, env.data
    coupled = coupled_dofs(model, joint)

    def lengths(value):
        data.qpos[env.qadr[joint]] = value
        for jid, poly in coupled:
            data.qpos[model.jnt_qposadr[jid]] = np.polyval(poly[::-1], value)
        mujoco.mj_forward(model, data)
        return data.actuator_length.copy()

    return (lengths(theta + h) - lengths(theta - h)) / (2 * h), coupled


# -- controllability ------------------------------------------------------

def test_the_root_is_unactuated_by_design(env):
    """A floating base. No muscle crosses it, and none should."""
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    moment = dense_moment(env.model, env.data)
    for joint in ROOT_JOINTS:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = int(env.model.jnt_dofadr[jid])
        assert np.abs(moment[:, dof]).max() < 1e-12, joint
    # every actuator is a tendon; none addresses a joint directly
    assert all(
        int(t) != int(mujoco.mjtTrn.mjTRN_JOINT) for t in env.model.actuator_trntype
    )


def test_every_independent_joint_has_muscles_both_ways(env):
    """Each of the nine dofs the policy owns can be driven either way.

    Read from the accelerations the full constrained model produces, so the
    knee's coupling and the pinned out-of-plane joints are accounted for.
    """
    model, data = env.model, env.data
    env.reset(seed=0)
    pose = data.qpos.copy()
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0          # isolate muscle action from falling
    try:
        def accel(activations):
            data.qpos[:] = pose
            data.qvel[:] = 0.0
            data.act[:] = 0.0
            data.ctrl[:] = 0.0
            if len(activations):
                data.act[activations] = 1.0
                data.ctrl[activations] = 1.0
            mujoco.mj_forward(model, data)
            return data.qacc.copy()

        passive = accel([])
        for joint in INDEPENDENT_JOINTS:
            dof = env.dadr[joint]
            raise_it, lower_it = [], []
            for i in range(model.nu):
                delta = accel([i])[dof] - passive[dof]
                if delta > 1e-6:
                    raise_it.append(i)
                elif delta < -1e-6:
                    lower_it.append(i)
            assert raise_it, joint
            assert lower_it, joint
            assert (accel(raise_it) - passive)[dof] > 1e-3, joint
            assert (accel(lower_it) - passive)[dof] < -1e-3, joint
    finally:
        model.opt.gravity[:] = gravity


def test_the_knee_can_extend(env):
    """The quadriceps extend the knee, through the patella coupling.

    Worth its own test because the obvious measurement says otherwise: on the
    `knee_angle` dof alone the vasti read -0.003 m, the same sign as the
    hamstrings. If the coupling were ever pinned or dropped by the planar
    restructuring this is the test that would catch it, and a model that cannot
    extend its knee cannot stand.
    """
    env.reset(seed=0)
    names = actuator_names(env.model)
    for side in ("r", "l"):
        dldq, coupled = moment_arms_through_coupling(env, "knee_angle_%s" % side, 0.5)
        assert len(coupled) == 7, "the knee drives seven dofs"

        def torque(group):
            wanted = tuple("%s_%s" % (mu, side) for mu in group)
            return sum(
                -env.model.actuator_gainprm[i, 2] * dldq[i]
                for i, name in enumerate(names) if name in wanted
            )

        # flexion is positive here, so extension torque is negative
        extension = torque(QUADRICEPS)
        flexion = torque(KNEE_FLEXORS)
        assert extension < -100.0, (side, extension)
        assert flexion > +100.0, (side, flexion)
        assert abs(extension) > abs(flexion), "extensors are the stronger group"


# -- falling and retreating are costs, not cliffs -------------------------

def test_falling_adds_no_penalty_of_its_own(env):
    """No explicit fall penalty: the forfeited episode is the whole price.

    At the 5.0 this used to be, the falling step scored -4.89 against a
    per-step maximum of 0.30 -- sixteen times the largest reward any step can
    earn, concentrated on one transition, and unrelated to how the fall came
    about.

    What the terminal step still carries is the forward payment, which is what
    it is for. That payment is proportional to distance, so it is a cost rather
    than a spike: this policy falls backwards and is charged for the ground it
    gave up, not for having fallen.
    """
    assert STAGE_W.reward.fall_penalty == 0.0
    per_step_max = STAGE_W.reward.alive + STAGE_W.reward.shaping_scale

    env.reset(seed=0)
    for _ in range(STAGE_W.episode_steps):
        _, reward, terminated, truncated, _ = env.step(
            np.zeros(env.n_act, dtype=np.float32)
        )
        if terminated:
            break
        if truncated:
            pytest.skip("this policy survived the episode")
    else:
        pytest.skip("no fall in one episode")

    # the terminal step is an ordinary step plus the forward payment, exactly
    ordinary = reward - env.terminal_bonus
    assert 0.0 <= ordinary <= per_step_max, ordinary
    assert env.terminal_bonus == pytest.approx(env.forward_bonus())
    # and that payment tracks the distance, rather than being a fixed penalty
    assert env.travel < 0.0, "this policy is expected to topple backwards"
    assert env.terminal_bonus < 0.0
    scale = (STAGE_W.forward_bonus * STAGE_W.episode_steps
             / STAGE_W.forward_reference_distance)
    assert env.terminal_bonus == pytest.approx(
        scale * env.travel * STAGE_W.backward_multiplier
        * env.steps / STAGE_W.episode_steps
    )


def test_falling_is_still_worse_than_surviving(env):
    """Removing the spike must not make falling free.

    Early termination is the price: the steps not taken, and the forward
    payment scaled by how little of the episode was survived.
    """
    stage = env.stage_spec
    env.reset(seed=0)
    base = env._start_tx
    env.data.qpos[env.qadr["pelvis_tx"]] = base + 0.5

    env.steps = 60                                    # fell early
    fell = env.forward_bonus() + 60 * stage.reward.alive
    env.steps = stage.episode_steps                   # same distance, survived
    stood = env.forward_bonus() + stage.episode_steps * stage.reward.alive
    assert stood > 10 * fell


def test_retreating_has_a_gradient_and_never_terminates(env):
    """Backwards is priced, not forbidden.

    The forward term is signed rather than clipped at zero, so there is a slope
    back towards forwards from anywhere, and no termination condition reads
    travel at all.
    """
    stage = env.stage_spec
    env.reset(seed=0)
    base = env._start_tx
    env.steps = stage.episode_steps

    seen = []
    for distance in (0.5, 0.0, -0.5, -1.0, -2.0):
        env.data.qpos[env.qadr["pelvis_tx"]] = base + distance
        mujoco.mj_forward(env.model, env.data)
        seen.append(env.forward_bonus())
        assert not env._is_fallen(), "travel is not a termination condition"

    assert all(b < a for a, b in zip(seen, seen[1:])), "strictly decreasing"
    assert seen[-1] < seen[-2] < 0.0, "retreating keeps costing, with no floor"
