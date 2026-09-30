# Muscle-actuated locomotion (v5)

A full-body **musculoskeletal** locomotion environment on MuJoCo, replacing
v4's planar 9-torque skeleton with MyoSuite's MyoFullBody model: 26 bodies,
53 qpos, **290 Hill-type muscles**, 82 kg.

![the model](figures/myo_model.png)

```
v5/
  myo_curriculum/
    rewards.py    loads v4's reward primitives (not a copy -- see below)
    stages.py     the two stages, as data
    env.py        the single environment class
    __init__.py   gymnasium registration + variant helper
  scripts/
    validate_env.py  smoke test: frame, Markov property, reward safety, gradient
    render.py        multi-view still or rollout figure
  tests/          42 tests
  figures/        rendered PNGs (referenced above)
```

No crutches yet. This is the "plain muscle locomotion first" step; the crutch
curriculum lives in `v4/` and is unaffected.

## Setup

MyoSuite needs **Python 3.10+**, which v4 cannot use -- it is pinned to 3.9 by
`gym<0.22`, which sconegym requires. So v5 gets its own environment:

```powershell
pip install uv
uv python install 3.12
uv venv --python 3.12 .venv-myo
uv pip install --python .venv-myo/Scripts/python.exe myosuite pytest matplotlib pillow
```

Then, from `v5/`:

```bash
../.venv-myo/Scripts/python.exe scripts/validate_env.py
../.venv-myo/Scripts/python.exe scripts/render.py
../.venv-myo/Scripts/python.exe -m pytest tests -q
```

Nothing here needs SCONE, sconegym, sconepy or a Hyfydy licence. The model
ships inside the `myosuite` wheel under Apache-2.0.

### The bigger MyoSkeleton is opt-in, and is your decision

`myobody.xml` is Apache-licensed and is what this package uses. MyoSuite can
additionally fetch a larger MyoSkeleton (~100 MB) from `myolab/myo_model`, but
only under a **non-commercial scientific research licence** that you have to
accept interactively:

```bash
../.venv-myo/Scripts/python.exe -m myosuite_init
```

That prompt is a licence agreement, so it is yours to accept, not something
this repo does for you. Nothing here depends on it. If you do accept it,
`MyoLocomotionEnv(model_path=...)` takes any MuJoCo muscle model, so switching
is one argument.

## What the model is

| | |
|---|---|
| bodies | 26 -- pelvis, full lumbar spine (L1-L5), torso, neck, head, both legs with patellae |
| qpos / qvel | 53 / 52 |
| muscles | 290 Hill-type, with 290 activation states |
| mass | 82.04 kg |
| speed | ~850 env steps/s single-threaded (`frame_skip=10`, so ~8.5k physics steps/s) |

**No arms.** `myobody.xml` is full-body from the pelvis up through the head,
but has no upper limbs. That does not matter for unaided locomotion and does
matter later: welding crutches needs forearms. When that time comes,
`myo_sim/body/myobody_simpleupper.xml` is the variant to use -- it has
`humerus/ulna/radius/hand` on both sides driven by 14 torque actuators plus 86
muscles, which is the same split v4 used (muscle-free torque arms holding the
crutches).

**Only 17 joints are independent.** The model has 46 non-root joints; 29 of
them are driven by equality constraints -- the knee's rolling contact
(`knee_angle_*_translation*`, `_rotation*`, `_beta_*`) follows `knee_angle_*`,
and the lumbar levels distribute the three trunk angles. `env.py` writes only
the 17 independent ones at reset and lets the solver resolve the rest;
perturbing a constrained joint directly would fight the constraint rather than
pose the model.

## What carries over from v4, and what does not

### The reward machinery carries over, by import

`v5/myo_curriculum/rewards.py` loads `v4/sconegym_crutch_v4/rewards.py` rather
than copying it. That file has no simulator, gym or Python-version dependency,
and it encodes design work worth keeping: every term is in `[0, 1]`, terms
compose as a weighted geometric mean so no term can be farmed in isolation, and
`termination_report()` proves the per-step reward cannot go negative. Copying
it was the exact failure v4 was written to undo, so v5 imports it by path --
which also skips v4's `__init__`, which imports gym 0.21.

### The terms do not carry over

Two kinds are new, because the body is:

- **Out-of-plane terms.** v4's model was strictly sagittal: every joint was a
  z-hinge, so falling sideways and turning were structurally impossible. Here
  they are the most common failure, hence `lateral` and `heading`, and
  termination checks trunk tilt as well as height.
- **An effort term.** 290 muscles are hugely overactuated -- many activation
  patterns produce the same motion and most are co-contraction a person would
  not use. `effort` selects among them. Nine torque actuators did not need it.
  It is deliberately loose (flat below a mean activation of 0.15): a tight
  effort penalty on an overactuated model suppresses motion before it
  suppresses co-contraction.

| stage | task | terms |
|---|---|---|
| **A** | hold a standing posture | height, upright, effort |
| **B** | walk forward at 1.2 m/s | + velocity, lateral, heading |

Stage A has **no velocity term at all**, rather than a velocity term with a
target of zero. A zero target rewards freezing, and would make the A→B
transition a discrete change in what the reward measures.

### Actions are activations, not torques

The policy emits `[-1, 1]` and the environment maps it to activation `[0, 1]`,
so **a zero action is half activation, not rest.** v4's action rate limiter is
kept, and `prev_action` stays in the observation for the reason v4 documents.

## Foot contact: two balls per foot, as in the cane model

The `.hfd` cane model this project is built around gives each foot exactly two
contact spheres -- `heel` and `toe`, radius 0.03, plus one per crutch tip.
MyoSuite's MyoFullBody instead wraps each foot in five capsules and an
ellipsoid: a rolling sole, with no clean heel/toe split to read gait phase
from. This environment replaces MyoSuite's scheme with the cane model's two
balls, so contact behaves the way the rest of the project assumes.

The swap happens through `MjSpec` at construction, so there is no second model
file to keep in step with the `myosuite` package and no mesh paths to rewrite.
The MyoSuite geoms are not deleted, only taken out of collision, so the foot
still renders as a foot. `ball_contacts=False` restores the stock scheme for a
comparison; the heel/toe grouping follows the flag.

The balls are **massless** (`density=0`): left at MuJoCo's default they added
0.3 kg and shifted the feet's inertia, which is not what a contact primitive
should do. Their positions come from this model's own sole geometry rather
than the `.hfd` numbers, whose calcn frame is scaled and oriented differently;
`BALL_CONTACTS` documents the derivation and a test pins the stance they give.

The heel ball sits on `calcn` and the toe ball on `toes`. The `.hfd` puts both
on `calcn`, but its own comment says why that was free -- "the mtp joint here
is locked 0..0 anyway, so this changes nothing kinematically". Here mtp is a
live joint with muscles crossing it, so the equivalent choice is to let the toe
ball follow the toes.

### What the swap exposed

Reducing eleven contact geoms to four made a latent defect obvious. The solved
stance had the feet **0.055 m apart laterally** -- narrower than one foot is
wide -- because `hip_adduction` was pinned to zero and nothing constrained
stance width. MyoSuite ships **14 leg-to-leg collision pairs**, and those
bypass `contype`/`conaffinity` entirely, so disabling the foot geoms did not
disable them: the foot-to-foot pair fired with **569 N** of spurious force,
more than half body weight, swamping the real ground reaction.

The stance solve now constrains stance width (0.17 m, a touch wider than the
model's 0.154 m hips) and fore-aft alignment. Left free the latter settled on a
0.110 m **split stance**, which loads the feet diagonally -- one heel and the
opposite toe.

### What is guaranteed, and what is not

All four balls **touch** the floor at every reset, to about 1e-7 m, with no
self-collision. Whether a given ball also **carries load** is the solver's to
decide: four coplanar point contacts against three equilibrium equations is
indeterminate -- the wobbly-table problem -- so one of them routinely comes out
at zero even while touching. That is reported by `validate_env.py` rather than
asserted.

And the heels unload within a few steps, because the body pitches forward.
Holding the stance is stage A's job, not the reset's.

## The standing stance is solved, not inherited

The shipped keyframe is a **mid-stride pose, not a stance**: the hips differ by
0.43 rad, `hip_rotation_r` is -35 degrees, the feet are 0.23 m apart along the
facing direction and the trunk is flexed 30 degrees. Dropping the model from it
lands it **on its toes with both heels 23 mm in the air** -- near-singular,
biased toward the ankle plantarflexors from step one, and with no heel contact
for a gait reward to read.

A symmetric pose cannot fix it either. With identical joint angles and level
hips, the right femur is **23.5 mm shorter than the left**, so one foot is
always off the ground.

So `_solve_stance_pose()` solves for the stance instead, by damped
Gauss-Newton over six leg angles, two trunk angles, pelvis height and root
pitch/roll, against:

- each foot's heel **and** toe at `z = 0` (plantigrade, both feet),
- the COM horizontally over the **base of support**,
- the trunk axis vertical, and
- stance width and fore-aft foot alignment.

It converges to a residual of about `1e-12`. `lat_bending` settles at
-0.126 rad, which is the model taking up its own leg-length difference -- what
a person with a leg-length discrepancy does.

Reset randomisation then breaks it again, in two ways: 0.02 rad at the ankle
tilts a 0.2 m foot by 4 mm and lifts the heel, and a hip or knee perturbation
moves a whole foot by up to 30 mm. `_seat_feet()` projects back onto the
constraints after randomising, sharing the same unknowns because a per-leg
solve is not enough -- the knee's lower limit is full extension, so once a leg
is straight it cannot lengthen and the per-leg Newton stalls a millimetre
short. Nine unknowns against six constraints, solved least-norm, costs about
8 ms per reset.

Measured over 40 resets: **both heels on the floor 40/40** (worst gap 1e-7 m),
total contact load 0.87-1.35 body weights, COM over the base to 1e-12 m.

Two honest caveats. A heel resting at `z = 0` does not always *carry* force --
the split between heel and toe is the contact solver's to make in a statically
indeterminate stance, and both heels are loaded on about 30 of 40 resets.
And without a trained policy the model pitches forward and unloads its heels
within about six steps; holding the stance is the policy's job, not the reset's.

### Contact load is reported heel and toe separately

`contact_loads()` returns `[heel_r, toe_r, heel_l, toe_l]` as fractions of body
weight, and that split is in the observation. A single per-foot total cannot
distinguish heel strike from toe-off, and that difference *is* gait phase.
`foot_contact_loads()` and `heel_contact_loads()` are the obvious reductions.

## Three things that were wrong first, and are now tested

These are the bugs a training run would have hidden rather than surfaced, so
each has a regression test.

**1. The body frames are locally y-up inside a z-up world.** The model's world
is z-up, but its body frames carry the OpenSim convention -- the torso frame's
own `+y` is what points at the sky. Reading a body's local `+z` as "up"
measured an **89° trunk tilt on a model standing perfectly straight**, and
every episode terminated on step 1. Posture is now derived from the
pelvis-to-head vector and heading from the hip-to-hip vector, which are
unambiguous whatever the frame convention.

**2. The neutral pose faces 109°, not +x.** `initial_forward_velocity` was
being written to the root's world-x dof, so stage B launched the model mostly
*sideways* -- the `lateral` term scored 0.23 at reset, punishing the
environment's own initial condition. The push now goes along the model's facing
direction, and everything directional is measured against the heading recorded
at reset. `lateral` now scores 1.00 there.

**3. Stage B's velocity target had no gradient from standstill.** At the
initial sigma of 0.35 against a 1.2 m/s target, a motionless model scored
`exp(-(1.2/0.35)²) = 7e-6`, which the term floor then flattened completely.
This is the same trap v4's stage D fell into and its README documents. Sigma is
now 0.80, so standstill scores 0.11 and every 0.1 m/s gained is worth
something. `validate_env.py` checks every term is off its floor at reset.

## Not done

In the order I would tackle them:

1. **Hold the stance.** The reset is now a balanced plantigrade stance, but
   nothing keeps it: the model pitches forward and unloads both heels within
   about six steps. That is what stage A is for.
2. **Train something.** There is no trained policy yet -- the rollouts in
   `render.py` are an untrained body falling over, which is what an untrained
   muscle body does. MyoSuite ships `myoLegWalk-v0` baselines worth reading
   first, and DEP-RL is designed for exactly this overactuation (it was v4's
   "Not done" item 3, and unlike v4's 9-torque model this body is the kind
   DEP-RL targets).
3. **A reference trajectory and RSI.** v4's `trajectory.py` loads OpenSim
   `.sto` files and its keyframe/chaining curriculum is the part most worth
   porting. It needs a reference whose dof names and sign conventions match
   this model -- and note the warning in v4's README section 11 about the
   existing reference disagreeing with the model on knee sign.
4. **Arms, then crutches**, on `myobody_simpleupper.xml`.
5. **Pathological gait.** MyoSuite ships `myoFati*` (fatigue) and `myoSarc*`
   (sarcopenia) environment variants. Those are the mechanism for modelling
   impairment directly, rather than inferring it from an assistive device.
