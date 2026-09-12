# Crutch curriculum v4

A rewrite of `sconegym_crutch_v3` that fixes four defects and collapses the four
copy-pasted stage classes into one environment. `v3` is untouched; nothing here
imports it.

```
v4/
  sconegym_crutch_v4/
    rewards.py    reward primitives + composition (no simulator dependency)
    stages.py     the four stages, as data
    env.py        the single environment class
    __init__.py   registration + variant helper
  configs/        tonic configs, one per stage
  notebooks/
    run_stages.ipynb  one episode per stage + reward plots
    figures/          rendered PNGs
  scripts/
    run_stages.py     rollout + plotting logic (the notebook calls this)
    reward_report.py  reward structure + safety analysis (numpy only)
    validate_env.py   environment smoke test (needs sconegym + Hyfydy)
  tests/          55 tests, runnable without gym or sconegym
```

## What changed and why

### 1. `prev_action` is in the observation

The action rate limiter makes the executed torque a function of the previous
action. In v3 that buffer was written and read every step but never appeared in
the observation, so the same observed state mapped to different executed
torques. The MDP was not Markov in the observation and the critic was fitting
returns to states that did not determine them — irreducible TD error that gets
worse the tighter the rate limit. At 0.05/step it takes 40 steps to cross the
action range, so the hidden state was large and slow-moving.

`env.py` now overrides `_get_obs` to append `prev_action`. Pass
`include_prev_action=False` to reproduce the old behaviour for an A/B.

### 2. Per-step reward cannot go negative

v3 stage D subtracted up to `0.45 × 5.0 = 2.25` per step for backward drift,
against a maximum `+1.55` per step and a **one-time** `−5.0` fall penalty.
Drifting backward for an episode cost about `−2250`; falling on step 3 cost
about `−5`. Terminating was the optimal policy, which reads as a stability
problem rather than a reward-spec problem.

Every term is now a value in `[0, 1]` where 1 is ideal, and penalties are terms
that fall toward 0 rather than quantities subtracted from the total. The step
reward is `alive + shaping_scale · compose(terms)`, bounded below by
`alive + shaping_scale · term_floor ≥ 0`. Termination-seeking is structurally
impossible, not merely tuned away.

`RewardSpec.termination_report(gamma)` reports the analysis for any spec, and
the environment warns at construction if a spec is unsafe. `scripts/reward_report.py`
prints it for all four stages.

### 3. Reward terms compose multiplicatively

v3 summed the terms, so standing still in stage C collected
`alive + height + posture ≈ 0.80` while ignoring `velocity` entirely. The
shaping terms are now combined as a **weighted geometric mean**, so every term
gates every other and the result stays in `[0, 1]`.

Measured effect (from `reward_report.py`): zeroing the heaviest term costs

| stage | terms | reward when heaviest term is ignored |
|---|---|---|
| A | 2 | 32% of best |
| B | 3 | 39% of best |
| C | 6 | 45% of best |
| D | 9 | 60% of best |

Note the trend: **gating power dilutes as term count grows**, because each
term's normalised exponent shrinks. Stage D's nine terms make it the weakest.
If D still refuses to walk, the lever is fewer terms or a lower `term_floor`,
not bigger weights. Set `composition="additive"` to ablate against the v3 form.

`term_floor` (default 0.05) keeps a zero-valued term from annihilating the whole
product, which matters for `pelvis_forward` — it is 0 at every episode start, so
with a hard floor of 0 stage D would have zero shaping reward and zero gradient
for the first steps of every episode.

### 4. The crutch contact force API fails loudly

v3 called `body.contact_force()` inside a `try/except` that printed once and
returned `0.0` for the rest of training. The consequences differed per stage:
in stage B a zero force landed on the Gaussian's tail and paid a constant
`+0.092`/step for a dangling crutch; in C and D it hit the hard `< 20 N` gate
and paid nothing. Either way the crutch reward — the entire point of stages B
through D — may have been measuring nothing, and v3's own warning string said
as much.

`env.py` probes the API at construction and **raises** if a stage weights a
crutch term and the probe fails. `strict_crutch=False` downgrades it to a
warning. `crutch_contact_force()` is public so you can read it directly, and
`validate_env.py` reports the force range over a rollout.

Two related fixes went in alongside:

- The hard `if force < 20.0: return 0.0` cliff (a `0 → 0.9` jump the policy can
  oscillate across) is replaced by `smoothstep` over `[cane_gate_lo_n,
  cane_gate_hi_n]`, default 10–30 N. A test asserts no jump exceeds 0.02 per
  0.1 N.
- `crutch_forward` and `pelvis_lag` now both measure against the **pelvis body**
  position. v3 mixed `model.com_pos().x` with the `pelvis_tx` dof between these
  two terms, so they disagreed about where the body was. Travel-based terms
  (`displacement`) still use `pelvis_tx`, which is zeroed at
  reset — that distinction is deliberate and documented in `env.py`.

### 5. One class instead of four

v3 had four ~550-line files that were copies of each other, which is how stage A
kept a linear velocity falloff while C and D used a Gaussian. There is now one
`CrutchCurriculumGym`, and stages are entries in `stages.STAGES`. A test asserts
every weighted term has an implementation, and another asserts the curriculum
only ever adds terms.

Also resolved: v3's stage D config declared `init_load` twice (`0.5` then
`0.4`); the second silently won. v4 uses `0.5` for all stages to match A–C.
Override it if `0.4` was intended.

### 6. Neutral-pose references, measured rather than assumed

The first real run of stage D showed four of its nine terms pinned at constants.
`scripts/calibrate.py` measured why:

| measurement | consequence |
|---|---|
| feet sit **+0.101 m** ahead of the pelvis body COM at rest | `pelvis_lag` scored `exp(-(0.101/0.05)^2) = 0.017` at rest — pinned for geometric reasons, not behavioural ones |
| crutches sit **+0.052 m** ahead, old margin was 0.050 | `crutch_forward` was a flat 1.0 and discriminated nothing |
| pelvis drifts **backward** 0.075 m under zero torque | `pelvis_forward = clip(travel/cap, 0, 1)` was 0 for an entire episode |

Both lag terms now measure deviation from `pelvis_foot_offset_ref` and
`crutch_offset_ref` in `TermParams`, which hold the measured values. Re-run
`calibrate.py` if the model or its init state changes. `pelvis_forward` is
dropped from stage D: it had no gradient and duplicates `velocity`.

Two stage-D tuning changes came from the same run. Its velocity target of
0.03 m/s was unreachable from a standstill (velocity never exceeded 0.043 while
stage C reached 0.993), so D now resets with C's forward push. And its crutch
sigma of 0.15 capped the term at 0.22 against C's 0.97, so it now matches C at
0.25 — unloading the crutch is still the goal, but it has to be reachable from
where stage C leaves the policy.

One thing the calibration did **not** support: `cane_target_load_fraction` is a
design goal, not a quantity to match. There is no quasi-static standing pose to
calibrate it against — with zero torque the model begins collapsing immediately
(COM height 0.922 to 0.858 over 36 steps, fall at step 73), so any "natural"
crutch load depends on the controller. The measured window median of 0.117
happens to sit near stage B's 0.15, which is reassuring but not a calibration.

Also worth watching: **18 of 36 steps carried zero crutch force.** Contact is
intermittent rather than steady. The smoothstep gate zeroes those steps, so the
term penalises chattering contact, which is probably right — but confirm it
once a trained policy exists.

## Setup

Python 3.9, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e "E:\Pooria\SCONE crane\Sconegym\sconegym"
```

`sconepy` is not a pip package -- it ships inside the SCONE install (found at
`C:\Program Files\SCONE\bin` on the training machine) and sconegym locates it
automatically. Loading a `.hfd` model additionally needs an **active Hyfydy
licence**; without one, `sconepy.load_model` raises before any v4 code runs.

Do not upgrade `gym`. The working environment uses a pre-0.22 release with the
old `registry.make(id, **kwargs)` API, and sconegym depends on it.

### What runs without a licence

Most of the project, which is deliberate -- the reward maths has no simulator
dependency:

| works without Hyfydy | needs Hyfydy |
|---|---|
| `pytest tests` (55 tests) | `scripts/validate_env.py` |
| `scripts/reward_report.py` | `scripts/run_stages.py` |
| notebook sections 1, 2, 7 | notebook sections 3-6 |

## Running it

```bash
# Reward structure and safety analysis. Needs only numpy.
python scripts/reward_report.py

# Tests. Needs numpy + pytest, no gym or sconegym.
python -m pytest tests -q

# Environment smoke test. Needs sconegym + Hyfydy.
python scripts/validate_env.py --stage A
python scripts/validate_env.py --stage D --steps 400

# Measure the neutral pose: crutch load as a fraction of body weight, and the
# pelvis-to-foot / pelvis-to-crutch offsets the lag terms reference.
python scripts/calibrate.py --stage B

# One episode per stage plus reward plots. Needs sconegym + Hyfydy.
# Uses half the logical cores; --cpu-fraction changes that.
python scripts/run_stages.py
python scripts/run_stages.py --policy random --seed 1 --cpu-fraction 0.5

# Or interactively:
jupyter lab notebooks/run_stages.ipynb
```

Section 7 of the notebook (`plot_gating`) needs neither gym nor a Hyfydy
licence, so the reward-gating analysis runs on any machine with numpy and
matplotlib.

### CPU budget

`scripts/run_stages.py` sets `OMP_/MKL_/OPENBLAS_/NUMEXPR_/VECLIB_*_NUM_THREADS`
at import time, before numpy loads its BLAS, and calls `torch.set_num_threads`
when torch is present. Those are cooperative limits. For a hard cap install
`psutil` and the process affinity mask is set too:

```bash
pip install psutil
```

Training, from the repository root with `v4/` on `PYTHONPATH`:

```bash
python -m deprl.main v4/configs/stage_A.yaml
```

## Configuration

**Stage configuration lives in `stages.py`, not in the YAML.** v3's configs
carried an `env_args` block, but nothing in the package read it and it was never
confirmed that tonic forwards it to the environment — which means the v3 runs may
have used class defaults rather than the YAML coefficients throughout. v4 does
not read `env_args` at all.

To change a coefficient, either edit `stages.py` or register a variant, which
routes overrides through gym's registration kwargs and therefore definitely
reaches the constructor:

```python
import sconegym_crutch_v4 as scv4
scv4.register_variant("D", "stage_d_tuned-v1", w_backward=0.25, term_floor=0.02)
```

Override namespaces, checked in order: `w_<term>` for a reward weight; then
`alive`, `shaping_scale`, `fall_penalty`, `term_floor`, `composition`; then any
`TermParams` field; then any `StageSpec` field. **Unknown names raise** — a YAML
typo should stop the run, not train 10M steps against a default nobody chose.
The v3 coefficient names (`posture_reward_coeff`, `backward_penalty_coeff`, …)
are gone and will raise if used.

Note that overriding `alive` does not adjust `shaping_scale`, so the shipped
invariant `alive + shaping_scale == 1` (max step reward of exactly 1.0) no
longer holds if you change one without the other.

## Consequences you need to know

**v3 checkpoints are not loadable.** The observation grew by 9 dimensions, so
the actor's input layer no longer matches. Either retrain from stage A or
construct with `include_prev_action=False`, which keeps the old layout along
with the non-Markov behaviour it causes. The `before_training` fields in
`configs/stage_{B,C,D}.yaml` are placeholders reading `REPLACE_ME` rather than
v3's absolute paths into `C:\Users\FUM Care\...`, which made the curriculum
unreproducible on any other machine.

**Reward magnitudes are not comparable to v3.** Every stage now tops out at
exactly 1.0 per step. v3 stage D ranged from about `+1.55` to `−2.25`. Do not
compare learning curves across versions.

## Not done

Deliberately out of scope for this pass, in the order I would tackle them:

1. **Critic warm-up and return normalisation at stage transitions.** The reward
   function still changes discretely between stages, so the critic still
   arrives mis-calibrated. Freezing the policy and training the critic alone for
   50–100k steps after each transfer, plus PopArt-style return normalisation, is
   the cheap fix.
2. **Stage-conditioned policy.** Feed the coefficient vector into the
   observation and train one policy across a continuum of settings. This removes
   the discrete transitions entirely and is the same mechanism a
   patient-parameter-conditioned twin would need later.
3. **DEP ablation.** DEP-RL targets overactuated musculoskeletal systems; this
   model has 9 torque actuators and 0 muscles. Run stage A with DEP on and off —
   either result is worth knowing.
4. **Automatic curriculum** (ALP-GMM or similar) once coefficients are
   conditioning inputs.
5. **Symmetry augmentation as a faded prior**, which is also the one inductive
   bias you must eventually release if the goal is modelling pathological
   asymmetry.
