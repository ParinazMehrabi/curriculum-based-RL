"""Measure the model's neutral-pose geometry and quasi-static crutch load.

Several reward terms compare a quantity against an assumed ideal of zero, or
against a target load fraction inherited from v3. This script measures what the
model actually does so those numbers can be set from evidence instead of guessed.

It reports, for the settled standing pose:

  * crutch contact force as a fraction of body weight  -> cane_target_load_fraction
  * pelvis-to-foot x offset                            -> pelvis_lag reference
  * pelvis-to-crutch x offset                          -> crutch_forward margin

Zero torque is applied, so the model eventually falls. Only the window before
the fall is quasi-static; the script reports where that window ends.

    python scripts/calibrate.py
    python scripts/calibrate.py --stage B --steps 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_V4 = Path(__file__).resolve().parents[1]
if str(REPO_V4) not in sys.path:
    sys.path.insert(0, str(REPO_V4))

import numpy as np

import gym
import sconegym  # noqa: F401

import sconegym_crutch_v4 as scv4


def _x(env, body_name):
    body = env._find_body(body_name)
    if body is None:
        return None
    try:
        return float(env._vec_x(body.com_pos()))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="B", help="any stage that needs crutch sensing")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make(scv4.env_id_for(args.stage), strict_crutch=False)
    u = env.unwrapped
    bw = u._body_weight_n
    zero = np.zeros(9, dtype=np.float32)

    print("=" * 78)
    print("Calibration against the settled standing pose (zero torque)")
    print("=" * 78)
    print("stage           :", u.stage_spec.name)
    print("body weight (N) : %.2f" % bw)
    print("init_load       : %.2f" % u.init_load)
    print()

    u.reset(seed=args.seed)

    rows = []
    fell_at = None
    for i in range(args.steps):
        pelvis = _x(u, "pelvis")
        rows.append(
            {
                "step": i,
                "force": u.crutch_contact_force(),
                "pelvis": pelvis,
                "calcn_r": _x(u, "calcn_r"),
                "calcn_l": _x(u, "calcn_l"),
                "crutch_r": _x(u, "Crutch_r"),
                "crutch_l": _x(u, "Crutch_l"),
                "com_y": u._vec_y(u.model.com_pos()),
                "travel": u._travel_x(),
            }
        )
        if u._is_fall():
            fell_at = i
            break
        u.step(zero)

    if fell_at is None:
        print("model stayed upright for all %d steps" % len(rows))
        window = rows
    else:
        print("model fell at step %d" % fell_at)
        window = rows[: max(1, int(fell_at * 0.5))]
    print("using the first %d steps as the quasi-static window" % len(window))
    print()

    def col(name):
        return np.asarray(
            [r[name] for r in window if r[name] is not None], dtype=float
        )

    force = col("force")
    frac = force / bw
    print("-- crutch load -------------------------------------------------")
    print("force (N)        : min %.1f  median %.1f  mean %.1f  max %.1f"
          % (force.min(), np.median(force), force.mean(), force.max()))
    print("fraction of BW   : min %.4f  median %.4f  mean %.4f  max %.4f"
          % (frac.min(), np.median(frac), frac.mean(), frac.max()))
    print("steps at zero    : %d of %d" % (int((force <= 0).sum()), force.size))
    print()
    print("  suggested cane_target_load_fraction = %.3f  (window median)"
          % float(np.median(frac)))
    print("  current stage values: " + ", ".join(
        "%s=%.3f" % (k, s.terms.cane_target_load_fraction)
        for k, s in scv4.STAGES.items() if s.needs_crutch_force))
    print()

    pelvis = col("pelvis")
    print("-- neutral pose geometry (x, metres) ---------------------------")
    print("pelvis           : median %.4f" % np.median(pelvis))
    for name in ("calcn_r", "calcn_l", "crutch_r", "crutch_l"):
        c = col(name)
        if c.size == 0:
            print("%-17s: body not found" % name)
            continue
        offs = c - pelvis[: c.size]
        print("%-17s: median %+.4f   offset from pelvis %+.4f" %
              (name, np.median(c), np.median(offs)))
    print()

    foot = np.minimum(col("calcn_r"), col("calcn_l"))
    foot_off = np.median(foot - pelvis[: foot.size])
    print("  rearmost foot minus pelvis = %+.4f m" % foot_off)
    if foot_off > 0.01:
        print("  -> the feet sit AHEAD of the pelvis by default, so pelvis_lag")
        print("     measured against zero is permanently penalised. Use this as")
        print("     the reference offset, not 0.0.")
    elif foot_off < -0.01:
        print("  -> the pelvis sits ahead of the feet by default.")
    else:
        print("  -> pelvis and feet are roughly aligned; a zero reference is fine.")
    print()

    cr, cl = col("crutch_r"), col("crutch_l")
    if cr.size and cl.size:
        crutch_off = np.median(
            np.minimum(cr, cl) - pelvis[: min(cr.size, cl.size)]
        )
        print("  rearmost crutch minus pelvis = %+.4f m" % crutch_off)
        print("  current crutch_forward_margin = %.3f m"
              % scv4.STAGES["D"].terms.crutch_forward_margin)
        if crutch_off > scv4.STAGES["D"].terms.crutch_forward_margin:
            print("  -> the crutches are always ahead of the pelvis, so")
            print("     crutch_forward is a constant 1.0 and discriminates nothing.")
    print()

    com_y = col("com_y")
    print("-- height ------------------------------------------------------")
    print("com height       : start %.4f  end %.4f  (fall threshold %.2f)"
          % (com_y[0], com_y[-1], u.min_com_height))
    print("travel x         : start %+.4f  end %+.4f" % (window[0]["travel"], window[-1]["travel"]))

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
