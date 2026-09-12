"""Print the reward structure and safety analysis for every stage.

Runs with numpy alone -- no gym, no sconegym, no Hyfydy -- so it works on a
laptop as well as the training machine.

    python scripts/reward_report.py
    python scripts/reward_report.py --gamma 0.995
"""
from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "sconegym_crutch_v4"
PKG_NAME = "_scv4_report"


def load():
    if PKG_NAME not in sys.modules:
        pkg = types.ModuleType(PKG_NAME)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[PKG_NAME] = pkg
    return (
        importlib.import_module(PKG_NAME + ".rewards"),
        importlib.import_module(PKG_NAME + ".stages"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()

    rewards, stages = load()

    print("=" * 78)
    print("v4 curriculum reward report (gamma = %g)" % args.gamma)
    print("=" * 78)

    all_safe = True
    for key in stages.STAGE_ORDER:
        stage = stages.STAGES[key]
        spec = stage.reward
        weights = spec.active_weights
        total_w = sum(weights.values())
        report = spec.termination_report(args.gamma)

        print()
        print("stage %s  (%s)" % (key, stage.name))
        print("-" * 78)
        print(
            "  alive %.2f   shaping_scale %.2f   floor %.2f   %s   fall_penalty %.1f"
            % (spec.alive, spec.shaping_scale, spec.term_floor, spec.composition, spec.fall_penalty)
        )
        print("  target_vel %.3f   init_load %.2f   episode_steps %d"
              % (stage.target_vel, stage.init_load, stage.episode_steps))
        print("  crutch: force=%s pose=%s" % (stage.needs_crutch_force, stage.needs_crutch_pose))
        print("  terms (weight, normalised exponent):")
        for name in sorted(weights, key=lambda k: -weights[k]):
            print(
                "    %-16s %.2f   %.3f"
                % (name, weights[name], weights[name] / total_w)
            )

        best, _ = spec.compose({t: 1.0 for t in spec.required_terms})
        worst, _ = spec.compose({t: 0.0 for t in spec.required_terms})
        print("  step reward range : %.4f .. %.4f" % (worst, best))
        print("  min_step_reward   : %.4f" % report["min_step_reward"])
        print("  terminate value   : %.4f" % report["terminate_value"])
        print("  worst continuation: %.4f" % report["worst_continuation"])
        print("  verdict           : %s" % report["verdict"])
        if report["termination_preferred"]:
            all_safe = False

        # What does ignoring the single heaviest term cost?
        if len(weights) > 1:
            heaviest = max(weights, key=lambda k: weights[k])
            farmed = {t: 1.0 for t in spec.required_terms}
            farmed[heaviest] = 0.0
            farmed_total, _ = spec.compose(farmed)
            print(
                "  ignoring %-14s -> %.4f (%.0f%% of best)"
                % (heaviest, farmed_total, 100.0 * farmed_total / max(best, 1e-9))
            )

    print()
    print("=" * 78)
    print("all stages safe from termination-seeking:", all_safe)
    print("=" * 78)
    return 0 if all_safe else 1


if __name__ == "__main__":
    sys.exit(main())
