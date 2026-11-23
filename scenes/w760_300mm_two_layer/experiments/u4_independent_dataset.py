"""Aggregate frozen-observer U4 trials and compute bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRIALS = [
    ("035_retract25um", 0.35, "u4_pose_sled_035_retract25um.json", "weak_recovery"),
    ("070_observer", 0.70, "u4_pose_sled_070_observer.json", "weak_reentry"),
    ("100_observer", 1.00, "u4_pose_sled_100_observer.json", "strong_boundary"),
    ("035_depen01c", 0.35, "u4_pose_sled_035_depen01c.json", "strong_first_diagnostic"),
    ("035_observer5", 0.35, "u4_pose_sled_035_observer5.json", "strong_first_diagnostic"),
    ("035_depen01b", 0.35, "u4_pose_sled_035_depen01b.json", "strong_first_diagnostic"),
    ("035_retract30um", 0.35, "u4_pose_sled_035_retract30um.json", "over_retracted_diagnostic"),
]


def bootstrap(values, statistic, seed=20261124, samples=20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "estimate": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in draws])
    return {"n": int(len(values)), "estimate": float(statistic(values)),
            "ci95": [float(np.percentile(estimates, 2.5)),
                     float(np.percentile(estimates, 97.5))]}


def main(root, output):
    root = Path(root)
    rows = []
    for name, target, filename, category in TRIALS:
        data = json.loads((root / filename).read_text())
        weak = data["weak_reload"]
        observer = data["observer"]
        force = data["truth"]["side_force_after_reload_N"]
        s6 = observer.get("first_correct_s6")
        rows.append({
            "name": name,
            "target_N": target,
            "category": category,
            "measured_min_N": force["min"],
            "measured_median_N": force["median"],
            "measured_max_N": force["max"],
            "weak_ticks": force["ticks_in_0.1_to_1_N"],
            "strong_ticks": weak["strong_contact_ticks"],
            "strict_independent_weak_episodes": weak["positive_contact_episodes"],
            "s6_delay_s": observer.get("s6_delay_from_sustained_reload_s"),
            "recovery_event_delay_s": observer.get("recovery_event_delay_s"),
            "false_s6_ticks": observer.get("false_s6", {}).get("positive_ticks", 0),
            "false_side_loaded_ticks": observer.get("false_side_loaded", {}).get("positive_ticks", 0),
            "s6_truth_force_N": None if s6 is None else s6.get("truth_side_N"),
        })
    accepted = [row for row in rows if row["category"] in ("weak_recovery", "weak_reentry", "strong_boundary")]
    weak_positive = [row for row in accepted if row["weak_ticks"] >= 3]
    main_summary = None
    main_path = root / "main_validation_replay_summary.json"
    if main_path.exists():
        main_summary = json.loads(main_path.read_text())
    delays = [row["s6_delay_s"] for row in accepted if row["s6_delay_s"] is not None]
    false_rates = [row["false_s6_ticks"] for row in accepted]
    result = {
        "dataset_definition": {
            "source": "independent U4 pose-controlled support trials",
            "frozen_observer": True,
            "strict_weak_positive": "at least 3 prior off ticks, at least 3 consecutive 0.1–1 N ticks, and no earlier >1 N contact in the same contact episode",
            "accepted_trials": [row["name"] for row in accepted],
            "diagnostic_trials": [row["name"] for row in rows if row not in accepted],
        },
        "trials": rows,
        "aggregate": {
            "accepted_trials": len(accepted),
            "weak_interval_trials": len(weak_positive),
            "strict_independent_weak_positive_trials": sum(row["strict_independent_weak_episodes"] > 0 for row in accepted),
            "s6_delay_s": bootstrap(delays, np.mean),
            "false_s6_ticks_per_trial": bootstrap(false_rates, np.mean),
            "weak_interval_recall": sum(row["weak_ticks"] >= 3 and row["s6_delay_s"] is not None for row in accepted) / len(weak_positive) if weak_positive else None,
            "zero_false_s6_trials": sum(row["false_s6_ticks"] == 0 for row in accepted),
        },
        "main_validation_replay": main_summary,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
