"""Compare quiet wrist force with native suction and supported cup separation."""

from collections import Counter
from pathlib import Path
import argparse
import json
import math

from compare_vision_states import load
from seven_observer import SevenObserver
from seven_reference import MIN_IMPULSE_NS, SevenReference
from task_state_labels import TASK_STATES
from thesis_metrics import confusion_metrics


WRIST_FORCE_N = .30
QUIET_TIME_S = .050
ARMS = ("left", "right")


def cup_contact(snapshot):
    return any(
        pair["label"].endswith("_cup") and pair["current_step_observed"]
        and pair["normal_impulse_Ns"] is not None
        and (pair["normal_impulse_Ns"] > MIN_IMPULSE_NS
             or (pair["separation_m"] is not None and pair["separation_m"] <= 0.))
        for pair in snapshot["pairs"]
    )


def withdrawal_steps(header):
    # Phase markers select scoring samples. The observers never receive them.
    events = sorted((round(event["sim_time_s"] * header["hz"]), event["phase"], event["status"])
                    for event in header.get("phase_times_for_scoring_only", ()))
    at = {(phase, status): step for step, phase, status in events}
    if header["case"] == "a":
        keys = (("upper_release", "OPENING_BOTH_SIDE_GRIPPERS"),
                ("A_withdraw_up", "CLEAR_PHYSX_EXECUTION"),
                ("upper_A_capture", "LANDED_RELEASED_RETRACTED"))
    elif header["case"] in ("u4", "u5"):
        keys = (("upper_B_front_unload", "RUNNING"), ("upper_B_park", "RUNNING"),
                ("upper_B_release", "TOOLS_PARKED"))
    else:
        return {}
    if any(key not in at for key in keys):
        return {}
    return dict(zip(("start", "separation_end", "parked"), (at[key] for key in keys)))


def score(path):
    header, samples = load(path)
    config = header["config"]
    reference = SevenReference(config, header["hz"], header["reference_positions"])
    observers = {"fused_5mm": SevenObserver(config, "fusion_compensated", revised=True),
                 "fused_1p5mm": SevenObserver(config, "fusion_compensated", revised=True,
                                             separation_margin_m=.0015)}
    traces = {name: [] for name in ("wrist_quiet", *observers)}
    physical_quiet = Counter()
    examples = {}
    release = {}
    withdrawal = withdrawal_steps(header)
    checkpoints = {}
    both_sealed_before = False
    quiet_since = None
    quiet_touch_steps = longest_quiet_touch = 0
    for sensor, physical in samples:
        now, step = sensor["time_s"], sensor["physics_step"]
        assert (physical["physics_step"], physical["time_s"]) == (step, now)
        truth = reference.update(physical)
        neighbors = sensor.get("optical_neighbors", header["optical_neighbors"])
        forces = {arm: math.hypot(*sensor[arm + "_wrench_world"][:3]) for arm in ARMS}
        if max(forces.values()) >= WRIST_FORCE_N:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        estimates = {name: model.update(sensor, neighbors) for name, model in observers.items()}
        predictions = {name: estimate["task_state"] for name, estimate in estimates.items()}
        quiet = quiet_since is not None and now - quiet_since >= QUIET_TIME_S - 1e-7
        predictions["wrist_quiet"] = "S0" if quiet else "UNKNOWN"
        if not header.get("score_after_step", 0) < step <= header["stop_step"]:
            continue
        label = truth["task_state"]
        for name, predicted in predictions.items():
            traces[name].append({"step": step, "truth": label, "pred": predicted})
        for marker in ("separation_end", "parked"):
            if step == withdrawal.get(marker):
                estimate = estimates["fused_5mm"]
                checkpoints[marker] = {"step": step, "truth": label, "pred": estimate["task_state"],
                                       "cup_estimates": estimate["tools"], "wrist_force_norm_N": forces,
                                       "camera_age_s": estimate["camera_age_s"]}
        snapshot = physical["cube_contacts"]
        if not snapshot["valid"]:
            continue
        sealed = bool(truth["sealed_arms"])
        touching = cup_contact(snapshot)
        both_sealed_before |= len(truth["sealed_arms"]) == 2
        if both_sealed_before and not sealed:
            release.setdefault("both_seals_open", {"step": step, "time_s": now,
                                                   "state": label, "wrist_force_norm_N": forces})
            if label == "S0":
                release.setdefault("first_s0", {"step": step, "time_s": now})
            if quiet:
                release.setdefault("first_wrist_quiet", {"step": step, "time_s": now})
        evidence = ("native_sealed" if sealed else
                    "unsealed_cup_contact" if touching else "unsealed_no_observed_cup_contact")
        physical_quiet["samples"] += 1
        physical_quiet["native_sealed_samples"] += sealed
        quiet_touch_steps = quiet_touch_steps + 1 if quiet and touching else 0
        longest_quiet_touch = max(longest_quiet_touch, quiet_touch_steps)
        if not quiet:
            continue
        physical_quiet["both_wrist_quiet"] += 1
        physical_quiet[evidence] += 1
        if evidence != "unsealed_no_observed_cup_contact" and evidence not in examples:
            examples[evidence] = {
                "step": step, "state": label, "sealed_arms": truth["sealed_arms"],
                "bottom_support": truth["support_objects"], "wrist_force_norm_N": forces,
                "wrist_torque_norm_Nm": {arm: math.hypot(*sensor[arm + "_wrench_world"][3:]) for arm in ARMS},
            }
    report = {"physical_counts": dict(physical_quiet), "examples": examples,
              "longest_quiet_touch_steps": longest_quiet_touch, "release_events": release}
    if withdrawal:
        scored = traces["fused_5mm"]
        outside_withdrawal = [row for row in scored
                              if not withdrawal["start"] < row["step"] < withdrawal["separation_end"]]
        report["withdrawal_acceptance"] = {
            "markers": withdrawal, "checkpoints": checkpoints,
            "excluded_steps": len(scored) - len(outside_withdrawal),
            "s1_withdrawal_excluded": confusion_metrics(outside_withdrawal, TASK_STATES)["S1"],
            "s1_excluded_truth_steps": sum(row["truth"] == "S1" for row in scored)
            - sum(row["truth"] == "S1" for row in outside_withdrawal),
        }
    return traces, report


def summarize(traces):
    return {name: confusion_metrics(trace, TASK_STATES)["S0"] for name, trace in traces.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--light-touch", type=Path)
    args = parser.parse_args()
    reports, pooled = {}, {}
    physical = Counter()
    for path in args.clips:
        traces, report = score(path)
        reports[path.name] = {"S0": summarize(traces), **report}
        physical.update(report["physical_counts"])
        for name, rows in traces.items():
            pooled.setdefault(name, []).extend(rows)
    result = {"wrist_force_max_N": WRIST_FORCE_N, "quiet_time_s": QUIET_TIME_S,
              "clips": reports, "pooled_S0": summarize(pooled),
              "pooled_physical_counts": dict(physical)}
    if args.light_touch:
        traces, report = score(args.light_touch)
        result["light_touch_control"] = {"source": args.light_touch.name, "S0": summarize(traces), **report}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
