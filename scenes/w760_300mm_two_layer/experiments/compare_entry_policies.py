"""Summarize physical upper-5 trials and replay first-gate policy decisions."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from skill_selection import Skill, entry_event, select_skill


POLICIES = ("fixed", "gap_fsm", "task_aware")
PHYSICAL_POLICIES = (*POLICIES, "task_aware_centered", "task_aware_sequential", "historical_realign")


def compare_observation(gate, edge):
    event = entry_event(
        gate["left_gap_m"], gate["right_gap_m"], gate["projected_cube_width_m"],
        edge, gate["minimum_gap_per_side_m"],
    )
    if event == "ENTRY_CLEAR":
        return event, dict.fromkeys(POLICIES, "CONTINUE_APPROACH")
    skill = select_skill(Skill.APPROACH, event)
    return event, {
        "fixed": "REFUSE_ENTRY",
        "gap_fsm": "REALIGN",
        "task_aware": "REALIGN" if skill == Skill.REALIGN else "REFUSE_ENTRY",
    }


def half_width_x(pose, edge):
    w, x, y, z = pose["orientation_wxyz"]
    projection = (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))
    return edge / 2 * sum(map(abs, projection))


def read_trial(run, policy):
    report = json.loads((run / "runner_report.json").read_text())
    experiment = report["upper5_experiment"]
    if experiment["entry_realign"] != (policy != "fixed"):
        raise ValueError(f"{run}: declared policy disagrees with the actual realignment setting")
    centering = experiment.get("center_correction")
    if centering is True:
        centering = "simultaneous"
    expected_centering = {"task_aware_centered": "simultaneous", "task_aware_sequential": "sequential"}.get(policy)
    if (centering or None) != expected_centering:
        raise ValueError(f"{run}: declared policy disagrees with the actual centering setting")
    row = report["upper_row"]
    input_hash = row["input_checkpoint"]["sha256"]
    edge = report["scene_parameters"]["cube_size_m"]
    phases = report["phases"]
    gates = [p for p in phases if p["phase"] == "upper_B_straight_entry_gate"]
    if not gates:
        raise ValueError(f"{run}: no recorded upper-5 entry gate")
    initial = gates[0]
    event, replay = compare_observation(initial, edge)
    corrected = next((p for p in gates if p.get("after_realign")), None)
    turning = next((p for p in phases if p["phase"] == "upper_B_yaw_recovery"
                    and p["status"] == "ROTATING_ON_LOWER_ROW"), {})
    recovery = any(p["phase"] == "upper_B_yaw_recovery" for p in phases)
    requested = recovery or any(
        p["phase"] == "upper_B_skill_selection" and p.get("next_skill") == Skill.REALIGN
        for p in phases
    )
    push_started = any(
        p["phase"] == "upper_B_rear" and p["status"] == "RUNNING" for p in phases
    )
    rear = next((p for p in phases if p["phase"] == "upper_B_rear"
                 and p["status"] == "FORCE_CONFIRMED"), {})
    seated = any(
        p["phase"] == "upper_slot_complete" and p.get("slot") == 5 and p["status"] == "SEATED_STABLE"
        for p in phases
    )
    support = next((entry["support"] for entry in row["completed_slots"] if entry["slot"] == 5), {})
    wrist = report.get("upper_b_wrist_watch", {})
    if seated:
        outcome = "SEATED_AFTER_REALIGN" if recovery else "SEATED_WITHOUT_REALIGN"
    elif wrist.get("stop"):
        outcome = "WRIST_CUTOFF_AFTER_REALIGN" if recovery else "WRIST_CUTOFF"
    elif push_started:
        outcome = "STOPPED_DURING_PUSH"
    elif requested:
        outcome = "STOPPED_AFTER_REALIGN_BEFORE_PUSH"
    elif initial["status"] == "BLOCKED":
        outcome = "REFUSED_BEFORE_REALIGN_OR_PUSH"
    else:
        outcome = "STOPPED_BEFORE_PUSH"

    neighbors = {p["cube_id"]: p for p in row["initial_upper_poses"]}
    left, right = neighbors[9], neighbors[10]
    inner_left = left["position_m"][0] + half_width_x(left, edge)
    inner_right = right["position_m"][0] - half_width_x(right, edge)
    x = initial["cube_position_m"][0]
    width = half_width_x({"orientation_wxyz": initial["cube_orientation_wxyz"]}, edge)
    gate_mm = 1000 * min(initial["left_gap_m"], initial["right_gap_m"])
    aligned_gate_mm = gate_mm + 500 * (initial["projected_cube_width_m"] - edge)
    runner_snapshot = run / "runner_snapshot.py"
    return {
        "run": run.name,
        "physical_policy": policy,
        "scene": report["scene_id"],
        "checkpoint_sha256": input_hash,
        "runner_snapshot_sha256": (hashlib.sha256(runner_snapshot.read_bytes()).hexdigest()
                                   if runner_snapshot.is_file() else None),
        "source_visual_runner_sha256": report["source_sha256"]["fr3_ab_visual_runner.py"],
        "yaw_perturbation_deg": experiment["requested_delta_yaw_deg"],
        "initial_true_x_mm": 1000 * x,
        "vertical_force_feedback": experiment["vertical_force_feedback"],
        "required_entry_gap_mm": 1000 * initial["minimum_gap_per_side_m"],
        "required_wrist_limit_N": wrist.get("force_limit_N", 9.0),
        "first_observation": event,
        "min_observed_gap_mm": gate_mm,
        "predicted_gap_after_yaw_only_mm": aligned_gate_mm,
        "independent_initial_x_envelope_gap_mm": 1000 * min(x - width - inner_left,
                                                             inner_right - x - width),
        "replayed_entry_decisions": replay,
        "actual_gate_statuses": [g["status"] for g in gates],
        "actual_realign_requested": requested,
        "actual_realign_started": recovery,
        "centering_path": centering,
        "actual_push_started": push_started,
        "actual_outcome": outcome,
        "actual_seated": seated,
        "physical_peak_wrist_N": (wrist.get("peak_force_N") or {}).get("value"),
        "physical_rear_hold_steps": rear.get("witnessed_target_contact_ticks"),
        "physical_seated_support_steps": support.get("positive_steps"),
        "physical_seated_support_total_steps": support.get("samples"),
        "physical_recheck_min_gap_mm": (None if corrected is None else
                                       1000 * min(corrected["left_gap_m"], corrected["right_gap_m"])),
        "planned_center_shift_x_mm": 1000 * turning.get("planned_center_shift_x_m", 0.),
        "physical_recheck_x_change_mm": (None if corrected is None else
                                         1000 * (corrected["cube_position_m"][0] - x)),
        "physical_recheck_elapsed_sim_s": (None if corrected is None else
                                          corrected["sim_time_s"] - initial["sim_time_s"]),
        "final_status": report["status"],
    }


def summarize(trials):
    groups = defaultdict(list)
    for t in trials:
        key = (t["scene"], t["checkpoint_sha256"], t["yaw_perturbation_deg"],
               round(t["initial_true_x_mm"], 1), t["vertical_force_feedback"],
               t["required_entry_gap_mm"], t["required_wrist_limit_N"])
        groups[key].append(t)
    matched = []
    for key, group in groups.items():
        counts = {}
        for policy in PHYSICAL_POLICIES:
            if policy == "historical_realign":
                continue
            same_policy = [t for t in group if t["physical_policy"] == policy]
            if same_policy:
                counts[policy] = {
                    "physical_trials": len(same_policy),
                    "seated": sum(t["actual_seated"] for t in same_policy),
                    "realign_started": sum(t["actual_realign_started"] for t in same_policy),
                    "push_started": sum(t["actual_push_started"] for t in same_policy),
                }
        matched.append({
            "scene": key[0], "checkpoint_sha256": key[1], "yaw_perturbation_deg": key[2],
            "initial_true_x_mm_rounded_to_0_1": key[3],
            "vertical_force_feedback": key[4], "required_entry_gap_mm": key[5],
            "required_wrist_limit_N": key[6], "physical_counts_by_policy": counts,
            "runs": [t["run"] for t in group],
        })
    return {
        "physical_results": trials,
        "matched_physical_conditions": matched,
        "replayed_decisions_are_decision_only": True,
        "scope": "One measured gate per trial for decision replay. Physical counts use only executed policies.",
        "true_gap_scope": "Measured active and neighbor poses projected on X. This is a side clearance envelope, not a full collision check.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("POLICY", "RUN"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for policy, _ in args.run:
        if policy not in PHYSICAL_POLICIES:
            parser.error(f"Unknown physical policy: {policy}")
    result = summarize([read_trial(Path(run), policy) for policy, run in args.run])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
