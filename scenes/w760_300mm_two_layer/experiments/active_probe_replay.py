"""Audit causal command/encoder/wrench excitation at the skill-4 boundary."""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from fusion_skill_boundary import (BACK, STROKES, Observer, Task, live_pairs,
                                   task_plane, vision_layout)


CASES = {
    **STROKES,
    "u1_reseat": ("upper_B_seam_rear_reseat", Task(7, "left", 1, 1, BACK)),
    "u3_side": ("upper_B_side", Task(9, "right", 0, -1, "/World/Cube_7", (BACK,))),
}
WINDOW_S = .5
AVERAGE_END_S = 8 / 120
MIN_COMMAND_M = .000040
MAX_COMMAND_M = .000130
MAX_TCP_FOLLOW_FRACTION = .35
MIN_FORCE_RISE_N = .30
TARGET_BAND_M = .002
TERMINAL_WALL_MAX_GAP_M = .00045
TERMINAL_NEIGHBOR_MAX_GAP_M = .0018
TRUTH_HOLD_S = .25


def command_kinematics(isaac_sim_dir):
    exts = isaac_sim_dir / "exts"
    sys.path.insert(0, str(exts / "isaacsim.robot_motion.lula" / "pip_prebundle"))
    import lula

    fr3 = exts / "isaacsim.robot_motion.motion_generation" / "motion_policy_configs" / "FR3"
    urdf = fr3 / "fr3.urdf"
    description = fr3 / "rmpflow" / "fr3_robot_description.yaml"
    model = lula.load_robot(str(description), str(urdf))
    return model.kinematics(), {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                for file in (urdf, description)}


def response(window, average_end):
    first = np.mean(window[:average_end], axis=0)
    last = np.mean(window[-average_end:], axis=0)
    command, tcp, force = last - first
    return {"command_advance_m": float(command), "encoder_advance_m": float(tcp),
            "axial_force_rise_N": float(force)}


def review(run, name, kinematics):
    phase, task = CASES[name]
    report = json.loads((run / "runner_report.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    config = report["scene_parameters"]
    hz = report["simulation_clock"]["control_hz"]
    events = [row for row in report["phases"]
              if row["phase"] == phase and row.get("cube_id") == task.cube_id]
    start = next(row for row in events if row["status"] == "RUNNING")
    end = next((row for row in events if row["status"] == "FORCE_CONFIRMED"), None)
    first = round(start["sim_time_s"] * hz) + 1
    last = (round(end["sim_time_s"] * hz) if end
            else report["simulation_clock"]["control_steps"])
    cube = f"/World/Cube_{task.cube_id}"
    half = config["cube_size_m"] / 2
    layout = vision_layout(report, task, first / hz)
    observer = Observer("fusion", task, half, task_plane(config, task), layout, hz)
    terminal_gap = (TERMINAL_WALL_MAX_GAP_M if task.target_path == BACK
                    else TERMINAL_NEIGHBOR_MAX_GAP_M)
    yaw = config["fr3"]["bases"][task.arm][3]
    basis = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                      [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    projection = task.sign * basis[task.axis]
    offset = np.asarray(metadata["tcp_offset_sensor_m"][task.arm])

    def project(q):
        pose = kinematics.pose(np.asarray(q, dtype=float), "fr3_link8")
        return float(projection @ (pose.translation + pose.rotation.matrix() @ offset))

    window = deque(maxlen=round(WINDOW_S * hz) + 1)
    bearing = deque(maxlen=round(TRUTH_HOLD_S * hz))
    average_end = round(AVERAGE_END_S * hz)
    rows = translated = target_steps = excited = estimates = matched = passive = passive_matched = 0
    terminal_estimates = terminal_matched = 0
    sat_estimates = sat_matched = 0
    first_detection = first_unsupported = first_native = None
    first_terminal_detection = first_unsupported_terminal = None
    first_sat_detection = first_unsupported_sat = None
    max_base_variation = max_command = 0.
    base_origin = None
    sidecar = run / "upper5_active_probe.jsonl"
    probe_rows = ({row["physics_step"]: row for line in sidecar.read_text().splitlines()
                   if (row := json.loads(line))["arm"] == task.arm and first <= row["physics_step"] <= last}
                  if name == "u5_rear" and sidecar.exists() else {})
    probe_bearings = {}
    probe_command_errors = []
    probe_estimates = probe_matches = 0
    contacts = run / "contact_states"
    with gzip.open(run / "wrench.jsonl.gz", "rt") as commands, \
            gzip.open(contacts / "sensor_inputs.jsonl.gz", "rt") as sensors, \
            gzip.open(contacts / "reference.jsonl.gz", "rt") as truth:
        for command, sample, reference in zip(commands, sensors, truth, strict=True):
            q, sample, reference = json.loads(command), json.loads(sample), json.loads(reference)
            tick = q["physics_step"]
            if tick > last:
                break
            if tick < first:
                continue
            if tick != sample["physics_step"] or tick != reference["physics_step"] or tick != first + rows:
                raise ValueError(f"{run}: unmatched sensor/command/truth at {tick}")
            rows += 1
            wrist = q[task.arm]
            tcp = task.sign * wrist["tcp_position_m"][task.axis]
            base = tcp - project(wrist["q_rad"])
            if base_origin is None:
                base_origin = base
            max_base_variation = max(max_base_variation, abs(base - base_origin))
            requested = project(q["control"]["position_targets_rad"][task.arm])
            force = np.asarray(sample[f"{task.arm}_wrench_world"][:3])
            axial = float(-task.sign * force[task.axis])
            observed = observer.update(sample, force, axial, float(np.linalg.norm(force)))
            window.append((requested, tcp, axial))

            # Ground truth stays out of all the observation and response calculations.
            if reference["active_cube_id"] != task.cube_id or not reference["cube_contacts"]["valid"]:
                raise ValueError(f"{run}: invalid independent reference at {tick}")
            pairs = live_pairs(reference, cube)
            target = task.target_path in pairs and f"{task.arm}_cup" in pairs.values()
            target_steps += target
            bearing.append(target)
            sustained_target = len(bearing) == bearing.maxlen and sum(bearing) >= .8 * len(bearing)
            if tick in probe_rows:
                probe_bearings[tick] = target
                probe_command_errors.append(abs(requested + base_origin - probe_rows[tick]["command_y_m"]))
            if sustained_target and first_native is None:
                first_native = tick
            if observed["state"] == "TARGET_CANDIDATE":
                passive += 1
                passive_matched += sustained_target
            if len(window) != window.maxlen:
                continue
            evidence = response(list(window), average_end)
            advance = evidence["command_advance_m"]
            max_command = max(max_command, advance)
            near = observed["target_gap_m"]
            sat = observed.get("target_sat_gap_m")
            axis_eligible = near is not None and abs(near) <= TARGET_BAND_M
            sat_eligible = sat is not None and -TARGET_BAND_M <= sat <= terminal_gap
            translated += axis_eligible and advance > 0.
            if not MIN_COMMAND_M <= advance <= MAX_COMMAND_M:
                continue
            excited += axis_eligible
            if (abs(evidence["encoder_advance_m"]) > MAX_TCP_FOLLOW_FRACTION * advance
                    or evidence["axial_force_rise_N"] < MIN_FORCE_RISE_N):
                continue
            result = {"physics_step": tick, "time_s": sample["time_s"], **evidence,
                      "native_target_steps": int(sum(bearing))}
            if sat_eligible:
                sat_estimates += 1
                sat_matched += sustained_target
                sat_result = {**result, "estimated_gap_m": sat}
                if first_sat_detection is None:
                    first_sat_detection = sat_result
                if not sustained_target and first_unsupported_sat is None:
                    first_unsupported_sat = sat_result
            if not axis_eligible:
                continue
            estimates += 1
            matched += sustained_target
            result["estimated_gap_m"] = near
            if first_detection is None:
                first_detection = result
            if not sustained_target and first_unsupported is None:
                first_unsupported = result
            # Apply the existing controller's tighter positive-gap limit to the same candidates.
            if near <= terminal_gap:
                terminal_estimates += 1
                terminal_matched += sustained_target
                if tick in probe_rows:
                    probe_estimates += 1
                    probe_matches += sustained_target
                if first_terminal_detection is None:
                    first_terminal_detection = result
                if not sustained_target and first_unsupported_terminal is None:
                    first_unsupported_terminal = result

    if not rows or max_base_variation > 2e-5:
        raise ValueError(f"{run}: archived TCP and FR3 encoder FK disagree or the base moved during {name}")
    logged_probe = None
    if probe_rows:
        last_ticks = list(probe_rows)[-bearing.maxlen:]
        logged_probe = {"first_tick": min(probe_rows), "last_tick": max(probe_rows),
                        "logged_steps": len(probe_rows),
                        "native_target_steps": sum(probe_bearings.values()),
                        "native_target_steps_at_end": sum(probe_bearings[tick] for tick in last_ticks),
                        "final_window_logged_steps": len(last_ticks),
                        "terminal_probe_candidates": probe_estimates,
                        "terminal_candidates_with_sustained_target": probe_matches,
                        "max_command_fk_difference_m": max(probe_command_errors),
                        "runtime_response": end.get("tactile_probe") if end else None}
        event = next((row for row in events if row["status"] == "SENSOR_PROBE_CANDIDATE"), None)
        if event is not None:
            tick = round(event["sim_time_s"] * hz)
            samples = [probe_bearings[i] for i in range(tick - bearing.maxlen + 1, tick + 1)
                       if i in probe_bearings]
            logged_probe["sensor_candidate"] = {
                "physics_step": tick, "estimated_gap_m": event["estimated_gap_m"],
                "response": event["tactile_probe"], "reference_frames": len(samples),
                "native_target_frames": sum(samples),
            }
    return {"task": vars(task), "run": run.name, "phase": phase, "frames": rows,
            "first_sustained_physical_target_step": first_native, "physical_target_frames": target_steps,
            "near_target_forward_command_windows": translated, "bounded_excitation_windows": excited,
            "maximum_observed_command_advance_m": max_command, "maximum_base_fk_variation_m": max_base_variation,
            "probe_candidates": estimates, "candidates_with_sustained_target": matched,
            "passive_candidates": passive, "passive_candidates_with_sustained_target": passive_matched,
            "first_probe_candidate": first_detection, "first_candidate_without_sustained_target": first_unsupported,
            "terminal_probe_candidates": terminal_estimates,
            "terminal_candidates_with_sustained_target": terminal_matched,
            "first_terminal_probe_candidate": first_terminal_detection,
            "first_terminal_candidate_without_sustained_target": first_unsupported_terminal,
            "sat_neighbor_probe_candidates": sat_estimates,
            "sat_neighbor_candidates_with_sustained_target": sat_matched,
            "first_sat_neighbor_probe_candidate": first_sat_detection,
            "first_sat_neighbor_candidate_without_sustained_target": first_unsupported_sat,
            "logged_active_probe": logged_probe,
            "run_error": report.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-sim-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, action="append", nargs=2, metavar=("RUN", "STROKE"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    kinematics, models = command_kinematics(args.isaac_sim_dir)
    cases = [review(Path(run), name, kinematics) for run, name in args.case]
    result = {
        "mode": "Causal replay on recorded excitation, no simulated command was changed",
        "filters": {"window_s": WINDOW_S, "average_end_s": AVERAGE_END_S,
                    "command_range_m": [MIN_COMMAND_M, MAX_COMMAND_M],
                    "max_encoder_follow_fraction": MAX_TCP_FOLLOW_FRACTION,
                    "min_axial_force_rise_N": MIN_FORCE_RISE_N, "camera_geometry_band_m": TARGET_BAND_M,
                    "exploratory_terminal_gap_m": {"min": -TARGET_BAND_M,
                                                   "wall_max": TERMINAL_WALL_MAX_GAP_M,
                                                   "neighbor_max": TERMINAL_NEIGHBOR_MAX_GAP_M}},
        "threshold_origin": "FR3 re-seat controller window, displacement and force-rise thresholds; fixed maximum from lateral probe travel",
        "exploratory_neighbor_geometry": "Reuse separating-axis clearance for the nominated neighbor with the same terminal bounds and response rule",
        "truth_window_s": TRUTH_HOLD_S,
        "truth_reference": "At least 80% of trailing frames have both the expected PhysX target and the active cup pair loaded",
        "robot_model_sha256": models,
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
