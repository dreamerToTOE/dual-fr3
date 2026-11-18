"""Check the five supply/A/B1/alternating-B2 executions and physical contact evidence."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import gzip
import json
import math
from pathlib import Path


HOLD_STATUSES = ("FORCE_CONFIRMED", "MAINTAINED_DURING_APPROACH")


def final_row_checks(report):
    issues = []
    final = report.get("final_row_observations", {})
    rows = final.get("cubes", [])
    if (final.get("source") != "rear_RGB_marker_support_plane"
            or [row.get("cube_id") for row in rows] != [1, 2, 3, 4, 5]):
        return ["fresh final RGB observations of all five Cubes are unavailable"]
    last_release = max((row.get("sim_time_s", 0.) for row in report.get("phases", [])
                        if row.get("phase") == "B2_push_release"), default=math.inf)
    if not math.isfinite(final.get("time_s", math.nan)) or final["time_s"] <= last_release:
        issues.append("final row observations do not follow the last release")
    targets = {row["cube_id"]: row["target_center_m"] for row in report.get("completed_cubes", [])}
    for row in rows:
        cube_id = row["cube_id"]
        position, rotation = row.get("position_m", []), row.get("rotation_world", [])
        goal = targets.get(cube_id, [])
        if (len(position) != 3 or len(goal) != 3
                or not all(math.isfinite(value) for value in (*position, *goal))
                or math.dist(position, goal) > .01):
            issues.append(f"Cube_{cube_id} final row position exceeds 10 mm or is unavailable")
        if (len(rotation) != 3 or any(len(axis) != 3 for axis in rotation)
                or not all(math.isfinite(value) for axis in rotation for value in axis)
                or abs(math.atan2(rotation[1][0], rotation[0][0])) > math.radians(3.)):
            issues.append(f"Cube_{cube_id} final row yaw exceeds 3 degrees or is unavailable")
    if not issues and [row["cube_id"] for row in sorted(rows, key=lambda row: row["position_m"][0])] != [1, 3, 5, 4, 2]:
        issues.append("final row order is not Cube 1, 3, 5, 4, 2 from left to right")
    return issues


def sequence_checks(report):
    issues = []
    if report.get("status") != "PASS_FIVE_CUBE_SENSOR_SEQUENCE":
        issues.append("runner did not complete the five-Cube sensor sequence")
    if report.get("prepared_scene_conditions"):
        issues.append("the run included pre-staged Cubes")
    if report.get("attachment_model") != "Isaac_Sim_Surface_Gripper" or not report.get("surface_gripper_validated"):
        issues.append("native suction carrying is unverified")
    if report.get("physx_contacts", {}).get("unexpected_contact_count") != 0:
        issues.append("unintended robot contact is unverified or present")
    completed = report.get("completed_cubes", [])
    if (not isinstance(completed, list) or any(not isinstance(row, dict) for row in completed)
            or [row.get("cube_id") for row in completed] != [1, 2, 3, 4, 5]):
        issues.append("completed Cube IDs must be exactly 1, 2, 3, 4, 5")
    sequence = report.get("packing_sequence", [])
    if [row.get("push_direction") for row in sequence] != [-1, 1, -1, 1, 0]:
        issues.append("lateral push order must alternate left/right/left/right/straight")
    phases = report.get("phases", [])
    starts = []
    previous_release = -math.inf
    for cube_id in range(1, 6):
        rows = [row for row in phases if row.get("cube_id") == cube_id]
        staged = any(item.get("cube_id") == cube_id and "staging_offset_from_rear_m" in item
                     for item in sequence)
        expected = [
            ("supply_arrival", "READY_FOR_RGB_LOCALIZATION"),
            ("A_attach", "SURFACE_GRIPPERS_CLOSED"),
            ("A_lift", "REACHED_WITH_SURFACE_GRIPPERS"),
            ("A_over_front", "REACHED_WITH_SURFACE_GRIPPERS"),
            ("A_landing", "REACHED_WITH_SURFACE_GRIPPERS"),
            ("A_release", "REACHED_AFTER_SUCTION_RELEASE"),
            ("B1_push", "REACHED_INTERMEDIATE_LANE" if staged else "REACHED_WITH_PHYSX_CONTACT"),
            ("B2_hold", "FORCE_CONFIRMED"),
            ("B2_push_release", "CENTER_CUBE_STRAIGHT_PUSH_COMPLETE" if cube_id == 5
             else "REACHED_WITH_PHYSX_CONTACT"),
        ]
        selected = []
        for name, status in expected:
            statuses = HOLD_STATUSES if name == "B2_hold" and cube_id < 5 else (status,)
            matched = next((row for row in rows if row["phase"] == name and row["status"] in statuses), None)
            if matched is None:
                issues.append(f"Cube_{cube_id} missing {name} / {status}")
            else:
                selected.append(matched)
        if len(selected) != len(expected):
            continue
        times = [row["sim_time_s"] for row in selected]
        if (not all(math.isfinite(time) for time in times) or times[0] <= previous_release
                or times != sorted(times) or len(set(times)) != len(times)):
            issues.append(f"Cube_{cube_id} stages are out of order")
            continue
        previous_release = times[-1]
        starts.append(int(round(times[0] * 120)))
        source, lift, over, landing = (row["cube_position_m"] for row in selected[:5] if row["phase"] != "A_attach")
        if (not all(math.isfinite(value) for point in (source, lift, over, landing) for value in point)
                or lift[2] - source[2] < .15 or over[1] - source[1] < .40 or landing[1] <= .35):
            issues.append(f"Cube_{cube_id} actual A transport displacement is missing")
        if cube_id < 5:
            direction = -1 if cube_id in (1, 3) else 1
            approach = next((row for row in rows if row["phase"] == "B2_high_approach"), {})
            goal = sequence[cube_id - 1].get("goal_center_m") if len(sequence) == 5 else None
            if not isinstance(goal, list) or len(goal) != 3 or not math.isfinite(goal[0]):
                issues.append(f"Cube_{cube_id} lateral target is unavailable")
                continue
            goal_x = goal[0]
            expected_progress = max(.005, abs(goal_x - selected[6]["cube_position_m"][0]) - .01)
            progress = selected[-1].get("progress_m", 0.)
            if approach.get("direction") != direction or not math.isfinite(progress) or progress < expected_progress:
                issues.append(f"Cube_{cube_id} lacks the required signed B2 push")
        elif any(row["phase"] == "B2_high_approach" for row in rows):
            issues.append("Cube_5 incorrectly executed a lateral B2 push")
    if len(starts) != 5 or starts != sorted(starts):
        issues.append("five ordered supply execution windows are unavailable")
    return issues, starts


def force_checks(directory, report):
    """Audit every force sample and each uninterrupted role-aware support window."""
    limits = {"A": (30., 3.), "B1": (30., 3.), "B2": (10., 1.)}
    issues = []
    phases = report.get("phases", [])
    arrivals = sorted((int(round(row["sim_time_s"] * 120)), row["cube_id"])
                      for row in phases if row.get("phase") == "supply_arrival")
    if [cube_id for _, cube_id in arrivals] != [1, 2, 3, 4, 5]:
        issues.append("force audit requires five ordered supply windows")
    starts = [step for step, _ in arrivals]
    sequence = {row["cube_id"]: row for row in report.get("packing_sequence", [])
                if isinstance(row, dict) and "cube_id" in row}
    cubes = {}
    for cube_id in range(1, 6):
        role = sequence.get(cube_id, {})
        support = role.get("support_side")
        selected = [row for row in phases if row.get("cube_id") == cube_id]
        begin = next((row for row in selected if row.get("phase") == "B2_hold"
                      and row.get("status") in HOLD_STATUSES), None)
        end = next((row for row in selected if row.get("phase") == "B2_support_unload"
                    and row.get("status") == "RUNNING"), None)
        if "staging_offset_from_rear_m" in role:
            # After the lateral arm clears, the support arm becomes the rear pushing arm.
            end = next((row for row in selected if row.get("phase") == "B2_push_release"
                        and row.get("status") == "PUSH_ARM_RETRACTED_SUPPORT_HOLDING"), None)
        window = (None if begin is None or end is None else
                  [int(round(row["sim_time_s"] * 120)) for row in (begin, end)])
        if support not in ("left", "right") or window is None or window[1] < window[0]:
            issues.append(f"Cube_{cube_id} support role or complete held window is unavailable")
            window = None
        if cube_id < 5 and window is not None:
            release = next((row for row in selected if row.get("phase") == "B2_push_release"
                            and row.get("status") == "PUSH_ARM_RETRACTED_SUPPORT_HOLDING"), None)
            if release is None or not window[0] <= round(release["sim_time_s"] * 120) <= window[1]:
                issues.append(f"Cube_{cube_id} held window does not include push-arm release")
        cubes[cube_id] = {
            "cube_id": cube_id, "support_side": support, "push_side": role.get("push_side"),
            "stage_peaks": {name: {
                "samples": 0,
                "force_norm_max_N": {"left": 0., "right": 0.},
                "tcp_torque_norm_max_Nm": {"left": 0., "right": 0.},
                **({"internal_force_norm_max_N": 0.} if name == "A" else {}),
            } for name in limits},
            "held_window_steps": window,
            "support_held": {"samples": 0, "contact_frames": 0, "in_band_frames": 0,
                             "longest_dropout_frames": 0, "_dropout": 0,
                             "normal_force_min_observed_N": None, "normal_force_max_observed_N": None},
        }
    path = directory / "wrench.jsonl.gz"
    previous_step = previous_time = None
    invalid_samples = clock_errors = 0
    try:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                step, time_s = row["physics_step"], row["time_s"]
                if (not isinstance(step, int) or not math.isfinite(time_s)
                        or previous_step is not None and (
                            step != previous_step + 1
                            or abs(time_s - previous_time - 1 / 120) > 1e-7)):
                    clock_errors += 1
                previous_step, previous_time = step, time_s
                index = bisect_right(starts, step) - 1
                if not 0 <= index < len(arrivals):
                    continue
                cube = cubes.get(arrivals[index][1])
                if cube is None:
                    continue
                window = cube["held_window_steps"]
                held = window is not None and window[0] <= step <= window[1]
                stats = cube["support_held"]
                if held:
                    stats["samples"] += 1
                try:
                    vectors = {side: row[side]["tcp_gravity_corrected_world"] for side in ("left", "right")}
                    if any(len(value) != 6 or not all(math.isfinite(x) for x in value)
                           for value in vectors.values()):
                        raise ValueError("invalid wrench")
                except (KeyError, TypeError, ValueError):
                    invalid_samples += 1
                    if held:
                        stats["_dropout"] += 1
                        stats["longest_dropout_frames"] = max(stats["longest_dropout_frames"], stats["_dropout"])
                    continue
                phase = str(row.get("phase", ""))
                stage = next((name for name in limits if phase.startswith(name)), None)
                if stage is not None:
                    peak = cube["stage_peaks"][stage]
                    peak["samples"] += 1
                    for side, wrench in vectors.items():
                        peak["force_norm_max_N"][side] = max(peak["force_norm_max_N"][side], math.hypot(*wrench[:3]))
                        peak["tcp_torque_norm_max_Nm"][side] = max(
                            peak["tcp_torque_norm_max_Nm"][side], math.hypot(*wrench[3:]))
                    if stage == "A":
                        internal = math.hypot(*((a - b) / 2 for a, b in zip(vectors["left"][:3], vectors["right"][:3])))
                        peak["internal_force_norm_max_N"] = max(peak["internal_force_norm_max_N"], internal)
                if held:
                    normal = -vectors[cube["support_side"]][1]
                    contact, in_band = normal > .1, .1 < normal <= 6.
                    stats["contact_frames"] += contact
                    stats["in_band_frames"] += in_band
                    stats["_dropout"] = 0 if contact else stats["_dropout"] + 1
                    stats["longest_dropout_frames"] = max(stats["longest_dropout_frames"], stats["_dropout"])
                    minimum, maximum = stats["normal_force_min_observed_N"], stats["normal_force_max_observed_N"]
                    stats["normal_force_min_observed_N"] = normal if minimum is None else min(minimum, normal)
                    stats["normal_force_max_observed_N"] = normal if maximum is None else max(maximum, normal)
    except (OSError, EOFError, ValueError, TypeError, KeyError) as error:
        issues.append(f"force recording is unavailable or incomplete ({error})")
    if invalid_samples or clock_errors:
        issues.append(f"force recording has {invalid_samples} invalid wrenches and {clock_errors} 120 Hz clock errors")
    for cube_id, cube in cubes.items():
        for stage, (max_force, max_torque) in limits.items():
            peak = cube["stage_peaks"][stage]
            passed = (peak["samples"] > 0
                      and max(peak["force_norm_max_N"].values()) <= max_force
                      and max(peak["tcp_torque_norm_max_Nm"].values()) <= max_torque
                      and peak.get("internal_force_norm_max_N", 0.) <= 15.)
            peak["status"] = "PASS" if passed else "FAIL"
            if not passed:
                issues.append(f"Cube_{cube_id} {stage} force feedback is missing or exceeds its limits")
        stats = cube["support_held"]
        stats.pop("_dropout")
        count = stats["samples"]
        stats["contact_coverage"] = stats["contact_frames"] / count if count else 0.
        stats["normal_force_in_band_proportion"] = stats["in_band_frames"] / count if count else 0.
        stats["longest_contact_dropout_s"] = stats["longest_dropout_frames"] / 120
        window = cube["held_window_steps"]
        passed = (window is not None and count == window[1] - window[0] + 1
                  and stats["contact_coverage"] >= .98
                  and stats["normal_force_in_band_proportion"] >= .98
                  and stats["longest_contact_dropout_s"] <= .05)
        stats["status"] = "PASS" if passed else "FAIL"
        if not passed:
            issues.append(f"Cube_{cube_id} support held-window coverage or dropout fails")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues, "cubes": list(cubes.values()),
            "limits": {"A_force_N": 30., "A_torque_Nm": 3., "A_internal_N": 15.,
                       "B1_force_N": 30., "B1_torque_Nm": 3., "B2_force_N": 10., "B2_torque_Nm": 1.,
                       "support_force_band_N": [.1, 6.], "min_coverage": .98, "max_dropout_s": .05},
            "window_basis": "all raw held samples through push-arm release, ending before support unloading"}


def physical_contact_checks(directory, report, starts):
    issues, windows, evidence = [], [], []
    rear_wall = "/World/Environment/wall_back"
    targets = ("/World/Environment/wall_left", "/World/Environment/wall_right",
               "/World/Cube_1", "/World/Cube_2", rear_wall)
    staged_ids = {item["cube_id"] for item in report.get("packing_sequence", [])
                  if "staging_offset_from_rear_m" in item}
    for cube_id in range(1, 6):
        phases = [row for row in report.get("phases", []) if row.get("cube_id") == cube_id]

        def event(phase, *statuses):
            return next((row for row in phases if row["phase"] == phase and row["status"] in statuses), {})

        def tick(row):
            time_s = row.get("sim_time_s", math.nan)
            return round(time_s * 120) if math.isfinite(time_s) else -1

        arrive = tick(event("supply_arrival", "READY_FOR_RGB_LOCALIZATION"))
        attach = tick(event("A_attach", "SURFACE_GRIPPERS_CLOSED"))
        detach = tick(event("A_release", "RELEASING_SURFACE_GRIPPERS"))
        touch = tick(event("B1_touch", "TACTILE_APPROACH"))
        hold_event = event("B2_hold", *HOLD_STATUSES)
        hold = tick(hold_event)
        rear_window = (touch, hold)  # Rear acquisition loads the wall before the initial B2 hold.
        target_window = rear_window
        maintained_approach = cube_id < 5 and hold_event.get("status") == "MAINTAINED_DURING_APPROACH"
        neighbor_target = cube_id in (3, 4)
        if cube_id < 5:
            seat = event("B2_push_seat", "FORCE_CONFIRMED")
            seat_end = tick(seat)
            probe = seat.get("probe_normal_samples_N", [])
            seat_start = tick(event("B2_push_seat", "RUNNING"))
            unload = tick(event("B2_push_unload", "RUNNING"))
            sync = tick(event("B2_contact_sync", "FORCE_CONFIRMED"))
            valid_probe = (isinstance(probe, list) and bool(probe)
                           and all(isinstance(force, (int, float)) and math.isfinite(force) for force in probe))
            if maintained_approach and valid_probe:
                if cube_id in staged_ids:
                    valid_probe = len(probe) == 30 and min(probe) >= 1.5 and sum(probe) / 30 >= 1.8
                    baseline = seat.get("sliding_baseline_N")
                    if baseline is not None:
                        travel = seat.get("measured_axial_travel_m", math.inf)
                        valid_probe = (valid_probe and isinstance(baseline, (int, float)) and math.isfinite(baseline)
                                       and sum(probe) / 30 >= baseline + .4
                                       and isinstance(travel, (int, float)) and 0. <= travel <= .0005)
                elif neighbor_target:
                    valid_probe = len(probe) == 30 and min(probe) >= 6.
                else:
                    valid_probe = len(probe) == 3 and probe[0] >= 6. and sum(probe) / 3 >= 4.
            probe_start = seat_end - len(probe) + 1 if valid_probe else -1
            if (not 0 <= hold < seat_start <= probe_start <= seat_end <= unload
                    or (maintained_approach and not hold < sync <= seat_start)):
                issues.append(f"Cube_{cube_id} paused B2 seating confirmation window is unavailable")
                probe_start = seat_end = -1
            target_window = (probe_start, seat_end)
            if cube_id in staged_ids:
                side_clear = tick(event("B2_push_release", "PUSH_ARM_RETRACTED_SUPPORT_HOLDING"))
                rear_start = tick(event("B2_staged_rear_seat", "RUNNING"))
                rear_seat = event("B2_staged_rear_seat", "FORCE_CONFIRMED")
                rear_forces = rear_seat.get("probe_normal_samples_N", [])
                rear_end = tick(rear_seat)
                rear_window = (rear_end - 29, rear_end)
                if (not seat_end <= side_clear <= rear_start <= rear_window[0] <= rear_end
                        or rear_end > tick(event("B2_support_unload", "RUNNING"))
                        or not isinstance(rear_forces, list) or len(rear_forces) != 30
                        or not all(isinstance(f, (int, float)) and math.isfinite(f) for f in rear_forces)
                        or rear_forces[0] < 5. or sum(rear_forces) / 30 < 3.):
                    issues.append(f"Cube_{cube_id} missing rear seating confirmation after lateral withdrawal")
                    rear_window = (-1, -1)
        if not 0 <= arrive < attach < detach < touch < hold:
            issues.append(f"Cube_{cube_id} supply, carry or rear-load window is unavailable")
        windows.append(((arrive, attach - 1), (attach + 1, detach), rear_window, target_window))
        evidence.append({"cube_id": cube_id, "reference_frames": 0, "supply_impulse_frames": 0,
                         "paired_attachment_frames": 0, "rear_impulse_frames": 0,
                         "required_rear_impulse_frames": 30 if cube_id in staged_ids else 1,
                         "target_impulse_frames": 0, "target_partner": targets[cube_id - 1],
                         "required_target_impulse_frames": 30 if maintained_approach and neighbor_target else 1,
                         "target_window_steps": list(target_window), "rear_window_steps": list(rear_window)})

    reference_path = directory / "contact_states/reference.jsonl.gz"
    if len(starts) == 5 and reference_path.is_file():
        count = 0
        try:
            with gzip.open(reference_path, "rt") as stream:
                for count, line in enumerate(stream, start=1):
                    frame = json.loads(line)
                    step = frame["physics_step"]
                    if step != count:
                        raise ValueError(f"reference physics step {step} does not match sample {count}")
                    index = bisect_right(starts, step) - 1
                    if index < 0 or frame["cube_contacts"]["valid"] is not True:
                        continue
                    values = evidence[index]
                    values["reference_frames"] += 1
                    supply, carry, rear, target = windows[index]
                    if carry[0] <= step <= carry[1]:
                        values["paired_attachment_frames"] += all(
                            frame["attachments"].get(side) is True for side in ("left", "right"))
                    active = f"/World/Cube_{index + 1}"
                    loaded_partners = set()
                    for pair in frame["cube_contacts"]["pairs"]:
                        impulse = pair["normal_impulse_Ns"]
                        if (pair["current_step_observed"] is True and impulse is not None
                                and math.isfinite(impulse) and impulse > 1e-8):
                            first, second = pair["collider0"], pair["collider1"]
                            if active == first:
                                loaded_partners.add(second)
                            elif active == second:
                                loaded_partners.add(first)
                    values["supply_impulse_frames"] += (
                        supply[0] <= step <= supply[1] and "/World/Environment/supply" in loaded_partners)
                    values["rear_impulse_frames"] += rear[0] <= step <= rear[1] and rear_wall in loaded_partners
                    values["target_impulse_frames"] += target[0] <= step <= target[1] and targets[index] in loaded_partners
            if count != report.get("simulation_clock", {}).get("control_steps"):
                raise ValueError("the reference recording does not cover the complete simulation clock")
        except (OSError, EOFError, ValueError, TypeError, KeyError) as error:
            issues.append(f"physical reference recording is invalid ({error})")
        for row in evidence:
            for key in ("supply_impulse_frames", "paired_attachment_frames"):
                if row[key] == 0:
                    issues.append(f"Cube_{row['cube_id']} missing physical evidence {key}")
            if row["rear_impulse_frames"] < row["required_rear_impulse_frames"]:
                issues.append(f"Cube_{row['cube_id']} loaded the rear wall in {row['rear_impulse_frames']} "
                              f"of {row['required_rear_impulse_frames']} required confirmation frames")
            if row["target_impulse_frames"] < row["required_target_impulse_frames"]:
                issues.append(f"Cube_{row['cube_id']} loaded the intended target in {row['target_impulse_frames']} "
                              f"of {row['required_target_impulse_frames']} required confirmation frames")
    else:
        issues.append("physical contact and attachment reference frames are unavailable")
    return issues, evidence


def analyze(directory):
    report = json.loads((directory / "runner_report.json").read_text())
    issues, starts = sequence_checks(report)
    issues.extend(final_row_checks(report))
    forces = force_checks(directory, report)
    issues.extend(forces["issues"])
    contact_issues, evidence = physical_contact_checks(directory, report, starts)
    issues.extend(contact_issues)
    completed = report.get("completed_cubes", [])
    for row in completed:
        if not isinstance(row, dict):
            continue
        position = row.get("camera_final_center_m", [])
        goal = row.get("target_center_m", [])
        rotation = row.get("rotation_world", [])
        if (len(position) != 3 or len(goal) != 3
                or not all(math.isfinite(value) for value in (*position, *goal))
                or math.dist(position, goal) > .01):
            issues.append(f"Cube_{row.get('cube_id')} lacks a finite camera-verified final slot")
        if len(rotation) != 3 or any(len(axis) != 3 for axis in rotation):
            issues.append(f"Cube_{row.get('cube_id')} lacks the final visual orientation")
    return {"status": "PASS_FULL_FIVE_CUBE_SEQUENCE" if not issues else "FAIL",
            "runner_status": report.get("status"), "issues": issues, "physical_evidence": evidence,
            "force_validation": forces,
            "scope": "all five supply arrivals, actual A transport, B1 rear loading and prescribed B2/straight sequence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if result["status"] == "PASS_FULL_FIVE_CUBE_SEQUENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
