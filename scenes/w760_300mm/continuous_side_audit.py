"""Audit a visible, continuous W760 first-layer run from PhysX, encoder and wrist logs."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import gzip
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr3_row_acceptance import final_row_checks, force_checks, physical_contact_checks, sequence_checks


CTRL_HZ = 120
A_ALIASES = {
    "A_lift_and_align": "A_lift",
    "A_loaded_rail_transfer": "A_over_front",
    "A_vertical_descent": "A_landing",
}
WALL_NAMES = ("wall_back", "wall_front", "wall_left", "wall_right")


def event(report, cube, phase, status):
    match = next((row for row in report["phases"] if (row["cube_id"], row["phase"], row["status"])
                  == (cube, phase, status)), None)
    return None if match is None else round(CTRL_HZ * match["sim_time_s"])


def side_checks(directory, report, front, half):
    """Verify both seals and the signed native cup normals for every carried step."""
    issues = []
    cubes = []
    for cube in range(1, 6):
        begin = event(report, cube, "A_attach", "SURFACE_GRIPPERS_CLOSED")
        end = event(report, cube, "A_release", "RELEASING_SURFACE_GRIPPERS")
        verified = event(report, cube, "A_release", "REACHED_AFTER_SUCTION_RELEASE")
        cubes.append({
            "cube_id": cube,
            "faces_by_arm": {"left": "+Y", "right": "-Y"} if cube in (2, 4) else {"left": "-X", "right": "+X"},
            "attach_step": begin,
            "release_step": end,
            "vision_verified_landing_step": verified,
            "expected_carried_steps": end - begin if begin is not None and end is not None else None,
            "carried_steps": 0,
            "both_seals_closed_steps": 0,
            "expected_cup_axis_alignment_min": None,
            "front_wall_clearance_mm_min": None,
            "floor_impulse_steps_before_landing_verification": 0,
        })
    arrivals = sorted((round(row["sim_time_s"] * CTRL_HZ), row["cube_id"]) for row in report["phases"]
                      if row["phase"] == "supply_arrival" and row["status"] == "READY_FOR_RGB_LOCALIZATION")
    starts = [step for step, _ in arrivals]
    frame_count = outside_carry = 0
    try:
        with gzip.open(directory / "contact_states/reference.jsonl.gz", "rt") as ref, \
             gzip.open(directory / "contact_states/sensor_inputs.jsonl.gz", "rt") as sensors:
            for frame_count, lines in enumerate(zip(ref, sensors, strict=True), 1):
                frame, sensor = map(json.loads, lines)
                step = frame["physics_step"]
                if step != sensor["physics_step"] or step != frame_count:
                    raise ValueError(f"misaligned sensor and reference clock at {step}")
                index = bisect_right(starts, step) - 1
                if index < 0:
                    outside_carry += any(frame["attachments"].values())
                    continue
                cube = arrivals[index][1]
                row = cubes[cube - 1]
                begin, end = row["attach_step"], row["release_step"]
                if begin is not None and begin < step and (end is None or step <= end):
                    row["carried_steps"] += 1
                    row["both_seals_closed_steps"] += all(frame["attachments"].get(side) is True
                                                         for side in ("left", "right"))
                    if frame["cube_contacts"]["valid"] is not True:
                        raise ValueError(f"invalid physical reference during Cube_{cube} A carry at {step}")
                    rotation = frame["cube_rotation_world"]
                    axis, left_inward = (1, -1) if cube in (2, 4) else (0, 1)
                    alignment = min(
                        sign * sum(component * coord[axis] for component, coord in
                                   zip(sensor["cup_normals_world"][side], rotation, strict=True))
                        for side, sign in (("left", left_inward), ("right", -left_inward))
                    )
                    if not math.isfinite(alignment):
                        raise ValueError(f"nonfinite cup-axis measurement at step {step}")
                    previous = row["expected_cup_axis_alignment_min"]
                    row["expected_cup_axis_alignment_min"] = alignment if previous is None else min(previous, alignment)
                    center, size = front
                    _, y, z = frame["cube_position_m"]
                    if abs(y - center[1]) <= half * sum(map(abs, rotation[1])) + size[1] / 2:
                        gap = (z - half * sum(map(abs, rotation[2])) - center[2] - size[2] / 2) * 1000
                        previous = row["front_wall_clearance_mm_min"]
                        row["front_wall_clearance_mm_min"] = gap if previous is None else min(previous, gap)
                else:
                    outside_carry += any(frame["attachments"].values())
                    verified = row["vision_verified_landing_step"]
                    if end is not None and verified is not None and end < step <= verified:
                        row["floor_impulse_steps_before_landing_verification"] += any(
                            pair["label"] == "floor" and pair["current_step_observed"]
                            and (pair.get("normal_impulse_Ns") or 0.) > 1e-8
                            and f"/World/Cube_{cube}" in (pair["collider0"], pair["collider1"])
                            for pair in frame["cube_contacts"]["pairs"]
                        )
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        issues.append(f"side grasp sensor/reference recording is invalid ({exc})")
    for row in cubes:
        checks = (row["carried_steps"] == row["both_seals_closed_steps"] == row["expected_carried_steps"]
                  and row["carried_steps"] > 0
                  and (row["expected_cup_axis_alignment_min"] or 0.) > .98
                  and (row["front_wall_clearance_mm_min"] or 0.) > 2.
                  and row["floor_impulse_steps_before_landing_verification"] > 0)
        row["status"] = "PASS" if checks else "INCOMPLETE_OR_FAILED"
        if not checks:
            issues.append(f"Cube_{row['cube_id']} full side grip, front-wall clearance or landed contact is unverified")
    if outside_carry:
        issues.append(f"{outside_carry} samples have suction attached outside the five A carry windows")
    if frame_count != report.get("simulation_clock", {}).get("control_steps"):
        issues.append("side grasp sensor/reference recording does not cover the complete control clock")
    return issues, cubes


def cube4_support(directory, report, force_validation):
    """Locate exact low-force spans with timestamps in the original C4 held window."""
    cube4 = next(row for row in force_validation["cubes"] if row["cube_id"] == 4)
    start = event(report, 4, "B2_hold", "MAINTAINED_DURING_APPROACH")
    end = event(report, 4, "B2_push_release", "PUSH_ARM_RETRACTED_SUPPORT_HOLDING")
    low = []
    if start is not None:
        try:
            with gzip.open(directory / "wrench.jsonl.gz", "rt") as trace:
                for line in trace:
                    sample = json.loads(line)
                    step = sample["physics_step"]
                    if step < start:
                        continue
                    if end is not None and step > end:
                        break
                    normal = -sample["right"]["tcp_gravity_corrected_world"][1]
                    if not math.isfinite(normal) or normal > .1:
                        continue
                    if low and low[-1]["last_step"] == step - 1:
                        low[-1]["last_step"] = step
                        low[-1]["end_phase"] = sample["phase"]
                    else:
                        low.append({"first_step": step, "last_step": step,
                                    "start_phase": sample["phase"], "end_phase": sample["phase"]})
        except (OSError, EOFError, KeyError, ValueError, TypeError):
            pass  # The shared force audit records an unavailable or malformed trace.
    for span in low:
        span["duration_ms"] = 1000 * (span["last_step"] - span["first_step"] + 1) / CTRL_HZ
    stats = cube4["support_held"]
    return {
        "arm": "right",
        "held_window_steps": (start, end),
        "normal_force_band_N": [.1, 6.],
        "in_band_fraction": stats["normal_force_in_band_proportion"] if stats["samples"] else None,
        "samples": stats["samples"],
        "dropout_count": len(low),
        "longest_dropout_ms": (stats["longest_contact_dropout_s"] * 1000 if stats["samples"]
                               else max((span["duration_ms"] for span in low), default=None)),
        "longest_dropout": max(low, key=lambda span: span["duration_ms"], default=None),
        "limits": {"minimum_in_band_fraction": .98, "maximum_dropout_ms": 50.},
        "status": stats["status"] if start is not None and end is not None else "INCOMPLETE",
    }


def row_summary(rows, half):
    if len(rows) != 5 or set(row["cube_id"] for row in rows) != set(range(1, 6)):
        return None
    oriented = []
    for row in sorted(rows, key=lambda row: row["position_m"][0]):
        if "rotation_world" in row:
            axis = row["rotation_world"][0]
        else:
            w, x, y, z = row["orientation_wxyz"]
            axis = (1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y))
        extent = half * sum(map(abs, axis))
        x = row["position_m"][0]
        oriented.append((row["cube_id"], x - extent, x + extent))
    gaps = {f"{left[0]}-{right[0]}": 1000 * (right[1] - left[2])
            for left, right in zip(oriented[:-1], oriented[1:], strict=True)}
    return {"order_left_to_right": [row[0] for row in oriented], "oriented_x_projected_gap_mm": gaps}


def analyze(directory):
    report = json.loads((directory / "runner_report.json").read_text())
    # Physical side operations use more descriptive phase names than the original run.
    phases = list(report["phases"])
    for row in phases:
        if row["phase"] in A_ALIASES and row["status"] == "CLEAR_PHYSX_EXECUTION":
            report["phases"].append(dict(row, phase=A_ALIASES[row["phase"]], status="REACHED_WITH_SURFACE_GRIPPERS"))
    issues, starts = sequence_checks(report)
    issues.extend(final_row_checks(report))
    continuous = report.get("continuous_run", {})
    if report.get("first_layer_side_only") is not True or continuous.get("restored_cube_ids") != []:
        issues.append("the report does not identify an uninterrupted side-only run with all cubes supplied live")
    force_validation = force_checks(directory, report)
    issues.extend(force_validation["issues"])
    contact_issues, physical = physical_contact_checks(directory, report, starts)
    issues.extend(contact_issues)
    walls = {name: (center, size) for name, center, size in
             report["scene_parameters"]["static_boxes"] if name in WALL_NAMES}
    half = report["scene_parameters"]["cube_size_m"] / 2
    if set(walls) != set(WALL_NAMES) or any(abs(size[2] - .30) > 1e-9 for _, size in walls.values()):
        issues.append("the four walls are not all 300 mm tall")
    width = None
    if {"wall_left", "wall_right"} <= walls.keys():
        (left, left_size), (right, right_size) = walls["wall_left"], walls["wall_right"]
        width = right[0] - right_size[0] / 2 - left[0] - left_size[0] / 2
    if width is None or abs(width - .76) > 1e-9:
        issues.append("inner tray width differs from 760 mm")
    side_cubes = []
    if "wall_front" in walls:
        side_issues, side_cubes = side_checks(directory, report, walls["wall_front"], half)
        issues.extend(side_issues)
    mode = json.loads((directory / "vision/metadata.json").read_text()).get("execution_mode")
    if mode != "visible":
        issues.append("visible Isaac Sim mode is unverified")
    clock = report.get("simulation_clock", {})
    if not (clock.get("control_hz") == CTRL_HZ
            and clock.get("control_steps") == clock.get("physics_steps")
            and clock.get("uncommanded_physics_steps") == 0):
        issues.append("control and PhysX advance on different clocks or have uncommanded steps")
    rgb = row_summary(report.get("final_row_observations", {}).get("cubes", []), half)
    truth = row_summary(report.get("final_physical_cube_poses", []), half)
    if truth is not None and truth["order_left_to_right"] != [1, 3, 5, 4, 2]:
        issues.append("the physical final row is not 1, 3, 5, 4, 2 from left to right")
    return {
        "status": "PASS_CONTINUOUS_VISIBLE_SIDE_SUCTION" if not issues else "INCOMPLETE_OR_FAILED",
        "runner_status": report.get("status"),
        "runner_error": report.get("error"),
        "execution_mode": mode,
        "tray_inner_width_mm": width * 1000 if width is not None else None,
        "wall_heights_mm": {name: size[2] * 1000 for name, (_, size) in walls.items()},
        "completed_cubes": [row["cube_id"] for row in report.get("completed_cubes", [])],
        "unexpected_robot_contacts": report.get("physx_contacts", {}).get("unexpected_contact_count"),
        "physics_clock": clock,
        "side_suction": side_cubes,
        "cube4_support": cube4_support(directory, report, force_validation),
        "physical_evidence": physical,
        "final_row_rgb": rgb,
        "final_row_physx": truth,
        "force_validation": force_validation,
        "issues": issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    output = args.output or args.directory / "continuous_side_acceptance.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "runner_error", "completed_cubes",
                                                  "cube4_support", "issues")}, indent=2))
    raise SystemExit(result["status"] != "PASS_CONTINUOUS_VISIBLE_SIDE_SUCTION")
