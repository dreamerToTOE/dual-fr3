"""Analyze virtual wrist wrench feedback; task phase names are evaluation metadata."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np


SIDES = ("left", "right")
SHAPES = {
    "raw_parent_on_tool_local": (6,), "tool_gravity_local": (6,),
    "gravity_corrected_local": (6,), "gravity_corrected_world": (6,),
    "sensor_position_m": (3,), "sensor_orientation_wxyz": (4,),
    "q_rad": (7,), "qdot_rad_s": (7,),
}
TCP_SHAPES = {
    "tcp_gravity_corrected_local": (6,), "tcp_gravity_corrected_world": (6,),
    "tcp_position_m": (3,),
}


def _valid_side(row, side, shapes=SHAPES):
    try:
        for key, shape in shapes.items():
            value = np.asarray(row[side][key], dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _tcp_transform_validation(rows, metadata):
    if metadata.get("schema_version", 1) not in (2, 3):
        return {"status": "UNVERIFIED", "reason": "legacy recording has no required TCP feedback"}
    limits = {
        "force_local_N": 1e-9, "force_world_N": 1e-9,
        "moment_shift_local_Nm": 1e-9, "moment_shift_world_Nm": 1e-9,
        "reverse_shift_local_Nm": 1e-9, "reverse_shift_world_Nm": 1e-9,
        "tcp_position_m": 1e-6, "frame_rotation_relative": 1e-6,
    }
    result = {"status": "FAIL", "limits": limits, "max_errors": dict.fromkeys(limits, 0.0),
              "valid_side_samples": dict.fromkeys(SIDES, 0), "invalid_field_samples": 0,
              "failed_relation_samples": 0, "by_phase": {}}
    try:
        offsets = {side: np.asarray(metadata["tcp_offset_sensor_m"][side], dtype=float) for side in SIDES}
        if any(r.shape != (3,) or not np.all(np.isfinite(r)) for r in offsets.values()):
            raise ValueError("invalid TCP offset")
    except (KeyError, TypeError, ValueError):
        return {**result, "all_fields_finite": False, "reason": "missing or invalid TCP offset metadata"}
    for row in rows:
        phase = result["by_phase"].setdefault(row.get("phase", "unknown"), {})
        for side in SIDES:
            stats = phase.setdefault(side, {"samples_checked": 0, "invalid_field_samples": 0,
                                            "max_errors": dict.fromkeys(limits, 0.0)})
            if not (_valid_side(row, side) and _valid_side(row, side, TCP_SHAPES)):
                result["invalid_field_samples"] += 1
                stats["invalid_field_samples"] += 1
                continue
            values = row[side]
            q = np.asarray(values["sensor_orientation_wxyz"], dtype=float)
            norm = float(np.linalg.norm(q))
            if norm < 1e-12 or abs(norm - 1) > 1e-5:
                result["invalid_field_samples"] += 1
                stats["invalid_field_samples"] += 1
                continue
            w, x, y, z = q / norm
            rotation = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ])
            sensor_local = np.asarray(values["gravity_corrected_local"], dtype=float)
            sensor_world = np.asarray(values["gravity_corrected_world"], dtype=float)
            tcp_local = np.asarray(values["tcp_gravity_corrected_local"], dtype=float)
            tcp_world = np.asarray(values["tcp_gravity_corrected_world"], dtype=float)
            offset = offsets[side]
            world_offset = np.asarray(values["tcp_position_m"], dtype=float) - np.asarray(
                values["sensor_position_m"], dtype=float)
            rotation_error = max(
                np.linalg.norm(rotation @ local[part:part+3] - world[part:part+3])
                / max(1.0, np.linalg.norm(world[part:part+3]))
                for local, world in ((sensor_local, sensor_world), (tcp_local, tcp_world))
                for part in (0, 3))
            errors = {
                "force_local_N": float(np.max(np.abs(tcp_local[:3] - sensor_local[:3]))),
                "force_world_N": float(np.max(np.abs(tcp_world[:3] - sensor_world[:3]))),
                "moment_shift_local_Nm": float(np.max(np.abs(
                    tcp_local[3:] - sensor_local[3:] + np.cross(offset, sensor_local[:3])))),
                "moment_shift_world_Nm": float(np.max(np.abs(
                    tcp_world[3:] - sensor_world[3:] + np.cross(world_offset, sensor_world[:3])))),
                "reverse_shift_local_Nm": float(np.max(np.abs(
                    tcp_local[3:] + np.cross(offset, tcp_local[:3]) - sensor_local[3:]))),
                "reverse_shift_world_Nm": float(np.max(np.abs(
                    tcp_world[3:] + np.cross(world_offset, tcp_world[:3]) - sensor_world[3:]))),
                "tcp_position_m": float(np.max(np.abs(world_offset - rotation @ offset))),
                "frame_rotation_relative": float(rotation_error),
            }
            result["valid_side_samples"][side] += 1
            stats["samples_checked"] += 1
            if any(errors[key] > limits[key] for key in limits):
                result["failed_relation_samples"] += 1
            for key, error in errors.items():
                result["max_errors"][key] = max(result["max_errors"][key], error)
                stats["max_errors"][key] = max(stats["max_errors"][key], error)
    result["all_fields_finite"] = not result["invalid_field_samples"]
    if rows and result["all_fields_finite"] and not result["failed_relation_samples"]:
        result["status"] = "PASS"
    return result


def _stats(values):
    if not len(values):
        return {"samples": 0}
    values = np.asarray(values, dtype=float)
    return {
        "samples": len(values),
        "component_mean": np.mean(values, axis=0).tolist(),
        "component_abs_p95": np.percentile(np.abs(values), 95, axis=0).tolist(),
        "component_abs_max": np.max(np.abs(values), axis=0).tolist(),
        "force_norm_p95_N": float(np.percentile(np.linalg.norm(values[:, :3], axis=1), 95)),
        "force_norm_max_N": float(np.max(np.linalg.norm(values[:, :3], axis=1))),
        "torque_norm_p95_Nm": float(np.percentile(np.linalg.norm(values[:, 3:], axis=1), 95)),
        "torque_norm_max_Nm": float(np.max(np.linalg.norm(values[:, 3:], axis=1))),
    }


def _last_static(rows, phase):
    selected = [row for row in rows if row.get("phase") == phase]
    if len(selected) < 30:
        return None
    selected = selected[-30:]
    if not all(_valid_side(row, side) for row in selected for side in SIDES):
        return None
    if max(np.max(np.abs(row[side]["qdot_rad_s"])) for row in selected for side in SIDES) > 0.01:
        return None
    return selected


def _direction(rows, phase, side, component, sign):
    # Exclude stopped endpoints and use the middle 60% of actual joint motion.
    selected = [row for row in rows if row.get("phase") == phase and _valid_side(row, side)
                and np.max(np.abs(row[side]["qdot_rad_s"])) > 0.001]
    if len(selected) < 5:
        return {"status": "UNVERIFIED", "samples": len(selected), "reason": "missing moving window"}
    start, stop = len(selected) // 5, len(selected) - len(selected) // 5
    selected = selected[start:stop]
    force = np.asarray([row[side]["gravity_corrected_world"][component] for row in selected])
    median = float(np.median(force))
    proportion = float(np.mean(sign * force > 0.1))
    if sign * median > 0.1:
        status = "CONSISTENT" if proportion >= 0.8 else "INTERMITTENT_FORCE"
    elif abs(median) <= 0.1:
        status = "NO_SUSTAINED_FORCE"
    else:
        status = "DIRECTION_MISMATCH"
    return {"status": status, "samples": len(selected), "component": ("Fx", "Fy", "Fz")[component],
            "median_N": median, "expected_sign": sign,
            "expected_direction_above_0_1N_proportion": proportion,
            "window_basis": "middle 60 percent of joint-moving samples"}


def _b2_force_control_validation(rows, metadata):
    """Check declared contact windows without filtering on motion or force."""
    cfg = metadata.get("b2_force_control_validation")
    if cfg is None:
        return {"status": "UNVERIFIED", "reason": "legacy run has no force-control acceptance contract"}
    result = {"status": "FAIL", "held_phases": {}, "releases": {}, "phase_peaks": {}}
    try:
        held = cfg["held_phases"]
        releases = {side: cfg["release_phases"][side] for side in SIDES}
        limits = {name: float(cfg[name]) for name in (
            "normal_force_min_N", "normal_force_max_N", "min_contact_coverage", "max_dropout_s",
            "max_force_N", "max_tcp_torque_Nm", "max_inward_m", "minimum_retreat_m")}
        if (not isinstance(held, (list, tuple)) or not held or "B2_push" not in held
                or not all(isinstance(p, str) for p in held)
                or any(not isinstance(phases, (list, tuple)) or not phases
                       or not all(isinstance(p, str) for p in phases) for phases in releases.values())
                or not all(np.isfinite(value) for value in limits.values())
                or not 0 < limits["normal_force_min_N"] < limits["normal_force_max_N"]
                or not 0 < limits["min_contact_coverage"] <= 1
                or any(limits[name] < 0 for name in ("max_dropout_s", "max_inward_m"))
                or any(limits[name] <= 0 for name in ("max_force_N", "max_tcp_torque_Nm", "minimum_retreat_m"))):
            raise ValueError("invalid acceptance contract")
    except (KeyError, TypeError, ValueError):
        return {**result, "reason": "invalid force-control acceptance contract"}
    result.update(limits=limits, release_phases=releases)
    phases = set(row.get("phase") for row in rows)
    required = {"B2_push"} | set(phase for names in releases.values() for phase in names)
    result["missing_required_phases"] = sorted(required - phases)

    def valid(selected, side):
        return bool(selected) and all(_valid_side(row, side) and _valid_side(row, side, TCP_SHAPES)
                                      for row in selected)

    def peaks(selected, side):
        force = np.asarray([row[side]["gravity_corrected_world"][:3] for row in selected])
        torque = np.asarray([row[side]["tcp_gravity_corrected_world"][3:] for row in selected])
        return {"samples": len(selected),
                "force_norm_p95_N": float(np.percentile(np.linalg.norm(force, axis=1), 95)),
                "force_norm_max_N": float(np.max(np.linalg.norm(force, axis=1))),
                "tcp_torque_norm_p95_Nm": float(np.percentile(np.linalg.norm(torque, axis=1), 95)),
                "tcp_torque_norm_max_Nm": float(np.max(np.linalg.norm(torque, axis=1)))}

    for name in dict.fromkeys(row.get("phase", "") for row in rows if str(row.get("phase", "")).startswith("B2")):
        selected = [row for row in rows if row.get("phase") == name]
        result["phase_peaks"][name] = {}
        for side in SIDES:
            if not valid(selected, side):
                result["phase_peaks"][name][side] = {"status": "FAIL", "reason": "invalid force/TCP feedback"}
                continue
            metrics = peaks(selected, side)
            metrics["status"] = ("PASS" if metrics["force_norm_max_N"] <= limits["max_force_N"]
                                  and metrics["tcp_torque_norm_max_Nm"] <= limits["max_tcp_torque_Nm"] else "FAIL")
            result["phase_peaks"][name][side] = metrics
    for name in held:
        selected = [row for row in rows if row.get("phase") == name]
        if not selected:
            continue
        if not valid(selected, "left"):
            result["held_phases"][name] = {"status": "FAIL", "reason": "invalid support feedback"}
            continue
        normal = -np.asarray([row["left"]["gravity_corrected_world"][1] for row in selected])
        contact = normal > limits["normal_force_min_N"]
        in_band = contact & (normal <= limits["normal_force_max_N"])
        run, longest, previous = 0, 0, None
        for row, touching in zip(selected, contact):
            if previous is not None and row["physics_step"] != previous + 1:
                run = 0
            run = 0 if touching else run + 1
            longest = max(longest, run)
            previous = row["physics_step"]
        coverage, band_coverage = float(np.mean(contact)), float(np.mean(in_band))
        metrics = {
            **peaks(selected, "left"), "normal_force_min_observed_N": float(np.min(normal)),
            "normal_force_median_N": float(np.median(normal)), "normal_force_max_observed_N": float(np.max(normal)),
            "contact_coverage": coverage, "normal_force_in_band_proportion": band_coverage,
            "longest_contact_dropout_s": longest / 120,
            "window_basis": "all declared held samples; no qdot filter or phase trimming",
        }
        metrics["status"] = ("PASS" if coverage >= limits["min_contact_coverage"]
                              and band_coverage >= limits["min_contact_coverage"]
                              and metrics["longest_contact_dropout_s"] <= limits["max_dropout_s"] else "FAIL")
        result["held_phases"][name] = metrics
    for side, names in releases.items():
        indices = [index for index, row in enumerate(rows) if row.get("phase") in names]
        selected = [rows[index] for index in indices]
        if (not indices or indices[0] == 0 or not valid(selected, side)
                or not _valid_side(rows[indices[0] - 1], side, TCP_SHAPES)):
            result["releases"][side] = {"status": "FAIL", "reason": "missing release data or preceding TCP reference"}
            continue
        origin = np.asarray(rows[indices[0] - 1][side]["tcp_position_m"])
        direction = np.array([1., 0., 0.]) if side == "right" else np.array([0., -1., 0.])
        positions = np.asarray([row[side]["tcp_position_m"] for row in selected])
        outward = np.r_[0., (positions - origin) @ direction]
        metrics = {
            **peaks(selected, side), "phases": names,
            "reference_physics_step": rows[indices[0] - 1]["physics_step"],
            "outward_direction_world": direction.tolist(),
            "final_outward_distance_m": float(outward[-1]),
            "maximum_inward_excursion_m": float(max(0., -np.min(outward))),
            "maximum_backtrack_m": float(np.max(np.maximum.accumulate(outward) - outward)),
        }
        metrics["status"] = ("PASS" if metrics["final_outward_distance_m"] >= limits["minimum_retreat_m"]
                              and metrics["maximum_inward_excursion_m"] <= limits["max_inward_m"]
                              and metrics["maximum_backtrack_m"] <= limits["max_inward_m"]
                              and metrics["force_norm_max_N"] <= limits["max_force_N"]
                              and metrics["tcp_torque_norm_max_Nm"] <= limits["max_tcp_torque_Nm"] else "FAIL")
        result["releases"][side] = metrics
    if (not result["missing_required_phases"] and result["held_phases"]
            and all(item["status"] == "PASS" for item in result["held_phases"].values())
            and all(item["status"] == "PASS" for item in result["releases"].values())
            and all(item["status"] == "PASS" for phase in result["phase_peaks"].values() for item in phase.values())):
        result["status"] = "PASS"
    return result


def _phase_force_control_validation(rows, metadata, prefix):
    """Check every stage sample, including contact transitions and stopped motion."""
    cfg = metadata.get(prefix.lower() + "_force_control_validation")
    if cfg is None:
        return {"status": "UNVERIFIED", "reason": f"legacy run has no {prefix} force acceptance contract"}
    result = {"status": "FAIL", "phase_peaks": {}}
    try:
        required = cfg["required_phases"]
        names = ["max_force_N", "max_tcp_torque_Nm"] + (["max_internal_force_N"] if prefix == "A" else [])
        limits = {key: float(cfg[key]) for key in names}
        mandatory = ({"A_approach_transit", "A_attach", "A_transfer", "A_lift", "A_over_front", "A_release"}
                     if prefix == "A" else {"B1"})
        if (not isinstance(required, (list, tuple)) or not required
                or not all(isinstance(phase, str) and phase.startswith(prefix) for phase in required)
                or not mandatory.issubset(required)
                or not all(np.isfinite(value) and value > 0 for value in limits.values())):
            raise ValueError("invalid acceptance contract")
    except (KeyError, TypeError, ValueError):
        return {**result, "reason": f"invalid {prefix} force acceptance contract"}
    phases = list(dict.fromkeys(row.get("phase", "") for row in rows
                                if str(row.get("phase", "")).startswith(prefix)))
    result.update(limits=limits, missing_required_phases=sorted(set(required) - set(phases)),
                  window_basis=f"all {prefix} samples; no motion filter, endpoint trimming or force clipping")
    peak_internal = 0.0
    for phase in phases:
        selected = [row for row in rows if row.get("phase") == phase]
        metrics = {"status": "FAIL", "samples": len(selected)}
        result["phase_peaks"][phase] = metrics
        if not all(_valid_side(row, side) and _valid_side(row, side, TCP_SHAPES)
                   for row in selected for side in SIDES):
            metrics["reason"] = "invalid force/TCP feedback"
            continue
        if prefix == "A":
            forces = np.asarray([[row[side]["gravity_corrected_world"][:3] for side in SIDES]
                                 for row in selected])
            internal = np.linalg.norm((forces[:, 0] - forces[:, 1]) / 2, axis=1)
            metrics["internal_force_norm_max_N"] = float(np.max(internal))
            peak_internal = max(peak_internal, metrics["internal_force_norm_max_N"])
        for side in SIDES:
            stats = _stats([row[side]["tcp_gravity_corrected_world"] for row in selected])
            metrics[side] = {"force_norm_max_N": stats["force_norm_max_N"],
                             "tcp_torque_norm_max_Nm": stats["torque_norm_max_Nm"],
                             "raw_force_norm_max_N": _stats([row[side]["raw_parent_on_tool_local"]
                                                              for row in selected])["force_norm_max_N"]}
        if ((prefix != "A" or metrics["internal_force_norm_max_N"] <= limits["max_internal_force_N"])
                and all(metrics[side]["force_norm_max_N"] <= limits["max_force_N"]
                        and metrics[side]["tcp_torque_norm_max_Nm"] <= limits["max_tcp_torque_Nm"]
                        for side in SIDES)):
            metrics["status"] = "PASS"
    if prefix == "A":
        result["peak_internal_force_norm_N"] = peak_internal
        result["internal_force_definition"] = "norm((F_left - F_right) / 2), world axes; includes unequal payload sharing"
    if (not result["missing_required_phases"] and result["phase_peaks"]
            and all(metrics["status"] == "PASS" for metrics in result["phase_peaks"].values())):
        result["status"] = "PASS"
    return result


def analyze(rows, metadata=None):
    """Return strict finite-data checks, phase statistics and independent load checks."""
    rows, metadata = list(rows), metadata or {}
    phases = list(dict.fromkeys(row.get("phase", "unknown") for row in rows))
    valid_counts = {side: sum(_valid_side(row, side) for row in rows) for side in SIDES}
    invalid_clock = 0
    times, steps = [], []
    for row in rows:
        try:
            time_s, step = float(row["time_s"]), float(row["physics_step"])
            if not np.isfinite(time_s) or not np.isfinite(step) or step != int(step):
                raise ValueError("invalid clock")
            times.append(time_s)
            steps.append(int(step))
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_clock += 1
    monotonic = bool(len(rows) > 0 and not invalid_clock
                     and np.all(np.diff(times) > 0) and np.all(np.diff(steps) == 1))
    clock_120hz = bool(len(times) >= 2 and not invalid_clock
                       and np.allclose(np.diff(times), 1 / 120, rtol=0, atol=1e-7))
    finite = bool(rows and all(count == len(rows) for count in valid_counts.values())
                  and not invalid_clock)
    tcp_validation = _tcp_transform_validation(rows, metadata)
    b2_validation = _b2_force_control_validation(rows, metadata)
    a_validation = _phase_force_control_validation(rows, metadata, "A")
    b1_validation = _phase_force_control_validation(rows, metadata, "B1")
    finite = finite and tcp_validation.get("all_fields_finite", True)
    result = {
        "units": {"force": "N", "torque": "N m", "time": "s", "position": "m"},
        "samples": len(rows), "valid_side_samples": valid_counts,
        "invalid_clock_samples": invalid_clock, "all_samples_finite": finite,
        "clock_monotonic_no_missing_steps": monotonic,
        "clock_120hz": clock_120hz,
        "phase_counts": dict(Counter(row.get("phase", "unknown") for row in rows)),
        "by_phase": {}, "acceptance": {},
        "calibration_status": ("PASS" if metadata.get("calibration_passed") is True else
                               "FAIL" if metadata.get("calibration_passed") is False else "UNVERIFIED"),
        "full_flow_status": metadata.get("task_status", "UNVERIFIED"),
        "tcp_transform_validation": tcp_validation,
        "b2_force_control_validation": b2_validation,
        "a_force_control_validation": a_validation,
        "b1_force_control_validation": b1_validation,
        "tcp_physical_calibration_status": (
            "PASS" if metadata.get("tcp_calibration_passed") is True else
            "FAIL" if metadata.get("tcp_calibration_passed") is False else "UNVERIFIED"),
    }
    for phase in phases:
        result["by_phase"][phase] = {}
        for side in SIDES:
            selected = [row[side] for row in rows if row.get("phase", "unknown") == phase
                        and _valid_side(row, side)]
            result["by_phase"][phase][side] = {
                name: _stats([row[name] for row in selected])
                for name in ("raw_parent_on_tool_local", "gravity_corrected_local", "gravity_corrected_world")
            }
    acceptance = result["acceptance"]
    initialization = _last_static(rows, "initialization")
    if initialization is None:
        acceptance["static_initialization"] = {"status": "UNVERIFIED", "reason": "missing 30 valid static samples"}
    else:
        checks = {}
        for side in SIDES:
            stats = _stats([row[side]["gravity_corrected_local"] for row in initialization])
            checks[side] = {"force_norm_p95_N": stats["force_norm_p95_N"],
                            "torque_norm_p95_Nm": stats["torque_norm_p95_Nm"]}
        passed = all(check["force_norm_p95_N"] < 0.3 and check["torque_norm_p95_Nm"] < 0.03
                     for check in checks.values())
        acceptance["static_initialization"] = {"status": "PASS" if passed else "FAIL", "samples": 30,
                                                "limits": {"force_N": 0.3, "torque_Nm": 0.03}, **checks}
    try:
        mass = float(metadata["cube_mass_kg_reference"])
        gravity = np.asarray(metadata["gravity_world_m_s2"], dtype=float)
        if not np.isfinite(mass) or mass <= 0 or gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("invalid Cube mass or gravity reference")
        expected_force = mass * gravity
    except (KeyError, TypeError, ValueError):
        expected_force = None
    for phase in ("A_lift", "A_over_front"):
        selected = _last_static(rows, phase)
        if selected is None or expected_force is None:
            acceptance[phase + "_payload"] = {
                "status": "UNVERIFIED",
                "reason": ("missing Cube mass or gravity reference metadata" if expected_force is None
                           else "missing 30 valid static samples"),
            }
            continue
        force = np.asarray([np.asarray(row["left"]["gravity_corrected_world"][:3])
                            + row["right"]["gravity_corrected_world"][:3] for row in selected])
        error_p95 = float(np.percentile(np.linalg.norm(force - expected_force, axis=1), 95))
        acceptance[phase + "_payload"] = {
            "status": "PASS" if error_p95 < 0.5 else "FAIL", "samples": 30,
            "expected_combined_force_world_N": expected_force.tolist(),
            "combined_force_world_mean_N": np.mean(force, axis=0).tolist(),
            "combined_force_error_norm_p95_N": error_p95, "error_limit_N": 0.5,
        }
    acceptance["B1_left_push_direction"] = _direction(rows, "B1", "left", 1, -1)
    acceptance["B2_right_push_direction"] = _direction(rows, "B2_push", "right", 0, 1)
    acceptance["B2_left_support"] = _direction(rows, "B2_push", "left", 1, -1)
    support = acceptance["B2_left_support"]
    result["support_task_issue"] = support["status"] in (
        "NO_SUSTAINED_FORCE", "DIRECTION_MISMATCH", "INTERMITTENT_FORCE")
    result["support_task_issue"] |= any(item["status"] == "FAIL"
                                        for item in b2_validation.get("held_phases", {}).values())
    transfer = [row for row in rows if row.get("phase") == "A_transfer"
                and all(_valid_side(row, side) for side in SIDES)]
    result["A_attachment_internal_load"] = {"status": "UNVERIFIED", "samples": len(transfer)}
    if transfer:
        forces = np.asarray([[row[side]["gravity_corrected_world"][:3] for side in SIDES]
                             for row in transfer])
        opposing_norm = np.linalg.norm((forces[:, 0] - forces[:, 1]) / 2, axis=1)
        index = int(np.argmax(opposing_norm))
        result["A_attachment_internal_load"] = {
            "status": "OBSERVED", "samples": len(transfer),
            "definition": "norm((F_left - F_right) / 2), world axes; diagnostic only",
            "peak_internal_force_norm_N": float(opposing_norm[index]),
            "peak_time_s": transfer[index]["time_s"],
            "peak_physics_step": transfer[index]["physics_step"],
            "combined_force_at_peak_world_N": np.sum(forces[index], axis=0).tolist(),
            "individual_force_at_peak_world_N": {side: forces[index, i].tolist() for i, side in enumerate(SIDES)},
        }
    result["release_transients"] = {}
    result["release_task_issue"] = False
    release_names = b2_validation.get("release_phases", {
        "right": ["B2_push_transit"], "left": ["B2_push_transit_transit"]})
    for side, names in release_names.items():
        selected = [row for row in rows if row.get("phase") in names and _valid_side(row, side)]
        if not selected:
            continue
        wrenches = np.asarray([row[side]["gravity_corrected_world"] for row in selected])
        norms = np.linalg.norm(wrenches[:, :3], axis=1)
        index = int(np.argmax(norms))
        push_p95 = result["by_phase"].get("B2_push", {}).get(side, {}).get(
            "gravity_corrected_world", {}).get("force_norm_p95_N", 0.0)
        review = bool(norms[index] > max(10.0, 10 * push_p95))
        result["release_transients"][side] = {
            "phase": " + ".join(names), "peak_force_N": float(norms[index]),
            "peak_wrench_world": wrenches[index].tolist(),
            "peak_time_s": selected[index]["time_s"],
            "peak_physics_step": selected[index]["physics_step"],
            "push_force_p95_N": push_p95, "requires_review": review,
            "review_rule": "peak > max(10 N, 10 * push P95); screening heuristic, not a safety rating",
        }
        result["release_task_issue"] |= review
    result["release_task_issue"] |= any(item["status"] == "FAIL"
                                        for item in b2_validation.get("releases", {}).values())
    checks = [value["status"] for name, value in acceptance.items() if name != "B2_left_support"]
    if not finite or not monotonic or not clock_120hz or tcp_validation["status"] == "FAIL":
        result["status"] = "INVALID_DATA"
    elif ("FAIL" in checks or "DIRECTION_MISMATCH" in checks or result["calibration_status"] == "FAIL"
          or metadata.get("schema_version") in (2, 3) and result["tcp_physical_calibration_status"] == "FAIL"):
        result["status"] = "SENSOR_CHECKS_FAILED"
    elif ("UNVERIFIED" in checks or "NO_SUSTAINED_FORCE" in checks or "INTERMITTENT_FORCE" in checks
          or result["calibration_status"] != "PASS"
          or metadata.get("schema_version") in (2, 3) and result["tcp_physical_calibration_status"] != "PASS"
          or result["full_flow_status"] not in ("PASS_B1_B2_PROXY_CONTACT", "PASS_A_B1_B2_SENSOR_CONTROL")):
        result["status"] = "INCOMPLETE"
    else:
        result["status"] = ("PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS" if result["release_task_issue"]
                            or "FAIL" in (b2_validation["status"], a_validation["status"], b1_validation["status"]) else
                            "PASS_FORCE_FEEDBACK_WITH_SUPPORT_WARNING" if result["support_task_issue"] else
                            "PASS_FORCE_FEEDBACK")
    result["task_force_status"] = (
        "FULL_FLOW_FORCE_FAIL" if "FAIL" in (a_validation["status"], b1_validation["status"], b2_validation["status"])
        or result["status"] in ("INVALID_DATA", "SENSOR_CHECKS_FAILED") else
        "FULL_FLOW_FORCE_PASS" if a_validation["status"] == b1_validation["status"] == b2_validation["status"] == "PASS"
        and result["status"] == "PASS_FORCE_FEEDBACK" and tcp_validation["status"] == "PASS"
        and result["tcp_physical_calibration_status"] == "PASS" else "UNVERIFIED")
    result["limitations"] = [
        "Gravity compensation does not remove tool inertia during motion.",
        "Payload check uses reference Cube mass and gravity metadata in a static suspended window.",
        "Individual arm loads need not split equally in the closed-chain attachment proxy.",
        "Driven attachment proxies can create large opposing arm loads; this alone is not a sensor calibration fault.",
        "Joint-moving windows do not independently establish Cube motion or wall contact.",
    ]
    return result


def plot_wrenches(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(rows)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    for side, force_ax, torque_ax in (("left", axes[0], axes[1]), ("right", axes[2], axes[3])):
        wrenches = np.asarray([row[side]["gravity_corrected_world"] if _valid_side(row, side)
                               else [np.nan] * 6 for row in rows]).reshape((-1, 6))
        for index, label in enumerate(("X", "Y", "Z")):
            force_ax.plot(times, wrenches[:, index], label=label, linewidth=0.8)
            torque_ax.plot(times, wrenches[:, index + 3], label=label, linewidth=0.8)
        force_ax.set_ylabel(f"{side.title()} force (N)")
        torque_ax.set_ylabel(f"{side.title()} torque (N m)")
    if rows:
        def stage(row):
            phase = str(row.get("phase", ""))
            return next((name for name in ("B2", "B1", "A") if phase.startswith(name)), "Setup")
        start = 0
        for end in range(1, len(rows) + 1):
            if end != len(rows) and stage(rows[end]) == stage(rows[start]):
                continue
            name = stage(rows[start])
            color = {"A": "#dbeafe", "B1": "#dcfce7", "B2": "#ffedd5", "Setup": "#eeeeee"}[name]
            for ax in axes:
                ax.axvspan(times[start], times[end - 1], color=color, alpha=0.4, zorder=-1)
            axes[0].text((times[start] + times[end - 1]) / 2, 1.02, name,
                         transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9)
            start = end
    for ax in axes:
        ax.axhline(0, color="black", alpha=0.25, linewidth=0.7)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Tool-on-sensor wrench at flange, world axes, tool gravity removed")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_tcp_torques(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout="constrained")
    for column, side in enumerate(SIDES):
        sensor = np.asarray([row[side]["gravity_corrected_world"] if _valid_side(row, side)
                             else [np.nan] * 6 for row in rows]).reshape((-1, 6))
        tcp = np.asarray([row[side]["tcp_gravity_corrected_world"] if _valid_side(row, side, TCP_SHAPES)
                          else [np.nan] * 6 for row in rows]).reshape((-1, 6))
        for index, component in enumerate(("Tx", "Ty", "Tz")):
            ax = axes[index, column]
            ax.plot(times, sensor[:, index + 3], label="At sensor", linewidth=0.8, linestyle="--")
            ax.plot(times, tcp[:, index + 3], label="At TCP", linewidth=0.8)
            ax.set_ylabel(f"{component} (N m)")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
        axes[0, column].set_title(side.title())
        axes[-1, column].set_xlabel("Simulation time (s)")
    fig.suptitle("Sensor vs TCP moment reference; world axes, tool gravity removed")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_analysis(directory):
    directory = Path(directory)
    with gzip.open(directory / "wrench.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    metadata = json.loads((directory / "metadata.json").read_text())
    summary = analyze(rows, metadata)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_wrenches(rows, directory / "wrench.png")
    contact_rows = [row for row in rows if str(row.get("phase", "")).startswith("B")]
    if contact_rows:
        plot_wrenches(contact_rows, directory / "wrench_contact.png")
    if metadata.get("schema_version") in (2, 3):
        plot_tcp_torques(rows, directory / "tcp_wrench.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing wrench.jsonl.gz and metadata.json")
    args = parser.parse_args()
    summary = write_analysis(args.directory)
    print(json.dumps({"status": summary["status"], "samples": summary["samples"],
                      "task_force_status": summary["task_force_status"],
                      "a_force_control_validation": {key: value for key, value in
                                                     summary["a_force_control_validation"].items()
                                                     if key != "phase_peaks"},
                      "b1_force_control_validation": {key: value for key, value in
                                                      summary["b1_force_control_validation"].items()
                                                      if key != "phase_peaks"},
                      "b2_force_control_validation": {key: value for key, value in
                                                      summary["b2_force_control_validation"].items()
                                                      if key != "phase_peaks"},
                      "tcp_physical_calibration_status": summary["tcp_physical_calibration_status"],
                      "tcp_transform_validation": {key: value for key, value in
                                                   summary["tcp_transform_validation"].items() if key != "by_phase"},
                      "acceptance": summary["acceptance"]}, indent=2, allow_nan=False))
