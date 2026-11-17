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


def _valid_side(row, side):
    try:
        for key, shape in SHAPES.items():
            value = np.asarray(row[side][key], dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


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
    for side, phase in (("right", "B2_push_transit"), ("left", "B2_push_transit_transit")):
        selected = [row for row in rows if row.get("phase") == phase and _valid_side(row, side)]
        if not selected:
            continue
        wrenches = np.asarray([row[side]["gravity_corrected_world"] for row in selected])
        norms = np.linalg.norm(wrenches[:, :3], axis=1)
        index = int(np.argmax(norms))
        push_p95 = result["by_phase"].get("B2_push", {}).get(side, {}).get(
            "gravity_corrected_world", {}).get("force_norm_p95_N", 0.0)
        review = bool(norms[index] > max(10.0, 10 * push_p95))
        result["release_transients"][side] = {
            "phase": phase, "peak_force_N": float(norms[index]),
            "peak_wrench_world": wrenches[index].tolist(),
            "peak_time_s": selected[index]["time_s"],
            "peak_physics_step": selected[index]["physics_step"],
            "push_force_p95_N": push_p95, "requires_review": review,
            "review_rule": "peak > max(10 N, 10 * push P95); screening heuristic, not a safety rating",
        }
        result["release_task_issue"] |= review
    checks = [value["status"] for name, value in acceptance.items() if name != "B2_left_support"]
    if not finite or not monotonic or not clock_120hz:
        result["status"] = "INVALID_DATA"
    elif "FAIL" in checks or "DIRECTION_MISMATCH" in checks or result["calibration_status"] == "FAIL":
        result["status"] = "SENSOR_CHECKS_FAILED"
    elif ("UNVERIFIED" in checks or "NO_SUSTAINED_FORCE" in checks or "INTERMITTENT_FORCE" in checks
          or result["calibration_status"] != "PASS"
          or result["full_flow_status"] != "PASS_B1_B2_PROXY_CONTACT"):
        result["status"] = "INCOMPLETE"
    else:
        result["status"] = ("PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS" if result["release_task_issue"] else
                            "PASS_FORCE_FEEDBACK_WITH_SUPPORT_WARNING" if result["support_task_issue"] else
                            "PASS_FORCE_FEEDBACK")
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
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing wrench.jsonl.gz and metadata.json")
    args = parser.parse_args()
    summary = write_analysis(args.directory)
    print(json.dumps({"status": summary["status"], "samples": summary["samples"],
                      "acceptance": summary["acceptance"]}, indent=2, allow_nan=False))
