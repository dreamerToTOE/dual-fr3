"""Replay the frozen fusion method against independent PhysX labels for S0–S3."""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
from statistics import median, quantiles
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


LEGACY = "f35b3d257eed29aa7419ace6cc7011ebd0baee44"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LEGACY_MODULES = ("seven_observer", "seven_commands", "active_probe_replay",
                  "fusion_skill_boundary", "fusion_upper5", "score_upper5", "task_state_labels")
SENSOR_KEYS = ("time_s", "camera", "tcp_positions_m", "cup_normals_world",
               "left_wrench_world", "right_wrench_world")
STATES = ("S0", "S1", "S2", "S3")
UNKNOWN = "UNKNOWN"
MIN_IMPULSE_NS = 1e-5
LOWER = {f"/World/Cube_{i}" for i in (1, 2, 3, 4, 5, 6, 12, 13, 14, 15)}


def optical_neighbors(report, cube_id):
    records = {}
    for frame in report.get("neighbor_observations", []):
        if frame["cube_id"] != cube_id:
            continue
        for box in frame["neighbors"]:
            stamp = box.get("capture_time_s")
            if stamp is None or box["name"] == f"Cube_{cube_id}":
                continue
            arrivals = [a["ended_s"] for a in report["neighbor_observation_attempts"]
                        if a["cube_id"] == cube_id and a["started_s"] <= stamp <= a["ended_s"]]
            if arrivals:
                records.setdefault("/World/" + box["name"], {
                    "position_m": box["position_m"], "rotation_world": box["rotation_world"],
                    "capture_time_s": stamp, "received_time_s": min(arrivals),
                })
    return records


def physical_label(row):
    """Label upright upper-row traces using current native contact and suction."""
    snapshot = row["cube_contacts"]
    if not snapshot["valid"]:
        return UNKNOWN, set(), set()
    active = f"/World/Cube_{row['active_cube_id']}"
    touching, unresolved, bearing = set(), set(), set()
    supported = False
    for pair in snapshot["pairs"]:
        if active not in (pair["collider0"], pair["collider1"]):
            continue
        other = pair["collider1"] if pair["collider0"] == active else pair["collider0"]
        label = pair["label"]
        fresh = pair["current_step_observed"]
        separation = pair["separation_m"]
        load = pair["normal_impulse_Ns"]
        loaded = fresh and load is not None and load > MIN_IMPULSE_NS
        if label in ("left_cup", "right_cup"):
            side = label[:-4]
            if loaded or pair["current_separation_observed"] and separation is not None and separation <= 0:
                touching.add(side)
            elif not pair["current_separation_observed"] or separation is None:
                unresolved.add(side)
        elif loaded:
            axes = pair.get("normal_axis_impulse_abs_Ns")
            vertical = axes is None or axes[2] >= max(axes[:2])
            if label in ("floor", "supply") or other in LOWER:
                supported |= vertical
            elif label == "cube_neighbor" or label.startswith("wall_"):
                if axes is None or max(axes[:2]) > axes[2]:
                    bearing.add(other)
    seals = {side for side, present in row["attachments"].items() if present}
    clear = {"left", "right"} - touching - unresolved - seals
    if supported:
        state = "S2" if seals else "S1" if touching else "S0" if len(clear) == 2 else UNKNOWN
    else:
        # The experiment covers supported transport and lifting of upright boxes.
        state = "S3" if seals and not bearing else UNKNOWN
        clear = set()
    return state, clear, bearing


def observed_label(result):
    state = result["state"]
    if state in ("S5", "S6"):
        return "S1"
    if state == "S4":
        return "S3"
    if state == "S3":
        return "S2" if "inferred_seal" in result["tools"].values() else UNKNOWN
    return state


def action_windows(report, hz):
    opened, windows = {}, {}
    for phase in report["phases"]:
        name = phase["phase"]
        if name not in ("upper_B_side", "upper_B_rear") and not name.endswith("_dynamic_withdraw"):
            continue
        key = phase["cube_id"], name
        if phase["status"] == "RUNNING":
            opened[key] = phase
        elif key in opened and phase["status"] in ("FORCE_CONFIRMED", "TOOL_CLEAR", "S0_CONFIRMED"):
            start = opened.pop(key)
            target = start.get("contact_target")
            arms = start.get("arms", ())
            if target or arms:
                windows[key] = (round(start["sim_time_s"] * hz), round(phase["sim_time_s"] * hz),
                                target, arms, phase["status"] == "S0_CONFIRMED")
    return windows


def event_score(rows, hz, persistence):
    truth_start = next((step for step, truth, _ in rows if truth is True), None)
    streak, triggers = 0, []
    for step, truth, predicted in rows:
        streak = streak + 1 if predicted else 0
        if streak == persistence:
            verdict = "correct" if truth is True else "false" if truth is False else "unverified"
            triggers.append({"step": step, "verdict": verdict})
    first = triggers[0] if triggers else None
    return {"truth_first_step": truth_start, "scored_steps": len(rows), "triggers": triggers,
            "first_outcome": first["verdict"] if first else "miss" if truth_start is not None else "no_event",
            "first_delay_ms": round((first["step"] - truth_start) * 1000 / hz, 3)
            if first and first["verdict"] == "correct" and truth_start is not None else None}


def state_scores(confusion):
    scores = {}
    for state in STATES:
        ref = sum(value for (truth, _), value in confusion.items() if truth == state)
        pred = sum(confusion[truth, state] for truth in STATES)
        correct = confusion[state, state]
        scores[state] = {"ref": ref, "pred": pred, "correct": correct,
                         "recall": correct / ref if ref else None,
                         "precision": correct / pred if pred else None,
                         "f1": 2 * correct / (ref + pred) if ref + pred else None}
    known = sum(score["ref"] for score in scores.values())
    return {"states": scores, "known_steps": known,
            "accuracy": sum(score["correct"] for score in scores.values()) / known if known else None,
            "confusion": {s: {p: confusion[s, p] for p in (*STATES, UNKNOWN)} for s in STATES}}


def profiles():
    yield "wrench_fk", 0, .005
    for delay in (0, 100, 200):
        for mode in ("vision_fk", "fusion_arrival", "fusion_compensated"):
            yield mode, delay, .005
        yield "fusion_compensated", delay, .003


def load_idle_windows(path, report_digest, cubes):
    manifest = json.loads(path.read_text())
    if manifest["runner_report_sha256"] != report_digest:
        raise ValueError("Idle windows were marked on a different runner report")
    windows = {}
    names = set()
    for cube in cubes:
        entries = sorted((row for row in manifest["windows"] if row["cube_id"] == cube),
                         key=lambda row: row["after_step"])
        if not entries:
            raise ValueError(f"No verified between-skill interval for Cube_{cube}")
        previous = 0
        for row in entries:
            name = row["name"]
            if name in names or not previous <= row["after_step"] < row["through_step"]:
                raise ValueError(f"Repeated, empty or overlapping idle window: {name}")
            names.add(name)
            previous = row["through_step"]
        windows[cube] = entries
    return windows


def replay(run, output, cubes, isaac_dir, idle_manifest=None):
    from seven_commands import targets
    from seven_observer import SevenObserver
    from active_probe_replay import command_kinematics

    report_bytes = (run / "runner_report.json").read_bytes()
    report = json.loads(report_bytes)
    idle = load_idle_windows(idle_manifest, sha256(report_bytes).hexdigest(), cubes) if idle_manifest else {}
    config = report["scene_parameters"]
    hz = report["simulation_clock"]["control_hz"]
    kin, robot_sha = command_kinematics(isaac_dir)
    yaws = {side: base[3] for side, base in config["fr3"]["bases"].items()}
    offsets = {side: np.asarray(value) for side, value in
               json.loads((run / "metadata.json").read_text())["tcp_offset_sensor_m"].items()}
    names = ("contact_states/sensor_inputs.jsonl.gz", "contact_states/reference.jsonl.gz", "wrench.jsonl.gz")
    windows = {} if idle else action_windows(report, hz)
    neighbors = {cube: optical_neighbors(report, cube) for cube in cubes}
    current = None
    counts, events, ages, unavailable, step_ranges = {}, {}, {}, {}, {}
    idle_counts = {}
    if idle:
        idle_counts = {row["name"]: Counter() for entries in idle.values() for row in entries}
    columns = tuple(profiles())
    result = {"run": run.name, "runner_status": report["status"], "legacy_revision": LEGACY,
              "hz": hz, "robot_sha256": robot_sha, "sources_sha256": {},
              "camera_neighbor_counts": {k: len(v) for k, v in neighbors.items()}}
    for name in (*names, "runner_report.json", "metadata.json"):
        result["sources_sha256"][name] = sha256((run / name).read_bytes()).hexdigest()
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(run / name, "rt")) for name in names]
        for sensor, physical, wrench in zip_longest(*(map(json.loads, s) for s in streams)):
            if sensor is None or physical is None or wrench is None:
                raise ValueError("The PhysX, observation and command streams differ in length")
            step = sensor["physics_step"]
            pair = step, sensor["time_s"]
            if any((r["physics_step"], r["time_s"]) != pair for r in (physical, wrench)):
                raise ValueError(f"Unpaired sample at step {step}")
            cube = physical["active_cube_id"]
            if cube not in cubes or (idle and step > idle[cube][-1]["through_step"]):
                continue
            if current != cube:
                current = cube
                observers = {(mode, delay, margin): SevenObserver(config, mode, delay_ms=delay, revised=True,
                                                                  separation_margin_m=margin)
                             for mode, delay, margin in columns}
                counts[cube] = {key: Counter() for key in columns}
                events[cube] = {key: {name: [] for cid, name in windows if cid == cube} for key in columns}
                ages[cube] = {key: [] for key in columns}
                unavailable[cube] = Counter()
                step_ranges[cube] = [step, step - 1]
            if step != step_ranges[cube][1] + 1:
                raise ValueError(f"Missing Cube_{cube} sample at step {step}")
            step_ranges[cube][1] = step
            inputs = {key: sensor[key] for key in SENSOR_KEYS}
            inputs["tcp_command_positions_m"] = targets(wrench, sensor, kin, yaws, offsets)
            idle_window = next((row for row in idle.get(cube, ())
                                if row["after_step"] < step <= row["through_step"]), None)
            if idle and idle_window is None:
                # Keep the historical sensor evidence used for delay compensation and seal tracking.
                for observer in observers.values():
                    observer.update(inputs, neighbors[cube])
                continue
            label, clear, bearing = physical_label(physical)
            if idle_window:
                idle_counts[idle_window["name"]][label] += 1
            active_windows = [(name, target, arms, all_clear)
                              for (cid, name), (start, end, target, arms, all_clear) in windows.items()
                              if cid == cube and start < step <= end]
            for key, observer in observers.items():
                estimate = observer.update(inputs, neighbors[cube])
                state = observed_label(estimate)
                counts[cube][key][label, state] += 1
                age = estimate["camera_age_s"]
                if age is not None and 0 <= age <= .40:
                    ages[cube][key].append(1000 * age)
                elif key[0] != "wrench_fk":
                    unavailable[cube][key] += 1
                for name, target, arms, all_clear in active_windows:
                    if target:
                        truth = target in bearing if physical["cube_contacts"]["valid"] else None
                        detected = target in estimate["boundaries"]
                    elif all_clear:
                        truth, detected = None if label == UNKNOWN else label == "S0", state == "S0"
                    else:
                        truth = None if label == UNKNOWN else set(arms) <= clear
                        detected = all(estimate["tools"][side] == "separated" for side in arms)
                    events[cube][key][name].append((step, truth, detected))
    if set(counts) != set(cubes):
        raise ValueError(f"Missing requested cubes: {set(cubes) - set(counts)}")
    if idle:
        result["score_scope"] = "between_skills_only"
        result["idle_windows"] = []
        for entries in idle.values():
            for row in entries:
                reference = idle_counts[row["name"]]
                if reference.total() != row["through_step"] - row["after_step"]:
                    raise ValueError(f"Incomplete idle window: {row['name']}")
                result["idle_windows"].append({**row, "reference_counts": reference})
    result["cubes"] = {}
    for cube in cubes:
        modes = {}
        for key, confusion in counts[cube].items():
            mode, delay, margin = key
            name = f"{mode}_{delay}ms" + ("_K5_3mm" if margin == .003 else "")
            age = ages[cube][key]
            modes[name] = {
                **state_scores(confusion), "unknown_reference_steps": sum(
                    n for (truth, _), n in confusion.items() if truth == UNKNOWN),
                "usable_camera_age_ms": {"median": median(age), "p95": quantiles(age, n=20)[-1]}
                if len(age) > 1 else None,
                "camera_unavailable_steps": unavailable[cube][key] if mode != "wrench_fk" else None,
                "events": {name: event_score(rows, hz, 6 if name.endswith("_dynamic_withdraw") else 3)
                           for name, rows in events[cube][key].items()},
            }
        result["cubes"][cube] = {"step_range": step_ranges[cube], "modes": modes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cube", nargs="+", type=int, required=True, choices=range(7, 12))
    parser.add_argument("--isaac-sim-dir", type=Path, required=True)
    parser.add_argument("--idle-windows", type=Path,
                        help="Score only the listed verified between-skill intervals")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="fr3-frozen-observer-") as legacy:
        for module in LEGACY_MODULES:
            path = (HERE / (module + ".py")).relative_to(REPO)
            contents = subprocess.check_output(["git", "-C", str(REPO), "show", f"{LEGACY}:{path}"])
            (Path(legacy) / (module + ".py")).write_bytes(contents)
        sys.path.insert(0, legacy)
        replay(args.run, args.output, args.cube, args.isaac_sim_dir, args.idle_windows)


if __name__ == "__main__":
    main()
