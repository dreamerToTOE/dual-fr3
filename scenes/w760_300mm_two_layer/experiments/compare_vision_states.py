"""Score RGB-D box fitting and rear RGB ArUco with the same S0-S6 observer."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np

from active_probe_replay import command_kinematics
from fusion_upper5 import FRONT_LOAD_N
from seven_commands import DEFAULT_ISAAC_DIR, targets
from seven_dataset import CAMERA, PAIR, REFERENCE, SENSOR, select
from seven_observer import SevenObserver
from seven_reference import SevenReference
from seven_replay import optical_audit
from thesis_metrics import STATES, confusion_metrics
from u4_contact_observer import U4ContactObserver
from u4_handover_score import events


CUBES = {"a": 7, "u4": 10, "u5": 11}


def phase_window(report, case, cube_id, hz):
    phases = {(e["phase"], e["status"]): round(e["sim_time_s"] * hz)
              for e in report["phases"] if e.get("cube_id") == cube_id}
    if case == "a":
        start = next(step for (phase, _), step in phases.items() if phase == "upper_supply")
        stop = phases.get(("upper_A_capture", "LANDED_RELEASED_RETRACTED"))
        if stop is None:
            stop = phases["upper_B_strategy", "SELECTED"]
    else:
        start, stop = 0, phases["upper_slot_complete", "SEATED_STABLE"]
    if not start < stop:
        raise ValueError(f"Invalid scoring window for Cube_{cube_id}")
    return phases, start, stop


def optical_neighbors(report, cube_id):
    records = {}
    for frame in report.get("neighbor_observations", []):
        if frame["cube_id"] != cube_id:
            continue
        for box in frame["neighbors"]:
            stamp = box.get("capture_time_s")
            if stamp is None or box["name"] == f"Cube_{cube_id}":
                continue
            attempts = [a["ended_s"] for a in report.get("neighbor_observation_attempts", [])
                        if a["cube_id"] == cube_id and a["started_s"] <= stamp <= a["ended_s"]]
            if attempts:
                records.setdefault("/World/" + box["name"], {
                    **select(box, ("position_m", "rotation_world", "capture_time_s")),
                    "received_time_s": min(attempts),
                })
    return records


def prepare(run, case, output, checkpoint=None):
    report_path = run / "runner_report.json"
    report = json.loads(report_path.read_text())
    config = report["scene_parameters"]
    hz = report["simulation_clock"]["control_hz"]
    cube_id = CUBES[case]
    phases, start, stop = phase_window(report, case, cube_id, hz)
    upper = report["upper_row"]
    positions = {
        str(box["cube_id"]): box["position_m"]
        for key in ("initial_lower_poses", "restored_lower_poses_at_start", "initial_upper_poses")
        for box in upper.get(key, []) if box["cube_id"] != cube_id
    }
    inputs = run / "contact_states/sensor_inputs.jsonl.gz"
    references = run / "contact_states/reference.jsonl.gz"
    wrench = run / "wrench.jsonl.gz"
    metadata = run / "metadata.json"
    files = [report_path, inputs, references, wrench, metadata]
    checkpoint_sha = report.get("comparison_checkpoint_sha256")
    if checkpoint is not None:
        supplied_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha and checkpoint_sha != supplied_sha:
            raise ValueError("The supplied checkpoint differs from the recorded input")
        checkpoint_sha = supplied_sha
    cfg = select(config, ("cube_size_m", "floor_top_m", "physics", "static_boxes", "supply_cube_center_m"))
    header = {"case": case, "source": run.name, "cube_id": cube_id, "hz": hz, "config": cfg,
              "reference_positions": positions, "optical_neighbors": optical_neighbors(report, cube_id),
              "phase_times_for_scoring_only": [
                  {"phase": phase, "status": status, "sim_time_s": step / hz}
                  for (phase, status), step in phases.items()
              ],
              "score_after_step": start, "stop_step": stop,
              "vision_source": report["vision"].get("observation", "rear D455 RGB ArUco"),
              "checkpoint_sha256": checkpoint_sha,
              "reference_monitor_sha256": report.get("reference_monitor_sha256"),
              "source_sha256": {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files}}
    if case != "a":
        kinematics, model_sha = command_kinematics(DEFAULT_ISAAC_DIR)
        base_yaws = {side: base[3] for side, base in config["fr3"]["bases"].items()}
        offsets = {side: np.asarray(value) for side, value in
                   json.loads(metadata.read_text())["tcp_offset_sensor_m"].items()}
        header["tcp_command_model_sha256"] = model_sha
    output.parent.mkdir(parents=True, exist_ok=True)
    last_frame = None
    ticks = visual_frames = last = 0
    with (gzip.open(inputs, "rt") as obs, gzip.open(references, "rt") as gt,
          gzip.open(wrench, "rt") as ft, gzip.open(output, "wt", compresslevel=6) as handle):
        def write(value):
            handle.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
        write(header)
        for sensor, physical, raw_wrist in zip_longest(*(map(json.loads, s) for s in (obs, gt, ft))):
            if sensor is None or physical is None or raw_wrist is None:
                raise ValueError("The three raw streams differ in length")
            tick, now = sensor["physics_step"], sensor["time_s"]
            if any((r["physics_step"], r["time_s"]) != (tick, now) for r in (physical, raw_wrist)):
                raise ValueError(f"Unpaired physics step {tick}")
            if tick > stop:
                break
            if physical["active_cube_id"] != cube_id:
                continue
            if ticks and tick != last + 1:
                raise ValueError(f"Noncontiguous clip at {tick}")
            row = select(sensor, SENSOR)
            packet = sensor.get("camera")
            if packet and packet["frame_id"] != last_frame:
                last_frame = packet["frame_id"]
                if last_frame > 0:
                    visual_frames += 1
                row["camera"] = select(packet, CAMERA)
            if case != "a":
                row["tcp_command_positions_m"] = targets(raw_wrist, sensor, kinematics, base_yaws, offsets)
                row["right_command_y_m"] = row["tcp_command_positions_m"]["right"][1]
            reference = select(physical, REFERENCE)
            contacts = physical["cube_contacts"]
            reference["cube_contacts"] = {"valid": contacts["valid"], "pairs": [
                select(p, PAIR) for p in contacts["pairs"]
            ]}
            write([row, reference])
            ticks += 1
            last = tick
    if last != stop:
        raise ValueError(f"Incomplete clip: last step {last}/{stop}")
    print(f"{case} {run.name}: {ticks} steps, {visual_frames} exposures -> {output}")


def load(path):
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        samples = list(map(json.loads, stream))
    return header, samples


def summarize(trace, hz):
    values = confusion_metrics(trace)
    values["_overall"]["truth_unknown_steps"] = sum(row["truth"] not in STATES for row in trace)
    values["s6_event"] = {
        "truth": events(trace, "truth", "S6"),
        "hit": events([dict(row, hit=row["truth"] == row["pred"] == "S6") for row in trace], "hit"),
    }
    first = values["s6_event"]["truth"]["first_step"]
    confirmation = values["s6_event"]["hit"]["first_3step_confirm"]
    values["s6_event"]["first_3step_delay_ms"] = (
        (confirmation - first) * 1000 / hz if first is not None and confirmation is not None else None
    )
    values["confusion"] = {
        state: {p: sum(r["truth"] == state and r["pred"] == p for r in trace)
                for p in (*STATES, "UNKNOWN")}
        for state in STATES
    }
    return values


def score(path, specialized):
    header, rows = load(path)
    hz = header["hz"]
    reference = SevenReference(header["config"], hz, header["reference_positions"])
    neighbors = header["optical_neighbors"]
    observer = SevenObserver(header["config"], "fusion_compensated", revised=True)
    u4_observer = U4ContactObserver(header["config"], handover=True) if specialized else None
    trace, u4_trace = [], []
    approach_s1_misses = Counter()
    phases = {(p["phase"], p["status"]): round(p["sim_time_s"] * hz)
              for p in header["phase_times_for_scoring_only"]}
    first = phases.get(("upper_B_side", "RUNNING"))
    last = phases.get(("upper_B_withdraw", "RUNNING"), phases.get(("upper_B_front_withdraw", "RUNNING")))
    release = phases.get(("upper_release", "OPENING_BOTH_SIDE_GRIPPERS"))
    start = header.get("score_after_step")
    if start is None:
        start = next((step for (phase, _), step in phases.items() if phase == "upper_supply"), 0)
    if u4_observer and (first is None or last is None):
        raise ValueError("U4 replay requires the side-push and withdrawal markers")
    previous = None
    for sensor, physical in rows:
        step, now = sensor["physics_step"], sensor["time_s"]
        if ((physical["physics_step"], physical["time_s"]) != (step, now)
                or (previous is not None and step != previous + 1)):
            raise ValueError(f"Unpaired or missing step in {path}: {step}")
        previous = step
        label = reference.update(physical)["state"]
        # The incident snapshot is available to the independent labeler only.
        optical = sensor.get("optical_neighbors", neighbors)
        result = observer.update(sensor, optical)
        if start < step <= header["stop_step"]:
            trace.append({"step": step, "truth": label, "pred": result["state"]})
            if header["case"] == "a" and label == "S1" and result["state"] != label:
                forces = np.asarray([sensor[arm + "_wrench_world"][:3] for arm in ("left", "right")])
                normals = np.asarray([sensor["cup_normals_world"][arm] for arm in ("left", "right")])
                compression = -np.sum(forces * normals, axis=1)
                loaded = (np.abs(compression) >= FRONT_LOAD_N) | (np.abs(forces[:, 2]) > .9)
                key = ("single_compression" if loaded.sum() == 1 and max(compression) >= FRONT_LOAD_N
                       else f"{loaded.sum()}_loaded_arms_without_single_compression")
                approach_s1_misses[key] += 1
        if u4_observer:
            inputs = dict(sensor)
            command = inputs.get("tcp_command_positions_m", {}).get("right")
            if "right_command_y_m" not in inputs and command:
                inputs["right_command_y_m"] = command[1]
            u4 = u4_observer.update(inputs, optical)
            if first <= step <= last:
                u4_trace.append({"step": step, "truth": label, "pred": u4["state"]})
    if previous != header["stop_step"] or not trace:
        raise ValueError(f"Incomplete trace in {path}")
    stats = summarize(trace, hz)
    if u4_observer:
        stats["u4_side_to_withdraw"] = summarize(u4_trace, hz)
        unload = phases["upper_B_side", "FORCE_CONFIRMED"]
        rear = phases["upper_B_rear", "RUNNING"]
        stats["u4_s5_by_motion"] = {
            name: confusion_metrics([row for row in u4_trace if low <= row["step"] < high])["S5"]
            for name, low, high in (
                ("lateral_push", first, unload),
                ("arm_handover", unload, rear),
                ("rear_push_through_seating", rear, last + 1),
            )
        }
    if header["case"] == "a" and release is not None:
        stats["a_s1_misses_by_wrist_evidence"] = dict(approach_s1_misses)
        stats["s0_timing"] = {
            name: confusion_metrics(part)["S0"] for name, part in (
                ("before_final_release", [r for r in trace if r["step"] <= release]),
                ("after_final_release", [r for r in trace if r["step"] > release]),
            )
        }
    return {"source": header["source"], "case": header["case"], "hz": hz,
            "step_window": [start + 1, header["stop_step"]],
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_sha256": header.get("checkpoint_sha256"),
            "reference_monitor_sha256": header.get("reference_monitor_sha256"),
            "metrics": stats, "optics": optical_audit(rows, "camera")}, trace


def compare(manifest, output):
    entries = json.loads(manifest.read_text())["clips"]
    reports = {}
    pooled, by_case = {}, {}
    for entry in entries:
        report, trace = score(manifest.parent / entry["path"], entry.get("specialized", False))
        method = entry["method"]
        reports[entry["id"]] = {**report, "method": method}
        if entry.get("pool", True):
            pooled.setdefault(method, []).extend(trace)
            accuracy = report["metrics"]["_overall"]["accuracy"]
            by_case.setdefault(method, {}).setdefault(report["case"], []).append(accuracy)
        print(f"Scored {entry['id']}", flush=True)
    aggregate = {method: confusion_metrics(trace) for method, trace in pooled.items()}
    equal_case_accuracy = {method: float(np.mean([np.mean(group) for group in cases.values()]))
                           for method, cases in by_case.items()}
    result = {"fusion": "SevenObserver fusion_compensated, revised, 5 mm tool separation for both optics",
              "ground_truth": "SevenReference from each trajectory's PhysX samples",
              "clips": reports, "pooled_steps": aggregate, "equal_case_accuracy": equal_case_accuracy}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("prepare")
    extract.add_argument("run", type=Path)
    extract.add_argument("case", choices=CUBES)
    extract.add_argument("output", type=Path)
    extract.add_argument("--checkpoint", type=Path)
    replay = commands.add_parser("score")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run, args.case, args.output, args.checkpoint)
    else:
        compare(args.manifest, args.output)


if __name__ == "__main__":
    main()
