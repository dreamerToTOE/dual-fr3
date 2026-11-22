"""Extract the two independent streams for one continuous, single-box pairing."""
import argparse
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


CAMERA = ("frame_id", "status", "capture_time_s", "received_time_s", "position_m",
          "cube_rotation_world", "reprojection_error_px", "supported_position_m",
          "supported_rotation_world", "supported_capture_time_s")
SENSOR = ("time_s", "physics_step", "tcp_positions_m", "cup_normals_world",
          "left_wrench_world", "right_wrench_world")
REFERENCE = ("time_s", "physics_step", "active_cube_id", "cube_position_m",
             "cube_rotation_world", "attachments")
PAIR = ("collider0", "collider1", "label", "current_step_observed", "separation_m",
        "normal_impulse_Ns", "normal_axis_impulse_abs_Ns", "touching", "current_separation_observed")


def select(row, fields):
    return {name: row[name] for name in fields if name in row}


def extract(run, clip, output, allow_incomplete=False):
    report = json.loads((run / "runner_report.json").read_text())
    config = report["scene_parameters"]
    hz = report["simulation_clock"]["control_hz"]
    cube_id = {"a": 7, "u4": 10, "u5": 11}[clip]
    events = [e for e in report["phases"] if e.get("cube_id") == cube_id]
    end_phase = "upper_slot_complete" if clip != "a" else "upper_A_capture"
    endings = [e for e in events if e["phase"] == end_phase]
    if not endings and not (allow_incomplete and clip == "u4" and report["status"] == "STOPPED"):
        raise ValueError(f"No {end_phase} for cube {cube_id}: inspect capture end before scoring.")
    end = round(endings[-1]["sim_time_s"] * hz) if endings else report["simulation_clock"]["control_steps"]
    poses = report["upper_row"]
    positions = {str(row["cube_id"]): row["position_m"]
                 for key in ("initial_lower_poses", "restored_lower_poses_at_start", "initial_upper_poses")
                 for row in poses.get(key, []) if row["cube_id"] != cube_id}
    # Neighbor snapshots describe boxes placed before this clip. The active box is never projected.
    optical = {}
    for packet in report.get("neighbor_observations", []):
        if packet["cube_id"] != cube_id:
            continue
        for neighbor in packet["neighbors"]:
            stamp = neighbor["capture_time_s"]
            arrivals = [a["ended_s"] for a in report.get("neighbor_observation_attempts", [])
                        if a["cube_id"] == cube_id and a["started_s"] <= stamp <= a["ended_s"]]
            if arrivals and neighbor["name"] != f"Cube_{cube_id}":
                path = "/World/" + neighbor["name"]
                # Retain the first mapped pose so later measurements cannot leak into earlier times.
                optical.setdefault(path, {**select(neighbor, ("position_m", "rotation_world", "capture_time_s")),
                                          "received_time_s": min(arrivals)})
    raw, truth = (run / "contact_states" / (name + ".jsonl.gz") for name in ("sensor_inputs", "reference"))
    paths = [raw, truth, run / "runner_report.json"]
    blur_path = run / "vision/blurred_camera.jsonl"
    blur = {}
    if blur_path.exists():
        paths.append(blur_path)
        with blur_path.open() as file:
            for line in file:
                packet = json.loads(line)
                blur[packet["frame_id"]] = select(packet, CAMERA)
    header = {"case": clip, "cube_id": cube_id, "source": run.name, "hz": hz, "stop_step": end,
              "config": select(config, ("cube_size_m", "floor_top_m", "physics", "static_boxes",
                                       "supply_cube_center_m")),
              "reference_positions": positions, "optical_neighbors": optical, "has_blur": bool(blur),
              "phase_times_for_scoring_only": [select(e, ("phase", "status", "sim_time_s"))
                                                for e in events if e["sim_time_s"] <= end / hz + 1e-7],
              "source_sha256": {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if not endings:
        header["incomplete_run"] = {"status": report["status"], "reason": report.get("error")}
    count = 0
    last_frame_id = None
    with gzip.open(output, "wt", compresslevel=6) as out, gzip.open(raw, "rt") as obs, gzip.open(truth, "rt") as ref:
        def emit(data):
            out.write(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
        emit(header)
        for sensor, physical in zip_longest(obs, ref):
            if sensor is None or physical is None:
                raise ValueError("Mismatched stream lengths")
            sensor, physical = json.loads(sensor), json.loads(physical)
            tick = sensor["physics_step"]
            if tick != physical["physics_step"] or sensor["time_s"] != physical["time_s"]:
                raise ValueError(f"Unpaired step {tick}")
            if tick > end:
                break
            if physical["active_cube_id"] != cube_id:
                continue
            packet = sensor.get("camera")
            row = select(sensor, SENSOR)
            if packet and packet["frame_id"] != last_frame_id:
                last_frame_id = packet["frame_id"]
                row["camera"] = select(packet, CAMERA)
                if last_frame_id in blur:
                    row["blur"] = blur[last_frame_id]
            gt = select(physical, REFERENCE)
            contacts = physical["cube_contacts"]
            gt["cube_contacts"] = {"valid": contacts["valid"], "pairs": [select(p, PAIR) for p in contacts["pairs"]]}
            emit([row, gt])
            count += 1
    if not count:
        raise ValueError("Empty physical clip")
    print(f"{clip}: {count} paired ticks -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("case", choices=("a", "u4", "u5"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Keep an interrupted U4 contact trial through its last physical step")
    args = parser.parse_args()
    extract(args.run, args.case, args.output, args.allow_incomplete)
