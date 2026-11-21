"""Project the issued right-arm targets onto world Y at the measured rail position."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from active_probe_replay import command_kinematics


def extract(run, clip, isaac_dir, output):
    with gzip.open(clip, "rt") as f:
        header = json.loads(next(f))
        first = json.loads(next(f))[0]["physics_step"]
    if header["case"] != "u4" or header["source"] != run.name:
        raise ValueError("Expected a U4 pairing from this physics run")
    sensor_sha = hashlib.sha256((run / "contact_states/sensor_inputs.jsonl.gz").read_bytes()).hexdigest()
    if sensor_sha != header["source_sha256"]["contact_states/sensor_inputs.jsonl.gz"]:
        raise ValueError("Camera/wrist source differs from paired clip")
    report = json.loads((run / "runner_report.json").read_bytes())
    yaw = report["scene_parameters"]["fr3"]["bases"]["right"][3]
    axis = np.array([math.sin(yaw), math.cos(yaw), 0.])
    offset = np.array(json.loads((run / "metadata.json").read_bytes())["tcp_offset_sensor_m"]["right"])
    model, model_sha = command_kinematics(isaac_dir)

    def local_y(q):
        pose = model.pose(np.asarray(q), "fr3_link8")
        return float(axis @ (pose.translation + pose.rotation.matrix() @ offset))

    commands = []
    source = run / "wrench.jsonl.gz"
    last = header["stop_step"]
    with gzip.open(source, "rt") as f, gzip.open(clip, "rt") as paired:
        next(paired)
        for line in f:
            row = json.loads(line)
            tick = row["physics_step"]
            if tick < first:
                continue
            if tick > last:
                break
            observed = json.loads(next(paired))[0]
            if (tick != first + len(commands) or tick != observed["physics_step"]
                    or row["time_s"] != observed["time_s"]):
                raise ValueError(f"Misaligned right-arm command at {tick}")
            arm = row["right"]
            base_y = arm["tcp_position_m"][1] - local_y(arm["q_rad"])
            commands.append(base_y + local_y(row["control"]["position_targets_rad"]["right"]))
    if not commands or first + len(commands) - 1 != last:
        raise ValueError("Incomplete command stream")
    with gzip.open(output, "wt") as f:
        json.dump({"source": run.name, "first_step": first, "last_step": last,
                   "sensor_inputs_sha256": sensor_sha, "robot_sha256": model_sha,
                   "wrench_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "target_y_m": commands}, f, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "clip", "isaac_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    extract(args.run, args.clip, args.isaac_dir, args.output)
