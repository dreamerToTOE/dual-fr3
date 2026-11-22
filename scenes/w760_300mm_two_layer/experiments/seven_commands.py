"""Join issued FR3 TCP commands to an existing paired observation clip."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from active_probe_replay import command_kinematics


ARMS = ("left", "right")
SENSOR_SOURCE = "contact_states/sensor_inputs.jsonl.gz"
DEFAULT_ISAAC_DIR = Path("/home/ubuntu2004/isaacsim-4.5.0")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets(row, sensor, kinematics, base_yaws, offsets):
    commands = {}
    issued = row.get("control", {}).get("position_targets_rad", {})
    for arm in ARMS:
        if issued.get(arm) is None:
            continue
        observed = np.asarray(sensor["tcp_positions_m"][arm])
        measured = row[arm]
        if not np.allclose(measured["tcp_position_m"], observed, rtol=0., atol=1e-8):
            raise ValueError(f"{arm} TCP disagrees with the paired clip at {row['physics_step']}")
        yaw = base_yaws[arm]
        c, s = np.cos(yaw), np.sin(yaw)

        def tool_position(q):
            pose = kinematics.pose(np.asarray(q), "fr3_link8")
            return pose.translation + pose.rotation.matrix() @ offsets[arm]

        x, y, z = tool_position(issued[arm]) - tool_position(measured["q_rad"])
        # The measured TCP supplies that step's world translation, including rail motion.
        commands[arm] = (observed + np.array([c*x - s*y, s*x + c*y, z])).tolist()
    return commands


def attach_commands(run, clip, isaac_dir, output):
    if clip.resolve() == output.resolve():
        raise ValueError("Write the command-enriched clip to a new path")
    source = run / "wrench.jsonl.gz"
    cfg = json.loads((run / "runner_report.json").read_text())["scene_parameters"]
    metadata = json.loads((run / "metadata.json").read_text())
    base_yaws = {arm: cfg["fr3"]["bases"][arm][3] for arm in ARMS}
    offsets = {arm: np.asarray(metadata["tcp_offset_sensor_m"][arm]) for arm in ARMS}
    kinematics, model_sha = command_kinematics(isaac_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(clip, "rt") as paired, gzip.open(source, "rt") as wrench:
        header = json.loads(next(paired))
        if (header["source"] != run.name
                or header["source_sha256"][SENSOR_SOURCE] != digest(run / SENSOR_SOURCE)):
            raise ValueError("Raw wrist commands are from a different physical run")
        header["tcp_commands"] = {"wrench_sha256": digest(source), "robot_sha256": model_sha}
        rows = (json.loads(line) for line in wrench)
        wrist = next(rows, None)
        last = count = 0
        with gzip.open(output, "xt", compresslevel=6) as enriched:
            enriched.write(json.dumps(header, allow_nan=False, separators=(",", ":")) + "\n")
            for item in paired:
                sensor, physical = json.loads(item)
                tick = sensor["physics_step"]
                while wrist is not None and wrist["physics_step"] < tick:
                    wrist = next(rows, None)
                if (wrist is None or wrist["physics_step"] != tick or wrist["time_s"] != sensor["time_s"]
                        or physical["physics_step"] != tick or physical["time_s"] != sensor["time_s"]
                        or (count and tick != last + 1)):
                    raise ValueError(f"No aligned TCP command sample at step {tick}")
                commands = targets(wrist, sensor, kinematics, base_yaws, offsets)
                sensor.pop("tcp_command_positions_m", None)
                if commands:
                    sensor["tcp_command_positions_m"] = commands
                enriched.write(json.dumps([sensor, physical], allow_nan=False,
                                          separators=(",", ":")) + "\n")
                last = tick
                count += 1
            if not count or last != header["stop_step"]:
                raise ValueError("Incomplete paired clip")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "clip", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--isaac-sim-dir", type=Path, default=DEFAULT_ISAAC_DIR)
    args = parser.parse_args()
    count = attach_commands(args.run, args.clip, args.isaac_sim_dir, args.output)
    print(f"{count} paired steps -> {args.output}")
