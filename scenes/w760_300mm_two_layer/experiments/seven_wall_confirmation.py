"""Replay frozen U5 rear-wall nominations with the prior command/response confirmation."""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from active_probe_replay import (AVERAGE_END_S, MAX_COMMAND_M, MAX_TCP_FOLLOW_FRACTION,
                                 MIN_COMMAND_M, MIN_FORCE_RISE_N, TRUTH_HOLD_S, WINDOW_S,
                                 command_kinematics, response)
from fusion_skill_boundary import BACK, live_pairs
from seven_observer import SevenObserver


def interval(header):
    events = [e for e in header["phase_times_for_scoring_only"] if e["phase"] == "upper_B_rear"]
    t = {e["status"]: round(e["sim_time_s"] * header["hz"]) for e in events}
    return t["RUNNING"] + 1, t["FORCE_CONFIRMED"]


def extract(run, directory, header, output):
    model, checksums = command_kinematics(directory)
    meta = json.loads((run / "metadata.json").read_bytes())
    cfg = json.loads((run / "runner_report.json").read_bytes())["scene_parameters"]
    offset = np.asarray(meta["tcp_offset_sensor_m"]["left"])
    yaw = cfg["fr3"]["bases"]["left"][3]
    axis = [np.sin(yaw), np.cos(yaw), 0]
    def y(q):
        wrist = model.pose(np.asarray(q, dtype=float), "fr3_link8")
        return float(axis @ (wrist.translation + wrist.rotation.matrix() @ offset))
    first, last = interval(header)
    source = run / "wrench.jsonl.gz"
    requests, bases = [], []
    with gzip.open(source, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if first <= row["physics_step"] <= last:
                if row["physics_step"] != first + len(requests):
                    raise ValueError("Discontinuous controller stream")
                bases.append(row["left"]["tcp_position_m"][1] - y(row["left"]["q_rad"]))
                requests.append(y(row["control"]["position_targets_rad"]["left"]))
    if len(requests) != last - first + 1 or np.ptp(bases) > .00002:
        raise ValueError("Missing command or moving base during the stroke")
    with gzip.open(output, "wt") as f:
        json.dump({"first_step": first, "last_step": last, "base_y_m": bases[0],
                   "max_base_range_m": float(np.ptp(bases)), "robot_sha256": checksums,
                   "wrench_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "target_y_relative_to_base_m": requests}, f)


def run(clip, commands, output):
    with gzip.open(clip, "rt") as f, gzip.open(commands, "rt") as c:
        header = json.loads(next(f))
        command = json.load(c)
        if header["case"] != "u5":
            raise ValueError("Only the U5 rear stroke is scored")
        hz = header["hz"]
        first, last = interval(header)
        if (first, last) != (command["first_step"], command["last_step"]):
            raise ValueError("Command projection and evaluated stroke disagree")
        est = SevenObserver(header["config"], "fusion_compensated")
        window = deque(maxlen=round(WINDOW_S * hz) + 1)
        bearing = deque(maxlen=round(TRUTH_HOLD_S * hz))
        nominated, confirmed, any_bearing, stable_bearing = set(), set(), set(), set()
        evidence = None
        for line in f:
            sample, ref = json.loads(line)
            tick = sample["physics_step"]
            prediction = est.update(sample, neighbors=header["optical_neighbors"])
            if tick < first or tick > last:
                continue
            if tick != ref["physics_step"] or not ref["cube_contacts"]["valid"]:
                raise ValueError("No matched physical reference")
            window.append([command["target_y_relative_to_base_m"][tick - first],
                           sample["tcp_positions_m"]["left"][1], -sample["left_wrench_world"][1]])
            pairs = live_pairs(ref, "/World/Cube_" + str(header["cube_id"]))
            loaded = BACK in pairs and "left_cup" in pairs.values()
            bearing.append(loaded)
            if loaded:
                any_bearing.add(tick)
            if len(bearing) == bearing.maxlen and sum(bearing) >= .8 * len(bearing):
                stable_bearing.add(tick)
            if BACK not in prediction["boundaries"]:
                continue
            nominated.add(tick)
            if len(window) == window.maxlen:
                delta = response(list(window), round(AVERAGE_END_S * hz))
                distance = delta["command_advance_m"]
                if (MIN_COMMAND_M <= distance <= MAX_COMMAND_M and
                        abs(delta["encoder_advance_m"]) <= MAX_TCP_FOLLOW_FRACTION * distance and
                        delta["axial_force_rise_N"] >= MIN_FORCE_RISE_N):
                    confirmed.add(tick)
                    if evidence is None:
                        evidence = delta
    def score(ticks):
        return {"first_step": min(ticks, default=None), "candidate_steps": len(ticks),
                "candidate_episodes": sum(t - 1 not in ticks for t in ticks),
                "without_current_bearing": len(ticks - any_bearing),
                "without_sustained_bearing": len(ticks - stable_bearing),
                "episodes_starting_before_first_contact": sum(t - 1 not in ticks and t < min(any_bearing)
                                                             for t in ticks) if any_bearing else None}
    report = {"run": header["source"], "control_hz": hz, "scored_steps": [first, last],
              "method": "Frozen seven-state raw ArUco/FK/wrench wall nominations intersected with archived command/response windows",
              "comparison": "Same run was previously scored by the local active_probe_replay rule; held out only from seven-observer development",
              "confirmed_windows_overlap": True, "current_bearing_steps": len(any_bearing),
              "sustained_bearing_steps": len(stable_bearing), "first_current_bearing": min(any_bearing, default=None),
              "first_sustained_bearing": min(stable_bearing, default=None),
              "frozen_passive": score(nominated), "confirmed": score(confirmed), "first_confirmation":evidence}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("commands", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--extract", type=Path, metavar="ARCHIVED_RUN")
    parser.add_argument("--isaac-sim-dir", type=Path)
    args = parser.parse_args()
    if args.extract:
        if not args.isaac_sim_dir:
            parser.error("--extract needs --isaac-sim-dir")
        with gzip.open(args.clip, "rt") as f:
            extract(args.extract, args.isaac_sim_dir, json.loads(next(f)), args.commands)
    run(args.clip, args.commands, args.report)
