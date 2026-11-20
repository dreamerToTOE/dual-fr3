"""Score the physical outcomes of the Upper5 straight-push experiment."""

import argparse
from collections import Counter, deque
import gzip
import json
import math
from pathlib import Path


UPPER_CUBE = "/World/Cube_11"
NEIGHBORS = {"/World/Cube_9", "/World/Cube_10"}
BACK_WALL = "/World/Environment/wall_back"
MIN_IMPULSE_NS = 1e-5
MOVE_WINDOW_S = .25
MOVE_MIN_M = .000020
STALL_WINDOW_S = .5
STALL_MAX_SPAN_M = .000040
STALL_MIN_REAR_GAP_M = .001


def measure(frame, wall_y, half_size):
    position = frame["cube_position_m"]
    rotation = frame["cube_rotation_world"]
    contacts = frame["cube_contacts"]
    impulse = Counter()
    for pair in contacts["pairs"]:
        present = {pair["collider0"], pair["collider1"]}
        load = pair["normal_impulse_Ns"] or 0.
        if UPPER_CUBE not in present or not pair["current_step_observed"] or load <= MIN_IMPULSE_NS:
            continue
        other = (present - {UPPER_CUBE}).pop()
        impulse["left_cup" if pair["label"] == "left_cup" else other] += load
    return {
        "physics_step": frame["physics_step"],
        "time_s": frame["time_s"],
        "cube_position_m": position,
        "cube_rotation_world": rotation,
        "yaw_deg": math.degrees(math.atan2(rotation[1][0], rotation[0][0])),
        "rear_gap_m": wall_y - position[1] - half_size * sum(map(abs, rotation[1])),
        "valid": contacts["valid"] and frame["active_cube_id"] == 11,
        "back_contact": BACK_WALL in impulse,
        "left_cup_contact": "left_cup" in impulse,
        "neighbor_impulse_Ns": {name: impulse[name] for name in sorted(NEIGHBORS & impulse.keys())},
    }


def label_frames(frames, hz):
    short = deque(maxlen=round(MOVE_WINDOW_S * hz) + 1)
    long = deque(maxlen=round(STALL_WINDOW_S * hz) + 1)
    previous = None
    for row in frames:
        state = "UNKNOWN"
        if not row["valid"] or (previous is not None and row["physics_step"] != previous + 1):
            short.clear()
            long.clear()
        previous = row["physics_step"]
        if row["valid"]:
            short.append(row)
            long.append(row)
            moving = (len(short) == short.maxlen and
                      row["cube_position_m"][1] - short[0]["cube_position_m"][1] >= MOVE_MIN_M)
            neighbor = bool(row["neighbor_impulse_Ns"])
            if row["back_contact"] and row["rear_gap_m"] <= STALL_MIN_REAR_GAP_M:
                state = "AT_BACK_TARGET"
            elif neighbor and len(long) == long.maxlen:
                y = [sample["cube_position_m"][1] for sample in long]
                loaded = sum(bool(sample["neighbor_impulse_Ns"]) and sample["left_cup_contact"]
                             for sample in long)
                if (loaded >= .8 * len(long) and max(y) - min(y) <= STALL_MAX_SPAN_M
                        and all(not sample["back_contact"] and sample["rear_gap_m"] > STALL_MIN_REAR_GAP_M
                                for sample in long)):
                    state = "BLOCKED_EARLY"
            if state == "UNKNOWN" and moving and row["left_cup_contact"]:
                state = "NEIGHBOR_CONTACT_MOVING" if neighbor else "NORMAL_ADVANCE"
        yield {**row, "state": state}


def score(run):
    report = json.loads((run / "runner_report.json").read_text())
    hz = report["simulation_clock"]["control_hz"]
    phases = [item for item in report["phases"] if item.get("cube_id") == 11]
    gate = next((item for item in reversed(phases) if item["phase"] == "upper_B_straight_entry_gate"), {})
    config = report["scene_parameters"]
    half_size = config["cube_size_m"] / 2
    _, back, size = next(item for item in config["static_boxes"] if item[0] == "wall_back")
    wall_y = back[1] - size[1] / 2

    start = next((i for i, item in enumerate(phases) if item["phase"] == "upper_B_rear"
                  and item["status"] == "RUNNING"), None)
    touch = next((round(item["sim_time_s"] * hz) + 1 for item in phases
                  if item["phase"] == "upper_B_front_touch"), None)
    first = last = None
    samples = []
    pre_push_neighbor = None
    stop = report["simulation_clock"]["control_steps"]
    if start is not None:
        first = round(phases[start]["sim_time_s"] * hz) + 1
        following = next((item for item in phases[start + 1:] if item["phase"] == "upper_B_rear"
                          and item["status"] == "FORCE_CONFIRMED"), None)
        last = round(following["sim_time_s"] * hz) if following is not None else stop
    if first is not None or touch is not None:
        trace_start = touch if touch is not None else first
        with gzip.open(run / "contact_states" / "reference.jsonl.gz", "rt") as stream:
            for line in stream:
                frame = json.loads(line)
                tick = frame["physics_step"]
                if tick > (last if last is not None else stop):
                    break
                if tick < trace_start:
                    continue
                row = measure(frame, wall_y, half_size)
                if first is not None and tick >= first:
                    samples.append(row)
                elif pre_push_neighbor is None and row["valid"] and row["neighbor_impulse_Ns"]:
                    pre_push_neighbor = row
    labels = list(label_frames(samples, hz))

    counts = Counter(row["state"] for row in labels)
    neighbor = next((row for row in labels if row["valid"] and row["neighbor_impulse_Ns"]), None)
    rear = next((row for row in labels if row["state"] == "AT_BACK_TARGET"), None)
    stalled = next((row for row in labels if row["state"] == "BLOCKED_EARLY"), None)
    seated = any(item["phase"] == "upper_slot_complete" and item["status"] == "SEATED_STABLE" for item in phases)
    wrist_watch = report.get("upper_b_wrist_watch", {})
    wrist_stop = wrist_watch.get("stop")
    if not samples:
        if pre_push_neighbor:
            outcome = "FRONT_TOUCH_NEIGHBOR_CONTACT"
        elif gate.get("status") == "BLOCKED" and start is None:
            outcome = "ENTRY_GATE_REFUSED"
        elif report.get("error") and start is None:
            outcome = "PRE_PUSH_WRIST_CUTOFF" if wrist_stop else "PRE_PUSH_ERROR"
        else:
            outcome = "NO_PHYSICAL_PUSH_TRACE"
    elif stalled:
        outcome = "STALL_THEN_SEATED" if seated else "PHYSICAL_STALL"
    elif seated:
        outcome = "NEIGHBOR_CONTACT_THEN_SEATED" if neighbor else "SEATED_WITHOUT_LOADED_NEIGHBOR_CONTACT"
    elif wrist_stop and first <= wrist_stop["physics_step"] <= last:
        outcome = "PUSH_WRIST_CUTOFF_WITH_NEIGHBOR_CONTACT" if neighbor else "PUSH_WRIST_CUTOFF"
    elif neighbor:
        outcome = "NEIGHBOR_CONTACT_WITHOUT_CONFIRMED_STALL"
    elif report.get("error"):
        outcome = "PUSH_ERROR_WITHOUT_CONFIRMED_STALL"
    else:
        outcome = "UNRESOLVED_PUSH"

    result = {
        "outcome": outcome,
        "runner_status": report["status"],
        "runner_error": report.get("error"),
        "runner_active_phase": report.get("active_phase"),
        "peak_wrist_force": wrist_watch.get("peak_force_N"),
        "wrist_stop": wrist_stop,
        "experiment": report.get("upper5_experiment"),
        "entry_gate": gate,
        "first_push_tick": first,
        "last_push_tick": last,
        "expected_push_frames": 0 if first is None else max(0, last - first + 1),
        "valid_push_frames": sum(row["valid"] for row in labels),
        "state_counts": dict(counts),
        "seated_stable": seated,
        "push_delta_y_mm": 1e3 * (samples[-1]["cube_position_m"][1] - samples[0]["cube_position_m"][1]) if samples else None,
        "final_rear_gap_mm": 1e3 * samples[-1]["rear_gap_m"] if samples else None,
        "first_loaded_neighbor": neighbor,
        "first_loaded_neighbor_during_front_touch": pre_push_neighbor,
        "first_native_wall_contact": rear,
        "first_measured_stall": stalled,
        "neighbor_contact_frame_count": sum(bool(row["neighbor_impulse_Ns"]) and row["valid"] for row in labels),
        "wall_contact_frame_count": counts.get("AT_BACK_TARGET", 0),
        "reference_rules": {"min_current_impulse_Ns": MIN_IMPULSE_NS,
                            "moving_window_s": MOVE_WINDOW_S, "moving_min_forward_mm": MOVE_MIN_M * 1e3,
                            "stall_window_s": STALL_WINDOW_S, "stall_max_y_span_mm": STALL_MAX_SPAN_M * 1e3,
                            "stall_min_rear_gap_mm": STALL_MIN_REAR_GAP_M * 1e3,
                            "stall_min_simultaneous_neighbor_and_cup_fraction": .8},
    }
    return result, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--labels", type=Path, help="Write independent per-tick truth labels as JSONL")
    args = parser.parse_args()
    result, labels = score(args.run)
    if args.labels is not None:
        with args.labels.open("w") as stream:
            for row in labels:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
