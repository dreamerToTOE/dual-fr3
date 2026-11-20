"""Replay the transition from supported motion (skill 3) to seating (skill 4)."""

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path

import numpy as np

from fusion_upper5 import (CAMERA_HOLD_S, CAMERA_MAX_AGE_S, FRONT_LOAD_N, MOTION_WINDOW_S,
                          axes_for, neighbor_clearance, yaw_of)
from score_upper5 import MIN_IMPULSE_NS, STALL_MAX_SPAN_M, STALL_WINDOW_S


BACK = "/World/Environment/wall_back"
UPPER_BOXES = {f"/World/Cube_{i}" for i in range(7, 12)}
MODES = ("vision_fk", "wrench_fk", "fusion")
STATES = ("ADVANCING", "TARGET_CONTACT", "NO_TARGET_CONTACT", "UNEXPECTED_CONTACT_MOVING",
          "UNEXPECTED_CONTACT", "BLOCKED_EARLY", "OVERLOAD_WHILE_ADVANCING")
CONTACT_BAND_M = .002
ADVANCE_MIN_M = .00005
WRIST_LIMIT_N = 9.
CONFIRM_S = .10
LOAD_CONFIRM_S = .25


@dataclass(frozen=True)
class Task:
    cube_id: int
    arm: str
    axis: int
    sign: int
    target_path: str
    retained: tuple[str, ...] = ()


STROKES = {
    "u4_side": ("upper_B_side", Task(10, "left", 0, 1, "/World/Cube_8")),
    "u4_rear": ("upper_B_rear", Task(10, "right", 1, 1, BACK, ("/World/Cube_8",))),
    "u5_rear": ("upper_B_rear", Task(11, "left", 1, 1, BACK)),
    "u2_reseat": ("upper_B_seam_rear_reseat", Task(8, "right", 1, 1, BACK)),
}


def live_pairs(reference, active):
    """Use only present-step impulses, excluding speculative and stale contacts."""
    hits = {}
    for row in reference["cube_contacts"]["pairs"]:
        if not row["current_step_observed"] or (row["normal_impulse_Ns"] or 0.) <= MIN_IMPULSE_NS:
            continue
        a, b = row["collider0"], row["collider1"]
        if a == active:
            hits[b] = row["label"]
        elif b == active:
            hits[a] = row["label"]
    return hits


def vision_layout(report, task, at):
    """Use the ArUco layout snapshot whose observation completed before push."""
    observations = []
    for packet in report.get("neighbor_observations", []):
        if packet["cube_id"] != task.cube_id:
            continue
        for obj in packet["neighbors"]:
            name = obj["name"]
            stamp = obj["capture_time_s"]
            # The median pose is available only once the complete exposure window was received.
            available = [attempt["ended_s"] for attempt in report.get("neighbor_observation_attempts", [])
                         if attempt["cube_id"] == task.cube_id
                         and attempt["started_s"] <= stamp <= attempt["ended_s"] <= at]
            if available and f"/World/{name}" in UPPER_BOXES:
                observations.append((stamp, f"/World/{name}",
                                     (np.asarray(obj["position_m"][:2]),
                                      axes_for(yaw_of(obj["rotation_world"])))))
    return {name: pose for _, name, pose in sorted(observations, key=lambda row: row[0])}


def task_plane(config, task):
    """The known wall plane or nominal target pose from the packing plan."""
    axis, sign = task.axis, task.sign
    if task.target_path == BACK:
        _, center, size = next(row for row in config["static_boxes"] if row[0] == "wall_back")
        return center[axis] - sign * size[axis] / 2
    index = int(task.target_path.rsplit("_", 1)[-1]) - 7
    return config["goal_cube_centers_m"][index][axis] - sign * config["cube_size_m"] / 2


class Observer:
    def __init__(self, mode, task, half, plane, neighbors, hz):
        self.mode, self.task, self.half, self.plane = mode, task, half, plane
        self.layout = {} if mode == "wrench_fk" else neighbors
        self.other_neighbors = {name: pose for name, pose in self.layout.items()
                                if name not in {task.target_path, *task.retained}}
        self.frame, self.camera = None, None
        self.positions = deque(maxlen=round(MOTION_WINDOW_S * hz) + 1)
        self.blocking = deque(maxlen=round(STALL_WINDOW_S * hz) + 1)
        self.sliding_load = deque(maxlen=round(hz))
        self.provisional_force_steps = 0
        self.load_confirm_steps = round(LOAD_CONFIRM_S * hz)

    def update(self, sample, force, axial, resultant):
        task, half = self.task, self.half
        axis, sign, arm = task.axis, task.sign, task.arm
        now = sample["time_s"]
        tcp = np.asarray(sample["tcp_positions_m"][arm][:2])
        normal = np.asarray(sample["cup_normals_world"][arm][:2])
        faces_push = sign * normal[axis] > .95
        force_enabled = self.mode != "vision_fk"
        loaded = faces_push and force_enabled and axial >= FRONT_LOAD_N
        camera = sample.get("camera")
        if self.mode != "wrench_fk" and camera and camera["frame_id"] != self.frame:
            self.frame = camera["frame_id"]
            if camera.get("supported_position_m") is not None:
                stamp = camera["supported_capture_time_s"]
                if stamp <= camera["received_time_s"] <= now + 1e-8:
                    self.camera = (stamp, np.asarray(camera["supported_position_m"][:2]),
                                   yaw_of(camera["supported_rotation_world"]))

        xy = None
        yaw = 0.
        optical_fresh = False
        if self.camera is not None:
            stamp, optical, yaw = self.camera
            age = now - stamp
            optical_fresh = 0 <= age <= CAMERA_MAX_AGE_S
            if optical_fresh or (loaded and 0 <= age <= CAMERA_HOLD_S):
                xy = optical.copy()
        radius = half if xy is None else half * np.abs(axes_for(yaw)[axis]).sum()
        if loaded:
            inferred = float(tcp[axis] + half * normal[axis])
            if xy is not None:
                xy[axis] = inferred
        elif self.mode == "vision_fk" and xy is not None and faces_push and (
                abs(tcp[axis] - (xy[axis] - sign * radius)) <= .012
                and abs(tcp[1 - axis] - xy[1 - axis]) <= half):
            inferred = float(xy[axis])
        else:
            self.positions.clear()
            self.blocking.clear()
            self.provisional_force_steps = 0
            return {"state": "UNKNOWN", "target_gap_m": None, "predicted_advance_m": None,
                    "camera_fresh": optical_fresh, "force_confirmation_candidate": False}

        plane = self.plane
        target = self.layout.get(task.target_path)
        if target is not None:
            other, axes = target
            plane = other[axis] - sign * half * np.abs(axes[axis]).sum()
        gap = float(sign * (plane - inferred) - radius)
        target_sat_gap = (neighbor_clearance(xy, yaw, {task.target_path: target}, half)
                          if target is not None and xy is not None else None)
        self.positions.append(inferred)
        advance = None if len(self.positions) < self.positions.maxlen else (
            sign * (self.positions[-1] - self.positions[0]))
        geometric_side = (xy is not None and bool(self.other_neighbors) and
                          neighbor_clearance(xy, yaw, self.other_neighbors, half) <= CONTACT_BAND_M)
        side_signal = abs(force[1 - axis]) >= max(.45, .17 * axial)
        # A retained side contact can exert transverse force while the new pusher moves.
        if xy is not None:
            unexpected = geometric_side and (not force_enabled or side_signal)
        else:
            unexpected = side_signal and not task.retained
        unexpected = unexpected and gap > CONTACT_BAND_M
        self.blocking.append((unexpected, inferred))
        sustained_block = (len(self.blocking) == self.blocking.maxlen
                           and sum(hits for hits, _ in self.blocking) >= .8 * len(self.blocking)
                           and np.ptp([value for _, value in self.blocking]) <= STALL_MAX_SPAN_M)

        state = "UNKNOWN"
        if advance is not None:
            if loaded and resultant >= WRIST_LIMIT_N and advance >= ADVANCE_MIN_M:
                state = "OVERLOAD_WHILE_ADVANCING"
            elif sustained_block:
                state = "BLOCKED_EARLY"
            elif advance >= ADVANCE_MIN_M:
                state = "UNEXPECTED_CONTACT_MOVING" if unexpected else "ADVANCING"
            elif unexpected:
                state = "UNEXPECTED_CONTACT"
            elif abs(gap) <= CONTACT_BAND_M:
                state = "TARGET_CANDIDATE"
            else:
                state = "NO_TARGET_CONTACT"
        # Compare the already used terminal gap and load-rise thresholds with the passive candidate.
        # We only accumulate a baseline while independently advancing, away from the candidate band.
        if loaded and advance is not None and advance >= ADVANCE_MIN_M and gap > CONTACT_BAND_M:
            self.sliding_load.append(axial)
        wall = task.target_path == BACK
        close = abs(gap) <= (.00045 if wall else .0018)
        enough_baseline = len(self.sliding_load) >= self.load_confirm_steps
        force_rise = axial - float(np.median(self.sliding_load)) if enough_baseline else None
        loaded_stop = (state == "TARGET_CANDIDATE" and close and force_rise is not None
                       and force_rise >= (1. if wall else .60))
        self.provisional_force_steps = self.provisional_force_steps + 1 if loaded_stop else 0
        return {"state": state, "target_gap_m": gap, "predicted_advance_m": advance,
                "target_sat_gap_m": target_sat_gap,
                "camera_fresh": optical_fresh, "force_rise_N": force_rise,
                "force_confirmation_candidate": self.provisional_force_steps >= self.load_confirm_steps}


def reference_state(frame, task, positions, blocking, stopped):
    active = f"/World/Cube_{task.cube_id}"
    if not frame["cube_contacts"]["valid"] or frame.get("active_cube_id") != task.cube_id:
        positions.clear()
        blocking.clear()
        return {"state": "UNKNOWN"}
    hits = live_pairs(frame, active)
    target_contact = task.target_path in hits
    cup_contact = f"{task.arm}_cup" in hits.values()
    unwanted = (hits.keys() & UPPER_BOXES) - {task.target_path, *task.retained}
    if BACK in hits and task.target_path != BACK:
        unwanted.add(BACK)
    x = frame["cube_position_m"][task.axis]
    positions.append(x)
    blocking.append((bool(unwanted and not target_contact and cup_contact), x))
    advance = None if len(positions) < positions.maxlen else task.sign * (positions[-1] - positions[0])
    state = "UNKNOWN"
    if cup_contact and target_contact:
        state = "TARGET_CONTACT"
    elif cup_contact and advance is not None:
        if stopped and advance >= ADVANCE_MIN_M:
            state = "OVERLOAD_WHILE_ADVANCING"
        elif unwanted and len(blocking) == blocking.maxlen and (
                sum(hit for hit, _ in blocking) >= .8 * len(blocking)
                and np.ptp([value for _, value in blocking]) <= STALL_MAX_SPAN_M):
            state = "BLOCKED_EARLY"
        elif advance >= ADVANCE_MIN_M:
            state = "UNEXPECTED_CONTACT_MOVING" if unwanted else "ADVANCING"
        elif unwanted:
            state = "UNEXPECTED_CONTACT"
        else:
            state = "NO_TARGET_CONTACT"
    return {"state": state, "target_contact": target_contact, "cup_contact": cup_contact,
            "retained_contacts": sorted(hits.keys() & set(task.retained)),
            "unexpected_contacts": sorted(unwanted), "physical_advance_m": advance}


def review(run, stroke, trace):
    phase, task = STROKES[stroke]
    report = json.loads((run / "runner_report.json").read_text())
    cfg = report["scene_parameters"]
    clock = report["simulation_clock"]
    hz = clock["control_hz"]
    events = [row for row in report["phases"] if row["phase"] == phase
              and row.get("cube_id") == task.cube_id]
    beginning = next(row for row in events if row["status"] == "RUNNING")
    ending = next((row for row in events if row["status"] == "FORCE_CONFIRMED"), None)
    first = round(beginning["sim_time_s"] * hz) + 1
    last = clock["control_steps"] if ending is None else round(ending["sim_time_s"] * hz)
    wrist_stop = (report.get("upper_b_wrist_watch", {}).get("stop") or {}).get("physics_step")
    layout = vision_layout(report, task, first / hz)
    plane = task_plane(cfg, task)
    half = cfg["cube_size_m"] / 2
    observers = {mode: Observer(mode, task, half, plane, layout, hz) for mode in MODES}
    task_fields = asdict(task)
    positions = deque(maxlen=round(MOTION_WINDOW_S * hz) + 1)
    blocking = deque(maxlen=round(STALL_WINDOW_S * hz) + 1)
    rows = []
    directory = run / "contact_states"
    with gzip.open(directory / "sensor_inputs.jsonl.gz", "rt") as sensors, \
            gzip.open(directory / "reference.jsonl.gz", "rt") as references:
        for sensor_line, reference_line in zip(sensors, references, strict=True):
            sample, truth = json.loads(sensor_line), json.loads(reference_line)
            step = sample["physics_step"]
            if step > last:
                break
            if step < first:
                continue
            if step != truth["physics_step"]:
                raise ValueError(f"{run}: sensor and truth clocks disagree at step {step}")
            if step != first + len(rows):
                raise ValueError(f"{run}: discontinuous samples in {stroke} at step {step}")
            force = np.asarray(sample[f"{task.arm}_wrench_world"][:3])
            axial = float(-task.sign * force[task.axis])
            resultant = float(np.linalg.norm(force))
            outputs = {mode: observer.update(sample, force, axial, resultant)
                       for mode, observer in observers.items()}
            evaluated = reference_state(truth, task, positions, blocking, step == wrist_stop)
            row = {"case_id": f"{run.name}:{stroke}", "physics_step": step,
                   "time_s": sample["time_s"], "task": task_fields, "axial_force_N": axial,
                   "resultant_force_N": resultant, "reference": evaluated["state"],
                   "predictions": {mode: observation["state"] for mode, observation in outputs.items()},
                   "reference_detail": evaluated, "prediction_detail": outputs}
            rows.append(row)
            if trace is not None:
                trace.write(json.dumps(row, allow_nan=False) + "\n")
    if not rows:
        raise ValueError(f"{run}: no samples for {stroke}")
    return summarize(rows, hz, layout, report.get("error"))


def summarize(rows, hz, layout, error):
    scored = [row for row in rows if row["reference"] != "UNKNOWN"]
    if not scored:
        raise ValueError("No valid contact reference samples")
    counts = Counter(row["reference"] for row in scored)

    def first_stable_step(stages):
        contact = 0
        for step, active in stages:
            contact = contact + 1 if active else 0
            if contact >= round(CONFIRM_S * hz):
                return step
        return None

    physical = first_stable_step((r["physics_step"], r["reference"] == "TARGET_CONTACT") for r in rows)
    metrics = {}
    for mode in MODES:
        confusion = {state: Counter() for state in counts}
        for row in scored:
            confusion[row["reference"]][row["predictions"][mode]] += 1
        target = first_stable_step((r["physics_step"], r["predictions"][mode] == "TARGET_CANDIDATE") for r in rows)
        force_stops = [r for r in scored if r["prediction_detail"][mode]["force_confirmation_candidate"]]
        proposed_confirmation = force_stops[0]["physics_step"] if force_stops else None
        counterpart = {state: ("TARGET_CANDIDATE" if state == "TARGET_CONTACT" else state) for state in counts}
        metrics[mode] = {
            "candidate_vs_contact_accuracy": sum(values[counterpart[state]]
                                                 for state, values in confusion.items()) / len(scored),
            "coverage": 1 - sum(values["UNKNOWN"] for values in confusion.values()) / len(scored),
            "candidate_recall": {state: values[counterpart[state]] / counts[state]
                                 for state, values in confusion.items()},
            "confusion": confusion, "first_sustained_candidate_step": target,
            "sustained_candidate_timing_s": (None if target is None or physical is None
                                          else (target - physical) / hz),
            "candidate_without_native_contact_ticks": sum(values["TARGET_CANDIDATE"]
                                                           for state, values in confusion.items()
                                                           if state != "TARGET_CONTACT"),
            "first_force_confirmation_candidate_step": proposed_confirmation,
            "force_confirmation_candidate_timing_s": (None if proposed_confirmation is None or physical is None
                                                       else (proposed_confirmation - physical) / hz),
            "force_confirmation_candidate_ticks": len(force_stops),
            "force_confirmation_candidate_without_native_contact_ticks": sum(
                r["reference"] != "TARGET_CONTACT" for r in force_stops),
        }
    return {"task": rows[0]["task"], "reference_counts": counts, "scored_ticks": len(scored),
            "absent_states": [state for state in STATES if state not in counts],
            "first_sustained_physical_target_step": physical,
            "retained_contact_ticks": sum(bool(r["reference_detail"].get("retained_contacts")) for r in rows),
            "causal_rgb_neighbor_paths": sorted(layout), "runner_error": error, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=2, required=True, metavar=("RUN", "STROKE"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if any(stroke not in STROKES for _, stroke in args.case):
        parser.error(f"STROKE must be one of {', '.join(STROKES)}")
    modes = {"vision_fk": "Supported-plane ArUco and TCP geometry",
             "wrench_fk": "Wrist wrench, TCP motion and nominal boundary",
             "fusion": "Wrist wrench, TCP motion and observed boundary / lateral geometry"}
    results = {"modes": modes, "contact_band_m": CONTACT_BAND_M, "motion_window_s": MOTION_WINDOW_S,
               "advance_min_m": ADVANCE_MIN_M, "sustained_target_s": CONFIRM_S,
               "candidate_interpretation": "Near and slow can enter skill 4. It never confirms the intended object.",
               "force_confirmation_candidate": "Exploratory 250 ms load-rise criterion using existing "
               "terminal thresholds; baseline requires 250 ms of advancing samples outside the candidate band.",
               "threshold_provenance": "Fixed engineering criteria, including existing terminal thresholds "
               "for the exploratory force-rise comparison; no parameter fitting",
               "wrist_cutoff_reference": "Runner wrist event plus independently measured true cube advance",
               "context": "Predeclared arm, direction, and intended or retained object; phase records only delimit scoring",
               "cases": {}}
    trace = None if args.trace is None else args.trace.open("w")
    try:
        for path, stroke in args.case:
            run = Path(path)
            results["cases"][f"{run.name}:{stroke}"] = review(run, stroke, trace)
    finally:
        if trace is not None:
            trace.close()
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
