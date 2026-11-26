"""Audit where frozen S5 estimates lose evidence on archived physical runs."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from active_probe_replay import MAX_COMMAND_M, MAX_TCP_FOLLOW_FRACTION, MIN_COMMAND_M, MIN_FORCE_RISE_N
from fusion_upper5 import FRONT_LOAD_N
from seven_observer import SevenObserver
from seven_reference import SevenReference
from u4_contact_observer import REAR, SIDE, U4ContactObserver


class BoundaryAudit(SevenObserver):
    """Record the inputs and outputs of the existing observer gates."""

    def __init__(self, config):
        super().__init__(config, "fusion_compensated", revised=True)

    def update(self, sensor, neighbors=None):
        self.direct = self.geometry = None
        self.responses = {}
        self.last_output = super().update(sensor, neighbors)
        return self.last_output

    def _boundaries(self, center, yaw, push, neighbors, now, loaded_faces=None):
        hits = super()._boundaries(center, yaw, push, neighbors, now, loaded_faces)
        if push is not None:
            self.direct = hits
            self.geometry = super()._boundaries(center, yaw, None, neighbors, now, loaded_faces)
        return hits

    def _confirm_boundaries(self, candidates, responses):
        self.responses = responses
        return super()._confirm_boundaries(candidates, responses)


def bearing_force(frame, contacts, hz):
    axis_impulses = []
    active = f"/World/Cube_{frame['active_cube_id']}"
    for pair in frame["cube_contacts"]["pairs"]:
        members = (pair["collider0"], pair["collider1"])
        if active not in members or not pair["current_step_observed"]:
            continue
        other = members[1] if members[0] == active else members[0]
        if other in contacts:
            axes = pair.get("normal_axis_impulse_abs_Ns")
            axis_impulses.append(max(axes[:2]) if axes is not None else pair["normal_impulse_Ns"])
    if not axis_impulses:
        raise ValueError("An S5 reference sample is missing its bearing impulse")
    return hz * sum(axis_impulses)


def motion_windows(header):
    hz = header["hz"]
    marks = {(p["phase"], p["status"]): round(p["sim_time_s"] * hz)
             for p in header["phase_times_for_scoring_only"]}
    if header["case"] == "u4":
        start = marks["upper_B_side", "RUNNING"]
        seated = marks["upper_B_side", "FORCE_CONFIRMED"]
        rear = marks["upper_B_rear", "RUNNING"]
        release = marks.get(("upper_B_withdraw", "RUNNING"), marks.get(("upper_B_front_withdraw", "RUNNING")))
        return (("lateral", start, seated), ("exchange", seated, rear), ("rear", rear, release + 1))
    return (("insertion", 1, header["stop_step"] + 1),)


def identified_path(trace, target, state):
    if state["state"] == "S5":
        return "correct_state" if target in state["boundaries"] else "correct_state_wrong_object"
    if target in state["boundaries"]:
        return "identified_along_with_another_boundary"
    if trace.direct is None:
        return "grounded_push_branch_not_eligible"
    if target not in trace.geometry:
        return "outside_geometry_candidate_set"
    if target not in trace.direct:
        return "push_not_towards_candidate"
    if sum(face == trace.direct[target] for face in trace.direct.values()) > 1:
        return "ambiguous_same_normal_candidates"
    change = [r for (_, direction), r in trace.responses.items() if direction == trace.direct[target]]
    if not change:
        return "no_contiguous_response_on_contact_normal"
    return "contact_response_not_confirmed"


def response_gates(audit, target):
    if audit.direct is None or target not in audit.direct:
        return ()
    response = [r for (_, normal), r in audit.responses.items() if normal == audit.direct[target]]
    failures = []
    for r in response:
        command = r["command_advance_m"]
        if command < MIN_COMMAND_M:
            failures.append("advance_below_minimum")
        if command > MAX_COMMAND_M:
            failures.append("advance_above_maximum")
        if abs(r["encoder_advance_m"]) > MAX_TCP_FOLLOW_FRACTION * abs(command):
            failures.append("tcp_follows_too_much")
        if r["axial_force_rise_N"] < MIN_FORCE_RISE_N:
            failures.append("force_rise_below_minimum")
    return set(failures)


def arm_compression(sensor, arm):
    force = np.asarray(sensor[arm + "_wrench_world"])
    normal = np.asarray(sensor["cup_normals_world"][arm])
    return -float(force[:3] @ normal)


def audit(path):
    with gzip.open(path, "rt") as file:
        header = json.loads(next(file))
        reference = SevenReference(header["config"], header["hz"], header["reference_positions"])
        general = BoundaryAudit(header["config"])
        use_specialist = header["case"] == "u4"
        if use_specialist:
            observer = U4ContactObserver(header["config"], handover=True)
            observer.base = general
        windows = motion_windows(header)
        sections = {name: {key: Counter() for key in
                           ("truth_objects", "generic_prediction", "generic_gate", "response_failures",
                            "camera_pose_age", "geometry_candidate",
                            "specialized_prediction", "specialized_side_miss_conditions",
                            "s5_by_load", "s1_geometry_candidates")}
                    for name, _, _ in windows}
        for line in file:
            sensor, frame = json.loads(line)
            step = sensor["physics_step"]
            if (step, sensor["time_s"]) != (frame["physics_step"], frame["time_s"]):
                raise ValueError(f"Unpaired sample at {step}")
            truth = reference.update(frame)
            neighbors = sensor.get("optical_neighbors", header["optical_neighbors"])
            if use_specialist:
                specialized = observer.update(sensor, neighbors)
                result = general.last_output
            else:
                result = general.update(sensor, neighbors)
                specialized = None
            phase = next((name for name, start, stop in windows if start <= step < stop), None)
            if phase is None or truth["state"] not in ("S5", "S1"):
                continue
            s = sections[phase]
            if truth["state"] == "S1":
                s["s1_geometry_candidates"]["true_s1_steps"] += 1
                if general.direct:
                    s["s1_geometry_candidates"]["steps_with_push_aligned_candidate"] += 1
                continue
            # Both U4 and U5 have one physical object for the S5 samples in these clips.
            target, = truth["horizontal_objects"]
            s["truth_objects"][target] += 1
            s["generic_prediction"][result["state"]] += 1
            gate = identified_path(general, target, result)
            s["generic_gate"][gate] += 1
            if gate == "contact_response_not_confirmed":
                s["response_failures"].update(response_gates(general, target))
            age = result["camera_age_s"]
            s["camera_pose_age"]["fresh" if age is not None and 0 <= age <= .40 else "missing_or_stale"] += 1
            if general.geometry is None:
                s["geometry_candidate"]["not_evaluated"] += 1
            else:
                key = "present" if target in general.geometry else "absent"
                s["geometry_candidate"][key] += 1
            load = bearing_force(frame, (target,), header["hz"])
            band = "below_0.1N" if load < .1 else "0.1_to_1N" if load < 1. else "at_least_1N"
            s["s5_by_load"][band + "_truth"] += 1
            if result["state"] == "S5":
                s["s5_by_load"][band + "_generic_correct"] += 1
            if specialized is not None:
                s["specialized_prediction"][specialized["state"]] += 1
                if specialized["state"] == "S5":
                    s["s5_by_load"][band + "_specialized_correct"] += 1
                elif target == SIDE:
                    misses = s["specialized_side_miss_conditions"]
                    left_compression, right_compression = (arm_compression(sensor, arm) for arm in ("left", "right"))
                    right = sensor["right_wrench_world"]
                    forward = sensor["cup_normals_world"]["right"][1] > .95 and right_compression >= FRONT_LOAD_N
                    if specialized["state"] == "S6":
                        misses["also_reported_rear"] += 1
                    elif not forward:
                        misses["right_arm_not_loaded_forward"] += 1
                    elif abs(left_compression) >= FRONT_LOAD_N:
                        misses["left_arm_still_loaded"] += 1
                    elif not observer.lateral_seen:
                        misses["no_retained_handover_context"] += 1
                    elif age is None or not 0 <= age <= .40:
                        misses["stale_without_handover"] += 1
                    elif specialized["side_raw"] is False:
                        misses["side_reported_absent"] += 1
                    elif (specialized["side_gap_m"] is None
                          or not (-.008 if specialized["supported_view_used"] else -.012)
                          <= specialized["side_gap_m"] <= .012):
                        misses["outside_specialized_side_gap_band"] += 1
                    elif specialized["side_raw"] is None and not (
                            right[0] < -.15 or (right[5] < -.055 and right[0] < .35)):
                        misses["right_wrench_no_signed_side_evidence"] += 1
                    else:
                        misses["other_with_handover_context"] += 1
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())["clips"]
    names = ("aruco_u4", "fitting_u4", "aruco_u5", "fitting_u5")
    results = {}
    for entry in entries:
        if entry["id"] not in names:
            continue
        results[entry["id"]] = audit(args.manifest.parent / entry["path"])
        print(f"Audited {entry['id']}", flush=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
