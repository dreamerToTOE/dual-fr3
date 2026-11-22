"""Score the live U4 reload experiment against the two exact PhysX contact pairs."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

from seven_reference import MIN_IMPULSE_NS
from u4_handover_score import ACTIVE, SIDE, events
from seven_replay import BACK


def bearing(snapshot, boundary, axis, hz):
    if not snapshot["valid"]:
        return None, None
    for pair in snapshot["pairs"]:
        if {pair["collider0"], pair["collider1"]} == {ACTIVE, boundary}:
            impulse = pair["normal_impulse_Ns"]
            if not pair["current_step_observed"] or impulse is None:
                return None, None
            components = pair.get("normal_axis_impulse_abs_Ns")
            if impulse <= MIN_IMPULSE_NS:
                return False, 0. if components is None else components[axis] * hz
            if components is None or components[axis] < .9 * max(components):
                return None, None
            return components[axis] > MIN_IMPULSE_NS, components[axis] * hz
    return False, 0.


def read_stream(path):
    with gzip.open(path, "rt") as frames:
        for line in frames:
            yield json.loads(line)


def score(run):
    report_path = run / "runner_report.json"
    report = json.loads(report_path.read_text())
    probe = report["u4_side_recovery_probe"]
    hz = report["wrist_force_sensors"]["frequency_hz"]
    streams = [run / "contact_states" / f"{name}.jsonl.gz"
               for name in ("reference", "u4_handover", "sensor_inputs")]
    predicted = {row["physics_step"]: row for row in read_stream(streams[1])
                 if probe["start_step"] <= row["physics_step"] <= probe["end_step"]}
    sensors = {row["physics_step"]: row for row in read_stream(streams[2])
               if row["physics_step"] in predicted}
    rows = []
    for physical in read_stream(streams[0]):
        step = physical["physics_step"]
        if step not in predicted or physical["active_cube_id"] != 10:
            continue
        estimate, sensor = predicted[step], sensors[step]
        if not physical["time_s"] == estimate["time_s"] == sensor["time_s"]:
            raise ValueError(f"Streams disagree at step {step}")
        side, side_N = bearing(physical["cube_contacts"], SIDE, 0, hz)
        rear, rear_N = bearing(physical["cube_contacts"], BACK, 1, hz)
        right = sensor["right_wrench_world"]
        rows.append({"step": step, "time_s": estimate["time_s"],
                     "truth_side": side, "truth_side_N": side_N,
                     "truth_rear": rear, "truth_rear_N": rear_N,
                     "state": estimate["state"], "side_loaded": estimate["side_loaded"],
                     "rear_loaded": estimate["rear_loaded"], "side_raw": estimate["side_raw"],
                     "side_event": estimate["side_event"], "boundaries": estimate["boundaries"],
                     "handover_active": estimate["handover_active"], "side_gap_m": estimate["side_gap_m"],
                     "camera_age_s": estimate["camera_age_s"],
                     "side_masked": estimate["u4_silent_loss_probe_active"],
                     "right_fx_N": right[0], "right_mz_Nm": right[5]})

    if [row["step"] for row in rows] != list(range(probe["start_step"], probe["end_step"] + 1)):
        raise ValueError("The probe has missing or duplicated samples")
    return summarize(rows, report, probe, hz), rows, streams + [report_path, run / "runner_snapshot.py"]


def summarize(rows, report, probe, hz):
    loss = next((row for row in rows if row["side_event"] == "side_load_lost_silent"
                 and row["state"] == "S5" and row["side_loaded"] is False), None)
    reload_window = [row for row in rows if row["step"] > probe["reload_command_step"]]
    joint_truth = [{**row, "both": row["truth_side"] is True and row["truth_rear"] is True}
                   for row in reload_window]
    truth_runs = events(joint_truth, "both")
    confirmation = truth_runs["first_3step_confirm"]
    onset = None if confirmation is None else confirmation - 2
    unmasked = [row for row in rows if row["step"] > probe["unmask_step"] and not row["side_masked"]]
    if any(row["side_masked"] for row in reload_window):
        raise ValueError("Reload must use unmasked observations")
    negative = [row for row in unmasked if row["truth_side"] is False and row["truth_rear"] is True]
    masked_negative = [row for row in rows if row["side_masked"]
                       and row["truth_side"] is False and row["truth_rear"] is True]
    dwell = [row for row in unmasked if row["step"] <= probe["reload_command_step"]]
    false_events = [row["step"] for row in negative if row["side_event"] == "side_load_recovered"]
    false_frames = [{**row, "false_s6": row["state"] == "S6",
                     "false_side": row["side_loaded"] is True} for row in negative]
    recoveries = [row for row in unmasked if row["side_event"] == "side_load_recovered"]
    s6 = next((row for row in reload_window if onset is not None and row["step"] >= onset
               and row["truth_side"] is True and row["truth_rear"] is True
               and row["state"] == "S6" and row["side_loaded"] is True and row["rear_loaded"] is True
               and set(row["boundaries"]) == {SIDE, BACK}), None)
    recovery = next((row for row in recoveries if onset is not None and row["step"] >= onset
                     and row["truth_side"] is True and row["truth_rear"] is True
                     and row["state"] == "S6" and row["side_loaded"] is True and row["rear_loaded"] is True
                     and set(row["boundaries"]) == {SIDE, BACK}), None)
    s6_frames = [{**row, "both": (row["truth_side"] is True and row["truth_rear"] is True
                                  and row["state"] == "S6" and row["side_loaded"] is True
                                  and row["rear_loaded"] is True and set(row["boundaries"]) == {SIDE, BACK})}
                 for row in reload_window]
    raw_evidence = []
    if recovery is not None:
        for row in reversed(unmasked):
            if row["step"] > recovery["step"]:
                continue
            if row["side_raw"] is not True:
                break
            raw_evidence.append(row)
    side_forces = [row["truth_side_N"] for row in joint_truth if row["both"]]
    return {"scene": report["scene_id"], "frequency_hz": hz, "status": report["status"],
            "error": report.get("error"), "loss_injection": report.get("u4_silent_loss_probe"), "probe": probe,
            "wrist_envelope": report.get("upper_b_wrist_watch"), "pre_perturbation_observation": rows[0],
            "confirmation_of_side_loss": loss,
            "truth": {"side_pair": [ACTIVE, SIDE], "rear_pair": [ACTIVE, BACK],
                      "minimum_axis_impulse_Ns": MIN_IMPULSE_NS,
                      "both_loaded_after_command": truth_runs, "sustained_reload_start_step": onset,
                      "side_force_after_reload_N": (None if not side_forces else
                                                    {"min": min(side_forces), "median": median(side_forces),
                                                     "max": max(side_forces),
                                                     "ticks_in_0.1_to_1_N": sum(.1 <= f <= 1. for f in side_forces)}),
                      "rear_truth_counts_during_probe": dict(Counter(str(row["truth_rear"]) for row in rows[1:])),
                      "unexpected_contacts": report["physx_contacts"]["unexpected_contact_count"]},
            "observer": {"first_correct_s6": s6, "first_true_recovery_event": recovery,
                         "s6_delay_from_sustained_reload_s": None if s6 is None else (s6["step"] - onset) / hz,
                         "recovery_event_delay_s": None if recovery is None else (recovery["step"] - onset) / hz,
                         "correct_s6_after_reload": events(s6_frames, "both"),
                         "consecutive_raw_side_evidence_at_event": list(reversed(raw_evidence)),
                         "unmasked_dwell_ticks": len(dwell),
                         "pre_reload_observation": dwell[-1] if dwell else None,
                         "unmasked_negative_ticks": len(negative),
                         "unmasked_negative_dwell_ticks": sum(row["step"] <= probe["reload_command_step"] for row in negative),
                         "truth_unknown_unmasked_ticks": sum(row["truth_side"] is None or row["truth_rear"] is None
                                                              for row in unmasked),
                         "masked_loss_negative_ticks": len(masked_negative),
                         "masked_loss_s6_ticks": sum(row["state"] == "S6" for row in masked_negative),
                         "false_recovery_event_steps": false_events,
                         "false_s6": events(false_frames, "false_s6"),
                         "false_side_loaded": events(false_frames, "false_side"),
                         "unmasked_recovery_events": recoveries}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result, rows, sources = score(args.run)
    result["source_run"] = args.run.name
    result["source_sha256"] = {str(path.relative_to(args.run)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    result["timeline"] = rows
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
