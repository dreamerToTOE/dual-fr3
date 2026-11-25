"""Reproduce the U4 online predictions and S5/S6 score from the archived run."""
import argparse
import gzip
import json
from pathlib import Path

from seven_reference import SevenReference
from u4_contact_observer import U4ContactObserver
from u4_handover_score import events, state_scores


def replay(run):
    report = json.loads((run / "runner_report.json").read_text())
    config = report["scene_parameters"]
    hz = report["simulation_clock"]["control_hz"]
    upper = report["upper_row"]
    positions = {
        str(box["cube_id"]): box["position_m"]
        for key in ("initial_lower_poses", "restored_lower_poses_at_start", "initial_upper_poses")
        for box in upper[key] if box["cube_id"] != 10
    }
    marks = {
        row["phase"]: round(row["sim_time_s"] * hz)
        for row in report["phases"] if row["cube_id"] == 10 and row["status"] == "RUNNING"
    }
    first, last = marks["upper_B_side"], marks["upper_B_withdraw"]
    observer = U4ContactObserver(config, handover=True)
    reference = SevenReference(config, hz, positions)
    data = run / "contact_states"
    rows = []
    count = 0
    with (
        gzip.open(data / "sensor_inputs.jsonl.gz", "rt") as inputs,
        gzip.open(data / "reference.jsonl.gz", "rt") as physics,
        gzip.open(data / "u4_handover.jsonl.gz", "rt") as online,
    ):
        streams = (map(json.loads, stream) for stream in (inputs, physics, online))
        for sensor, physical, recorded in zip(*streams, strict=True):
            count += 1
            step = sensor["physics_step"]
            if not all((p["physics_step"], p["time_s"]) == (step, sensor["time_s"])
                       for p in (physical, recorded)):
                raise ValueError(f"Unpaired physics step {step}")
            prediction = observer.update(sensor, neighbors=sensor.get("optical_neighbors", {}))
            fields = ("state", "boundaries", "side_loaded", "rear_loaded")
            if any(prediction[key] != recorded[key] for key in fields):
                raise ValueError(f"Online replay disagrees at physics step {step}")
            truth = reference.update(physical)
            if first <= step <= last:
                rows.append({
                    "step": step,
                    "truth_state": truth["state"],
                    "truth_objects": "|".join(sorted(truth["horizontal_objects"])),
                    "fusion_state": prediction["state"],
                    "fusion_objects": "|".join(sorted(prediction["boundaries"])),
                    "s6_hit": truth["state"] == prediction["state"] == "S6",
                })
    scores = state_scores(rows, "fusion")["totals"]
    for counts in scores.values():
        denominator = counts["true"] + counts["predicted_on_known"]
        counts["f1"] = 2 * counts["tp"] / denominator if denominator else None
    onset = events(rows, "truth_state", "S6")["first_step"]
    confirmation = events(rows, "s6_hit")["first_3step_confirm"]
    return {
        "matched_online_steps": count,
        "scored_steps": len(rows),
        "states": scores,
        "s6_truth_start_step": onset,
        "s6_first_3step_confirmation": confirmation,
        "s6_delay_s": (confirmation - onset) / hz if confirmation is not None else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.archive), indent=2, allow_nan=False))
