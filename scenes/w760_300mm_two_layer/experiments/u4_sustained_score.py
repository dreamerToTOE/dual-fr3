"""Evaluate duration-stratified U4 side-bearing events and compare two observers."""
import argparse
from collections import Counter
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path

from u4_handover_score import paired_rows
from u4_left_unload_score import MODELS, left_cup_reference
from u4_pulse_score import digest
from u4_side_continuous_score import comparison_windows, declarations, total


WINDOWS = ("handoff", "rear_stroke")
DURATIONS = ("1_to_5", "6_to_11", "at_least_12")


def duration_class(ticks):
    return DURATIONS[0] if ticks <= 5 else DURATIONS[1] if ticks <= 11 else DURATIONS[2]


def evaluate_bouts(rows, hz, prior=()):
    """Match each new declaration to one continuous PhysX state interval."""
    physics = []
    for state, group in groupby(rows, key=lambda r: r["truth_side"]):
        segment = list(group)
        physics.append({"state": state, "start": segment[0]["step"], "end": segment[-1]["step"],
                        "ticks": len(segment), "window": segment[0]["window"]})
    alarms = {}
    for model in MODELS:
        calls = [call for call in declarations([*prior, *rows], model, 3)
                 if call[0] >= rows[0]["step"]]
        stats = Counter()
        for _, _, positive, values in calls:
            key = "bearing" if positive else "clear"
            if any(v is (not positive) for v in values):
                stats[key+"_false"] += 1
            elif any(v is None for v in values):
                stats[key+"_unknown"] += 1
        for interval in physics:
            state = interval["state"]
            inside = [] if state is None else [
                end for start, end, called, _ in calls
                if called is state and interval["start"] <= start and end <= interval["end"]
            ]
            interval[model] = {"confirm_step": inside[0] if inside else None,
                               "latency_s": (inside[0]-interval["start"])/hz if inside else None}
            if inside:
                key = "bearing" if state else "clear"
                stats[key+"_"+duration_class(interval["ticks"])] += len(inside)
        alarms[model] = dict(stats)
    return physics, alarms


def first_pair(physics, n, zero, hz):
    # A leading clear block with no witnessed bearing does not have an observed loss onset.
    clear_index = next((i for i, r in enumerate(physics) if r["state"] is False and r["ticks"] >= n
                        and i and physics[i-1]["state"] is True), None)
    if clear_index is None:
        return {"loss": None, "reload": None, "first_observed_clear_censored": bool(
            any(r["state"] is False and r["ticks"] >= n for r in physics))}
    loss = physics[clear_index]
    reload = next((physics[i] for i in range(clear_index+1, len(physics))
                   if physics[i]["state"] is True and physics[i]["ticks"] >= n
                   and physics[i-1]["state"] is False), None)
    return {"loss": {**loss, "onset_s": (loss["start"]-zero)/hz},
            "reload": {**reload, "onset_s": (reload["start"]-zero)/hz} if reload else None,
            "first_observed_clear_censored": False}


def trial_events(rows, hz):
    exchange = [r for r in rows if r["window"] in WINDOWS]
    if not exchange or exchange[0]["window"] != "handoff":
        return None
    zero = exchange[0]["step"]-1
    prior = [r for r in rows if zero - hz < r["step"] <= zero]
    physics, alarms = evaluate_bouts(exchange, hz, prior)
    by_state = {}
    for state, label in ((True,"bearing"), (False,"clear")):
        intervals = [r for r in physics if r["state"] is state]
        by_state[label] = Counter(duration_class(r["ticks"]) for r in intervals)
    return {"unload_command_start_step": zero, "true_bouts": by_state,
            "last_physical_bout": {key: physics[-1][key] for key in ("state","ticks","end")},
            "first": {str(n): first_pair(physics, n, zero, hz) for n in (12,24)},
            "alerts": alarms}


def run(trial):
    import u4_sustained_observer as candidate

    observer = candidate.SustainedSideObserver()

    def compare(sensor, passive):
        result = observer.update(sensor, passive)
        wrench = sensor["left_wrench_world"][:3]
        normal = sensor["cup_normals_world"]["left"]
        return result, {"sustained_applied": result.get("sustained_applied", False),
                        "sustained_score": result.get("sustained_score"),
                        "sustained_reason": result.get("sustained_reason"),
                        "left_compression_N": -sum(a*b for a,b in zip(wrench, normal, strict=True)),
                        "left_tcp_x_m": sensor["tcp_positions_m"]["left"][0],
                        "right_tcp_x_m": sensor["tcp_positions_m"]["right"][0],
                        "right_tcp_y_m": sensor["tcp_positions_m"]["right"][1],
                        "side_gap_m": passive["side_gap_m"],
                        "camera_age_s": passive["camera_age_s"]}

    paired = trial/"paired.jsonl.gz"
    commands = trial/"right_commands.json.gz"
    header, _, _, rows = paired_rows(paired, commands, compare, reference_fields=left_cup_reference)
    files = [Path(candidate.__file__), Path(__file__).with_name("u4_contact_observer.py")]
    if hasattr(candidate, "PARAMETERS"):
        files.append(candidate.PARAMETERS)
    result = {"source": header["source"], "source_sha256": header["source_sha256"],
              "model_sha256": {p.name: digest(p) for p in files},
              "paired_sha256": digest(paired), "command_sha256": digest(commands),
              "last_step": header["stop_step"], "incomplete_run": header.get("incomplete_run"),
              "events": trial_events(rows, header["hz"]),
              "windows": comparison_windows(rows, header["hz"])}
    (trial/"counts.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with gzip.open(trial/"timeline.csv.gz","wt") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: int(v) if type(v) is bool else v for k,v in row.items()})


def summarize(root):
    records = []
    for p in sorted(root.glob("*/counts.json")):
        o = json.loads(p.read_text())
        if o["source"] != p.parent.name or (records and o["model_sha256"] != records[0]["model_sha256"]):
            raise ValueError(f"Inconsistent trial or frozen model {p}")
        records.append(o)
    if not records:
        raise ValueError("No scored trials")
    summary = {"trials": [r["source"] for r in records], "model_sha256": records[0]["model_sha256"],
               "windows": {w: total([r["windows"][w] for r in records if w in r["windows"]])
                           for w in (*WINDOWS, "early_handoff")}}
    events = [r["events"] for r in records if r["events"] is not None]
    summary["true_bouts"] = {key: {d: sum(e["true_bouts"][key].get(d,0) for e in events) for d in DURATIONS}
                             for key in ("clear", "bearing")}
    summary["alerts"] = {}
    for model in MODELS:
        counts = Counter()
        for e in events:
            counts.update(e["alerts"][model])
        summary["alerts"][model] = dict(counts)
    first = {}
    for n in ("12","24"):
        pairs = [e["first"][n] for e in events]
        first[n] = {name: {"targets": sum(p[name] is not None for p in pairs),
                           **{model: sum(p[name] is not None and p[name][model]["confirm_step"] is not None
                                         for p in pairs) for model in MODELS}}
                    for name in ("loss","reload")}
        first[n]["both"] = {
            model: sum(p["reload"] is not None and p["loss"][model]["confirm_step"] is not None
                       and p["reload"][model]["confirm_step"] is not None for p in pairs)
            for model in MODELS
        }
    summary["first"] = first
    (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--summary",action="store_true")
    args = parser.parse_args()
    (summarize if args.summary else run)(args.path)
