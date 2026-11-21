"""Compare a lateral probe with passive fusion on exactly the same measured U4 motion."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from u4_handover_score import events, paired_rows, ratio


MODELS = ("passive", "active")
SEGMENTS = ("pre", "out_neg", "hold_neg", "cross_pos", "hold_pos", "return", "post")
COUNTS = (12, 36, 12, 60, 12, 36, 24)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tally(truth, prediction):
    known = [(t, p) for t, p in zip(truth, prediction, strict=True) if t is not None]
    pos = sum(t for t, _ in known)
    neg = len(known) - pos
    tp = sum(t and p is True for t, p in known)
    fp = sum(not t and p is True for t, p in known)
    unknown_pos = sum(t and p is None for t, p in known)
    unknown_neg = sum(not t and p is None for t, p in known)
    return {"true": pos, "negative": neg, "unknown_truth": len(truth)-len(known), "tp": tp, "fp": fp,
            "unknown_on_positive": unknown_pos, "unknown_on_negative": unknown_neg,
            "recall": ratio(tp, pos), "fpr": ratio(fp, neg), "precision": ratio(tp, tp+fp),
            "coverage": ratio(len(known)-unknown_pos-unknown_neg, len(known))}


def scores(rows, model):
    def is_state(state, label):
        return None if state == "UNKNOWN" else state == label
    states = {s: tally([is_state(r["truth_state"], s) for r in rows],
                       [is_state(r[model+"_state"], s) for r in rows]) for s in ("S5", "S6")}
    side = tally([r["truth_side"] for r in rows], [r[model+"_side"] for r in rows])
    weak = [r for r in rows if r["truth_side"] is True and r["truth_side_N"] is not None
            and .1 <= r["truth_side_N"] < 1.]
    hit = sum(r[model+"_side"] is True for r in weak)
    return {"states": states, "side": side, "weak_side": {
        "true": len(weak), "tp": hit, "recall": ratio(hit, len(weak)),
        "unknown": sum(r[model+"_side"] is None for r in weak)}}


def stable_end(rows):
    last = [r["truth_side"] for r in rows[-3:]]
    return last[-1] if len(last) == 3 and last[-1] is not None and len(set(last)) == 1 else None


def exposure(rows):
    force = [r["truth_side_N"] for r in rows if r["truth_side_N"] is not None]
    return {"ticks": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "known_loaded": sum(r["truth_side"] is True for r in rows),
            "known_unloaded": sum(r["truth_side"] is False for r in rows),
            "unknown": sum(r["truth_side"] is None for r in rows),
            "weak": sum(r["truth_side"] is True and r["truth_side_N"] is not None and
                        .1 <= r["truth_side_N"] < 1. for r in rows),
            "median_side_N": median(force) if force else None, "end_3step": stable_end(rows)}


def episode(rows):
    selected = [r for r in rows if r["segment"]]
    if not selected:
        return {"complete_sequence": False, "status": "NO_PROBE_EXECUTED", "segments": {}}
    start = selected[0]["step"]
    incoming = [r for r in rows if start-12 <= r["step"] < start]
    prior = stable_end(incoming)
    parts = {s: [r for r in selected if r["segment"] == s] for s in SEGMENTS}
    before = stable_end(parts["pre"])
    during = [r for r in selected if r["segment"] not in ("pre", "post")]
    unloaded = events(during, "truth_side", False)["first_3step_confirm"]
    reloaded = (events([r for r in selected if r["step"] > unloaded], "truth_side")["first_3step_confirm"]
                if unloaded is not None else None)
    native = events(during, "truth_side")
    responses = [r for r in selected if r["probe_response"] is not None]
    if len(responses) > 1:
        raise ValueError("Expected a single retrospective probe decision")
    decision = responses[0] if responses else None
    conclusion = decision["probe_before_loaded"] if decision else None
    return {"segments": {s: exposure(rs) for s, rs in parts.items() if rs},
            "before_command_hold_12": exposure(incoming) if incoming else None,
            "before_command_hold_end_loaded": prior,
            "load_changed_during_pre_hold": prior != before if prior is not None and before is not None else None,
            "complete_sequence": [len(parts[s]) for s in SEGMENTS] == list(COUNTS),
            "pre_end_loaded": before, "post_end_loaded": stable_end(parts["post"]),
            "box_range_m": {axis: max(r[k] for r in selected) - min(r[k] for r in selected)
                            for axis, k in (("x","truth_box_x_m"), ("y","truth_box_y_m"))},
            "during_single_tick_load": native["first_step"], "during_load_ticks": native["positive_ticks"],
            "during_load_episodes": native["episodes"], "during_longest_sequence": native["longest_ticks"],
            "first_sustained_onset_during": native["first_3step_confirm"], "first_sustained_unload_during": unloaded,
            "first_reload_after_unload": reloaded,
            "single_tick_contact_during_previously_clear": native["first_step"] is not None if before is False else None,
            "sustained_contact_during_previously_clear": native["first_3step_confirm"] is not None if before is False else None,
            "retrospective_decision_step": decision["step"] if decision else None,
            "estimated_pre_loaded": conclusion, "retrospective_correct": (
                before == conclusion if before is not None and conclusion is not None else None),
            "retrospective_matches_preintervention": (
                prior == conclusion if prior is not None and conclusion is not None else None),
            "probe_response": decision["probe_response"] if decision else None}


def windows(rows, models=MODELS):
    selections = {"full_clip": rows}
    for win in ("handoff", "rear_stroke", "withdraw"):
        selections[win] = [r for r in rows if r["window"] == win]
    selections["handoff_and_rear"] = [r for r in rows if r["window"] in ("handoff", "rear_stroke")]
    pulse = [r for r in rows if r.get("segment")]
    if pulse:
        selections["probe"] = pulse
        post = [r["step"] for r in pulse if r["segment"] == "post"]
        for win in ("handoff", "rear_stroke"):
            selections[win+"_without_pulse"] = [r for r in selections[win] if not r["segment"]]
            if post:
                selections["after_return_"+win] = [r for r in selections[win] if r["step"] >= post[0]]
    return {k: {"ticks": len(part), **{m: scores(part, m) for m in models}}
            for k, part in selections.items() if part}


def archived_control(path, rear_ticks):
    with gzip.open(path/"timeline.csv.gz", "rt") as f:
        rows = [{"window": r["window"], "truth_state": r["truth_state"], "passive_state": r["fusion_state"],
                 "truth_side": {"": None, "1": True, "0": False}[r["truth_side"]],
                 "truth_side_N": float(r["truth_side_N"]) if r["truth_side_N"] else None,
                 "passive_side": {"": None, "1": True, "0": False}[r["fusion_side"]]}
                for r in csv.DictReader(f)]
    result = json.loads((path/"scores.json").read_text())
    rear = [r for r in rows if r["window"] == "rear_stroke"][:rear_ticks]
    return {"source": result["source"], "timeline_sha256": digest(path/"timeline.csv.gz"),
            "observer_sha256": result["observer_sha256"]["u4_contact_observer.py"],
            "common_rear_ticks": len(rear), "common_rear_prefix": scores(rear, "passive"),
            "windows": windows(rows, ("passive",))}


def run(source, command_source, telemetry, output, control=None):
    from u4_side_probe_observer import PulseObserver

    data = [json.loads(line) for line in telemetry.read_text().splitlines()]
    ticks = [d["physics_step"] for d in data]
    if ticks and ticks != list(range(ticks[0], ticks[0]+len(data))):
        raise ValueError("Discontinuous pulse telemetry")
    expected = [s for s, n in zip(SEGMENTS, COUNTS) for _ in range(n)]
    if [d["segment"] for d in data] != expected[:len(data)]:
        raise ValueError("Unknown pulse sequence")
    commands = dict(zip(ticks, data))
    observer = PulseObserver()

    def compare(sensor, passive):
        record = commands.get(sensor["physics_step"])
        pulse = None
        if record:
            if (abs(sensor["time_s"] - record["time_s"]) > 1e-6
                    or abs(sensor["tcp_positions_m"]["right"][0] - record["encoder_x_m"]) > 1e-7
                    or not np.allclose(sensor["right_wrench_world"], record["right_wrench_world"], rtol=0., atol=1e-6)):
                raise ValueError(f"Probe telemetry does not match sensor stream at {sensor['physics_step']}")
            pulse = {k: record[k] for k in ("physics_step", "segment", "command_dx_m")}
        active = observer.update(sensor, passive, pulse)
        return active, {"segment": record["segment"] if record else "",
                        "probe_before_loaded": active.get("probe_before_loaded"),
                        "probe_applied": active["probe_applied"],
                        "probe_response": active.get("probe_response")}

    header, cmd, _, rows = paired_rows(source, command_source, compare)
    if ticks and (ticks[0] < rows[0]["step"] or ticks[-1] > rows[-1]["step"]):
        raise ValueError("Probe outside the paired interval")
    for row in rows:
        for old, new in (("baseline", "passive"), ("fusion", "active")):
            for key in ("state", "objects", "side", "rear"):
                row[new+"_"+key] = row.pop(old+"_"+key)
    phase_markers = {(e["phase"], e["status"]) for e in header["phase_times_for_scoring_only"]}
    if bool(ticks) != (("upper_B_u4_side_probe", "STARTED") in phase_markers):
        raise ValueError("Probe telemetry disagrees with the physical runner's pulse event")
    result = {"source": header["source"], "hz": header["hz"], "source_sha256": header["source_sha256"],
              "incomplete_run": header.get("incomplete_run"), "last_step": header["stop_step"],
              "physical_status": "STOPPED" if header.get("incomplete_run") else "SEATED_STABLE",
              "complete_handoff": ("upper_B_rear","RUNNING") in phase_markers,
              "complete_rear": ("upper_B_rear","FORCE_CONFIRMED") in phase_markers,
              "wrench_sha256": cmd["wrench_archive_sha256"], "probe_sha256": digest(telemetry),
              "observer_sha256": {name: digest(Path(__file__).with_name(name))
                                  for name in ("u4_side_probe_observer.py", "u4_contact_observer.py")},
              "episode": episode(rows), "active_evidence_used": events(rows, "probe_applied"),
              "changed_side_predictions": sum(r["passive_side"] != r["active_side"] for r in rows),
              "changed_state_predictions": sum(r["passive_state"] != r["active_state"] for r in rows),
              "windows": windows(rows)}
    if control:
        rear = [r for r in rows if r["window"] == "rear_stroke"]
        archive = archived_control(control, len(rear))
        if archive["observer_sha256"] != result["observer_sha256"]["u4_contact_observer.py"]:
            raise ValueError("Historical control used a different passive observer")
        result["matched_no_pulse"] = archive
        common = rear[:archive["common_rear_ticks"]]
        result["common_rear_prefix"] = {m: scores(common, m) for m in MODELS}
    output.mkdir(parents=True, exist_ok=True)
    (output/"scores.json").write_text(json.dumps(result, indent=1, allow_nan=False)+"\n")
    with gzip.open(output/"timeline.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: int(v) if isinstance(v, bool) else json.dumps(v) if isinstance(v,dict) else v
                             for k,v in r.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "commands", "telemetry", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--control", type=Path, help="Archived U4 directory with the same initial offset")
    args = parser.parse_args()
    run(args.source, args.commands, args.telemetry, args.output, args.control)
