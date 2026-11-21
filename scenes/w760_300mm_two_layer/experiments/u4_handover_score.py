"""Pair causal U4 predictions with present-step PhysX, then score the arm exchange."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

from seven_observer import SevenObserver
from seven_reference import MIN_IMPULSE_NS, SevenReference
from seven_replay import BACK, UNKNOWN


SIDE = "/World/Cube_8"
ACTIVE = "/World/Cube_10"
LABELS = ("S5", "S6")
AXES = ("side", "rear")
WINDOWS = ("side_stroke", "handoff", "rear_stroke", "withdraw")


def anchors(header):
    events = {(e["phase"], e["status"]): round(e["sim_time_s"] * header["hz"])
              for e in header["phase_times_for_scoring_only"]}
    points = []
    for key in (("upper_B_side", "RUNNING"), ("upper_B_side_unload", "RUNNING"),
                ("upper_B_rear", "RUNNING"), ("upper_B_rear", "FORCE_CONFIRMED"),
                ("upper_slot_complete", "SEATED_STABLE")):
        if key not in events and header.get("incomplete_run"):
            if not points or header["stop_step"] > points[-1]:
                points.append(header["stop_step"])
            break
        points.append(events[key])
    if any(a >= b for a, b in zip(points, points[1:])):
        raise ValueError("Missing or misordered U4 exchange")
    return points


def load_at(snapshot, physical_label, target, axis, hz):
    """Return a ternary axis-bearing label and the projected normal force when available."""
    if not snapshot["valid"]:
        return None, None
    match = [p for p in snapshot["pairs"] if {p["collider0"], p["collider1"]} == {ACTIVE, target}]
    if not match:
        return False, 0.
    p, = match
    value = p.get("normal_impulse_Ns")
    if not p["current_step_observed"] or value is None:
        return None, None
    components = p.get("normal_axis_impulse_abs_Ns")
    force = components[axis] * hz if components is not None else None
    if value <= MIN_IMPULSE_NS:
        return False, force
    if target in physical_label["ambiguous_contacts"]:
        return None, force
    horizontal = target in physical_label["horizontal_objects"]
    if components is not None:
        horizontal = horizontal and components[axis] >= .9 * max(components)
    return horizontal, force


def predictions(result):
    # A seven-state baseline only names external objects for loaded-boundary states.
    # S0 says the tool is clear; a bearing remaining between boxes is unobserved.
    return [result.get(axis+"_loaded", True if target in result["boundaries"] else
                       None if result["state"] in (UNKNOWN, "S0") else False)
            for axis, target in zip(AXES, (SIDE, BACK))]


def events(rows, key, expected=True):
    """Report a single-tick onset and the first online confirmation after three consecutive ticks."""
    first = confirmed = last = prev = None
    streak = episodes = longest = positive = 0
    for row in rows:
        tick = row["step"]
        if row[key] == expected:
            if prev != tick - 1:
                streak = 0
            streak += 1
            positive += 1
            longest = max(longest, streak)
            if streak == 1:
                episodes += 1
            if first is None:
                first = tick
            if streak == 3 and confirmed is None:
                confirmed = tick
            last = tick
        else:
            streak = 0
        prev = tick
    return {"first_step": first, "first_3step_confirm": confirmed,
            "last_step": last, "positive_ticks": positive, "episodes": episodes, "longest_ticks": longest}


def ratio(n, d):
    return n / d if d else None


def object_scores(rows, model, axis, hz):
    tk, pk = "truth_"+axis, model+"_"+axis
    counts = Counter((r[tk], r[pk]) for r in rows)
    hit, fp = counts[True, True], counts[False, True]
    pos, known = sum(n for (t, p), n in counts.items() if t is True), sum(n for (t, p), n in counts.items() if t is not None)
    truth, predicted = events(rows, tk), events(rows, pk)
    first, stable, received = truth["first_step"], truth["first_3step_confirm"], predicted["first_3step_confirm"]
    # Distinguish false positives before any native load from predictions kept during unload/gaps.
    early, retained = [], []
    saw_load = False
    for row in rows:
        step = row["step"]
        false_positive = row[tk] is False and row[pk] is True
        early.append({"step": step, "positive": false_positive and not saw_load})
        retained.append({"step": step, "positive": false_positive and saw_load})
        saw_load |= row[tk] is True
    delay = lambda a, b: (a - b) / hz if a is not None and b is not None else None
    return {"known_truth": known, "truth_pos": pos, "truth_neg": known-pos, "truth_unknown": len(rows)-known,
            "tp": hit, "fp": fp, "tn": counts[False, False], "negative_on_positive": counts[True, False],
            "abstain_on_positive": counts[True, None], "abstain_on_negative": counts[False, None],
            "precision": ratio(hit, hit+fp), "recall_including_abstentions": ratio(hit, pos),
            "coverage_on_known": ratio(known-counts[True, None]-counts[False, None], known),
            "truth_events": truth, "predicted_events": predicted,
            "false_positives_before_first_window_load": events(early, "positive"),
            "false_positives_after_observed_window_load": events(retained, "positive"),
            "first_positive_minus_first_bearing_s": delay(predicted["first_step"], first),
            "confirmation_minus_first_bearing_s": delay(received, first),
            "confirmation_minus_3step_truth_s": delay(received, stable)}


def state_scores(rows, model):
    counts = Counter((r["truth_state"], r[model+"_state"]) for r in rows)
    known = sum(v for (t, p), v in counts.items() if t != UNKNOWN)
    metrics = {}
    for label in LABELS:
        actual = sum(v for (t, p), v in counts.items() if t == label)
        called = sum(v for (t, p), v in counts.items() if t != UNKNOWN and p == label)
        hit = counts[label, label]
        hits_with_objects = sum(r["truth_state"] == r[model+"_state"] == label
                                and r["truth_objects"] == r[model+"_objects"] for r in rows)
        metrics[label] = {"true": actual, "predicted_on_known": called, "tp": hit,
                          "tp_exact_objects": hits_with_objects, "unknown_on_true": counts[label, UNKNOWN],
                          "precision": ratio(hit, called), "recall": ratio(hit, actual)}
    return {"totals": metrics, "known_truth": known,
            "coverage_on_known": ratio(known - sum(v for (t, p), v in counts.items()
                                                   if t != UNKNOWN and p == UNKNOWN), known),
            "confusion": {t: {p: n for (truth, p), n in counts.items() if truth == t}
                          for t in sorted(set(r["truth_state"] for r in rows))}}


def summarize(rows, hz):
    out = {}
    for name, selected in [("full_clip", rows),
                           *[(win, [r for r in rows if r["window"] == win]) for win in WINDOWS]]:
        if not selected:
            raise ValueError(f"Empty {name}")
        out[name] = {"steps": [selected[0]["step"], selected[-1]["step"]], "ticks": len(selected)}
        for model in ("baseline", "fusion"):
            out[name][model] = {"states": state_scores(selected, model),
                                **{k: object_scores(selected, model, k, hz) for k in AXES}}
        side = [r for r in selected if r["truth_side"] is True and r["truth_side_N"] is not None]
        bins = {}
        for low, high in ((0., .1), (.1, 1.), (1., float("inf"))):
            subset = [r for r in side if low <= r["truth_side_N"] < high]
            bins[f"[{low:g},{high:g})"] = {
                "ticks": len(subset),
                **{m: {"hit": sum(r[m+"_side"] is True for r in subset),
                       "unknown": sum(r[m+"_side"] is None for r in subset)} for m in ("baseline", "fusion")}}
        out[name]["side_positive_load_bins_N"] = bins
    return out


def reacquisition(rows, hz):
    # First allow the side bearing to drop for three known steps after unload starts.
    # This cutoff only selects a scoring window. Both observers have run on the entire clip.
    exchange = [r for r in rows if r["window"] in ("handoff", "rear_stroke")]
    unloaded = events(exchange, "truth_side", False)["first_3step_confirm"]
    if unloaded is None:
        return {"three_step_unloaded": None}
    after = [r for r in exchange if r["step"] > unloaded]
    onset = events(after, "truth_side")["first_step"]
    out = {"three_step_unloaded": unloaded, "first_subsequent_load": onset}
    for model in ("baseline", "fusion"):
        out[model] = object_scores(after, model, "side", hz)
        if onset is not None:
            out[model]["negative_seen_before_reload"] = any(
                r[model+"_side"] is False and unloaded <= r["step"] < onset for r in exchange)
    return out


def paired_rows(source, command_source, compare_pulse=None):
    from u4_contact_observer import U4ContactObserver

    with gzip.open(command_source, "rt") as commands:
        cmd = json.load(commands)
    with gzip.open(source, "rt") as stream:
        header = json.loads(next(stream))
        if header["case"] != "u4" or header["cube_id"] != 10:
            raise ValueError("Expected the fourth upper box, Cube_10")
        if (header["source"] != cmd["source"] or header["stop_step"] != cmd["last_step"]
                or header["source_sha256"]["contact_states/sensor_inputs.jsonl.gz"] != cmd["sensor_inputs_sha256"]
                or len(cmd["target_y_m"]) != cmd["last_step"] - cmd["first_step"] + 1):
            raise ValueError("Command sidecar does not match the source and step range")
        hz, points = header["hz"], anchors(header)
        truth = SevenReference(header["config"], hz, header["reference_positions"])
        baseline = SevenObserver(header["config"], "fusion_compensated") if compare_pulse is None else None
        fusion = U4ContactObserver(header["config"])
        rows = []
        for line in stream:
            sensor, physical = json.loads(line)
            tick = sensor["physics_step"]
            if tick != physical["physics_step"] or tick != cmd["first_step"] + len(rows):
                raise ValueError(f"Missing or mismatched tick {tick}")
            sensor_with_command = dict(sensor, right_command_y_m=cmd["target_y_m"][tick - cmd["first_step"]])
            fusion_result = fusion.update(sensor_with_command, neighbors=header["optical_neighbors"])
            extra = {}
            if compare_pulse is None:
                baseline_result = baseline.update(sensor, neighbors=header["optical_neighbors"])
            else:
                baseline_result = fusion_result
                fusion_result, extra = compare_pulse(sensor_with_command, baseline_result)
            ref = truth.update(physical)
            win = next((w for w, a, b in zip(WINDOWS, points, points[1:]) if a < tick <= b), "other")
            row = {"step": tick, "time_s": sensor["time_s"], "window": win, "truth_state": ref["state"],
                   "truth_objects": "|".join(sorted(ref["horizontal_objects"]))}
            if compare_pulse is not None:
                row["truth_box_x_m"], row["truth_box_y_m"] = physical["cube_position_m"][:2]
            for axis, target, i in zip(AXES, (SIDE, BACK), (0, 1)):
                row["truth_"+axis], row["truth_"+axis+"_N"] = load_at(physical["cube_contacts"], ref, target, i, hz)
            for m, result in (("baseline", baseline_result), ("fusion", fusion_result)):
                row[m+"_state"] = result["state"]
                row[m+"_objects"] = "|".join(sorted(result["boundaries"]))
                row[m+"_side"], row[m+"_rear"] = predictions(result)
            for arm in ("left", "right"):
                for i, k in enumerate(("fx", "fy")):
                    row[arm+"_"+k+"_N"] = sensor[arm+"_wrench_world"][i]
            rows.append({**row, **extra})
    if not rows or rows[-1]["step"] != points[-1]:
        raise ValueError("U4 stop step not reached")
    return header, cmd, points, rows


def run(source, command_source, output):
    header, cmd, points, rows = paired_rows(source, command_source)
    hz = header["hz"]
    output.mkdir(parents=True, exist_ok=True)
    scores = {"source": header["source"], "hz": hz, "boundaries": [SIDE, BACK], "source_sha256": header["source_sha256"],
              "observer_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("u4_contact_observer.py", "seven_observer.py", "seven_reference.py")},
              "source_wrench_sha256": cmd["wrench_archive_sha256"], "robot_sha256": cmd["robot_sha256"],
              "phase_steps_for_scoring_only": dict(zip((*WINDOWS, "complete"), points)),
              "rear_first_step_by_force_N": {
                  str(level): next((r["step"] for r in rows if r["window"] == "rear_stroke" and
                                    r["truth_rear"] is True and r["truth_rear_N"] is not None and
                                    r["truth_rear_N"] >= level), None) for level in (.1, 1.)},
              "windows": summarize(rows, hz), "side_reacquisition": reacquisition(rows, hz)}
    with gzip.open(output / "timeline.csv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: int(v) if isinstance(v, bool) else v for k, v in r.items()})
    (output / "scores.json").write_text(json.dumps(scores, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_clip", type=Path)
    parser.add_argument("right_commands", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.paired_clip, args.right_commands, args.output)
