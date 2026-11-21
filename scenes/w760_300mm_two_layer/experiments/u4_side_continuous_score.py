"""Score continuously updated side bearing on the two synchronized U4 streams."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from u4_handover_score import paired_rows
from u4_left_unload_score import MODELS, evaluate, left_cup_reference
from u4_pulse_score import digest, scores, tally


def declarations(rows, model, n):
    """Yield the first confirmation step for every contiguous, known prediction."""
    history = []
    previous = None
    for row in rows:
        tick, call = row["step"], row[model + "_side"]
        if previous != (tick - 1, call):
            history.clear()
        previous = (tick, call)
        if call is not None and len(history) < n:
            history.append(row["truth_side"])
            if len(history) == n:
                yield tick-n+1, tick, call, history.copy()


def alerts(rows, model, n):
    """Check all steps of each sustained call against present-step PhysX."""
    result = {"clear": Counter(), "bearing": Counter()}
    for _, _, call, history in declarations(rows, model, n):
        outcome = ("incorrect" if (not call) in history else
                   "unverifiable" if None in history else "correct")
        result["bearing" if call else "clear"][outcome] += 1
    return {target: {key: counts[key] for key in ("correct", "incorrect", "unverifiable")}
            for target, counts in result.items()}


def event_matches(rows, events):
    if events is None:
        return None
    start = events["unload_command_start_step"]
    exchange = [r for r in rows if r["window"] in ("handoff", "rear_stroke")]
    traces = [r for r in rows if start-120 < r["step"] <= exchange[-1]["step"]]
    for name in ("from_unload_command_3", "from_unload_command_12"):
        event = events[name]
        n = event["minimum_consecutive_steps"]
        clear_start = event["true_clear_onset_step"]
        clear_end = event["clear_bout_last_observed_step"]
        reload_confirm = event["true_reload_confirm_step"]
        reload_start = reload_end = None
        if reload_confirm is not None:
            reload_start = reload_confirm-n+1
            reload_end = next((r["step"]-1 for r in exchange if r["step"] > reload_confirm
                               and r["truth_side"] is not True), exchange[-1]["step"])
        for model in MODELS:
            calls = list(declarations(traces, model, n))
            for key, positive, onset, end in (("clear", False, clear_start, clear_end),
                                               ("reload", True, reload_start, reload_end)):
                hit = next((last for first,last,label,_ in calls
                            if label is positive and onset is not None and onset <= first <= last <= end), None)
                event[model][f"new_{key}_declaration_step"] = hit
    return events


def comparison_windows(rows, hz):
    handoff = [r for r in rows if r["window"] == "handoff"]
    rear = [r for r in rows if r["window"] == "rear_stroke"]
    windows = {}
    for name, region in (("handoff", handoff), ("rear_stroke", rear),
                         ("early_handoff", handoff[:2*hz])):
        if not region:
            continue
        windows[name] = {
            model: {**scores(region, model),
                    "clear": tally([None if r["truth_side"] is None else not r["truth_side"] for r in region],
                                   [None if r[model+"_side"] is None else not r[model+"_side"] for r in region]),
                    "alerts": alerts(region, model, 3)}
            for model in MODELS
        }
    return windows


SENSITIVITY = (.5, 0.)


def threshold_rows(rows, model, width):
    middle = (model.clear_below + model.load_above) / 2
    lower = middle + width * (model.clear_below - middle)
    upper = middle + width * (model.load_above - middle)
    for row in rows:
        side, state = row["baseline_side"], row["baseline_state"]
        score = row["continuous_side_score"]
        if side is None and score is not None:
            side = False if score < lower else True if score > upper else None
            if side is not None:
                rear = row["baseline_rear"]
                state = ("S1", "S5", "S6")[int(side)+int(rear)] if rear is not None else "UNKNOWN"
        yield {**row, "fusion_side": side, "fusion_state": state}


def run(trial):
    from u4_side_continuous import ContinuousSideObserver, PARAMETERS

    observer = ContinuousSideObserver()

    def compare(sensor, passive):
        result = observer.update(sensor, passive)
        left = sensor["left_wrench_world"][:3]
        normal = sensor["cup_normals_world"]["left"]
        return result, {
            "left_compression_N": -sum(x*y for x,y in zip(left, normal, strict=True)),
            "right_shear_x_N": sensor["right_wrench_world"][0],
            "right_compression_y_N": -sensor["right_wrench_world"][1],
            "side_gap_m": passive["side_gap_m"],
            "camera_age_s": passive["camera_age_s"],
            "unload_event": False,
            "continuous_side_applied": result.get("continuous_side_applied", False),
            "continuous_side_score": result.get("continuous_side_score"),
            "continuous_side_reason": result.get("continuous_side_reason"),
        }

    paired = trial/"paired.jsonl.gz"
    commands = trial/"right_commands.json.gz"
    header, _, _, rows = paired_rows(paired, commands, compare, reference_fields=left_cup_reference)
    result = {
        "source": header["source"],
        "source_sha256": header["source_sha256"],
        "observer_sha256": {name: digest(Path(__file__).with_name(name))
                            for name in ("u4_contact_observer.py", "u4_side_continuous.py")},
        "calibration_sha256": digest(PARAMETERS),
        "paired_sha256": digest(paired),
        "command_sha256": digest(commands),
        "last_step": header["stop_step"],
        "incomplete_run": header.get("incomplete_run"),
        "events": event_matches(rows, evaluate(rows, header["hz"])["events"]),
        "windows": comparison_windows(rows, header["hz"]),
        "changed_side_steps": sum(r["baseline_side"] != r["fusion_side"] for r in rows),
    }
    result["threshold_sensitivity"] = {}
    for width in SENSITIVITY:
        alternative = list(threshold_rows(rows, observer, width))
        result["threshold_sensitivity"][str(width)] = {
            "events": event_matches(alternative, evaluate(alternative, header["hz"])["events"]),
            "windows": comparison_windows(alternative, header["hz"]),
        }
    (trial/"counts.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with gzip.open(trial/"timeline.csv.gz", "wt") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: int(v) if isinstance(v, bool) else v for k,v in row.items()})


def summarize(root):
    reports = []
    for path in sorted(root.glob("*/counts.json")):
        r = json.loads(path.read_text())
        if r["source"] != path.parent.name or (reports and r["observer_sha256"] != reports[0]["observer_sha256"]):
            raise ValueError(f"Inconsistent report {path}")
        reports.append(r)
    if not reports:
        raise ValueError("No trials to summarize")
    if len({r["calibration_sha256"] for r in reports}) != 1:
        raise ValueError("Trials used different calibrations")
    summary = {"trials": [r["source"] for r in reports], **summary_metrics(reports),
               "calibration_sha256": reports[0]["calibration_sha256"]}
    summary["threshold_sensitivity"] = {
        str(width): summary_metrics([r["threshold_sensitivity"][str(width)] for r in reports])
        for width in SENSITIVITY
    }
    (root/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


def summary_metrics(reports):
    summary = {"windows": {}, "events": {}}
    for window in ("handoff", "early_handoff", "rear_stroke"):
        all_scores = [r["windows"][window] for r in reports if window in r["windows"]]
        summary["windows"][window] = total(all_scores)
    for duration in ("from_unload_command_3", "from_unload_command_12"):
        events = [r["events"][duration] for r in reports if r["events"]]
        by_model = {}
        for model in MODELS:
            clear = [e for e in events if e["true_clear_confirm_step"] is not None and not e["initial_loss_unwitnessed"]]
            reloads = [e for e in clear if e["true_reload_confirm_step"] is not None]
            hit_clear = sum(e[model]["clear_confirm_within_first_true_bout_step"] is not None for e in clear)
            hit_reload = sum(e[model]["bearing_confirm_within_first_true_reload_bout_step"] is not None
                             for e in reloads)
            hit_both = sum(e[model]["clear_confirm_within_first_true_bout_step"] is not None and
                           e[model]["bearing_confirm_within_first_true_reload_bout_step"] is not None
                           for e in reloads)
            strict_clear = sum(e[model]["new_clear_declaration_step"] is not None for e in clear)
            strict_reload = sum(e[model]["new_reload_declaration_step"] is not None for e in reloads)
            strict_both = sum(e[model]["new_clear_declaration_step"] is not None and
                              e[model]["new_reload_declaration_step"] is not None for e in reloads)
            by_model[model] = {"clear_targets": len(clear), "clear_hits": hit_clear,
                               "reload_targets": len(reloads), "reload_hits": hit_reload,
                               "both_hits": hit_both, "new_clear_alerts": strict_clear,
                               "new_reload_alerts": strict_reload, "both_new_alerts": strict_both}
        summary["events"][duration] = by_model
    return summary


def total(all_scores):
    out = {"trials": len(all_scores)}
    for model in MODELS:
        groups = [r[model] for r in all_scores]
        out[model] = {}
        for key in ("weak_side", "side", "clear", "S5", "S6"):
            cells = [r["states"][key] if key in ("S5", "S6") else r[key] for r in groups]
            c = {n: sum(x.get(n, 0) for x in cells) for n in ("tp", "true", "fp", "negative")}
            # Report weak recall with its weak positives and the same known-clear false-alarm set.
            if key == "weak_side":
                c.update({n: sum(x["side"][n] for x in groups) for n in ("fp", "negative")})
            out[model][key] = {**c, "recall": c["tp"]/c["true"] if c["true"] else None,
                               "fpr": c["fp"]/c["negative"] if c["negative"] else None}
        out[model]["alerts"] = {
            target: {k: sum(x["alerts"][target][k] for x in groups)
                     for k in ("correct", "incorrect", "unverifiable")}
            for target in ("clear", "bearing")
        }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    (summarize if args.summary else run)(args.path)
