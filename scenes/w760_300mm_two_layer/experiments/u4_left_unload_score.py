"""Compare U4 passive and left-unload fusion against the same causal physical timeline."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from u4_handover_score import ACTIVE, MIN_IMPULSE_NS, paired_rows
from u4_pulse_score import digest, scores
from u4_pulse_summary import comparison


MODELS = ("baseline", "fusion")
WINDOWS = ("handoff", "rear_stroke", "handoff_and_rear", "after_left_cup_clear",
           "first_second_after_left_cup_clear")


def left_cup_reference(physical, hz):
    snapshot = physical["cube_contacts"]
    cups = [p for p in snapshot["pairs"] if p["label"] == "left_cup" and
            ACTIVE in (p["collider0"], p["collider1"])]
    measured = [p for p in cups if p["current_step_observed"] and p["normal_impulse_Ns"] is not None]
    complete = snapshot["valid"] and len(measured) == len(cups)
    positive = snapshot["valid"] and any(p["normal_impulse_Ns"] > MIN_IMPULSE_NS for p in measured)
    total_x = None
    if complete and all(p.get("normal_axis_impulse_abs_Ns") is not None for p in cups):
        total_x = hz * sum(p["normal_axis_impulse_abs_Ns"][0] for p in cups)
    return {"truth_left_cup": True if positive else False if complete else None,
            "truth_left_cup_x_N": total_x}


def first_streak(rows, key, value, n):
    streak = 0
    previous = None
    for row in rows:
        if row["step"] != previous:
            streak = 0
        streak = streak + 1 if row[key] is value else 0
        if streak == n:
            return row["step"]
        previous = row["step"] + 1
    return None


def event_timing(rows, origin, hz, n, prior_side=()):
    """Measure the first observed clear bout and subsequent sustained return of side load."""
    truth = first_streak(rows, "truth_side", False, n)
    onset = truth - n + 1 if truth is not None else None
    bout = []
    if onset is not None:
        for r in rows:
            if r["step"] < onset:
                continue
            if r["truth_side"] is not False:
                break
            bout.append(r)
    last = next((r["step"] for r in reversed([*prior_side, *rows]) if
                 onset is not None and r["step"] < onset and r["truth_side"] is True), None)
    reload = first_streak([r for r in rows if truth is not None and r["step"] > truth],
                          "truth_side", True, n)
    def seconds(step):
        return (step - origin) / hz if step is not None else None
    censored = onset is not None and (last is None or
                  onset == rows[0]["step"] and last != onset - 1)
    out = {"minimum_consecutive_steps": n, "last_observed_bearing_before_clear_step": last,
           "true_clear_onset_step": onset, "true_clear_onset_s": seconds(onset),
           "true_clear_confirm_step": truth, "true_clear_confirm_s": seconds(truth),
           "clear_bout_last_observed_step": bout[-1]["step"] if bout else None,
           "initial_loss_unwitnessed": censored, "true_reload_confirm_step": reload,
           "true_reload_confirm_s": seconds(reload)}
    reload_bout = []
    if reload is not None:
        for r in rows:
            if r["step"] < reload - n + 1:
                continue
            if r["truth_side"] is not True:
                break
            reload_bout.append(r)
    for model in MODELS:
        key = model + "_side"
        first = first_streak(rows, key, False, n)
        inside = first_streak(bout, key, False, n)
        prediction_return = first_streak([r for r in rows if first is not None and r["step"] > first],
                                         key, True, n)
        out[model] = {"first_clear_confirm_step": first, "first_clear_confirm_s": seconds(first),
                      "first_declaration_minus_true_confirm_s": (
                          (first - truth) / hz if first is not None and truth is not None and not censored else None),
                      "early_declaration": first < truth if first is not None and truth is not None else None,
                      "clear_confirm_within_first_true_bout_step": inside,
                      "missed_first_clear_bout": inside is None if truth is not None else None,
                      "matched_clear_delay_from_true_onset_s": (
                          (inside - onset) / hz if inside is not None and not censored else None),
                      "predicted_reload_after_predicted_clear_step": prediction_return,
                      "bearing_confirm_within_first_true_reload_bout_step": first_streak(reload_bout, key, True, n)}
    return out


def evaluate(rows, hz):
    exchange = [r for r in rows if r["window"] in ("handoff", "rear_stroke")]
    handoff = [r for r in exchange if r["window"] == "handoff"]
    if not handoff:
        return {"windows": {}, "events": None}
    origin = handoff[0]["step"] - 1
    prior = [r for r in rows if origin - hz < r["step"] <= origin]
    tool = first_streak(exchange, "truth_left_cup", False, 3)
    after = [r for r in exchange if tool is not None and r["step"] >= tool]
    windows = {"handoff": handoff, "rear_stroke": [r for r in exchange if r["window"] == "rear_stroke"],
               "handoff_and_rear": exchange, "after_left_cup_clear": after,
               "first_second_after_left_cup_clear": [r for r in after if r["step"] < tool + hz]}
    metrics = {}
    for window, selected in windows.items():
        if selected:
            metrics[window] = {"ticks": len(selected), "start": selected[0]["step"], "end": selected[-1]["step"],
                               "bearing_below_point_one_N": sum(r["truth_side"] is True and
                                   r["truth_side_N"] is not None and r["truth_side_N"] < .1 for r in selected),
                               **{m: scores(selected, m) for m in MODELS}}
    tool_prior = next((r["step"] for r in reversed([*prior, *exchange])
                       if tool is not None and r["step"] < tool-2 and r["truth_left_cup"] is True), None)
    return {"windows": metrics, "events": {"unload_command_start_step": origin,
            "first_observer_unload_event_step": next((r["step"] for r in exchange if r["unload_event"]), None),
            "last_left_cup_bearing_before_unload_step": tool_prior,
            "left_cup_clear_confirm_step": tool, "left_cup_clear_confirm_s": (tool-origin)/hz if tool else None,
            "from_unload_command_3": event_timing(exchange, origin, hz, 3, prior),
            "from_unload_command_12": event_timing(exchange, origin, hz, 12, prior),
            "after_left_cup_clear_3": (event_timing(after, tool, hz, 3,
                   [r for r in [*prior, *exchange] if r["step"] < tool]) if after else None)}}


def run(trial):
    from u4_left_unload_observer import LeftUnloadObserver
    observer = LeftUnloadObserver()

    def compare(sensor, passive):
        result = observer.update(sensor, passive)
        left = sensor["left_wrench_world"][:3]
        normal = sensor["cup_normals_world"]["left"]
        return result, {"left_compression_N": -sum(x*y for x,y in zip(left, normal, strict=True)),
                        "left_tcp_x_m": sensor["tcp_positions_m"]["left"][0],
                        "side_gap_m": passive["side_gap_m"], "camera_age_s": passive["camera_age_s"],
                        "unload_stage": result.get("unload_stage"), "unload_event": result["unload_event"],
                        "unload_applied": result["unload_applied"], "unload_evidence": result.get("unload_evidence")}

    header, cmd, _, rows = paired_rows(trial/"paired.jsonl.gz", trial/"right_commands.json.gz", compare,
                                       reference_fields=left_cup_reference)
    result = {"source": header["source"], "source_sha256": header["source_sha256"], "hz": header["hz"],
              "last_step": header["stop_step"], "incomplete_run": header.get("incomplete_run"),
              "wrench_sha256": cmd["wrench_archive_sha256"],
              "paired_sha256": digest(trial/"paired.jsonl.gz"),
              "command_sha256": digest(trial/"right_commands.json.gz"),
              "calibration_sha256": digest(Path(__file__).parent/"results/u4_left_unload_calibration.json"),
              "observer_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                    ("u4_contact_observer.py", "u4_left_unload_observer.py")},
              "changed_state_steps": sum(r["baseline_state"] != r["fusion_state"] for r in rows),
              "changed_side_steps": sum(r["baseline_side"] != r["fusion_side"] for r in rows),
              "unload_events": [r["step"] for r in rows if r["unload_event"]],
              "event_observations": [{k: r[k] for k in ("step", "unload_evidence", "truth_left_cup",
                                        "truth_side", "truth_side_N", "baseline_side", "fusion_side", "unload_applied")}
                                     for r in rows if r["unload_event"]],
              **evaluate(rows, header["hz"])}
    (trial/"counts.json").write_text(json.dumps(result, indent=1, allow_nan=False)+"\n")
    with gzip.open(trial/"timeline.csv.gz", "wt") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: int(v) if isinstance(v, bool) else json.dumps(v, allow_nan=False)
                             if isinstance(v, dict) else v for k, v in row.items()})


def aggregate(root):
    runs = []
    for path in sorted(root.glob("*/counts.json")):
        record = json.loads(path.read_text())
        if record["source"] != path.parent.name:
            raise ValueError(f"Score source disagrees with trial directory: {path}")
        if runs and record["observer_sha256"] != runs[0]["observer_sha256"]:
            raise ValueError("Trials used different observers")
        runs.append(record)
    if not runs:
        raise ValueError("No scored runs")
    output = {"trials": [r["source"] for r in runs], "windows": {}}
    for win in WINDOWS:
        data = [r["windows"][win] for r in runs if win in r["windows"]]
        output["windows"][win] = {"runs": len(data), "trials": [r["source"] for r in runs if win in r["windows"]],
                                  **{("weak_side" if metric == "handoff_weak" else metric):
                                      comparison([d["baseline"] for d in data], [d["fusion"] for d in data],
                                      metric, MODELS) for metric in ("handoff_weak", "S5", "S6")}}
    (root/"summary.json").write_text(json.dumps(output, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--summary", action="store_true", help="Pool observed windows across scored subdirectories")
    args = parser.parse_args()
    (aggregate if args.summary else run)(args.path)
