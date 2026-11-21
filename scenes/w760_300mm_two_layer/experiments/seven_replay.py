"""Replay the paired streams without feedback to the packing controller."""
import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from seven_observer import MODES, SevenObserver
from seven_reference import SevenReference


STATES = [f"S{i}" for i in range(7)]
UNKNOWN = "UNKNOWN"
BACK = "/World/Environment/wall_back"


def metrics(rows, hz):
    matrix = {s: Counter() for s in STATES}
    ref_counts = Counter(r["ref"] for r in rows)
    exact_objects = []
    for r in rows:
        if r["ref"] in matrix:
            matrix[r["ref"]][r["pred"]] += 1
        if r["ref"] in ("S5", "S6"):
            exact_objects.append(bool(r["pred"] == r["ref"] and r["reference_objects"] == r["boundaries"]))
    total = sum(ref_counts[s] for s in STATES)
    correct = sum(matrix[s][s] for s in STATES)
    unknown = sum(counts[UNKNOWN] for counts in matrix.values())
    recall = {s: matrix[s][s] / ref_counts[s] if ref_counts[s] else None for s in STATES}
    # Each stable truth switch gets one opportunity before the reference changes again.
    switches = []
    for state, group in itertools.groupby(enumerate(rows), key=lambda entry: entry[1]["ref"]):
        group = list(group)
        i, first = group[0]
        if (i == 0 or state not in STATES or rows[i-1]["ref"] not in STATES or len(group) < 3
                or rows[i-1]["step"] + 1 != first["step"]):
            continue
        match = next((r for _, r in group if r["pred"] == state), None)
        already = match is not None and match is first and rows[i-1]["pred"] == state
        switches.append({"to": state, "step": first["step"], "samples": len(group),
                         "outcome": "early" if already else "detected" if match else "miss",
                         "delay_ms": round((match["time_s"] - first["time_s"]) * 1000, 2)
                         if match is not None and not already else None})
    lags = [s["delay_ms"] for s in switches if s["delay_ms"] is not None]
    return {"ticks": len(rows), "reference_counts": dict(ref_counts), "reference_known": total,
            "correct": correct, "prediction_unknown_on_known": unknown,
            "unknown_predictions_all_ticks": sum(r["pred"] == UNKNOWN for r in rows),
            "coverage_on_known": (total-unknown)/total if total else None,
            "accuracy_including_abstentions": correct/total if total else None,
            "confusion": {s: dict(c) for s, c in matrix.items()}, "recall": recall,
            "exact_horizontal_objects_and_class": [sum(exact_objects), len(exact_objects)],
            "switches": switches, "switch_counts": dict(Counter(x["outcome"] for x in switches)),
            "median_detected_delay_ms": float(np.median(lags)) if lags else None}


def target_events(header, rows):
    result = {}
    for phase, path in (("upper_B_side", "/World/Cube_8"), ("upper_B_rear", BACK)):
        events = [e for e in header["phase_times_for_scoring_only"] if e["phase"] == phase]
        start = next((e for e in events if e["status"] == "RUNNING"), None)
        end = next((e for e in events if e["status"] == "FORCE_CONFIRMED"), None)
        if not start or not end or (phase == "upper_B_side" and header["case"] != "u4"):
            continue
        hz = header["hz"]
        stroke = [r for r in rows if round(start["sim_time_s"]*hz) < r["step"] <= round(end["sim_time_s"]*hz)]
        if not stroke:
            continue
        first_true = next((r["step"] for r in stroke if path in r["reference_objects"]
                           and r["ref"] in ("S5", "S6")), None)
        # A confirmation requires three consecutive predictions of this particular boundary.
        count = 0
        confirmations = []
        for r in stroke:
            count = count + 1 if path in r["boundaries"] and r["pred"] in ("S5", "S6") else 0
            if count == 3:
                confirmations.append((r["step"], r["ref"], path in r["reference_objects"]))
        result[phase] = {"target": path, "first_reference_step": first_true, "confirmed_episodes": len(confirmations),
                         "first_predicted_step": confirmations[0][0] if confirmations else None,
                         "early_episodes": sum(step < first_true and state in STATES and not touched
                                               for step, state, touched in confirmations) if first_true else None,
                         "false_episodes_on_known_truth": sum(state in STATES and not touched
                                                              for _, state, touched in confirmations)}
    return result


def plot(rows, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    names = ["PhysX reference", "Vision + FK", "Wrist + FK", "Fusion, arrival", "Fusion, compensated"]
    labels = STATES + [UNKNOWN]
    palette = ["#a6cee3", "#3f88c5", "#106b40", "#e67e22", "#7b48a6", "#cf3156", "#65432c", "#dddfe3"]
    t = [r["time_s"] for r in rows[MODES[0]]]
    bands = [[labels.index(r["ref"]) for r in rows[MODES[0]]]]
    bands.extend([labels.index(r["pred"]) for r in rows[m]] for m in MODES)
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    ax.imshow(bands, cmap=ListedColormap(palette), norm=BoundaryNorm(np.arange(9)-.5, 8),
              extent=[t[0], t[-1], 4.5, -.5], interpolation="nearest", aspect="auto", origin="upper")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Simulation time (s)")
    ax.set_title(title)
    ax.legend(handles=[Patch(color=c, label=s) for s, c in zip(labels, palette)],
              loc="upper center", bbox_to_anchor=(.5, -.28), ncol=8, frameon=False)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def replay(source, output):
    with gzip.open(source, "rt") as file:
        header = json.loads(next(file))
        samples = [json.loads(line) for line in file]
    cfg, hz, layout = header["config"], header["hz"], header["optical_neighbors"]
    labeler = SevenReference(cfg, hz=hz, object_positions=header["reference_positions"])
    reference = [labeler.update(physical) for _, physical in samples]
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    for image in (("rgb", "blur") if header["has_blur"] else ("rgb",)):
        for delay in (0, 100, 200):
            tracks = {}
            for mode in MODES:
                key = f"{image}_{delay}_{mode}"
                if mode == "wrench_fk" and (image != "rgb" or delay):
                    scores[key] = {"identical_to": "rgb_0_wrench_fk"}
                    continue
                observer = SevenObserver(cfg, mode, delay_ms=delay)
                trace = []
                for (sensor, physical), truth in zip(samples, reference):
                    packet = sensor.get("blur" if image == "blur" else "camera")
                    inputs = {k: v for k, v in sensor.items() if k not in ("blur", "camera")}
                    inputs["camera"] = packet
                    result = observer.update(inputs, neighbors=layout)
                    xyz = result.get("position_m")
                    error = (np.asarray(xyz) - physical["cube_position_m"]) * 1000 if xyz is not None else None
                    trace.append({"xy_error_mm": float(np.linalg.norm(error[:2])) if error is not None else None,
                                  "z_error_mm": float(error[2]) if error is not None else None,
                                  "step": sensor["physics_step"], "time_s": sensor["time_s"], "ref": truth["state"],
                                  "pred": result["state"], "reference_objects": sorted(truth["horizontal_objects"]),
                                  "boundaries": sorted(result["boundaries"]), "camera_age_s": result["camera_age_s"]})
                scores[key] = metrics(trace, hz)
                scores[key]["target_confirmation"] = target_events(header, trace)
                scores[key]["position_errors"] = {}
                for state in ("all", *STATES):
                    valid = [r for r in trace if r["xy_error_mm"] is not None and (state == "all" or r["ref"] == state)]
                    if valid:
                        errors = [r["xy_error_mm"] for r in valid]
                        scores[key]["position_errors"][state] = {
                            "n": len(valid), "xy_median_mm": float(np.median(errors)),
                            "xy_p95_mm": float(np.percentile(errors, 95)),
                            "signed_z_median_mm": float(np.median([r["z_error_mm"] for r in valid]))}

                if image == "rgb" and delay == 0:
                    tracks[mode] = trace
            if tracks:
                plot(tracks, {"a": "A  |  Lift, transport and landing", "u4": "Upper 4  |  Side contact and rear seating",
                              "u5": "Upper 5  |  Straight insertion and withdrawal"}[header["case"]],
                     output / (header["case"] + "_timeline.png"))
                with gzip.open(output / (header["case"] + "_timeline.csv.gz"), "wt") as file:
                    writer = csv.writer(file)
                    writer.writerow(["step", "time_s", "truth", "truth_objects", "truth_reason", "support_load_fraction",
                                     *MODES, "compensated_objects", "camera_age_s",
                                     "left_wrist_force_N", "right_wrist_force_N"])
                    for i, ((sensor, _), truth) in enumerate(zip(samples, reference)):
                        row = tracks["fusion_compensated"][i]
                        writer.writerow([row["step"], row["time_s"], row["ref"], "|".join(row["reference_objects"]),
                                         truth["reason"], truth["support_load_fraction"],
                                         *[tracks[m][i]["pred"] for m in MODES], "|".join(row["boundaries"]),
                                         row["camera_age_s"], *[round(float(np.linalg.norm(sensor[s+"_wrench_world"][:3])), 4)
                                                                for s in ("left", "right")]])
    destination = output / (header["case"] + "_scores.json")
    destination.write_text(json.dumps({"case": header["case"], "source": header["source"],
                                      "scores": scores}, indent=1, allow_nan=False)+"\n")
    print(f'{header["case"]}: {len(samples)} paired samples, {len(scores)} conditions -> {destination}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay(args.source, args.output)
