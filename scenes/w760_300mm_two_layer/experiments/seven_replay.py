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


def plot_transfer(samples, physical, estimated, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([s["time_s"] for s, _ in samples])
    weight = config["physics"]["cube_mass_kg"] * abs(config["physics"]["gravity_m_s2"][2])
    bottom = np.array([np.nan if r["support_load_fraction"] is None else r["support_load_fraction"]
                       for r in physical])
    wrist = np.array([-(s["left_wrench_world"][2] + s["right_wrench_world"][2]) / weight
                      for s, _ in samples])
    airborne = [i for i, r in enumerate(physical) if r["state"] == "S4"]
    if not airborne:
        return
    fig, panels = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, constrained_layout=True)
    for ax, label, reason, fallback in zip(panels, ("Lift-off", "Landing"),
                                           ("unloading_bottom", "loading_bottom"), (airborne[0], airborne[-1])):
        transfer_times = [t[i] for i, r in enumerate(physical) if r["state"] == "S3" and r["reason"] == reason]
        center = float(np.median(transfer_times)) if transfer_times else t[fallback]
        window = np.abs(t-center) <= 1.0
        ax.plot(t[window]-center, bottom[window], label="Bottom / PhysX", color="#136a8a")
        ax.plot(t[window]-center, wrist[window], label="Wrists / sensors", color="#ce650c")
        for y, seq, source, color in ((-.10, physical, "Reference S3", "#3f5889"),
                                      (-.22, estimated, "Estimated S3", "#934774")):
            ticks = window & np.array([(r.get("state", r.get("pred")) == "S3") for r in seq])
            ax.scatter(t[ticks]-center, np.full(ticks.sum(), y), s=8, marker="|", label=source, color=color)
        ax.axhline(1, lw=.5, color=".55", ls="--")
        ax.set(title=label, xlabel="Time from transfer midpoint (s)")
    panels[0].set_ylabel("Vertical load / box weight")
    panels[1].legend(fontsize=8, loc="upper right")
    fig.savefig(output, dpi=145)
    plt.close(fig)


def optical_audit(samples, image):
    """Score the raw marker estimate against the independent pose at exposure."""
    times = np.array([physical["time_s"] for _, physical in samples])
    positions = np.array([physical["cube_position_m"] for _, physical in samples])
    errors, ages = [], []
    packets = detected = 0
    for sensor, _ in samples:
        packet = sensor.get(image)
        if not packet:
            continue
        packets += 1
        if packet.get("status") != "detected" or packet.get("position_m") is None:
            continue
        detected += 1
        stamp = packet["capture_time_s"]
        if not times[0] <= stamp <= times[-1]:
            continue
        location = [np.interp(stamp, times, axis) for axis in positions.T]
        errors.append(1000 * (np.asarray(packet["position_m"]) - location))
        ages.append(1000 * (packet["received_time_s"] - stamp))
    result = {"packets": packets, "detected": detected, "paired": len(errors)}
    if errors:
        result.update(median_signed_xyz_mm=np.median(errors, axis=0).tolist(),
                      median_exposure_to_receive_ms=float(np.median(ages)))
    return result


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
                                  "reason": truth["reason"], "pred": result["state"], "reference_objects": sorted(truth["horizontal_objects"]),
                                  "boundaries": sorted(result["boundaries"]), "camera_age_s": result["camera_age_s"]})
                scores[key] = metrics(trace, hz)
                scores[key]["target_confirmation"] = target_events(header, trace)
                if header["case"] == "a":
                    scores[key]["transfer_by_direction"] = {
                        reason: {"n": sum(r["ref"] == "S3" and r["reason"] == reason for r in trace),
                                 "hits": sum(r["ref"] == r["pred"] == "S3" and r["reason"] == reason for r in trace)}
                        for reason in ("loading_bottom", "unloading_bottom")}
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
                if header["case"] == "a":
                    plot_transfer(samples, reference, tracks["fusion_compensated"], cfg, output / "a_transfer.png")
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
                                      "raw_optics": {k: optical_audit(samples, k) for k in
                                                     (("camera", "blur") if header["has_blur"] else ("camera",))},
                                      "scores": scores}, indent=1, allow_nan=False)+"\n")
    print(f'{header["case"]}: {len(samples)} paired samples, {len(scores)} conditions -> {destination}')


def replay_revision(source, output):
    tracks = {"frozen": [], "revised": []}
    with gzip.open(source, "rt") as stream:
        header = json.loads(next(stream))
        reference = SevenReference(header["config"], header["hz"], header["reference_positions"])
        observers = {name: SevenObserver(header["config"], "fusion_compensated", revised=name == "revised")
                     for name in tracks}
        command_samples = 0
        for line in stream:
            sensor, physical = json.loads(line)
            if sensor["physics_step"] != physical["physics_step"] or sensor["time_s"] != physical["time_s"]:
                raise ValueError("Unpaired reference sample")
            command_samples += bool(sensor.get("tcp_command_positions_m"))
            ref = reference.update(physical)
            row = {"step": sensor["physics_step"], "time_s": sensor["time_s"], "ref": ref["state"],
                   "reference_objects": sorted(ref["horizontal_objects"])}
            for name, observer in observers.items():
                result = observer.update(sensor, neighbors=header["optical_neighbors"])
                tracks[name].append({**row, "pred": result["state"], "boundaries": result["boundaries"]})
    if header["case"] in ("u4", "u5") and not command_samples:
        raise ValueError("Enrich the paired push clip with its issued TCP commands using seven_commands.py")
    scores = {}
    for name, rows in tracks.items():
        summary = metrics(rows, header["hz"])
        summary["target_confirmation"] = target_events(header, rows)
        summary["false_boundaries_during_s1"] = dict(Counter(
            path for row in rows if row["ref"] == "S1" for path in row["boundaries"]))
        scores[name] = summary
    output.mkdir(parents=True, exist_ok=True)
    path = output / (header["case"] + "_revision.json")
    path.write_text(json.dumps({"case": header["case"], "source": header["source"],
                                "command_samples": command_samples, "scores": scores},
                               indent=1, allow_nan=False) + "\n")
    print(f'{header["case"]}: {len(tracks["frozen"])} paired samples -> {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revised", action="store_true", help="Compare the revised fusion with the frozen baseline")
    args = parser.parse_args()
    (replay_revision if args.revised else replay)(args.source, args.output)
