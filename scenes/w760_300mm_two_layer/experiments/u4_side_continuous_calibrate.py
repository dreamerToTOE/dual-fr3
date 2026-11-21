"""Fit and leave-one-run-out check the continuous side observer on historical U4 runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from seven_reference import SevenReference
from u4_contact_observer import SIDE, U4ContactObserver
from u4_handover_score import anchors, load_at
from u4_left_unload_score import event_timing
from u4_side_continuous import FEATURES, PARAMETERS, WINDOW, SideFeatures


RIDGE = .15
ARCHIVES = ("u4_handover", "u4_left_unload", "u4_side_pulse")


def read_archive(path):
    commands = json.loads(gzip.decompress(path.with_name("right_commands.json.gz").read_bytes()))
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        timeline = header["phase_times_for_scoring_only"]
        if not any(e["phase"] == "upper_B_side_unload" and e["status"] == "RUNNING" for e in timeline):
            return None
        bounds = anchors(header)
        hz = header["hz"]
        if hz != 120:
            raise ValueError("Expected 120 Hz historical capture")
        rear_end = bounds[min(3, len(bounds)-1)]
        interventions = [round(e["sim_time_s"]*hz) for e in timeline
                         if e["phase"] == "upper_B_u4_side_probe" and e["status"] == "STARTED"]
        end = min([rear_end, *interventions])
        passive = U4ContactObserver(header["config"])
        features = SideFeatures()
        reference = SevenReference(header["config"], hz, header["reference_positions"])
        timing, vectors = [], []
        for line in stream:
            sensor, truth = json.loads(line)
            tick = sensor["physics_step"]
            if tick != truth["physics_step"]:
                raise ValueError("Unpaired U4 record")
            if tick > end:
                break
            sensor["right_command_y_m"] = commands["target_y_m"][tick-commands["first_step"]]
            prediction = passive.update(sensor, header["optical_neighbors"])
            x = features.update(sensor, prediction)
            phys = reference.update(truth)
            if tick <= bounds[1]:
                continue
            bearing, f = load_at(truth["cube_contacts"], phys, SIDE, 0, hz)
            timing.append({"step": tick, "handoff": tick <= bounds[2], "truth_side": bearing,
                           "force": f, "baseline_side": prediction["side_loaded"]})
            vectors.append(x if x is not None and prediction["side_loaded"] is None else np.full(len(FEATURES), np.nan))
    return timing, np.array(vectors), bounds[1]


def evidence_groups(runs):
    groups = []
    for rows, vectors, _ in runs:
        valid = np.isfinite(vectors).all(axis=1)
        for label in (False, True):
            relevant = np.array([r["truth_side"] is label for r in rows])
            x = vectors[valid & relevant]
            if len(x):
                groups.append((x, label))
    return groups


def fit(runs):
    groups = evidence_groups(runs)
    x = np.concatenate([v for v, _ in groups])
    y = np.concatenate([np.full(len(v), label, dtype=float) for v, label in groups])
    counts = {label: sum(g == label for _, g in groups) for label in (False, True)}
    w = np.concatenate([np.full(len(v), .5/(len(v)*counts[label])) for v, label in groups])
    center = w @ x
    scale = np.maximum(np.sqrt(w @ (x-center)**2), 1e-12)
    z = np.column_stack((np.ones(len(x)), (x-center)/scale))
    beta = np.zeros(z.shape[1])
    regularization = np.diag([0., *([RIDGE]*len(FEATURES))])
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(z @ beta, -40, 40)))
        curvature = (z.T*(w*p*(1-p))) @ z + regularization
        update = np.linalg.solve(curvature, z.T @ (w*(p-y)) + regularization @ beta)
        beta -= update
        if np.max(np.abs(update)) < 1e-9:
            break
    scores = z @ beta
    # Shortcut: these are calibration error bounds, not out-of-scene guarantees.
    clear_below = float(np.min(scores[y == 1]))
    load_above = float(np.max(scores[y == 0]))
    return {"center": center.tolist(), "scale": scale.tolist(), "intercept": float(beta[0]),
            "weights": beta[1:].tolist(), "clear_below": min(clear_below, load_above),
            "load_above": max(clear_below, load_above)}


def validate(run, model):
    events, x, origin = run
    value = model["intercept"] + ((x-model["center"])/model["scale"]) @ model["weights"]
    rows = [dict(r, fusion_side=True if q > model["load_above"] else
                 False if q < model["clear_below"] else r["baseline_side"])
            for r, q in zip(events, value, strict=True)]
    metrics = {}
    for name, window in (("handoff", [r for r in rows if r["handoff"]]),
                         ("rear", [r for r in rows if not r["handoff"]])):
        metrics[name] = {"ticks": len(window)}
        for model in ("baseline", "fusion"):
            pred = model + "_side"
            weak = [r for r in window if r["truth_side"] is True and r["force"] is not None and .1 <= r["force"] < 1]
            clear = [r for r in window if r["truth_side"] is False]
            loaded = [r for r in window if r["truth_side"] is True]
            metrics[name][model] = {
                "weak": len(weak), "weak_hits": sum(r[pred] is True for r in weak),
                "loaded": len(loaded), "loaded_hits": sum(r[pred] is True for r in loaded),
                "clear": len(clear), "clear_hits": sum(r[pred] is False for r in clear),
                "false_load": sum(r[pred] is True for r in clear),
                "false_clear": sum(r[pred] is False for r in loaded),
            }
    return {"windows": metrics, "first_transition": event_timing(rows, origin, 120, 3)}


def main(output):
    root = PARAMETERS.parent
    paths = [p for group in ARCHIVES for p in sorted((root/group).glob("*/paired.jsonl.gz"))]
    observed = {}
    for path in paths:
        run = read_archive(path)
        if run is not None:
            observed[str(path.parent.relative_to(root))] = (path, run)
    folds = {}
    for name, (path, run) in observed.items():
        held_out = fit([other for key, (_, other) in observed.items() if key != name])
        folds[name] = {"paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       **validate(run, held_out)}
    report = {"window_physics_steps": WINDOW, "sensor_features": FEATURES, "ridge": RIDGE,
              "fit_scope": "Historical U4 handoff and rear push data, strictly before any probe action. PhysX used only for calibration and scoring. Classes balanced per run and class. Labels with unavailable PhysX evidence omitted.",
              "thresholds": "Per-class training extrema on pooled scores, avoid overlapping bounds; strict inequalities with abstention in between.",
              "whole_run_holdout": folds, "model": fit([v for _, v in observed.values()])}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
