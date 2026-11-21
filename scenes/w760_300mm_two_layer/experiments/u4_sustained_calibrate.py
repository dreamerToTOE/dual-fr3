"""Fit and verify the force-motion model on archived, whole U4 trial prefixes."""
import argparse
import gzip
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from seven_reference import SevenReference
from u4_contact_observer import SIDE, U4ContactObserver
from u4_handover_score import anchors, load_at
from u4_pulse_score import digest
from u4_sustained_observer import FEATURES, PARAMETERS, WINDOW, CONFIRM, ForceMotion, SustainedSideObserver


ARCHIVES = ("u4_handover", "u4_left_unload", "u4_side_pulse", "u4_side_continuous")
SUSTAINED = 12
RIDGE = .03
OPPOSITE_TAIL = .01


def history(path):
    with gzip.open(path, "rt") as f, gzip.open(path.with_name("right_commands.json.gz"), "rt") as g:
        header, cmd = json.loads(next(f)), json.load(g)
        if header["source"] != cmd["source"] or header["stop_step"] != cmd["last_step"] or (
                header["source_sha256"]["contact_states/sensor_inputs.jsonl.gz"] != cmd["sensor_inputs_sha256"]):
            raise ValueError("Mismatched command and sensor archives")
        markers = header["phase_times_for_scoring_only"]
        if not any(mark["phase"] == "upper_B_side_unload" and mark["status"] == "RUNNING" for mark in markers):
            return None
        bounds = anchors(header)
        if header["hz"] != 120:
            raise ValueError("Expected 120 Hz capture")
        end = min(bounds[min(3, len(bounds)-1)], min(
            (round(mark["sim_time_s"]*120) for mark in markers if mark["phase"] == "upper_B_u4_side_probe"
             and mark["status"] == "STARTED"), default=header["stop_step"]))
        passive = U4ContactObserver(header["config"])
        truth = SevenReference(header["config"], 120, header["reference_positions"])
        features = ForceMotion()
        data, vectors = [], []
        for expected, line in enumerate(f, start=cmd["first_step"]):
            sensor, physical = json.loads(line)
            if sensor["physics_step"] != expected or physical["physics_step"] != expected:
                raise ValueError("Unsynchronized motion and contact truth")
            if expected > end:
                break
            sensor["right_command_y_m"] = cmd["target_y_m"][expected-cmd["first_step"]]
            prior = passive.update(sensor, header["optical_neighbors"])
            x = features.update(sensor)
            phys = truth.update(physical)
            if expected <= bounds[1]:
                continue
            bearing, force = load_at(physical["cube_contacts"], phys, SIDE, 0, 120)
            data.append((expected-bounds[1], expected <= bounds[2], bearing, force, prior["side_loaded"]))
            if x is None or prior["side_loaded"] is not None:
                x = np.full(len(FEATURES), np.nan)
            vectors.append(x)
    if data:
        return data, np.asarray(vectors)
    return None


def fit(runs):
    groups = []
    for events, x in runs:
        label = np.full(len(events), -1)
        for bearing, ticks in groupby(enumerate(events), key=lambda item: item[1][2]):
            indices = [i for i, _ in ticks]
            if bearing is not None:
                label[indices[SUSTAINED-1:]] = int(bearing)
        for c in (0, 1):
            points = x[(label == c) & np.isfinite(x).all(axis=1)]
            if len(points):
                groups.append((c, points))
    x = np.concatenate([x for c, x in groups])
    y = np.concatenate([np.full(len(x), c) for c, x in groups])
    per_class = {c: sum(label == c for label, _ in groups) for c in (0, 1)}
    w = np.concatenate([np.full(len(x), .5 / (len(x)*per_class[c])) for c, x in groups])
    mean = w @ x
    std = np.maximum(np.sqrt(w @ (x-mean)**2), 1e-11)
    z = np.column_stack((np.ones(len(x)), (x-mean)/std))
    penalty = np.diag([0., *([RIDGE]*len(FEATURES))])
    beta = np.zeros(z.shape[1])
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(z @ beta, -40., 40.)))
        hessian = (z.T * (w*p*(1-p))) @ z + penalty
        step = np.linalg.solve(hessian, z.T @ (w*(p-y)) + penalty @ beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    scores = z @ beta
    bounds = sorted((float(np.quantile(scores[y == 1], OPPOSITE_TAIL)),
                     float(np.quantile(scores[y == 0], 1-OPPOSITE_TAIL))))
    return {"center": mean.tolist(), "scale": std.tolist(), "weights": beta[1:].tolist(),
            "intercept": float(beta[0]), "clear_below": bounds[0], "load_above": bounds[1]}


def bouts(values, minimum):
    result = []
    for label, group in groupby(enumerate(values), key=lambda item: item[1]):
        ticks = [i for i, _ in group]
        if label is not None and len(ticks) >= minimum:
            result.append((label, ticks[0], ticks[-1]))
    return result


def validate(run, parameters):
    estimator = SustainedSideObserver(parameters)
    events, x = run
    scores = estimator.intercept + ((x-estimator.center)/estimator.scale) @ estimator.weights
    updated = [estimator.decision(float(s) if np.isfinite(s) else None) for s in scores]
    observed = [e[2] for e in events]
    passive = [e[4] for e in events]
    fusion = [p if p is not None else m for p, m in zip(passive, updated, strict=True)]
    physical = bouts(observed, SUSTAINED)
    clear = next((b for b in physical if b[0] is False), None)
    reload = next((b for b in physical if clear and b[0] is True and b[1] > clear[2]), None)
    summary = {}
    for name, pred in (("baseline", passive), ("fusion", fusion)):
        called = bouts(pred, CONFIRM)
        def confirmed(target):
            hit = next((first+CONFIRM-1 for label, first, _ in called
                        if target and label is target[0] and target[1] <= first <= target[2]-CONFIRM+1), None)
            return events[hit][0] if hit is not None else None
        summary[name] = {"first_clear_declaration": confirmed(clear), "first_reload_declaration": confirmed(reload)}
        for section, mask in (("handoff", [e[1] for e in events]), ("all", [True]*len(events))):
            pairs = [(p,t) for p,t,allowed in zip(pred, observed, mask, strict=True) if allowed]
            scores = {"clear": sum(p is False and t is False for p,t in pairs),
                      "false_clear": sum(p is False and t is True for p,t in pairs),
                      "loaded": sum(p is True and t is True for p,t in pairs),
                      "false_loaded": sum(p is True and t is False for p,t in pairs)}
            summary[name][section] = scores
    return {"first_sustained_clear": [events[i][0] for i in clear[1:]] if clear else None,
            "first_sustained_reload": [events[i][0] for i in reload[1:]] if reload else None,
            "results": summary}


def main(output):
    root = PARAMETERS.parent
    captures = {}
    for study in ARCHIVES:
        for paired in sorted((root/study).glob("*/paired.jsonl.gz")):
            run = history(paired)
            if run is not None:
                captures[str(paired.parent.relative_to(root))] = (paired, run)
    folds = {}
    for name, (path, run) in captures.items():
        other = [v for k, (_, v) in captures.items() if k != name]
        folds[name] = {"paired_sha256": digest(path), **validate(run, fit(other))}
    output.write_text(json.dumps({
        "training": "Only earliest physical prefixes through first rear-push confirmation, cut before any probe. Every fold withholds a complete trial.",
        "score": "Run- and class-balanced regularized logistic force-motion model with three consecutive supportive updates. The .01 opposite-class tail is a fixed rejection criterion on training scores and is no guarantee on held-out trials.",
        "truth_frames_required": SUSTAINED, "window_physics_steps": WINDOW, "event_confirmation_steps": CONFIRM,
        "ridge": RIDGE, "opposite_class_tail": OPPOSITE_TAIL, "features": FEATURES,
        "model": fit([v for _, v in captures.values()]), "whole_run_holdout": folds,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
