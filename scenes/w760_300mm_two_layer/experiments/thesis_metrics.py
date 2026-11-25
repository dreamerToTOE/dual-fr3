"""Compute run-level thesis metrics for frozen multisource state estimators.

The input manifest must split complete trajectories, never individual physics
steps. Each run points at a command-enriched paired.jsonl.gz clip.
"""

from __future__ import annotations

from collections import Counter
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from seven_observer import SevenObserver
from seven_reference import SevenReference


STATES = tuple(f"S{i}" for i in range(7))
MODALITIES = ("vision_only", "tactile_only", "fusion")


def load_clip(path: Path):
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError(f"Empty paired clip: {path}")
    steps = [sensor["physics_step"] for sensor, _ in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise ValueError(f"Non-contiguous paired clip: {path}")
    if header.get("stop_step") != steps[-1]:
        raise ValueError(
            f"Clip is not a complete run: stop_step={header.get('stop_step')} "
            f"last_step={steps[-1]} path={path}"
        )
    return header, rows


def classify(header, rows):
    if header["case"] in ("u4", "u5") and not any(
            sensor.get("tcp_command_positions_m") for sensor, _ in rows):
        raise ValueError(
            f"{header['source']} lacks issued TCP commands. Enrich it with seven_commands.py first."
        )
    reference = SevenReference(header["config"], header["hz"], header["reference_positions"])
    observers = {
        "vision_only": SevenObserver(header["config"], "vision_fk"),
        "tactile_only": SevenObserver(header["config"], "wrench_fk"),
        "fusion": SevenObserver(header["config"], "fusion_compensated", revised=True),
    }
    traces = {name: [] for name in MODALITIES}
    for sensor, physical in rows:
        if sensor["physics_step"] != physical["physics_step"] or sensor["time_s"] != physical["time_s"]:
            raise ValueError(f"Unpaired step in {header['source']}")
        truth = reference.update(physical)
        base = {
            "step": sensor["physics_step"],
            "time_s": sensor["time_s"],
            "truth": truth["state"],
        }
        for name, observer in observers.items():
            prediction = observer.update(sensor, neighbors=header["optical_neighbors"])
            traces[name].append({**base, "pred": prediction["state"]})
    return traces


def confusion_metrics(trace):
    matrix = {state: Counter() for state in STATES}
    for row in trace:
        if row["truth"] in matrix:
            matrix[row["truth"]][row["pred"]] += 1
    result = {}
    for state in STATES:
        truth_count = sum(matrix[state].values())
        predicted_count = sum(matrix[other][state] for other in STATES)
        tp = matrix[state][state]
        precision = tp / predicted_count if predicted_count else None
        recall = tp / truth_count if truth_count else None
        f1 = (2 * precision * recall / (precision + recall)
               if precision is not None and recall is not None and precision + recall else None)
        result[state] = {
            "tp": tp,
            "truth": truth_count,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unknown": matrix[state]["UNKNOWN"],
        }
    known = sum(sum(matrix[state].values()) for state in STATES)
    unknown = sum(matrix[state]["UNKNOWN"] for state in STATES)
    result["_overall"] = {
        "known_steps": known,
        "correct_steps": sum(matrix[state][state] for state in STATES),
        "unknown_steps": unknown,
        "coverage": (known - unknown) / known if known else None,
        "accuracy": sum(matrix[state][state] for state in STATES) / known if known else None,
    }
    return result


def event_metrics(trace, minimum_steps=3):
    events = []
    current = None
    for row in trace + [{"step": None, "time_s": None, "truth": None, "pred": None}]:
        if row["truth"] != current:
            if current in STATES and len(events[-1]["rows"]) >= minimum_steps:
                pass
            if current in STATES and events:
                events[-1]["rows"] = events[-1]["rows"]
            if row["truth"] in STATES:
                events.append({"state": row["truth"], "start_step": row["step"],
                               "start_time_s": row["time_s"], "rows": []})
            current = row["truth"]
        if events and row["truth"] == events[-1]["state"]:
            events[-1]["rows"].append(row)
    events = [event for event in events if len(event["rows"]) >= minimum_steps]
    by_state = {}
    for state in STATES:
        selected = [event for event in events if event["state"] == state]
        delays = []
        early = 0
        for event in selected:
            matching = next((row for row in event["rows"] if row["pred"] == state), None)
            prior = next((row for row in trace if row["step"] < event["start_step"]), None)
            if matching is None:
                continue
            if prior is not None and prior["pred"] == state:
                early += 1
                continue
            delays.append((matching["time_s"] - event["start_time_s"]) * 1000)
        by_state[state] = {
            "events": len(selected),
            "detected": len(delays),
            "missed": len(selected) - len(delays),
            "early": early,
            "detection_rate": len(delays) / len(selected) if selected else None,
            "median_delay_ms": float(np.median(delays)) if delays else None,
            "mean_delay_ms": float(np.mean(delays)) if delays else None,
        }
    return by_state


def mean_ci(values, rng, samples=5000):
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if clean.size == 0:
        return {"mean": None, "ci95": [None, None], "n_runs": 0}
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = np.mean(rng.choice(clean, size=clean.size, replace=True))
    return {
        "mean": float(np.mean(clean)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_runs": int(clean.size),
    }


def summarize(runs, seed=20261122):
    rng = np.random.default_rng(seed)
    result = {"seed": seed, "runs": runs, "modalities": {}}
    for modality in MODALITIES:
        result["modalities"][modality] = {"overall": {}, "states": {}, "events": {}}
        for metric in ("accuracy", "coverage"):
            result["modalities"][modality]["overall"][metric] = mean_ci(
                [run["metrics"][modality]["_overall"][metric] for run in runs], rng
            )
        for state in STATES:
            result["modalities"][modality]["states"][state] = {
                metric: mean_ci([run["metrics"][modality][state][metric] for run in runs], rng)
                for metric in ("precision", "recall", "f1")
            }
            result["modalities"][modality]["events"][state] = {
                metric: mean_ci([run["events"][modality][state][metric] for run in runs], rng)
                for metric in ("detection_rate", "median_delay_ms", "mean_delay_ms")
            }
    return result


def run(manifest_path: Path, output: Path):
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("runs", [])
    if not entries:
        raise ValueError("Manifest has no runs")
    split_names = {entry.get("split") for entry in entries}
    if split_names != {"development", "validation", "test"}:
        raise ValueError("Manifest must contain development, validation and test splits")
    seen = set()
    run_reports = []
    for entry in entries:
        name = entry["name"]
        if name in seen:
            raise ValueError(f"Duplicate run name: {name}")
        seen.add(name)
        clip_path = Path(entry["path"])
        if manifest.get("schema_version") == 2:
            clip_path = manifest_path.parent / clip_path
        header, rows = load_clip(clip_path)
        traces = classify(header, rows)
        run_reports.append({
            "name": name,
            "split": entry["split"],
            "case": header["case"],
            "source": header["source"],
            "steps": len(rows),
            "metrics": {modality: confusion_metrics(trace) for modality, trace in traces.items()},
            "events": {modality: event_metrics(trace) for modality, trace in traces.items()},
        })
    test_runs = [run for run in run_reports if run["split"] == "test"]
    summary = summarize(test_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "manifest": str(manifest_path),
        "split_counts": dict(Counter(run["split"] for run in run_reports)),
        "run_reports": run_reports,
        "test_summary": summary,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)
