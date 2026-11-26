"""Causal sensor-only recognition and separate physical-reference evaluation."""
from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


MODES = ("fusion", "vision_only", "force_only")


def reference_state(truth, config):
    """Use physical contacts/attachments and Cube truth only in this evaluator."""
    if not truth["cube_contacts"].get("valid", False):
        return {"state": "UNKNOWN", "reason": "invalid physical contact snapshot"}
    contacts = truth["cube_contacts"]["loaded_contacts"]
    position = np.asarray(truth["cube_position_m"])
    rotation = np.asarray(truth["cube_rotation_world"])
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError("Invalid physical Cube reference pose")
    half = config["cube_size_m"] / 2
    radius = half * np.sum(np.abs(rotation), axis=1)
    # Report solver-offset adjacency separately from force-bearing contact.
    boxes = {name: (np.asarray(center), np.asarray(size))
             for name, center, size in config["static_boxes"]}
    gaps = {
        "wall_back": boxes["wall_back"][0][1] - boxes["wall_back"][1][1] / 2 - position[1] - radius[1],
        "wall_left": position[0] - radius[0] - boxes["wall_left"][0][0] - boxes["wall_left"][1][0] / 2,
        "wall_right": boxes["wall_right"][0][0] - boxes["wall_right"][1][0] / 2 - position[0] - radius[0],
    }
    left = bool(contacts["left_cup"] or truth["attachments"]["left"])
    right = bool(contacts["right_cup"] or truth["attachments"]["right"])
    supported = bool(contacts["floor"] or contacts["supply"])
    back = bool(contacts["wall_back"] or abs(gaps["wall_back"]) <= 50e-6)
    side = bool(contacts["wall_left"] or contacts["wall_right"]
                or min(abs(gaps["wall_left"]), abs(gaps["wall_right"])) <= 50e-6)
    if not left and not right:
        state = "RELEASED_CORNER" if back and side and supported else "NO_TOOL_CONTACT"
    elif left and right and not supported:
        state = "BILATERAL_SUSPENDED"
    elif left and right and back and side:
        state = "DUAL_CORNER_CONTACT"
    elif left and right and back:
        state = "DUAL_REAR_CONTACT"
    elif left and back:
        state = "LEFT_REAR_CONTACT"
    elif left and supported and not right:
        state = "LEFT_FLOOR_CONTACT"
    elif left and right and supported:
        state = "BILATERAL_SUPPORTED"
    else:
        state = "UNKNOWN"
    return {"state": state, "tool_contact": {"left": left, "right": right},
            "supported": supported, "back_wall": back, "side_wall": side,
            "physical_wall_contact": {name: bool(contacts[name]) for name in gaps},
            "geometric_wall_adjacency": {name: bool(abs(gap) <= 50e-6) for name, gap in gaps.items()},
            "wall_gaps_m": gaps, "wall_adjacency_tolerance_m": 50e-6,
            "source": "PhysX contact lifecycle/impulses, attachment joints, independent Cube geometry"}


def recognition_metrics(predictions, truth):
    if len(predictions) != len(truth) or not truth:
        raise ValueError("Recognition evaluation needs paired nonempty physical samples")
    pairs = [(t["state"], p["state"]) for p, t in zip(predictions, truth)]
    decidable = [(a, b) for a, b in pairs if a != "UNKNOWN"]
    labels = sorted(set(a for a, _ in decidable))
    confusion = {label: dict(Counter(b for a, b in decidable if a == label)) for label in labels}
    by_state = {}
    for label in labels:
        actual = sum(a == label for a, _ in decidable)
        predicted = sum(b == label for _, b in decidable)
        correct = sum(a == b == label for a, b in decidable)
        precision = correct / predicted if predicted else 0.
        recall = correct / actual
        by_state[label] = {"reference_samples": actual, "predicted_samples": predicted,
                           "precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    covered = [(a, b) for a, b in decidable if b not in ("UNKNOWN", "INCONSISTENT")]
    events = []
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        label = pairs[start][0]
        if label != "UNKNOWN" and end - start >= 30:
            detected = next((i for i in range(start, end) if pairs[i][1] == label), None)
            events.append({"state": label, "reference_start_sample": start,
                           "duration_s": (end - start) / 120,
                           "recognition_delay_s": None if detected is None else (detected - start) / 120})
        start = end
    delays = [event["recognition_delay_s"] for event in events if event["recognition_delay_s"] is not None]
    return {
        "samples": len(pairs), "reference_known_samples": len(decidable),
        "reference_unknown_samples": len(pairs) - len(decidable),
        "accuracy_including_abstention_as_error": sum(a == b for a, b in decidable) / len(decidable) if decidable else None,
        "decision_coverage": len(covered) / len(decidable) if decidable else None,
        "accuracy_on_decided_samples": sum(a == b for a, b in covered) / len(covered) if covered else None,
        "macro_f1": float(np.mean([v["f1"] for v in by_state.values()])) if by_state else None,
        "prediction_counts": dict(Counter(b for _, b in pairs)),
        "predictions_on_unknown_reference": dict(Counter(b for a, b in pairs if a == "UNKNOWN")),
        "stable_contact_events": events,
        "stable_event_definition": "physical reference constant for >=0.25 s; accuracy still uses every sample",
        "stable_event_misses": sum(event["recognition_delay_s"] is None for event in events),
        "recognition_delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
        "by_state": by_state, "confusion_matrix": confusion,
    }


class ContactStateExperiment:
    """Observer only: recommendations are logged, never executed as commands."""

    def __init__(self, directory, config, *, camera_stream_source=None):
        from fr3_contact_state import ContactStateEstimator
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.camera_input = camera_stream_source or "causal RGB/ArUco observation"
        library_path = Path(__file__).with_name("fr3_contact_skills.json")
        metadata = {"schema_version": 1, "scene_config": config,
                    "skill_library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
                    "recognizer_sha256": hashlib.sha256(Path(__file__).with_name("fr3_contact_state.py").read_bytes()).hexdigest(),
                    "camera_input": self.camera_input,
                    "modes": {"fusion": "camera + wrist wrench + encoder FK + static geometry",
                              "vision_only": "camera + encoder geometry; no force input",
                              "force_only": "wrist wrench + encoder FK + static geometry; no camera"},
                    "reference_clock": "snapshot.physics_step_index is the absolute World step; log physics_step is relative control tick"}
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        self.models = {mode: ContactStateEstimator(config, mode=mode) for mode in MODES}
        self.predictions = {mode: [] for mode in MODES}
        self.references = []
        self.inputs = []
        self.streams = {name: gzip.open(self.directory / f"{name}.jsonl.gz", "wt")
                        for name in ("sensor_inputs", "reference", "predictions")}

    def append(self, *, physics_step, sensors, truth):
        # The recognizer receives only the explicitly constructed sensor boundary.
        outputs = {mode: model.update(**sensors) for mode, model in self.models.items()}
        reference = reference_state(truth, self.config)
        common = {"physics_step": physics_step, "time_s": sensors["time_s"]}
        for name, row in (("sensor_inputs", sensors), ("reference", {**truth, **reference}),
                          ("predictions", outputs)):
            self.streams[name].write(json.dumps({**common, **row}, allow_nan=False) + "\n")
        self.inputs.append(sensors)
        self.references.append(reference)
        for mode in MODES:
            self.predictions[mode].append(outputs[mode])
        return outputs["fusion"]

    def close(self, task_status="UNVERIFIED"):
        for stream in self.streams.values():
            stream.close()
        if not self.references:
            return {"status": "INCOMPLETE", "samples": 0}
        metrics = {mode: recognition_metrics(self.predictions[mode], self.references) for mode in MODES}
        # Occlusion stress changes only observed camera availability, not physical truth.
        from fr3_contact_state import ContactStateEstimator
        model = ContactStateEstimator(self.config, mode="fusion")
        masked = []
        for sensors in self.inputs:
            masked.append(model.update(**{**sensors, "camera": None}))
        metrics["fusion_camera_fully_unavailable"] = recognition_metrics(masked, self.references)
        summary = {
            "status": "COMPLETED_BASELINE_EVALUATION", "samples": len(self.references),
            "physical_task_status": task_status,
            "metrics": metrics,
            "input_boundary": [self.camera_input, "wrist TCP six-axis feedback", "encoder FK/tool geometry"],
            "excluded_from_recognizer": ["task phase", "skill ID", "Cube truth", "PhysX contact report", "attachment state"],
            "reference_definition": "independent physical contacts/attachments plus 50 um wall adjacency; no wrist force or task phase",
            "control": "observer only; skill recommendations do not change robot commands",
            "limitations": ["Single normal proxy-contact task is a baseline, not research-grade generalization evidence.",
                            "Unloaded geometric contact can be unobservable; UNKNOWN counts against accuracy/coverage.",
                            "Camera outage is sensor replay, not a simulated physical disconnection."],
        }
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        self.plot()
        return summary

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted({row["state"] for row in self.references}
                        | {row["state"] for rows in self.predictions.values() for row in rows})
        indices = {label: index for index, label in enumerate(labels)}
        times = [row["time_s"] for row in self.inputs]
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, layout="constrained")
        for ax, (name, rows) in zip(axes, [("Independent physical reference", self.references), *self.predictions.items()]):
            ax.step(times, [indices[row["state"]] for row in rows], where="post")
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_title(name)
        axes[-1].set_xlabel("Simulation time (s)")
        fig.savefig(self.directory / "state_timeline.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="A run's contact_states directory")
    args = parser.parse_args()
    metadata = json.loads((args.directory / "metadata.json").read_text())
    with gzip.open(args.directory / "sensor_inputs.jsonl.gz", "rt") as stream:
        inputs = [json.loads(line) for line in stream]
    with gzip.open(args.directory / "reference.jsonl.gz", "rt") as stream:
        truth = [json.loads(line) for line in stream]
    from fr3_contact_state import ContactStateEstimator
    results = {}
    keys = ("time_s", "left_wrench_world", "right_wrench_world", "tcp_positions_m", "cup_normals_world", "camera")
    for mode in MODES:
        model = ContactStateEstimator(metadata["scene_config"], mode=mode)
        outputs = [model.update(**{key: row[key] for key in keys}) for row in inputs]
        results[mode] = recognition_metrics(outputs, truth)
    (args.directory / "replay_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({mode: {key: value for key, value in metrics.items()
                            if key not in ("by_state", "confusion_matrix")}
                      for mode, metrics in results.items()}, indent=2))
