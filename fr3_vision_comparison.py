"""Record independent camera/reference streams and compare matched Cube poses."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def match_reference(history, capture_time_s, max_time_delta_s=1 / 240 + 1e-8):
    """Find a 120 Hz reference near the exposure time, never the readout time."""
    if (not history or not np.isfinite(capture_time_s)
            or capture_time_s < history[0]["time_s"] - 1e-8
            or capture_time_s > history[-1]["time_s"] + 1e-8):
        return None
    index = bisect_left(history, capture_time_s, key=lambda row: row["time_s"])
    candidates = history[max(0, index - 1):min(len(history), index + 1)]
    closest = min(candidates, key=lambda row: abs(row["time_s"] - capture_time_s))
    return closest if abs(closest["time_s"] - capture_time_s) <= max_time_delta_s else None


def _stage(phase):
    return next((stage for stage in ("B2", "B1", "A") if str(phase).startswith(stage)), "Setup")


def _position(value):
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("A position must contain three finite coordinates in meters")
    return value.tolist()


def _paired(row):
    return (row.get("truth_position_m") is not None
            and row.get("camera_position_m") is not None
            and row.get("paired", True))


def _metrics(rows):
    pairs = [row for row in rows if _paired(row)]
    detections = sum(row.get("camera_position_m") is not None for row in rows)
    result = {
        "attempts": len(rows),
        "camera_detections": detections,
        "matched": len(pairs),
        "misses": len(rows) - detections,
        "unpaired": detections - len(pairs),
        "detection_rate": detections / len(rows) if rows else None,
        "status_counts": dict(Counter(row.get("status", "unknown") for row in rows)),
        "xyz_bias_m": None,
        "xyz_mae_m": None,
        "euclidean_mean_m": None,
        "euclidean_median_m": None,
        "euclidean_rmse_m": None,
        "euclidean_p95_m": None,
        "euclidean_max_m": None,
    }
    if pairs:
        errors = np.asarray([np.asarray(row["camera_position_m"]) - row["truth_position_m"]
                             for row in pairs], dtype=float)
        norms = np.linalg.norm(errors, axis=1)
        result.update({
            "xyz_bias_m": np.mean(errors, axis=0).tolist(),
            "xyz_mae_m": np.mean(np.abs(errors), axis=0).tolist(),
            "euclidean_mean_m": float(np.mean(norms)),
            "euclidean_median_m": float(np.median(norms)),
            "euclidean_rmse_m": float(np.sqrt(np.mean(norms ** 2))),
            "euclidean_p95_m": float(np.percentile(norms, 95)),
            "euclidean_max_m": float(np.max(norms)),
        })
    return result


def summarize(records):
    """Metric units are meters; missing observations never become zero error."""
    records = list(records)
    phases = list(dict.fromkeys(row.get("phase", "unknown") for row in records))
    return {
        "units": {"positions": "m", "errors": "m", "detection_rate": "fraction"},
        "overall": _metrics(records),
        "by_phase": {phase: _metrics([row for row in records
                                      if row.get("phase", "unknown") == phase])
                     for phase in phases},
        "by_stage": {stage: _metrics([row for row in records
                                      if _stage(row.get("phase", "")) == stage])
                     for stage in dict.fromkeys(_stage(phase) for phase in phases)},
    }


def _write_analysis(directory, records, metadata):
    summary = {"metadata": metadata, **summarize(records)}
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "phase", "physics_step", "capture_time_s", "frame_id", "status", "paired",
                         "truth_x_m", "truth_y_m", "truth_z_m", "camera_x_m", "camera_y_m", "camera_z_m",
                         "error_x_m", "error_y_m", "error_z_m", "position_error_m"])
        for row in records:
            error = (np.asarray(row["camera_position_m"]) - row["truth_position_m"]).tolist() if _paired(row) else None
            writer.writerow([row.get(key) for key in
                             ("sample_id", "phase", "physics_step", "capture_time_s", "frame_id", "status")]
                            + [_paired(row)] + (row.get("truth_position_m") or [None] * 3)
                            + (row.get("camera_position_m") or [None] * 3) + (error or [None] * 3)
                            + [float(np.linalg.norm(error)) if error is not None else None])
    return summary


class VisionComparison:
    """Keep all observations; only this evaluator is given reference positions."""

    def __init__(self, directory, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.records = []
        self.sample_ids = set()
        self.streams = {name: (self.directory / f"{name}.jsonl").open("w")
                        for name in ("ground_truth", "camera", "comparison")}
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def append(self, *, sample_id, phase, physics_step, capture_time_s, truth_position_m,
               camera_position_m=None, status="detected", frame_id=None, frame_metadata=None, **extra):
        if sample_id in self.sample_ids:
            raise ValueError(f"Duplicate visual sample ID: {sample_id}")
        if not np.isfinite(capture_time_s):
            raise ValueError("Capture time must be finite")
        truth = _position(truth_position_m)
        camera = _position(camera_position_m)
        common = {"sample_id": sample_id, "phase": phase, "physics_step": physics_step,
                  "capture_time_s": capture_time_s, "frame_id": frame_id}
        truth_row = {**common, "truth_position_m": truth, "source": "isaacsim_reference"}
        camera_row = {**common, "camera_position_m": camera, "status": status,
                      "source": "rgb_aruco"}
        paired = truth is not None and camera is not None
        error = (np.asarray(camera) - truth).tolist() if paired else None
        row = {**extra, **common, "truth_position_m": truth, "camera_position_m": camera,
               "status": status, "paired": paired, "error_xyz_m": error,
               "position_error_m": float(np.linalg.norm(error)) if paired else None,
               "frame_metadata": frame_metadata or {}}
        # Metadata supplied by the caller stays in the evaluator, never the camera channel.
        encoded = {"ground_truth": json.dumps(truth_row, ensure_ascii=False, allow_nan=False),
                   "camera": json.dumps(camera_row, ensure_ascii=False, allow_nan=False),
                   "comparison": json.dumps(row, ensure_ascii=False, allow_nan=False)}
        for name, value in encoded.items():
            self.streams[name].write(value + "\n")
        self.records.append(row)
        self.sample_ids.add(sample_id)
        return row

    def finalize(self):
        for stream in self.streams.values():
            stream.close()
        return _write_analysis(self.directory, self.records, self.metadata)


def plot_comparison(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No observations to plot")
    times = np.asarray([row["capture_time_s"] for row in records], dtype=float)
    truth = np.asarray([row.get("truth_position_m") or [np.nan] * 3 for row in records], dtype=float) * 1000
    camera = np.asarray([row.get("camera_position_m") or [np.nan] * 3 for row in records], dtype=float) * 1000
    error = np.asarray([np.linalg.norm(camera[i] - truth[i]) if _paired(row) else np.nan
                        for i, row in enumerate(records)])
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for axis, label, column in zip(axes[:3], ("X", "Y", "Z"), range(3)):
        axis.plot(times, truth[:, column], color="#333333", linewidth=1.2, label="Isaac Sim reference")
        axis.plot(times, camera[:, column], color="#007c91", linewidth=0.9, alpha=.85, label="RGB / ArUco estimate")
        axis.set_ylabel(f"{label} (mm)")
    axes[3].plot(times, error, color="#b53832", linewidth=.9)
    axes[3].set_ylabel("3D error (mm)")
    finite_error = error[np.isfinite(error)]
    axes[3].set_ylim(0, max(1.0, float(np.max(finite_error)) * 1.1) if len(finite_error) else 1.0)
    axes[3].set_xlabel("Simulation capture time (s)")
    axes[0].legend(loc="best", frameon=False, ncol=2, fontsize=9)
    groups = [_stage(row.get("phase", "")) for row in records]
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(times) - 1
        for axis in axes:
            if index % 2 == 0:
                axis.axvspan(times[start], times[end], color="#e5e8eb", alpha=.45, zorder=-2)
            axis.axvline(times[start], color="#a5a5a5", linewidth=.5, alpha=.6)
        axes[0].text((times[start] + times[end]) / 2, 1.02, groups[start],
                     transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9)
    for axis in axes:
        axis.grid(True, color="#d9d9d9", linewidth=.5, alpha=.7)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Camera Cube position versus independent simulation reference", fontsize=12)
    fig.supxlabel("Missing camera observations remain gaps; error statistics use matched pairs only.", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Full comparison.jsonl from a completed run")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.comparison.read_text().splitlines() if line.strip()]
    directory = args.comparison.parent
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = _write_analysis(directory, records, metadata)
    plot_comparison(records, directory / "comparison.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
