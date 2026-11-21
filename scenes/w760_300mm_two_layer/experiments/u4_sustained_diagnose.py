"""Replay the archived +0.40 mm run around its first 54-step no-load bout."""
import argparse
from collections import Counter
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from fusion_upper5 import FRONT_LOAD_N
from u4_pulse_score import digest
from u4_contact_observer import U4ContactObserver
from u4_side_continuous import ContinuousSideObserver, PARAMETERS


ROOT = Path(__file__).with_name("results")
TRIAL = ROOT / "u4_side_continuous" / "forward04"


def stats(values):
    data = np.asarray(values)
    return {"min": data.min(axis=0).tolist(), "median": np.median(data, axis=0).tolist(),
            "max": data.max(axis=0).tolist(), "peak_to_peak": np.ptp(data, axis=0).tolist(),
            "last_minus_first": (data[-1] - data[0]).tolist()}


def normal(sensor, arm):
    return -float(np.dot(sensor[arm + "_wrench_world"][:3], sensor["cup_normals_world"][arm]))


def read_trial():
    with gzip.open(TRIAL / "timeline.csv.gz", "rt") as f:
        timeline = {int(r["step"]): r for r in csv.DictReader(f)}
    with gzip.open(TRIAL / "paired.jsonl.gz", "rt") as f:
        header = json.loads(next(f))
        paired = {}
        for line in f:
            sensor, reference = json.loads(line)
            step = sensor["physics_step"]
            if not (reference["physics_step"] == step and sensor["time_s"] == reference["time_s"]
                    == float(timeline[step]["time_s"])):
                raise ValueError(f"Misaligned packet at physical step {step}")
            paired[step] = sensor, reference
    if paired.keys() != timeline.keys() or header["source"] != TRIAL.name:
        raise ValueError("Paired stream does not match timeline")
    with gzip.open(TRIAL / "right_commands.json.gz", "rt") as f:
        targets = json.load(f)
    command = dict(zip(range(targets["first_step"], targets["last_step"] + 1), targets["target_y_m"], strict=True))
    if command.keys() != timeline.keys() or targets["source"] != TRIAL.name:
        raise ValueError("Right-arm command stream does not match timeline")
    return header, timeline, paired, command


def observations(steps, timeline, paired, command):
    data = [timeline[i] for i in steps]
    pairs = [next(p for p in paired[i][1]["cube_contacts"]["pairs"]
                  if {p["collider0"], p["collider1"]} == {"/World/Cube_10", "/World/Cube_8"})
             for i in steps]
    return {
        "steps": len(steps),
        "truth_side": dict(Counter(r["truth_side"] for r in data)),
        "passive_side": dict(Counter(r["baseline_side"] for r in data)),
        "fused_side": dict(Counter(r["fusion_side"] for r in data)),
        "continuous_reason": dict(Counter(r["continuous_side_reason"] for r in data)),
        "force_N": {k: stats([float(r[k]) for r in data]) for k in
                    ("truth_side_N", "truth_left_cup_x_N", "left_compression_N", "right_shear_x_N")},
        "right_normal_N": stats([normal(paired[i][0], "right") for i in steps]),
        "right_mz_Nm": stats([paired[i][0]["right_wrench_world"][5] for i in steps]),
        "tcp_world_um": {arm: stats([np.asarray(paired[i][0]["tcp_positions_m"][arm])*1e6 for i in steps])
                         for arm in ("left", "right")},
        "right_command_y_um": stats([command[i]*1e6 for i in steps]),
        "truth_box_world_um": stats([np.asarray(paired[i][1]["cube_position_m"])*1e6 for i in steps]),
        "truth_side_pair_separation_um": stats([p["separation_m"]*1e6 for p in pairs]),
        "truth_side_pair_touching_steps": sum(p["touching"] for p in pairs),
        "fused_side_gap_mm": stats([float(r["side_gap_m"])*1e3 for r in data if r["side_gap_m"]]),
        "continuous_score": stats([float(r["continuous_side_score"]) for r in data
                                   if r["continuous_side_score"]]),
    }


def diagnosis():
    header, timeline, paired, command = read_trial()
    hz = header["hz"]
    events = json.loads((TRIAL / "counts.json").read_text())["events"]
    time_zero = events["unload_command_start_step"]
    first = events["from_unload_command_3"]
    sustained = events["from_unload_command_12"]
    first_clear, clear_end = first["true_clear_onset_step"], first["clear_bout_last_observed_step"]
    clear = range(first_clear, clear_end + 1)
    if len(clear) != 54 or any(timeline[i]["truth_side"] != "0" or float(timeline[i]["truth_side_N"])
                              or timeline[i]["fusion_side"] == "0" for i in clear):
        raise ValueError("The archived 54-step loss miss did not reproduce")

    passive = U4ContactObserver(header["config"])
    continuous = ContinuousSideObserver()
    for i in range(min(paired), clear_end+1):
        sample = dict(paired[i][0], right_command_y_m=command[i])
        passive_result = passive.update(sample, neighbors=header["optical_neighbors"])
        result = continuous.update(sample, passive_result)
        if i in clear:
            for label, value in (("baseline_side", passive_result["side_loaded"]),
                                 ("fusion_side", result["side_loaded"])):
                if timeline[i][label] != {True: "1", False: "0", None: ""}[value]:
                    raise ValueError(f"Prediction replay differs at {i} for {label}")
            score = result["continuous_side_score"]
            archived = timeline[i]["continuous_side_score"]
            if (score is None) != (archived == "") or (score is not None
                                                       and not np.isclose(score, float(archived))):
                raise ValueError(f"Model score replay differs at {i}")

    after = events["left_cup_clear_confirm_step"]
    first_reload = first["true_reload_confirm_step"] - 2
    later_reload = sustained["true_reload_confirm_step"] - 11
    reload_end = later_reload
    while timeline[reload_end + 1]["truth_side"] == "1":
        reload_end += 1
    match_context = [i for i in range(clear_end + 1, first_reload + 53)
                     if timeline[i]["truth_side"] == "1" and timeline[i]["truth_left_cup"] == "0"]
    if not match_context:
        raise ValueError("No adjacent bearing windows with left cup clear")
    short_returns = [len(list(bout)) for truth, bout in groupby(
        (timeline[i]["truth_side"] for i in range(clear_end + 1, later_reload)))
        if truth == "1"]

    clear_t0, clear_t1 = (paired[step][0]["time_s"] for step in (first_clear, clear_end))
    cameras = [s["camera"] for i in clear for s, _ in (paired[i],)
               if "camera" in s and s["camera"]["status"] == "detected"]
    fresh = [c for c in cameras if clear_t0 <= c["capture_time_s"] <= c["received_time_s"] <= clear_t1]
    exclusions = []
    for i in clear:
        if timeline[i]["continuous_side_reason"] != "no_current_evidence":
            continue
        current = normal(paired[i][0], "right")
        recent = float(np.median([normal(paired[j][0], "right") for j in range(i-2, i+1)]))
        if not (current < .18 or recent < FRONT_LOAD_N):
            raise ValueError("Previously unclassified reason for no feature")
        exclusions.append({"step": i, "current_right_normal_N": current, "recent_median_right_normal_N": recent})
    report = {
        "source": "u4_side_continuous/forward04",
        "source_sha256": {name: digest(TRIAL / name) for name in
                          ("paired.jsonl.gz", "timeline.csv.gz", "counts.json", "right_commands.json.gz")},
        "calibration_sha256": digest(PARAMETERS),
        "matched_old_model_replay_steps": len(clear),
        "physics_hz": hz,
        "unload_command_start_step": time_zero,
        "first_loss_steps": [first_clear, clear_end],
        "first_loss_duration_s": len(clear)/hz,
        "left_cup_clear_confirm_step": after,
        "first_3step_reload_steps": [first_reload, first["true_reload_confirm_step"]],
        "first_12step_reload_steps": [later_reload, reload_end],
        "bearing_bout_counts_between_first_loss_and_12step_reload": dict(sorted(Counter(short_returns).items())),
        "preceding_12_steps": observations(range(first_clear-12, first_clear), timeline, paired, command),
        "54step_loss": observations(clear, timeline, paired, command),
        "loss_after_left_cup_confirmed_clear": observations(range(after, clear_end+1), timeline, paired, command),
        "nearby_bearing_with_left_cup_clear_step_ids": match_context,
        "nearby_bearing_with_left_cup_clear": observations(match_context, timeline, paired, command),
        "only_three_step_reload": observations(range(first_reload, first_reload + 3), timeline, paired, command),
        "sustained_reload_first_12steps": observations(range(later_reload, later_reload + 12), timeline, paired, command),
        "fresh_camera_frames_received_during_loss": len(fresh),
        "first_fresh_camera_delivery_after_loss_ms": (fresh[0]["received_time_s"]-clear_t0)*1e3,
        "fresh_camera_box_world_um": stats([np.asarray(c["position_m"])*1e6 for c in fresh]),
        "fresh_camera_transport_ms": stats([(c["received_time_s"] - c["capture_time_s"])*1e3 for c in fresh]),
        "loss_steps_excluded_by_right_normal_load_gate": exclusions,
    }
    return report, timeline, paired


def plot(path, report, timeline, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    first, end = report["first_loss_steps"]
    after = report["left_cup_clear_confirm_step"]
    hz = report["physics_hz"]
    steps = np.arange(first-12, end+21)
    t = (steps-first)/hz
    fig, ax = plt.subplots(4, 1, figsize=(8.4, 8), sharex=True)
    fig.subplots_adjust(left=.15, right=.85, hspace=.33, top=.92, bottom=.13)
    for a in ax:
        a.axvspan(0, (end-first+1)/hz, color="#e9e1f4", alpha=.6)
        a.axvline((after-first)/hz, color="#7961a2", linestyle=":", linewidth=1)
        a.spines[["top", "right"]].set_visible(False)
    def series(key):
        return [float(timeline[i][key]) if timeline[i][key] else np.nan for i in steps]
    ax[0].plot(t, series("truth_side_N"), label="两箱侧载 · PhysX", color="#007c72")
    ax[0].plot(t, series("left_compression_N"), label="左腕法向力", color="#a7521f")
    ax[1].plot(t, series("right_shear_x_N"), label="右腕 X 向力", color="#a7521f")
    ax[1].plot(t, [normal(paired[i][0], "right") for i in steps], label="右腕法向力", color="#4164a1")
    ax[2].plot(t, series("continuous_side_score"), label="旧连续判据分数", color="#754397")
    camera_steps = [i for i in steps if paired[i][0].get("camera", {}).get("status") == "detected"]
    ax[2].scatter([(i-first)/hz for i in camera_steps], [-2.65]*len(camera_steps),
                  label="有视觉检测送达", color="#4164a1", s=12)
    model = json.loads(PARAMETERS.read_text())["model"]
    for edge in (model["clear_below"], model["load_above"]):
        ax[2].axhline(edge, color="#777", linestyle="--", alpha=.6)
    for arm, a, scale, col in (("left", ax[3], 1e3, "#a7521f"), ("right", ax[3].twinx(), 1e6, "#4164a1")):
        x = np.array([paired[i][0]["tcp_positions_m"][arm][0] for i in steps])
        a.plot(t, (x-x[12])*scale, color=col)
        side, unit = ("左", "mm") if arm == "left" else ("右", "µm")
        a.set_ylabel(f"{side} TCP 的 X 位移 ({unit})", color=col)
        a.tick_params(axis="y", labelcolor=col)
        a.set_xlim(t[0], t[-1])
    ax[0].set_ylabel("力 (N)")
    ax[1].set_ylabel("力 (N)")
    ax[2].set_ylabel("分数")
    for a in ax[:3]:
        a.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    ax[3].set_xlabel("距 PhysX 箱间失载起点 (s)")
    fig.suptitle("旧 +0.40 mm 试次 · 连续 54 步无箱间侧载", fontsize=14)
    fig.text(.50, .038, "浅紫：箱间失载　竖点线：左杯连续 3 步确认卸载　横虚线：旧分数的两条判据边界",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report, timeline, paired = diagnosis()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(args.plot, report, timeline, paired)
