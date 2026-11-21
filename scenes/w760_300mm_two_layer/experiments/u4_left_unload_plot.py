"""Plot the handoff against measured left-cup and side-neighbor bearing."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def values(rows, key):
    return np.array([float(r[key]) if r[key] else np.nan for r in rows])


def plot(root, continuous=False):
    trials = []
    for path in sorted(root.glob("*/timeline.csv.gz")):
        with gzip.open(path, "rt") as stream:
            rows = list(csv.DictReader(stream))
        handoff = [int(r["step"]) for r in rows if r["window"] == "handoff"]
        if not handoff:
            continue
        zero = handoff[0] - 1
        if continuous:
            report = json.loads((path.parent / "counts.json").read_text())["events"]
            event = report["from_unload_command_3"]
            onset = event["true_clear_onset_step"]
            if onset is None:
                continue
            label = f"{path.parent.name} · 撤力后 {(onset-zero)/120:.3f} s"
            reload = event["true_reload_confirm_step"]
            trials.append((label, [r for r in rows if onset-90 <= int(r["step"]) <= onset+432],
                           onset, (reload-onset)/120 if reload else None))
        else:
            view = [r for r in rows if zero - 48 <= int(r["step"]) <= zero + 300]
            trials.append((path.parent.name, view, zero, None))
    if not trials:
        raise ValueError("No side-loss event recorded" if continuous else "No handoff recorded")
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    width = max(8.5, 3.3*len(trials) + 1.5)
    fig, grid = plt.subplots(3, len(trials), figsize=(width, 6.5),
                             squeeze=False, sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [2.3, 1, 1.65]})
    fig.subplots_adjust(top=.81, bottom=.17, left=1.37/width, right=.98, wspace=.14, hspace=.14)
    for column, (name, rows, zero, reload) in enumerate(trials):
        force, vision, states = grid[:, column]
        t = (values(rows, "step") - zero) / 120
        force.plot(t, values(rows, "truth_side_N"), color="#007c72", lw=1.15, label="箱与侧邻箱 · PhysX 法向力")
        if continuous:
            force.plot(t, values(rows, "right_shear_x_N"), color="#ae4e19", lw=1.0, label="右腕 X 向力")
        else:
            force.plot(t, values(rows, "left_compression_N"), color="#ae4e19", lw=1.0, label="左腕 · 工具法向压力")
        vision.plot(t, 1000*values(rows, "side_gap_m"), color="#4164a1", lw=1., label="D455 与运动学融合的侧隙")
        event = reload if continuous else next((float(x) for x, r in zip(t, rows) if r["unload_event"] == "1"), None)
        for ax in (force, vision, states):
            ax.axvline(0, color="#555", ls="--", lw=.85)
            if event is not None:
                ax.axvline(event, color="#885384", ls=":", lw=1.1)
            ax.set_xlim((-.75, 3.6) if continuous else (-.4, 2.5))
            ax.spines[["top", "right"]].set_visible(False)
        force.axhspan(.1, 1, color="#b4e2d6", alpha=.35, zorder=0, label="目标弱载范围")
        vision.axhline(0, color="#aaa", lw=.6, zorder=0)
        labels = ["左杯真值", "侧邻箱真值", "被动融合", "连续融合" if continuous else "撤力曲线融合"]
        for y, key in enumerate(("truth_left_cup", "truth_side", "baseline_side", "fusion_side")):
            obs = values(rows, key)
            for mask, color in ((obs == 1, "#007c72"), (obs == 0, "#b8c5cc"), (np.isnan(obs), "#f0dfb7")):
                states.fill_between(t, 3-y-.32, 3-y+.32, where=mask, step="mid", color=color, linewidth=0)
        states.set(yticks=[3, 2, 1, 0], yticklabels=labels, ylim=(-.55, 3.6))
        states.set_xlabel("距首次物理侧失载起点 (s)" if continuous else "距左臂开始撤力的时间 (s)")
        force.set_title(name)
    grid[0, 0].set_ylabel("力 (N)")
    grid[1, 0].set_ylabel("侧隙 (mm)")
    title = "上层第 4 箱 · 换臂与后推时的侧载判断" if continuous else "上层第 4 箱 · 换臂时左腕撤力与箱间承载"
    fig.suptitle(title, fontsize=16)
    h, l = grid[0, 0].get_legend_handles_labels()
    h2, l2 = grid[1, 0].get_legend_handles_labels()
    fig.legend(h+h2, l+l2, loc="upper center", bbox_to_anchor=(.50, .94), ncol=4, frameon=False)
    caption = "绿：承载　灰：无载　浅黄：待确认\n"
    if continuous:
        caption += "竖虚线：PhysX 首次侧失载起点　紫色点线：PhysX 首次连续 3 步确认回载"
    else:
        caption += "竖虚线：撤力开始　紫色点线：撤力特征计算完成"
    fig.text(.52, .046, caption, ha="center", fontsize=9)
    fig.savefig(root / "handoff.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    plot(args.path, args.continuous)
