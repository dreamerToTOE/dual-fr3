"""Draw the side and rear bearing traces for the three held-out U4 runs."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np


def plot(directory):
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    colors = ["#e0e4e6", "#11736d", "#efba63"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], len(colors))
    fig, axes = plt.subplots(3, 2, figsize=(12, 7))
    fig.subplots_adjust(left=.11, right=.98, bottom=.09, top=.83, hspace=.56, wspace=.30)
    for pair, (name, label) in zip(axes, (("nominal", "初始 Y 原位"), ("forward04", "初始 Y +0.40 mm"),
                                          ("back04", "初始 Y −0.40 mm"))):
        score = json.loads((directory / name / "scores.json").read_text())
        bounds = score["phase_steps_for_scoring_only"]
        with gzip.open(directory / name / "timeline.csv.gz", "rt") as f:
            rows = [r for r in csv.DictReader(f) if bounds["handoff"] < int(r["step"]) <= bounds["complete"]]
        if not rows:
            raise ValueError(f"Missing handover in {name}")
        hz = score["hz"]
        ticks = np.array([int(r["step"]) for r in rows])
        if np.any(np.diff(ticks) != 1):
            raise ValueError(f"Noncontiguous timeline in {name}")
        tracks = ["truth_side", "fusion_side", "truth_rear", "fusion_rear"]
        values = np.array([[{"0": 0, "1": 1, "": 2}[r[key]] for r in rows] for key in tracks])
        rear = np.flatnonzero((ticks > bounds["rear_stroke"]) & (values[2] == 1))
        for col, ax in enumerate(pair):
            origin = ticks[rear[0]] if col == 1 and len(rear) else bounds["rear_stroke"]
            t = (ticks - origin) / hz
            edges = [t[0] - .5 / hz, t[-1] + .5 / hz, 3.5, -.5]
            ax.imshow(values, origin="upper", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, extent=edges)
            ax.set_yticks(range(4), ["侧邻箱 · 真值", "侧邻箱 · 融合", "后墙 · 真值", "后墙 · 融合"])
            ax.axvline(0, lw=1, ls="--", color="#373c43")
            ax.axhline(1.5, color="white", lw=2)
            ax.set_title(label if col == 0 else ("后墙首次受力附近" if len(rear) else "未观察到后墙承载"), fontsize=11)
            if col == 1 and len(rear):
                ax.set_xlim(-1.5, 1.5)
    axes[-1, 0].set_xlabel("相对右臂开始向后推的时间 (s)")
    axes[-1, 1].set_xlabel("相对后墙首次受力的时间 (s)")
    fig.suptitle("上层第 4 箱换臂后的承载判断", fontsize=16)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, ("未承载", "承载", "待确认"))],
               ncol=3, loc="upper center", bbox_to_anchor=(.5, .935), frameon=False)
    fig.savefig(directory / "handover.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    plot(Path(sys.argv[1]))
