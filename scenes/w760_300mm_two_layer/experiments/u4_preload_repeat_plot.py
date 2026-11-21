"""Plot sensor compression and PhysX side load around independent U4 handoffs."""
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(root, history):
    all_cases = json.loads((root / "handoff_summary.json").read_text())["trials"]
    old_events = json.loads((history / "counts.json").read_text())["events"]
    old_offset = old_events["unload_command_start_step"]
    old_loss = old_events["from_unload_command_3"]
    start = old_loss["true_clear_onset_step"]
    end = old_loss["clear_bout_last_observed_step"]
    duration = end - start + 1
    captures = [(f"历史 · 首次失载 {duration} 步", history, old_offset, (start, end))]
    for case in all_cases:
        if case["handoff_observed"]:
            loss = case["first_sustained_handoff_loss"]
            span = (loss["start"], loss["end"]) if loss else None
            captures.append((case["source"], root / case["source"], case["handoff_start_step"], span))
    columns = min(len(captures), 4)
    lines = (len(captures) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(3.45 * columns, 3.25 * lines + 1.0),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(top=.78 if lines == 1 else .84, left=.066, right=.985,
                        bottom=.18 if lines == 1 else .1, wspace=.18, hspace=.43)
    for ax, (label, path, offset, span) in zip(axes.ravel(), captures):
        with gzip.open(path / "timeline.csv.gz", "rt") as source:
            rows = [r for r in csv.DictReader(source) if offset - 24 <= int(r["step"]) <= offset + 210]
        t = [(int(r["step"]) - offset) / 120 for r in rows]
        for key, color, title in (
                ("left_compression_N", "#ac541f", "左杯法向压力 · 腕力"),
                ("truth_side_N", "#08786e", "两箱间侧向力 · PhysX")):
            ax.plot(t, [float(r[key]) if r[key] else np.nan for r in rows], color=color,
                    lw=1.25, label=title)
        ax.axvline(0, color="#777", ls="--", lw=.75)
        if span:
            start, end = span
            ax.axvspan((start - offset) / 120, (end + 1 - offset) / 120, color="#dcd5f4",
                       zorder=0, label="首次持续失载")
        ax.set(xlabel="撤力起点为 0 (s)", title=label, xlim=(-.2, 1.75))
        ax.spines[["top", "right"]].set_visible(False)
    for extra in list(axes.ravel())[len(captures):]:
        extra.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("力 (N)")
    fig.suptitle("上层第 4 箱 · +0.40 mm 换臂受力对照", fontsize=15, y=.96)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(.5, .915))
    fig.savefig(root / "handoff_forces.png", dpi=170)


if __name__ == "__main__":
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    draw(Path(sys.argv[1]), Path(sys.argv[2]))
