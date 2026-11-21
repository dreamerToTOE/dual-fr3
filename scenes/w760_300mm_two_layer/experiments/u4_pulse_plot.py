"""Plot measured displacement and force for two selected U4 probe records."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(results, trials):
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(10, 5.5))
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.76, wspace=.20, hspace=.16)
    styles = {"command": "#3c4451", "encoder": "#5772b4", "side": "#087d71", "wrist": "#ac532b"}
    for col, trial in enumerate(trials):
        run = results / trial
        pulse = [json.loads(line) for line in (run / "u4_side_probe.jsonl").read_text().splitlines()]
        episode = json.loads((run / "scores.json").read_text())["episode"]
        bearing = {True: "承载", False: "未承载", None: "待确认"}[episode["pre_end_loaded"]]
        first = pulse[0]["physics_step"]
        with gzip.open(run / "timeline.csv.gz", "rt") as stream:
            truth = {int(r["step"]): r for r in csv.DictReader(stream)}
        t = np.array([v["time_s"] - pulse[0]["time_s"] for v in pulse])
        pre_x = np.median([v["encoder_x_m"] for v in pulse if v["segment"] == "pre"])
        movement, forces = axes[:, col]
        movement.plot(t, [v["command_dx_m"] * 1e3 for v in pulse], c=styles["command"], ls="--", label="右 TCP 指令")
        movement.plot(t, [(v["encoder_x_m"] - pre_x) * 1e3 for v in pulse], c=styles["encoder"], label="右 TCP 实际位移")
        side = [truth[v["physics_step"]]["truth_side_N"] for v in pulse]
        forces.plot(t, [float(f) if f else np.nan for f in side], c=styles["side"], lw=1.5, label="侧邻箱法向承载 · PhysX")
        forces.plot(t, [-v["right_wrench_world"][0] for v in pulse], c=styles["wrist"], lw=1.5, label="右腕 −Fx")
        for ax in (movement, forces):
            ax.axhline(0, color="#82909a", lw=.6, zorder=0)
            for section in ("out_neg", "cross_pos", "return", "post"):
                i = next(v["physics_step"] for v in pulse if v["segment"] == section) - first
                ax.axvline(t[i], color="#aab3bd", ls=":", lw=.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        movement.set_title(f"{trial}  ·  横移前{bearing}")
        forces.set_xlabel("从探测记录开始 (s)")
    axes[0, 0].set_ylabel("侧向位移 (mm)")
    axes[1, 0].set_ylabel("世界 X 方向力 (N)")
    handles, labels = [], []
    for ax in axes[:, 0]:
        hs, ls = ax.get_legend_handles_labels()
        handles.extend(hs)
        labels.extend(ls)
    fig.suptitle("右臂侧向探测的位移与受力", fontsize=15)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, .92), frameon=False)
    fig.savefig(results / "probe_response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("trials", nargs=2)
    args = parser.parse_args()
    plot(args.results, args.trials)
