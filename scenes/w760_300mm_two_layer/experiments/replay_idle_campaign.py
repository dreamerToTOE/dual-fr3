"""Replay the audited skill gaps in each checkpoint-linked 15-box source run."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_skill_states import state_scores


def replay_source(root, manifest, output, isaac_dir, run, cube):
    destination = output / f"{run}_cube{cube}.json"
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = [
        sys.executable, str(Path(__file__).with_name("replay_skill_states.py")), str(root / run),
        str(destination), "--cube", str(cube), "--isaac-sim-dir", str(isaac_dir),
        "--idle-windows", str(manifest),
    ]
    subprocess.run(command, env=env, check=True)
    return json.loads(destination.read_text())


def initialize_scores():
    return defaultdict(Counter)


def collect_scores(summary, cube):
    return {key: Counter({(truth, estimate): count
                          for truth, row in mode["confusion"].items()
                          for estimate, count in row.items()})
            for key, mode in summary["cubes"][str(cube)]["modes"].items()}


def merge_scores(destination, source):
    for key, counts in source.items():
        destination[key].update(counts)


def replay_campaign(root, manifest_path, output, isaac_dir, workers):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sequence = {int(cube): label for cube, label in manifest["boxes"].items()}
    if sorted(sequence) != list(range(1, 16)) or len(set(sequence.values())) != 15:
        raise ValueError("The campaign manifest must name the 15 distinct boxes")
    tasks = [(run, cube) for run, entry in manifest["runs"].items()
             for cube in sorted({window["cube_id"] for window in entry["windows"]})]
    if {cube for _, cube in tasks} != set(sequence):
        raise ValueError("Some boxes are missing audited skill gaps")
    output.mkdir(parents=True, exist_ok=True)
    overall = initialize_scores()
    rows, boxes = defaultdict(initialize_scores), defaultdict(initialize_scores)
    provenance = {label: {"windows": [], "unknown_reference_steps": 0} for label in sequence.values()}
    sources = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(replay_source, root, manifest_path, output, isaac_dir, run, cube)
                for run, cube in tasks]
        for (run, cube), job in zip(tasks, jobs, strict=True):
            result = job.result()
            if result["score_scope"] != "between_skills_only":
                raise ValueError(f"Unexpected scoring scope for {run}, Cube_{cube}")
            counts = collect_scores(result, cube)
            meta = provenance[sequence[cube]]
            windows = result["idle_windows"]
            samples = sum(window["through_step"] - window["after_step"] for window in windows)
            unknown = sum(window["reference_counts"].get("UNKNOWN", 0) for window in windows)
            if any(sum(mode.values()) != samples - unknown for mode in counts.values()):
                raise ValueError(f"Scored steps do not match the audited gaps for {run}, Cube_{cube}")
            for window in windows:
                meta["windows"].append({"run": run, **window})
            meta["unknown_reference_steps"] += unknown
            sources[run] = {"status": result["runner_status"], "sha256": result["sources_sha256"]}
            merge_scores(boxes[sequence[cube]], counts)
            merge_scores(rows[sequence[cube][0]], counts)
            merge_scores(overall, counts)

    summary = {
        "scope": "between_skills_only",
        "window_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "sources": sources,
        "legacy_revision": result["legacy_revision"],
        "robot_sha256": result["robot_sha256"],
        "hz": result["hz"],
        "boxes": {label: {**provenance[label], "modes": {key: state_scores(counts)
                                                        for key, counts in boxes[label].items()}}
                  for label in sorted(boxes, key=lambda name: ("RFU".index(name[0]), name[1:]))},
        "rows": {row: {key: state_scores(counts) for key, counts in modes.items()} for row, modes in rows.items()},
        "overall": {key: state_scores(counts) for key, counts in overall.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--isaac-sim-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    replay_campaign(args.run_root, args.manifest, args.output, args.isaac_sim_dir, args.jobs)
