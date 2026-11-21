"""Run three fresh, sequential U4 trials from the measured post-transfer checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCENE = Path(__file__).resolve().parents[1]
TRIALS = (("nominal", 0), ("forward04", .0004), ("back04", -.0004))


def checkpoint(source, dy):
    data = json.loads(source)
    if data["next_slot"] != 4 or data["landed"]["cube_id"] != 10:
        raise ValueError("Expected the U4 post-transfer checkpoint")
    data["fixture_last_slot"] = 4
    data["landed"]["position_m"][1] += dy
    data["u4_initial_condition"] = {
        "source_checkpoint_sha256": hashlib.sha256(source).hexdigest(),
        "active_box_world_Y_offset_m": dy,
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaac_python", type=Path)
    parser.add_argument("output", type=Path, help="New directory for independent physical runs")
    parser.add_argument("--side-probe", action="store_true",
                        help="Run early (3), late (2), a zero-amplitude sham and near-wall (2) trials")
    args = parser.parse_args()
    source = (SCENE / "validation/upper4_after_transfer_checkpoint.json").read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    trials = [(name, dy, None) for name, dy in TRIALS]
    if args.side_probe:
        trials = [(when + "_" + name, dy, when)
                  for when, subset in (("early", TRIALS), ("late", TRIALS[:2]),
                                       ("sham", TRIALS[:1]), ("near", TRIALS[:2]))
                  for name, dy in subset]
    failures = []
    for name, dy, timing in trials:
        input_path = args.output / (name + ".json")
        input_path.write_text(json.dumps(checkpoint(source, dy), indent=2) + "\n")
        run = args.output / name
        with (args.output / (name + ".log")).open("w") as log:
            command = [args.isaac_python, SCENE / "two_row_physx.py", run,
                       "--resume", input_path, "--close-on-finish"]
            if timing:
                command += ["--u4-side-probe", timing]
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode or not (run / "runner_report.json").is_file():
            if not args.side_probe:
                raise RuntimeError(f"{name}: capture exited {process.returncode}, inspect {name}.log")
            failures.append(name)
            continue
        report = json.loads((run / "runner_report.json").read_text())
        if not any(e["phase"] == "upper_slot_complete" and e["status"] == "SEATED_STABLE"
                   and e.get("cube_id") == 10 for e in report["phases"]):
            if not args.side_probe:
                raise RuntimeError(f"{name}: missing U4 SEATED_STABLE, inspect {name}.log")
            failures.append(name)
    if failures:
        raise SystemExit("Incomplete U4 trials (raw records retained): " + ", ".join(failures))


if __name__ == "__main__":
    main()
