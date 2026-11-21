"""Run fresh, sequential U4 trials from the measured post-transfer checkpoint."""
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--left-unload", action="store_true",
                      help="Observe natural left unload at three offsets; repeat in order until three unloads or six attempts")
    mode.add_argument("--side-probe", action="store_true",
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
    if args.left_unload:
        trials += [(name + "_repeat", dy, None) for name, dy in TRIALS]
    reached_unload = 0
    failures = []
    for name, dy, timing in trials:
        if args.left_unload and reached_unload == len(TRIALS):
            break
        input_path = args.output / (name + ".json")
        input_path.write_text(json.dumps(checkpoint(source, dy), indent=2) + "\n")
        run = args.output / name
        with (args.output / (name + ".log")).open("w") as log:
            command = [args.isaac_python, SCENE / "two_row_physx.py", run,
                       "--resume", input_path, "--close-on-finish"]
            if timing:
                command += ["--u4-side-probe", timing]
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        report_path = run / "runner_report.json"
        if not report_path.is_file() or (process.returncode and not args.left_unload):
            if not args.side_probe:
                raise RuntimeError(f"{name}: capture exited {process.returncode}, inspect {name}.log")
            failures.append(name)
            continue
        report = json.loads(report_path.read_text())
        seated = any(e["phase"] == "upper_slot_complete" and e["status"] == "SEATED_STABLE"
                     and e.get("cube_id") == 10 for e in report["phases"])
        if args.left_unload and (seated or report["status"] == "STOPPED"):
            reached_unload += any(e["phase"] == "upper_B_side_unload" and e["status"] == "RUNNING"
                                  and e.get("cube_id") == 10 for e in report["phases"])
            continue
        if not seated:
            if not args.side_probe:
                raise RuntimeError(f"{name}: missing U4 SEATED_STABLE, inspect {name}.log")
            failures.append(name)
    if args.left_unload and reached_unload < len(TRIALS):
        raise SystemExit(f"Reached left unload in {reached_unload} of six attempts (raw records retained)")
    if failures:
        raise SystemExit("Incomplete U4 trials (raw records retained): " + ", ".join(failures))


if __name__ == "__main__":
    main()
