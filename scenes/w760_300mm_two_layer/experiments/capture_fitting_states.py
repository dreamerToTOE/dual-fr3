"""Record a markerless A or upper-row trial with directional PhysX labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("case", choices=("a", "u4", "u5"))
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checkpoint_data = args.resume.read_bytes()
    checkpoint = json.loads(checkpoint_data)
    checkpoint_sha = hashlib.sha256(checkpoint_data).hexdigest()
    if checkpoint["next_front"] != 6 or checkpoint["next_slot"] != {"a": 1, "u4": 4, "u5": 5}[args.case]:
        parser.error("The checkpoint does not match this trial")
    if (args.case == "a") == bool(checkpoint.get("landed")):
        parser.error("A starts at the supply. U4/U5 start at a measured landing.")

    # Load the research logger before entering the baseline worktree. Its
    # directional impulse evidence changes the reference stream only.
    research = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(research))
    import fr3_contact_monitor

    monitor = Path(fr3_contact_monitor.__file__)
    monitor_sha = hashlib.sha256(monitor.read_bytes()).hexdigest()
    base = args.base_checkout.resolve()
    scene = base / "scenes/w760_300mm_two_layer"
    sys.path[:0] = [str(scene), str(base)]
    import markerless_rows
    import markerless_tracking

    install = markerless_tracking.install

    def install_capture(source):
        if args.case == "a":
            anchor = "            save_upper_checkpoint(slot, landed_now)"
            finish = '\n            record("upper_A_capture", "LANDED_RELEASED_RETRACTED", cube_id=task_id)'
        else:
            anchor = "            save_upper_checkpoint(slot + 1)"
            finish = ""
        return markerless_rows.replace_once(install(source), anchor, anchor + finish +
                                            f'\n            report["reference_monitor_sha256"] = "{monitor_sha}"'
                                            f'\n            report["comparison_checkpoint_sha256"] = "{checkpoint_sha}"'
                                            '\n            report["status"] = "CAPTURED_FITTING_STATES"'
                                            '\n            return 0')

    markerless_tracking.install = install_capture
    sys.argv = [str(scene / "markerless_rows.py"), str(args.output.resolve()),
                "--resume", str(args.resume.resolve()), "--headless", "--close-on-finish"]
    if args.check_only:
        sys.argv.append("--check-only")
    return markerless_rows.main()


if __name__ == "__main__":
    raise SystemExit(main())
