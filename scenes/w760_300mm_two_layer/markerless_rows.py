"""Replay the high-wall packing scene with only the fixed rear RGB-D view."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from two_row_physx import FRONT_IDS, LOWER_IDS, REPO, SCENE, build, row_guide_paths, start_state
from single_cube_square import replace_block
from source_hooks import replace_once


def rear_depth_only(source):
    source = replace_block(source, '        if args.five_cube_row:\n            from fr3_cube_supply import add_cube_marker',
                           '        for side, base in CONFIG["fr3"]["bases"].items():')
    source = replace_once(source, '            add_cube_marker(stage, path, CONFIG["cube_size_m"], CONFIG["vision"]["aruco_marker_size_m"],\n'
                          '                            CONFIG["cube_sequence"][index - 1]["marker_id"])\n', '')
    source = replace_block(source, '        supply_prim = UsdGeom.Camera.Define(', '        if not args.headless:\n            import omni.ui')
    source = replace_once(source, '        supply_camera.pause()\n',
                          '        vision_camera.add_distance_to_image_plane_to_frame()\n')
    source = source.replace('        from fr3_cube_supply import spawn_supply_cube\n',
                            '        from fr3_cube_supply import spawn_supply_cube as spawn_marked_cube\n'
                            '        from functools import partial\n'
                            '        spawn_supply_cube = partial(spawn_marked_cube, add_marker=False)\n')
    return source


def capture_layout(camera, cube_id, cube, restored, step, render, output):
    """Save genuine rear RGB-D and a separate static reference for this layout."""
    frame = camera.get_current_frame(clone=True)
    for _ in range(12):
        step(8)
        render()
        frame = camera.get_current_frame(clone=True)
        depth = np.asarray(frame.get("distance_to_image_plane", []))
        if depth.ndim >= 2 and np.isfinite(depth).any() and int(frame["rendering_frame"]) > 0:
            break
    else:
        raise RuntimeError("Rear depth did not initialize")
    np.savez_compressed(output / "layout.npz", rgba=frame["rgba"], depth_m=depth)
    bodies = {**restored, str(cube_id): cube}
    poses = {}
    for name, body in bodies.items():
        position, orientation = body.get_world_pose()
        poses[name] = {"position_m": position.tolist(), "orientation_wxyz": orientation.tolist()}
    (output / "layout_reference.json").write_text(json.dumps(poses, indent=2) + "\n")


def capture_source(source):
    # The static layout clips finish before the first optical observation can affect motion.
    return replace_once(source, '        current_task = {"cube_id": FIRST_ID, "marker_id": FIRST_ID + 22}',
                        '''        current_task = {"cube_id": FIRST_ID, "marker_id": FIRST_ID + 22}
        capture_layout(vision_camera, FIRST_ID, cube, {**lower_bodies, **upper_bodies},
                       step, render_only, output_directory)
        report["status"] = "CAPTURED_MARKERLESS_LAYOUT"
        return 0''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-finish", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--capture-layout", action="store_true")
    parser.add_argument("--observed-skills", action="store_true",
                        help="Use the experimental observation-driven skill runner")
    parser.add_argument("--sensor-front-bearing", action="store_true",
                        help="Use the sensor-only rear bearing transition for front slots 3 and 4")
    parser.add_argument("--start-empty", action="store_true",
                        help="Execute rear-row supply and placement before front and upper rows")
    parser.add_argument("--stop-after-rear", type=int, choices=range(1, 6))
    parser.add_argument("--stop-after-front", type=int, choices=range(1, 6),
                        help="Stop after the indicated lower front slot is verified")
    args = parser.parse_args()
    checkpoint = {}
    if args.resume:
        data = args.resume.read_bytes()
        checkpoint = json.loads(data)
        checkpoint["input_provenance"] = {"path": str(args.resume.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    if args.start_empty and args.resume:
        parser.error("Use either --start-empty or --resume.")
    rear = 1 if args.start_empty else checkpoint.get("next_rear", 6)
    if rear not in range(1, 7):
        parser.error("Invalid next rear slot.")
    if not args.observed_skills and (rear < 6 or args.stop_after_rear is not None):
        parser.error("Rear-row execution requires --observed-skills.")
    if args.sensor_front_bearing and not args.observed_skills:
        parser.error("--sensor-front-bearing requires --observed-skills.")
    if args.stop_after_rear is not None and args.stop_after_rear < rear:
        parser.error("The requested rear slot is already complete.")
    if args.capture_layout and args.observed_skills:
        parser.error("Static layout capture uses the baseline runtime.")
    if rear < 6:
        front, upper, saved = 1, 1, checkpoint.get("initial_boxes", [])
        if ((checkpoint and checkpoint.get("scene") != SCENE)
                or [box["cube_id"] for box in saved] != list(range(1, rear))
                or [box["cube_id"] for box in checkpoint.get("rear_completed", [])] != list(range(1, rear))
                or (checkpoint.get("rear_landed") and checkpoint["rear_landed"]["cube_id"] != rear)):
            parser.error("The rear-row checkpoint must contain exactly the completed rear placements.")
    else:
        front, upper, saved = start_state(checkpoint)
    if args.stop_after_front is not None and args.stop_after_front < front:
        parser.error("The requested stopping slot is already complete in the checkpoint")
    first_id = rear if rear < 6 else FRONT_IDS[front - 1] if front < 6 else 6 + upper
    output = args.output.resolve()
    source = rear_depth_only(build())
    if args.capture_layout:
        source = capture_source(source)
        source = replace_block(source, '        supply_estimator = make_vision_estimator(',
                               '        comparison_directory = ', '        supply_estimator = vision_estimator\n')
    else:
        from markerless_tracking import install
        source = install(source)
        if args.observed_skills:
            from dynamic_withdraw import install as dynamic_withdraw
            from observed_b1 import install as observed_b1
            from observed_rear import install as observed_rear
            from observed_transport import install as observed_transport

            source = observed_rear(dynamic_withdraw(observed_b1(observed_transport(source))))
            if args.sensor_front_bearing:
                from observed_front_bearing import install as observed_front_bearing
                source = observed_front_bearing(source)
    snapshot = output / "runner_snapshot.py"
    runner = compile(source, str(snapshot), "exec")
    if args.check_only:
        print(f"Markerless runner compiled, {len(saved)} restored boxes, current Cube_{first_id}")
        return 0

    with open("/tmp/fr3-upper-one-kit.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True)
        snapshot.write_text(source)
        sys.path.insert(0, str(REPO))
        sys.argv = [str(REPO / "fr3_ab_visual_runner.py"), "--five-cube-row", "--output-dir", str(output),
                    "--collision-workers", "3"]
        if args.headless:
            sys.argv.append("--headless")
        if args.close_on_finish or args.capture_layout:
            sys.argv.append("--close-on-finish")
        exec(runner, {"__file__": sys.argv[0], "__name__": "__main__", "SCENE": SCENE, "LOWER_IDS": LOWER_IDS,
                      "FRONT_IDS": FRONT_IDS, "START_FRONT": front, "START_SLOT": upper, "FIRST_ID": first_id,
                      "START_REAR": rear, "STOP_AFTER_REAR": args.stop_after_rear,
                      "STOP_AFTER_FRONT": args.stop_after_front, "INITIAL_BOXES": saved, "RUNTIME_GUIDES": row_guide_paths(),
                      "CHECKPOINT": checkpoint, "RELEASE_CHECKPOINT":
                          checkpoint.get("rear_landed") or checkpoint.get("front_landed") or checkpoint.get("landed"),
                      "SUPPORT_TOP_BEVEL_M": None, "UPPER_REST_OFFSET_M": None, "CUBE_CONTACT_OFFSET_M": None,
                      "capture_layout": capture_layout})


if __name__ == "__main__":
    raise SystemExit(main())
