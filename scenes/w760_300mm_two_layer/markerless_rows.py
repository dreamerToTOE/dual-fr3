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
    parser.add_argument("--stop-after-front", type=int, choices=range(1, 6),
                        help="Stop after the indicated lower front slot is verified")
    args = parser.parse_args()
    checkpoint = {}
    if args.resume:
        data = args.resume.read_bytes()
        checkpoint = json.loads(data)
        checkpoint["input_provenance"] = {"path": str(args.resume.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    front, upper, saved = start_state(checkpoint)
    if args.stop_after_front is not None and args.stop_after_front < front:
        parser.error("The requested stopping slot is already complete in the checkpoint")
    first_id = FRONT_IDS[front - 1] if front < 6 else 6 + upper
    output = args.output.resolve()
    source = rear_depth_only(build())
    if args.capture_layout:
        source = capture_source(source)
        source = replace_block(source, '        supply_estimator = make_vision_estimator(',
                               '        comparison_directory = ', '        supply_estimator = vision_estimator\n')
    else:
        from markerless_tracking import install
        source = install(source)
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
                      "STOP_AFTER_FRONT": args.stop_after_front, "INITIAL_BOXES": saved, "RUNTIME_GUIDES": row_guide_paths(),
                      "CHECKPOINT": checkpoint, "RELEASE_CHECKPOINT": checkpoint.get("front_landed") or checkpoint.get("landed"),
                      "SUPPORT_TOP_BEVEL_M": None, "UPPER_REST_OFFSET_M": None, "CUBE_CONTACT_OFFSET_M": None,
                      "capture_layout": capture_layout})


if __name__ == "__main__":
    raise SystemExit(main())
