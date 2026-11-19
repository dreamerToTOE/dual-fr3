"""Run the 760 mm, 300 mm-wall five-cube row with native opposed side suction."""
import argparse
import json
from pathlib import Path
import sys

from continuous_side_b import patch_b
from continuous_side_capture import patch_capture
from resume_cube3 import patch_a3_park
from side_transfer import patch_side_transfer
from source_hooks import replace_once


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


GUIDE_NAMES = (
    "bilateral_side_fixed_c1.json",
    "bilateral_side_c2_depth_faces.json",
    "bilateral_side_c3.json",
    "bilateral_side_c4_depth_faces.json",
    "bilateral_side_fixed_c5.json",
)


def patch_a(source):
    source = patch_side_transfer(source)
    source = replace_once(source, '        world.reset()\n', '''        import carb.settings
        import omni.replicator.core as replicator
        import omni.timeline
        replicator.orchestrator.set_capture_on_play(False)
        playback = omni.timeline.get_timeline_interface()
        world.reset()
        playback.set_end_time(3600.)
        playback.set_auto_update(False)
        playback.play(0., 3600. * playback.get_time_codes_per_seconds(), False)
        timeline_interrupts = {int(omni.timeline.TimelineEventType.PAUSE), int(omni.timeline.TimelineEventType.STOP)}
        def log_timeline_interrupt(event):
            if event.type in timeline_interrupts:
                print("FR3_TIMELINE_INTERRUPT " + json.dumps({"type": event.type,
                    "physics_step": world.current_time_step_index, "current_s": playback.get_current_time(),
                    "end_s": playback.get_end_time(), "stopped": playback.is_stopped(),
                    "modal_tool_active": carb.settings.get_settings().get("/app/tools/modal_tool_active")}), flush=True)
        timeline_sub = playback.get_timeline_event_stream().create_subscription_to_pop(log_timeline_interrupt)
        playback.commit()
        report["playback"] = {"start_s": playback.get_start_time(), "end_s": playback.get_end_time(),
                              "playing": playback.is_playing(), "mode": "application-commanded physics clock"}
        print("FR3_TIMELINE " + json.dumps(report["playback"]), flush=True)
''')
    source = replace_once(source, 'if not args.headless and tick % 2 == 0:',
                          'if not args.headless and tick % 4 == 0:')
    source = replace_once(source, '"control_hz": 120, "render_hz": 60,',
                          '"control_hz": 120, "render_hz": 30,')
    source = replace_once(source, '        "scope": "automatic_articulation_visual_demo",',
                          '''        "scope": "W760 300mm wall continuous five-cube row; opposing vertical face suction only",
        "first_layer_side_only": True, "visual_mode": not args.headless,
        "continuous_run": {"source": "one timeline, original supply spawn", "restored_cube_ids": [],
                           "side_guides": SIDE_GUIDE_NAMES},''')
    source = replace_once(source, '            if not within_limits():\n                raise RuntimeError(f"{phase} time-parameterized trajectory exceeds joint limits")',
                          '''            if phase == "A_empty_repark" and replay_source == "online_planner" and not within_limits():
                # Independent arm legs hold the other arm still at each waypoint.
                times, combined = plan_joint_trajectory(np.hstack([all_paths[side] for side in sides]),
                    max_joint_speed=max_joint_speed, sample_hz=120.0, stop_at_waypoints=True)
                samples = {side: combined[:, i * 7:(i + 1) * 7] for i, side in enumerate(sides)}
            if not within_limits():
                raise RuntimeError(f"{phase} time-parameterized trajectory exceeds joint limits")''')
    source = replace_once(source, '            grasp_seeds = {side:', '''            for side, base in rail_states.items():
                target = base.copy()
                target[1] = GUIDE["rail_supply_y_m"][side]
                if abs(target[1] - base[1]) > 1e-6:
                    move_robot_base(side, target)
            grasp_seeds = {side:''')
    source = replace_once(source, '                "phase": name, "samples": frames + 1,',
                          '                "phase": name, "cube_id": current_task["cube_id"], "samples": frames + 1,')
    source = replace_once(source, '        for cube_index, current_task in enumerate(sequence):',
                          '''        for cube_index, current_task in enumerate(sequence):
            GUIDE = SIDE_GUIDES[current_task["cube_id"]]
            FIRST_LOADED = next(i for i, p in enumerate(GUIDE["phases"]) if p["carrying"])''')
    source = replace_once(source, '                CONFIG["task"]["phase_A"]["landing_center_m"][0] = landing_x',
                          '''                CONFIG["task"]["phase_A"]["landing_center_m"] = GUIDE["landing_m"]
                report.setdefault("side_a_guide_by_cube", {})[str(current_task["cube_id"])] = {
                    "landing_m": GUIDE["landing_m"], "grasp_faces_world": GUIDE["grasp_faces_world"],
                    "rail_supply_y_m": GUIDE["rail_supply_y_m"], "rail_landing_y_m": GUIDE["rail_landing_y_m"],
                }''')
    return replace_once(source, '            from fr3_row_acceptance import final_row_checks',
                        '''            report["final_physical_cube_poses"] = []
            for index, body in enumerate(placed_bodies, start=1):
                position, quaternion = body.get_world_pose()
                report["final_physical_cube_poses"].append({"cube_id": index,
                    "position_m": position.tolist(), "orientation_wxyz": quaternion.tolist()})
            from fr3_row_acceptance import final_row_checks''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-finish", action="store_true")
    parser.add_argument("--collision-workers", type=int, default=3)
    parser.add_argument("--resume-cube3", action="store_true",
                        help="Use the recorded Cube 3 release checkpoint from the GUI acceptance")
    args = parser.parse_args()

    guides = {}
    for index, name in enumerate(GUIDE_NAMES, start=1):
        guide = json.loads((HERE / name).read_text())
        assert guide["slot"] == index and guide["status"] == "CLEAR_GEOMETRY"
        a, b = [guide["grasp_faces_world"][side] for side in ("left", "right")]
        assert a[2] == b[2] == 0 and sum(abs(x) for x in a) == .075
        assert all(x == -y for x, y in zip(a, b, strict=True))
        guides[index] = guide
    production = REPO / "fr3_ab_visual_runner.py"
    source = patch_b(patch_a(production.read_text()))
    checkpoint = None
    if args.resume_cube3:
        checkpoint = json.loads((HERE / "validation" / "checkpoint_cube3_release.json").read_text())
        assert [box["cube_id"] for box in checkpoint["boxes"]] == [1, 2, 3]
        source = patch_a3_park(source)
    source = patch_capture(source)
    output = args.output.resolve()
    snapshot = output / "runner_snapshot.py"
    runner = compile(source, str(snapshot), "exec")
    if args.check_only:
        print("Continuous A/B, five vertical-face guides, viewport snapshots: compiled.")
        return
    output.mkdir(parents=True)
    snapshot.write_text(source)
    sys.path.insert(0, str(REPO))
    sys.argv = [str(production), "--five-cube-row", "--output-dir", str(output),
                "--collision-workers", str(args.collision_workers)]
    if args.headless:
        sys.argv.append("--headless")
    if args.close_on_finish:
        sys.argv.append("--close-on-finish")
    exec(runner, {"__file__": str(production), "__name__": "__main__", "SIDE_GUIDES": guides,
                  "SIDE_GUIDE_NAMES": list(GUIDE_NAMES), "A3_PARK": checkpoint})


if __name__ == "__main__":
    main()
