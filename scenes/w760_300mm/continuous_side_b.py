"""Integrate the checked high-wall B postures into the continuous row runner."""
from pathlib import Path
import json

from source_hooks import replace_once


HERE = Path(__file__).resolve().parent


def patch_b(source):
    seeds = json.loads((HERE / "b_postures.json").read_text())
    outer = {int(slot): posture for slot, posture in seeds["outer"].items()}
    third = seeds["third"]
    source = replace_once(source, '        vision_camera.set_projection_type("pinhole")', '''        vision_camera.set_projection_type("pinhole")
        # The main rear view is blocked by Cube 1's short support approach.
        side_rear_prim = UsdGeom.Camera.Define(stage, "/World/LeftBayObservationCamera")
        side_rear_position = [.36, *vision_cfg["camera_position_m"][1:]]
        side_rear_prim.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*side_rear_position), Gf.Vec3d(*vision_cfg["camera_look_at_m"]),
            Gf.Vec3d(0, 0, 1)).GetInverse())
        side_rear_prim.CreateClippingRangeAttr(vision_camera_prim.GetClippingRangeAttr().Get())
        side_rear_prim.CreateFocalLengthAttr(vision_camera_prim.GetFocalLengthAttr().Get())
        side_rear_prim.CreateHorizontalApertureAttr(vision_camera_prim.GetHorizontalApertureAttr().Get())
        side_rear_prim.CreateVerticalApertureAttr(vision_camera_prim.GetVerticalApertureAttr().Get())
        side_rear_camera = world.scene.add(IsaacCamera(
            prim_path="/World/LeftBayObservationCamera", name="left_bay_observation_camera",
            frequency=vision_cfg["frequency_hz"], resolution=tuple(vision_cfg["resolution"])))
        side_rear_camera.set_projection_type("pinhole")''')
    source = replace_once(source, '        supply_camera.pause()\n        if vision_viewport is not None:',
                          '        supply_camera.pause()\n        side_rear_camera.pause()\n        if vision_viewport is not None:')
    source = replace_once(source, '        vision_estimator = make_vision_estimator(camera_matrix, camera_to_world)',
                          '''        vision_estimator = make_vision_estimator(camera_matrix, camera_to_world)
        side_rear_to_world = np.linalg.inv(np.asarray(side_rear_camera.get_view_matrix_ros(), dtype=float))
        side_rear_intrinsics = np.asarray(side_rear_camera.get_intrinsics_matrix(), dtype=float)
        side_rear_estimator = make_vision_estimator(side_rear_intrinsics, side_rear_to_world)
        primary_rear_view = vision_camera, vision_estimator
        report["vision"]["left_bay_camera"] = {
            "position_m": side_rear_position, "intrinsics": side_rear_intrinsics.tolist(),
            "camera_to_world_ros": side_rear_to_world.tolist(), "scope": "Cube 1 rear contact views"
        }''')
    source = replace_once(source, '        supported_pose_filter = PlanarPoseMedian()',
                          '        supported_pose_filter = PlanarPoseMedian()\n        last_supported_capture = None\n'
                          '        withdrawing_corner_tools = False')
    source = replace_once(source,
                          '            nonlocal latest_camera_observation, control_center, control_rotation, supported_pose_time',
                          '            nonlocal latest_camera_observation, control_center, control_rotation, supported_pose_time\n'
                          '            nonlocal last_supported_capture')
    source = replace_once(source, '            if push_faces and receive_time - capture_time <= .20:',
                          '            if push_faces and not withdrawing_corner_tools and 0 <= receive_time - capture_time <= .20:')
    source = replace_once(source,
                          '                        and raw_rotation[2, 2] >= math.cos(math.radians(35))):\n'
                          '                    supported_pose = vision_estimator.resting_pose(observation, CONFIG["floor_top_m"])',
                          '''                        and raw_rotation[2, 2] >= math.cos(math.radians(18))):
                    estimated_center = observation.get("cube_position_m")
                    floor_center_z = CONFIG["floor_top_m"] + CONFIG["cube_size_m"] / 2
                    if estimated_center is not None and abs(estimated_center[2] - floor_center_z) <= .040:
                        supported_pose = vision_estimator.resting_pose(observation, CONFIG["floor_top_m"])
                    else:
                        latest_camera_observation["control_pose_rejection"] = "marker_depth_off_support"''')
    source = replace_once(source, '                if supported_pose is not None:\n                    filtered_pose',
                          '''                if supported_pose is not None:
                    if last_supported_capture is None or capture_time - last_supported_capture > .09:
                        supported_pose_filter.reset()
                    last_supported_capture = capture_time
                    filtered_pose''')
    source = replace_once(source, '                if camera is supply_camera:\n                    camera.pause()',
                          '                if camera is supply_camera or (camera is side_rear_camera and camera is not vision_camera):\n'
                          '                    camera.pause()')
    source = replace_once(source,
                          '"phase": phase, "source": "supply_rgb" if camera is supply_camera else "rear_rgb",',
                          '"phase": phase, "source": ("supply_rgb" if camera is supply_camera else\n'
                          '                                           "left_bay_rgb" if camera is side_rear_camera else "rear_rgb"),')
    source = replace_once(source, '        def step(count=1):', '''        def observe_b_cube(phase):
            if current_task["cube_id"] == 1 and phase in ("B2_rear_pose", "B2_approach_pose"):
                side_rear_estimator.marker_id = vision_estimator.marker_id
                return observe_resting_cube(phase, side_rear_camera, side_rear_estimator, CONFIG["floor_top_m"])
            return observe_resting_cube(phase, vision_camera, vision_estimator, CONFIG["floor_top_m"])

        def step(count=1):''')
    source = replace_once(source, '            nonlocal seals_required, carry_offsets, contact_plan_rotation',
                          '            nonlocal seals_required, carry_offsets, contact_plan_rotation\n'
                          '            nonlocal vision_camera, vision_estimator, last_supported_capture, withdrawing_corner_tools')
    source = replace_once(source, '            move({support_side: q}, duration=1.0)\n            allowed_contacts.add(support_side)',
                          '''            move({support_side: q}, duration=1.0)
            if slot == 1:
                # Use the centered RGB frame to plan the approached face, then watch contact from the clear oblique view.
                vision_camera.pause()
                vision_camera, vision_estimator = side_rear_camera, side_rear_estimator
                vision_estimator.marker_id = current_task["marker_id"]
                vision_camera.resume()
                last_supported_capture = None
                record("B1_optical_view", "LEFT_BAY_CAMERA_ACTIVE")
            allowed_contacts.add(support_side)''')
    source = replace_once(source,
                          '                    "image_shape": list(rgba.shape),',
                          '                    "image_shape": list(rgba.shape),\n'
                          '                    "optical_view": "left_bay" if vision_camera is side_rear_camera else "rear_center",')
    source = replace_once(source, '                    y = left_hold.update(normal)', '''                    gain = left_hold.gain_m_per_Ns
                    if current_task["cube_id"] == 4 and contact_monitor.phase == "B2_push" and normal < .20:
                        left_hold.gain_m_per_Ns = 4e-4
                    try:
                        y = left_hold.update(normal)
                    finally:
                        left_hold.gain_m_per_Ns = gain''')
    source = replace_once(source, '        def settle_cube(phase):', '''        def park_for_final_rgb():
            record("B2_empty_repark", "RUNNING")
            move({side: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for side in robots})
            for side, base in rail_states.items():
                if not np.isclose(base[1], .20):
                    target = base.copy()
                    target[1] = .20
                    move_robot_base(side, target)

        def settle_cube(phase):''')
    begin = '''            if args.five_cube_row and abs(rail_states[support_side][1] - .40) > 1e-6:
                position, _ = tcp_fk(support_side, applied_targets[support_side])
                support_clearance_z = max(position[2], CONFIG["floor_top_m"] + half + .25)
                cartesian_move(support_side, [position[0], position[1], support_clearance_z],
                               2.0, "B1_support_lift")
                support_base = rail_states[support_side].copy()
                support_base[1] = .40
                move_robot_base(support_side, support_base)'''
    source = replace_once(source, begin, f'''            slot = current_task["cube_id"]
            outside_plan = {outer!r}.get(slot)
            third_box_seed = {third["precontact_q_rad"]!r}
            third_box_rail = {third["support_rail_y_m"]!r}
            support_y = outside_plan["support_y"] if outside_plan is not None else third_box_rail if slot == 3 else .40
            stations = {{support_side: support_y}}
            if outside_plan is not None:
                stations[push_side] = outside_plan["push_y"]
            # The completed A phase leaves the tools parked for empty rail travel.
            for side, y in stations.items():
                if not np.isclose(rail_states[side][1], y):
                    base = rail_states[side].copy()
                    base[1] = y
                    record("B1_rail_preparation", "RUNNING", arm=side, y_m=float(y))
                    move_robot_base(side, base)''')
    source = replace_once(source, '            front_height = .004 if staged else 0.',
                          '            front_height = .004 if staged or outside_plan is not None else 0.')
    source = replace_once(source,
                          '            q = solve(support_side, contact, front_tool_rotation, continuous=True, seed_override=b1_seed)',
                          '''            if outside_plan is not None or slot == 3:
                q = cartesian_joint_target(support_side,
                    np.asarray(outside_plan["b1_q"] if outside_plan is not None else third_box_seed),
                    contact, front_tool_rotation, retarget_guide=True)
            else:
                q = solve(support_side, contact, front_tool_rotation, continuous=True, seed_override=b1_seed)''')
    source = replace_once(source,
                          '''            b1_actual, rear_rotation = observe_resting_cube("B2_rear_pose", vision_camera, vision_estimator,
                                                           CONFIG["floor_top_m"])''',
                          '            b1_actual, rear_rotation = observe_b_cube("B2_rear_pose")')
    source = replace_once(source,
                          '''                position, _ = tcp_fk(support_side, applied_targets[support_side])
                cartesian_move(support_side, position + [0, 0, .004], 1.0, "B2_acquire")
                contact_plan_rotation = rear_rotation.copy()
                push_faces[support_side] = (1, np.array([0., -1., 0.]), np.array([0., -half, 0.]))''',
                          '''                if outside_plan is None:
                    position, _ = tcp_fk(support_side, applied_targets[support_side])
                    cartesian_move(support_side, position + [0, 0, .004], 1.0, "B2_acquire")
                contact_plan_rotation = rear_rotation.copy()
                push_faces[support_side] = (1, np.array([0., -1., 0.]), np.array([0., -half, front_height]))''')
    source = replace_once(source,
                          '''            b2_center, rear_rotation = observe_resting_cube("B2_approach_pose", vision_camera, vision_estimator,
                                                            CONFIG["floor_top_m"])''',
                          '            b2_center, rear_rotation = observe_b_cube("B2_approach_pose")')
    source = replace_once(source,
                          '            b2_transit_z = float(current_task.get("staged_transit_z_m", role_seeds["transit_z_m"]))',
                          '            b2_transit_z = 1.05 if outside_plan is not None else float(current_task.get("staged_transit_z_m", role_seeds["transit_z_m"]))')
    a = source.index('            position, _ = tcp_fk(push_side, applied_targets[push_side])',
                     source.index('            record("B2_high_approach"'))
    b = source.index('            outward = ', a)
    source = source[:a] + '            if outside_plan is None:\n' + ''.join(
        '    ' + line if line.strip() else line for line in source[a:b].splitlines(keepends=True)) + source[b:]
    source = replace_once(source,
                          '                    push_side, current_task.get("staged_precontact_q", case["precontact_q"][push_side]),',
                          '''                    push_side, (min((np.asarray(q) for q in outside_plan["b2_entry_q"]),
                                    key=lambda q: np.linalg.norm(tcp_fk(push_side, q)[0] - outside))
                                if outside_plan is not None else
                                current_task.get("staged_precontact_q", case["precontact_q"][push_side])),''')
    source = replace_once(source, '                move({push_side: descent[0]})', '''                if slot == 4:
                    record("B2_high_approach_home_via", "RUNNING", push_arm=push_side)
                    move({push_side: np.asarray(CONFIG["fr3"]["official_start_q_rad"])})
                move({push_side: descent[0]})''')
    source = replace_once(source, '            unload_normal(push_side, 0, push_sign, "B2_push_unload")', '''            if slot == 2:
                # At this confirmed corner seat only the tools move outward. The wrist can obscure the tag
                # until both tools clear; recheck the resting pose from an unobstructed view at the end.
                withdrawing_corner_tools = True
                record("B2_corner_withdrawal", "ENCODER_WRENCH_THEN_CLEAR_VIEW_RGB")
            unload_normal(push_side, 0, push_sign, "B2_push_unload")''')
    source = replace_once(source, '            settle_cube("B2_push_arm_release")', '''            if slot in (1, 2):
                # The remaining support wrist still covers the marker. Keep its force loop
                # active for the dwell, then require fresh stillness after both tools clear.
                record("B2_push_arm_release", "TACTILE_DWELL_BEFORE_EMPTY_REPARK")
                step(90)
            else:
                settle_cube("B2_push_arm_release")''')
    source = replace_once(source, '            max_speed = settle_cube("B2")', '''            park_for_final_rgb()
            if slot == 1:
                vision_camera.pause()
                vision_camera, vision_estimator = primary_rear_view
                vision_camera.resume()
                last_supported_capture = None
                record("B2_optical_view", "REAR_CENTER_CAMERA_RESTORED")
            max_speed = settle_cube("B2")
            if slot == 2:
                withdrawing_corner_tools = False
                if contact_plan_rotation[:, 0] @ control_rotation[:, 0] < math.cos(math.radians(3)):
                    raise RuntimeError("Clear-view RGB observed more than 3 degrees of Cube rotation after withdrawal")''')
    source = replace_once(source, '                speed = settle_cube("B2_center_cube")',
                          '                park_for_final_rgb()\n                speed = settle_cube("B2_center_cube")')
    return source
