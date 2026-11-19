"""Resume the final GUI acceptance segment at Cube 3 release."""
from source_hooks import replace_once


def patch_a3_park(source):
    source = replace_once(source, '"restored_cube_ids": [],', '"restored_cube_ids": [1, 2, 3],')
    source = replace_once(source, '        placed_bodies = [cube]\n', '''        def restore_box_pose(body, pose):
            body.set_world_pose(position=np.asarray(pose["cube_position_m"]),
                                orientation=np.asarray(pose["cube_orientation_wxyz"]))
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))

        placed_bodies = [cube]
        restore_box_pose(cube, A3_PARK["boxes"][0])
        from fr3_cube_supply import spawn_supply_cube
        prior_cube = spawn_supply_cube(stage, world, CONFIG, 1, contact_monitor)
        supply_transition = True
        step()
        prior_cube.initialize()
        restore_box_pose(prior_cube, A3_PARK["boxes"][1])
        placed_bodies.append(prior_cube)
        supply_transition = False
''')
    source = replace_once(source, '        report["completed_cubes"] = []\n',
                          '        report["completed_cubes"] = list(A3_PARK["completed"])\n')
    loop = '        for cube_index, current_task in enumerate(sequence):'
    source = replace_once(source, loop, loop + '\n            if cube_index < 2:\n                continue')
    start = source.index('                move({side: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for side in robots})',
                         source.index(loop))
    end = source.index('                from fr3_cube_supply import spawn_supply_cube', start)
    returning_to_supply = ''.join('    ' + line if line.strip() else line
                                  for line in source[start:end].splitlines(keepends=True))
    source = source[:start] + '                if cube_index > 2:\n' + returning_to_supply + source[end:]
    source = replace_once(source, '                placed_bodies.append(cube)\n', '''                placed_bodies.append(cube)
                if cube_index == 2:
                    restore_box_pose(cube, A3_PARK["boxes"][2])
''')
    source = replace_once(source, '        def run_cube():\n', '        def run_cube():\n            nonlocal control_center, control_rotation\n')
    start = source.index('            step(120)\n            record("A_approach",', source.index('        def run_cube():'))
    end = source.index('            set_contact_drive_mode(True)\n            allowed_contacts.discard("right")', start)
    original_a = ''.join('    ' + line if line.strip() else line
                         for line in source[start:end].splitlines(keepends=True))
    return source[:start] + '''            if current_task["cube_id"] == 3:
                estimate = A3_PARK["control_pose"]
                control_center = np.asarray(estimate["position_m"])
                control_rotation = np.asarray(estimate["rotation_world"])
                for side, saved_q in A3_PARK["q_rad"].items():
                    base = rail_states[side].copy()
                    base[1] = A3_PARK["rails"][side]
                    set_robot_base_pose(side, base)
                    q = np.asarray(saved_q)
                    robots[side].set_joint_positions(q, joint_indices=joints[side])
                    robots[side].set_joint_velocities(np.zeros(7), joint_indices=joints[side])
                    command_positions(side, q)
                allowed_contacts.clear()
                record("A3_landing_checkpoint", "RESTORED_MEASURED_RELEASE", source=A3_PARK["source"])
                step(60)
                park_after_side_transfer()
                step(120)
                cube_start, b1_rotation = observe_resting_cube("B1_landing_pose", vision_camera, vision_estimator,
                                                              CONFIG["floor_top_m"])
                landing_error = float(np.linalg.norm(cube_start - GUIDE["landing_m"]))
                if landing_error > .015:
                    raise RuntimeError(f"A release landing error {landing_error:.3f} m")
                record("A_release", "REACHED_AFTER_SUCTION_RELEASE", landing_error_m=landing_error)
            else:
''' + original_a + source[end:]
