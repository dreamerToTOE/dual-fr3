"""Measured first-front B1 boundary capture and tactile re-acquisition."""
import json
from textwrap import indent


from source_hooks import replace_once


def write_state(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def inject(source):
    # Registered body and neighbor transforms are restored before World.reset by the normal checkpoint loader.
    source = replace_once(source, '                    q = np.asarray(saved)\n                    robots[side].set_joint_positions',
                          '''                    q = np.asarray(saved)
                    if CHECKPOINT.get("front_b1") and side == CHECKPOINT["front_b1"]["support_arm"]:
                        tip, attitude = tcp_fk(side, q)
                        q = cartesian_joint_target(side, q, tip - [0., .004, 0.], attitude)
                    robots[side].set_joint_positions''')
    source = replace_once(source,'                record("front_A", "RESTORED_MEASURED_LANDING", slot=front_slot)',
                          '''                if CHECKPOINT.get("front_b1"):
                    if np.linalg.norm(control_center - CHECKPOINT["front_b1"]["rear_rgb"]) > .005:
                        raise RuntimeError("Reacquired RGB pose differs from the archived B1 handoff by over 5 mm")
                    record("front_B1_resume", "RESTORED_WITH_CUPS_CLEAR", slot=front_slot,
                           source_step=CHECKPOINT["front_b1"]["source_step"])
                else:
                    record("front_A", "RESTORED_MEASURED_LANDING", slot=front_slot)''')
    start = source.index('            # The completed A phase leaves the tools parked')
    end = source.index('            if staged:\n                record("B1_push"',start)
    resume = '''            handoff = CHECKPOINT.get("front_b1") if slot == START_FRONT else None
            if handoff:
                if support_side != handoff["support_arm"] or any(grippers.is_closed(a) for a in robots):
                    raise RuntimeError("Expected open side cups at the front B1 boundary")
                if push_side in stations and not np.isclose(rail_states[push_side][1], stations[push_side]):
                    base = rail_states[push_side].copy()
                    base[1] = stations[push_side]
                    move_robot_base(push_side, base)
                cube_start = np.asarray(handoff["cube_start_rgb"])
                b1_actual, rear_rotation = observe_b_cube("front_B1_resume_before_touch")
                staged = "staging_offset_from_rear_m" in current_task
                front_height = .004
                front_rotation = np.asarray(FRONT_ROTATION) if support_side == "left" else np.diag([-1., 1., -1.])
                initial_rear = overlapping_rear(b1_actual, rear_rotation)
                if not initial_rear:
                    raise RuntimeError("No visible rear box in the restored B1 lane")
                b1_target[1] = min(front_planes[name] for name in initial_rear) - half
                if staged and abs(b1_actual[1] - (b1_target[1] - current_task["staging_offset_from_rear_m"])) > .008:
                    raise RuntimeError("This B1 checkpoint belongs to a different staging lane; resume at A landing")
                contact_plan_rotation = rear_rotation.copy()
                allowed_contacts.add(support_side)
                push_faces[support_side] = (1, -rear_rotation[:, 1], np.array([0., -half, front_height]))
                touch_face(support_side, rear_rotation[:, 1], .012, "front_B1_resume_touch")
                stable = 0
                for _ in range(240):
                    step()
                    check_current_collision("front_B1_resume", contact_uncertainty_m=.005)
                    pairs = contact_monitor.cube_contact_snapshot()["pairs"]
                    loads = {p["label"] for p in pairs if p["current_step_observed"] and (p["normal_impulse_Ns"] or 0.) > 1e-5}
                    force = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
                    stable = stable + 1 if {"floor",support_side + "_cup"} <= loads and push_side + "_cup" not in loads and .4 <= force <= 9. else 0
                    if stable == 18:
                        break
                else:
                    raise RuntimeError("Could not recover floor and support-cup contact at stored B1 boundary")
                # The support wrist covers the marker after touch. Y is now measured at the loaded front face.
                b1_actual, rear_rotation = sensed_cube()
                report["target_observations"].append({"phase": "front_B1_resume_verified",
                    "source": "precontact_RGB_and_loaded_wrist_FK", "position_m": b1_actual.tolist()})
                if np.max(np.abs(joint_state()[push_side] - CHECKPOINT["parked_q"][push_side])) > .012:
                    raise RuntimeError("Passive arm did not remain at its B1 checkpoint")
                if np.linalg.norm(b1_actual - handoff["rear_rgb"]) > .005:
                    raise RuntimeError("Reacquiring the support cup displaced the restored B1 cube by over 5 mm")
                b1_progress = float(b1_actual[1] - cube_start[1])
                record("front_B1_resume", "FLOOR_AND_SUPPORT_CUP_REACQUIRED", source_step=handoff["source_step"],
                       normal_N=float(force), stable_ticks=stable)
            else:
'''
    regular = source[start:end]
    save = '''            current_pairs = contact_monitor.cube_contact_snapshot()
            boundary_support = {p["label"] for p in current_pairs["pairs"] if p["current_step_observed"]
                                and (p["normal_impulse_Ns"] or 0.) > 1e-5}
            if (slot in (1, 2, 3, 4) and not handoff and current_pairs["valid"] and {"floor", support_side + "_cup"} <= boundary_support
                    and push_side + "_cup" not in boundary_support and not any(grippers.is_closed(a) for a in robots)
                    and .5 < -last_force_sample[support_side]["tcp_gravity_corrected_world"][1] < 9.):
                from front_b1_checkpoint import write_state
                state = {"scene": SCENE, "next_front": slot, "next_slot": 1, "source_run": str(output_directory),
                         "initial_boxes": upper_pose_records(lower_bodies), "rear_initial": rear_initial,
                         "front_completed": report["front_row"]["completed"],
                         "front_landed": upper_pose_records({str(current_task["cube_id"]): cube})[0],
                         "parked_q": {a:q.tolist() for a,q in joint_state().items()},
                         "parked_base_y_m": {a:float(base[1]) for a,base in rail_states.items()},
                         "front_b1": {"source_run": str(output_directory), "source_step":tick, "support_arm":support_side,
                                      "cube_start_rgb":cube_start.tolist(), "rear_rgb":b1_actual.tolist(),
                                      "b1_progress_m":b1_progress, "left_normal_N":float(-last_force_sample[support_side]["tcp_gravity_corrected_world"][1])}}
                write_state(output_directory / "front_b1_checkpoint.json", state)
'''
    source = source[:start] + resume + indent(regular, '    ') + save + source[end:]
    return replace_once(source, '                record("B1_push", "REACHED_WITH_PHYSX_CONTACT", progress_m=b1_progress,',
                        '                record("front_B1_boundary" if handoff else "B1_push",\n'
                        '                       "REACQUIRED_FROM_ARCHIVE" if handoff else "REACHED_WITH_PHYSX_CONTACT", progress_m=b1_progress,')
