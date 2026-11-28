"""Check a front-row neighbor seat with D455 geometry, wrist force and measured TCP motion."""

import numpy as np

from source_hooks import replace_once


def contact_evidence(samples):
    """Compare the previous advance with the most recent 15 samples of a continuous seek."""
    if len(samples) < 30:
        return None
    positions, normals = np.asarray(samples, dtype=float).T
    preceding_advance = positions[14] - positions[0]
    recent_span = np.ptp(positions[15:])
    median_force = np.median(normals[15:])
    if preceding_advance >= 25e-6 and recent_span <= 12e-6 and median_force >= 5.:
        return {
            "preceding_advance_m": float(preceding_advance),
            "recent_span_m": float(recent_span),
            "median_wrist_normal_N": float(median_force),
        }
    return None


def install(source):
    source = replace_once(source, '            def verify_contact(label, paths, arm, axis, direction):',
                          SEEK + '            def verify_contact(label, paths, arm, axis, direction, *, audit_only=False):')
    source = replace_once(source,
                          '''                if sum(positive) < 18 or np.median(normals) < .25 or np.median(floor_loads) < .5:
                    raise RuntimeError(f"Front {slot} {label} lacked sustained floor/target contact: {evidence[label]}, expected {paths}")
                if label == "rear_seat":
                    current_task["rear_seat_verified"] = True
                record("front_B_" + label, "NATIVE_CONTACT_CONFIRMED", **evidence[label])''',
                          '''                corroborated = sum(positive) >= 18 and np.median(normals) >= .25 and np.median(floor_loads) >= .5
                if audit_only:
                    record("front_B_" + label, "PHYSX_CORROBORATED" if corroborated else "PHYSX_DISAGREEMENT",
                           **evidence[label])
                    return
                if not corroborated:
                    raise RuntimeError(f"Front {slot} {label} lacked sustained floor/target contact: {evidence[label]}, expected {paths}")
                if label == "rear_seat":
                    current_task["rear_seat_verified"] = True
                record("front_B_" + label, "NATIVE_CONTACT_CONFIRMED", **evidence[label])''')
    return replace_once(source, '''                rear_path = f"/World/Cube_{slot}"
                probe_rear_face(support_side, verify_path=rear_path)
                verify_contact("rear_seat", [rear_path], support_side, 1, 1)''',
                          '''                rear_name = f"Cube_{slot}"
                rear_path = "/World/" + rear_name
                observe_front_rear_bearing(support_side, rear_name)
                verify_contact("rear_seat", [rear_path], support_side, 1, 1, audit_only=True)''')


SEEK = '''
            def observe_front_rear_bearing(arm, name):
                from collections import deque
                from observed_front_bearing import contact_evidence

                phase = "front_rear_sensor_seek"
                speed, limit = .00045, .005
                record(phase, "RUNNING", arm=arm, target=name, velocity_m_s=speed, travel_limit_m=limit)
                tip, orientation = tcp_fk(arm, applied_targets[arm])
                history = deque(maxlen=30)
                increment = speed / 120
                for index in range(int(limit / increment) + 1):
                    wrench = np.asarray(last_force_sample[arm]["tcp_gravity_corrected_world"])
                    if not np.all(np.isfinite(wrench)) or np.linalg.norm(wrench[3:]) > 1.:
                        raise RuntimeError(f"{phase}: {arm} wrench outside limits: {wrench}")
                    position = np.asarray(last_force_sample[arm]["tcp_position_m"])
                    center, attitude = sensed_cube()
                    gap = front_planes[name] - half - position[1]
                    history.append((float(position[1]), float(-wrench[1])))
                    matched = contact_evidence(history)
                    if matched and -.002 <= gap <= .0012 and name in overlapping_rear(center, attitude):
                        current_task["rear_seat_verified"] = True
                        record(phase, "SENSOR_BEARING_CONFIRMED", arm=arm, target=name,
                               estimated_gap_m=float(gap), travel_m=index * increment, **matched)
                        return
                    travel = (index + 1) * increment
                    if travel > limit:
                        break
                    target = tip + [0., travel, 0.]
                    q = cartesian_joint_target(arm, applied_targets[arm], target, orientation)
                    command_positions(arm, q)
                    step()
                    check_current_collision(phase)
                raise RuntimeError(f"{phase}: {name} bearing was unconfirmed inside the 5 mm seek")

'''
