"""Measure unloaded gaps and confirm tactile contact from an upper landing checkpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import markerless_rows
import observed_rear
from source_hooks import replace_once


install_rear = observed_rear.install


def with_upper_probe(source):
    source = install_rear(source)
    source = replace_once(source, '            def approach_front():\n', '''            def audit_upper_touch(arm, inward, distance, phase):
                # Simulator pose is used only in this validation wrapper.
                tcp, _ = tcp_fk(arm, joint_state()[arm])
                center, quaternion = cube.get_world_pose()
                box_rotation = quat_to_rot_matrix(quaternion)
                axis = int(np.argmax(np.abs(box_rotation.T @ inward)))
                normal = box_rotation[:, axis]
                outward = -np.sign(normal @ inward) * normal
                gap = float((tcp - center) @ outward - half)
                record("upper_k0_audit", "PRECONTACT_ARRIVED", arm=arm, touch_phase=phase,
                       commanded_standoff_mm=FACE_STANDOFF_M * 1000, actual_gap_mm=gap * 1000,
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                touch_face(arm, inward, distance, phase)
                record("upper_k0_audit", "TOUCH_CONFIRMED", arm=arm, touch_phase=phase,
                       observation=report["target_observations"][-1],
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                if slot == 5 or phase == "upper_B_side_touch":
                    report["status"] = "PASS_UPPER_K0_PROBE"
                    raise SystemExit(0)

            def approach_front():
''')
    for arm, direction, phase in (("support_side", "front_normal", "front"),
                                  ("push_side", "-normal", "side")):
        arguments = f'({arm}, {direction}, .018, "upper_B_{phase}_touch")'
        source = replace_once(source, "touch_face" + arguments, "audit_upper_touch" + arguments)
    return source


if __name__ == "__main__":
    observed_rear.install = with_upper_probe
    markerless_rows.main()
