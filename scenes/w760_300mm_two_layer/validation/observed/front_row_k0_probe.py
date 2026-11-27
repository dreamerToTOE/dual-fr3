"""Check the front-row unloaded gap before each tactile approach from a landing checkpoint."""
import re
import sys
from pathlib import Path

scene = Path.cwd() / "scenes/w760_300mm_two_layer"
sys.path.insert(0, str(scene))
import markerless_rows
import observed_rear
from source_hooks import replace_once

install_rear = observed_rear.install

def with_front_probe(source):
    source = install_rear(source)
    start = source.index("        def run_front_b(")
    stop = source.index("\n        def ", start + 12)
    front = source[start:stop]
    front = re.sub(r"\btouch_face\(", "front_probe_touch(", front)
    front = replace_once(front, '''        def run_front_b(slot, observed_neighbors):
''', '''        def run_front_b(slot, observed_neighbors):
            approach_count = 0

            def front_probe_touch(side, inward, distance, phase):
                nonlocal approach_count
                actual_tcp, _ = tcp_fk(side, joint_state()[side])
                actual_cube, actual_q = cube.get_world_pose()
                box_rotation = quat_to_rot_matrix(actual_q)
                axis = int(np.argmax(np.abs(box_rotation.T @ inward)))
                normal = box_rotation[:, axis]
                outward = -np.sign(normal @ inward) * normal
                gap = float((actual_tcp - actual_cube) @ outward - half)
                record("front_k0_audit", "PRECONTACT_ARRIVED", arm=side, axis=axis, touch_phase=phase,
                       actual_gap_mm=gap * 1000.,
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                touch_face(side, inward, distance, phase)
                record("front_k0_audit", "TOUCH_CONFIRMED", arm=side, axis=axis, touch_phase=phase,
                       observation=report["target_observations"][-1],
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                if phase != "front_center_align_touch":
                    approach_count += 1
                if approach_count == (3 if slot in (1, 2) else 2 if slot in (3, 4) else 1):
                    report["status"] = "PASS_FRONT_K0_PROBE"
                    raise SystemExit(0)

''')
    return source[:start] + front + source[stop:]

if __name__ == "__main__":
    observed_rear.install = with_front_probe
    markerless_rows.main()
