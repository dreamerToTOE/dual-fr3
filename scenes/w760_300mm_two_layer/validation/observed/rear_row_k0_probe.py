"""Audit rear-row precontact gaps and tactile confirmation from a saved scene."""

import os
from pathlib import Path
import sys

scene = Path.cwd() / "scenes/w760_300mm_two_layer"
sys.path.insert(0, str(scene))

import markerless_rows
import observed_rear
from source_hooks import replace_once

original_install = observed_rear.install
gap_mm = float(os.environ.get("FR3_TEST_STANDOFF_MM", "7"))
lateral_gap_mm = float(os.environ.get("FR3_TEST_LATERAL_STANDOFF_MM", str(gap_mm)))

def install_probe(source):
    source = original_install(source)
    source = replace_once(source, "                distance = FACE_STANDOFF_M if slot == 2 else .010",
                          f"                distance = {gap_mm / 1000!r} if axis == 1 else {lateral_gap_mm / 1000!r}")
    source = replace_once(source, '''                move_to_observed_precontact(arm, destination, rotation @ tool, f"rear_{slot}_B1_approach", hint,''',
                          '''                record("rear_k0_probe", "APPROACH_START", approach_axis=axis, arm=arm,
                       nominal_gap_mm=distance * 1000)
                move_to_observed_precontact(arm, destination, rotation @ tool, f"rear_{slot}_B1_approach", hint,''')
    source = replace_once(source, '''                if not allowed_contacts:
                    set_contact_drive_mode(True)
                allowed_contacts.add(arm)''',
                          '''                actual_tcp, _ = tcp_fk(arm, joint_state()[arm])
                actual_cube, actual_q = cube.get_world_pose()
                actual_outward = -direction * quat_to_rot_matrix(actual_q)[:, axis]
                actual_gap = float((actual_tcp - actual_cube) @ actual_outward - half)
                record("rear_k0_probe", "ARRIVED_CLEAR", approach_axis=axis, arm=arm,
                       actual_gap_mm=actual_gap * 1000,
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                if not allowed_contacts:
                    set_contact_drive_mode(True)
                allowed_contacts.add(arm)''')
    source = replace_once(source, '''                touch_face(arm, inward, .018, f"rear_{slot}_B1_touch")''',
                          '''                touch_face(arm, inward, .018, f"rear_{slot}_B1_touch")
                record("rear_k0_probe", "TACTILE_CONFIRMED", approach_axis=axis, arm=arm,
                       observation=report["target_observations"][-1],
                       native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])
                if axis == 0 or slot == 5:
                    report["status"] = "PASS_APPROACH_PROBE"
                    raise SystemExit(0)''')
    return source

if __name__ == "__main__":
    observed_rear.install = install_probe
    markerless_rows.main()
