"""Install the native opposing side-suction A phase."""
from pathlib import Path

from source_hooks import replace_once


HERE = Path(__file__).resolve().parent


def patch_side_transfer(source):
    source = replace_once(source, '        CONFIG = row_config(CONFIG)', '''        CONFIG = row_config(CONFIG)
        CONFIG["scene_id"] += "_300mm_first_layer_continuous_side"
        for name, center, size in CONFIG["static_boxes"]:
            if name.startswith("wall_"):
                center[2] += size[2] / 2
                size[2] *= 2
        camera = CONFIG["vision"]
        camera["camera_position_m"][2] = 2.45
        camera["camera_look_at_m"][1:] = [.83, .87]
        camera["mount"]["pole_size_m"][2] += .65
        camera["resolution"] = [1280, 960]''')
    source = replace_once(source, '        def run_cube():',
                          (HERE / "first_layer_side_a5_tail.txt").read_text() + '\n        def run_cube():')
    source = replace_once(source, '            move(dual_target(supply, gap=.035, rotation=supply_rotation), duration=3.)',
                          '''            allowed_contacts.clear()
            grasp_seeds = {side: np.asarray(GUIDE["phases"][FIRST_LOADED]["q_rad"][side][0])
                           for side in robots}
            desired_rotation = supply_rotation @ np.asarray(GUIDE["tool_rotation_world"])
            face_normals = {side: np.asarray(offset, dtype=float) / half
                            for side, offset in GUIDE["grasp_faces_world"].items()}
            if (set(face_normals) != set(robots) or any(
                    normal[2] != 0 or np.count_nonzero(normal) != 1 or not np.isclose(np.linalg.norm(normal), 1.)
                    for normal in face_normals.values())
                    or not np.allclose(face_normals["left"], -face_normals["right"])):
                raise ValueError("A requires two opposing vertical box faces")
            approach = {side: cartesian_joint_target(
                            side, grasp_seeds[side], supply + (half + .035) * (supply_rotation @ normal),
                            desired_rotation, retarget_guide=True)
                        for side, normal in face_normals.items()}
            move(approach, duration=3.)''')
    source = replace_once(source,
                          '''            approach_targets = {side: supply + supply_rotation @ [sign * (half + .007), 0, 0]
                                for side, sign in (("left", -1), ("right", 1))}''',
                          '''            approach_targets = {side: supply + (half + .007) * (supply_rotation @ normal)
                                for side, normal in face_normals.items()}''')
    source = replace_once(source, '            normal = supply_rotation[:, 0]',
                          '            normal = supply_rotation @ face_normals["right"]')
    source = replace_once(source, '            try_attach_pair()',
                          '''            allowed_contacts.update(robots)
            set_contact_drive_mode(True)
            try_attach_pair()
            set_contact_drive_mode(False)''')
    before = source.index('            previous = supply\n')
    after = source.index('            record("A_release",', before)
    source = source[:before] + '''            previous = np.asarray(GUIDE["landing_m"])
            for phase in GUIDE["phases"]:
                if phase["carrying"]:
                    follow_side_guide(phase)
''' + source[after:]
    before = source.index('            release_segments = [("outward",')
    after = source.index('            step(120)\n            cube_start, b1_rotation', before)
    source = source[:before] + '''            for phase in GUIDE["phases"][FIRST_LOADED:]:
                if not phase["carrying"]:
                    follow_side_guide(phase)
            allowed_contacts.clear()
            park_after_side_transfer()
''' + source[after:]
    return source
