"""Feed each lower front-row box from the supply station with opposed side cups."""
import json

import numpy as np

from continuous_side_row import GUIDE_NAMES, HERE as FLOOR_DIR
from source_hooks import replace_once


def floor_guide(slot):
    if slot not in range(1, 6):
        raise ValueError(f"Invalid front-row slot {slot}")
    path = FLOOR_DIR / GUIDE_NAMES[slot - 1]
    guide = json.loads(path.read_text())
    faces = np.asarray([guide["grasp_faces_world"][arm] for arm in ("left", "right")])
    if (guide["slot"] != slot or guide["status"] != "CLEAR_GEOMETRY"
            or not np.array_equal(faces[0], -faces[1]) or faces[0, 2] != 0
            or np.count_nonzero(faces[0]) != 1 or not np.isclose(np.linalg.norm(faces[0]), .075)):
        raise ValueError(f"Guide {path.name} does not specify opposing vertical faces")
    guide["grasp_tcp_offsets_world"] = guide["grasp_faces_world"]
    if slot in (3, 5):
        park = next(p["q_rad"] for p in guide["unloaded_transits"] if p["phase"] == "return_home_after_withdrawal")
        left = .6 * np.asarray(park["left"][0]) + .4 * np.asarray(park["left"][1])
        # Retract the left tool around the already seated outer boxes before moving both arms home.
        if slot == 3:
            left[2] -= .07
        else:
            left[3] += .17
        park["left"].insert(1, left.tolist())
        park["right"].insert(1, park["right"][0].copy())
    if slot in (2, 4):
        # With the rear row occupied, raise the released cups as they leave the front/back faces.
        outward, upward = guide["phases"][-2:]
        if (outward["phase"], upward["phase"]) != ("withdraw_laterally", "withdraw_up") or (
                outward["carrying"] or upward["carrying"]):
            raise ValueError("Expected two unloaded withdrawal segments")
        guide["phases"][-2:] = [{
            "phase": "withdraw_up_and_out", "carrying": False,
            "cube_centers_m": upward["cube_centers_m"], "rail_paths_m": upward["rail_paths_m"],
            "q_rad": {arm: [outward["q_rad"][arm][0], *upward["q_rad"][arm][1:]]
                      for arm in ("left", "right")},
        }]
    return guide


def inject(source):
    source = replace_once(source, '            if cruise:\n                # Each axis cruises',
                          '''            # The front/rear grip uses the gradual rail start verified on slot 2.
            cruise = cruise and current_task.get("front_slot") not in (2, 4)
            if cruise:
                # Each axis cruises''')
    source = replace_once(source, '            report["upper_row"]["latest_release_support"] = {',
                          '''            row_report = (report["front_floor_landings"][str(current_task["front_slot"])]
                          if "front_slot" in current_task else report["upper_row"])
            row_report["latest_release_support"] = {''')
    source = replace_once(source, '            record("upper_support_confirmed", "FORCE_TRANSFERRED",',
                          '''            support_label = "A_front_floor_support" if "front_slot" in current_task else "upper_support_confirmed"
            record(support_label, "FORCE_TRANSFERRED",''')
    source = replace_once(source, '            record("upper_release", "OPENING_BOTH_SIDE_GRIPPERS")',
                          '''            release_label = "A_front_floor_release" if "front_slot" in current_task else "upper_release"
            record(release_label, "OPENING_BOTH_SIDE_GRIPPERS")''')
    source = replace_once(source, '                               "A_empty_repark")\n            for side, base in rail_states.items():',
                          '                               "A_empty_repark")\n'
                          '            if current_task.get("front_slot") in (1, 2):\n'
                          '                # B assigns both parked carriages their measured approach stations.\n'
                          '                return\n'
                          '            for side, base in rail_states.items():')
    return replace_once(source, '        def run_cube():', '''        def run_front_a(slot, observed_neighbors):
            from front_row_a import floor_guide

            nonlocal GUIDE, FIRST_LOADED, LOWER_PATHS
            floor = CONFIG["floor_top_m"]
            if current_task.get("front_slot") != slot or abs(SUPPORT_TOP - floor) > 1e-9:
                raise ValueError("Lower-row carry requires the current front slot and floor support plane")
            GUIDE = floor_guide(slot)
            if slot == 5:
                from front_row_skills import front_geometry
                center_x = front_geometry(slot, observed_neighbors, CONFIG["cube_size_m"], floor, 0.)[0]
                offset_x = float(center_x - GUIDE["landing_m"][0])
                GUIDE["landing_m"][0] = float(center_x)
                for phase in GUIDE["phases"]:
                    if phase["phase"] in ("loaded_rail_transfer", "vertical_descent"):
                        centers = np.asarray(phase["cube_centers_m"], dtype=float)
                        fraction = (np.linspace(0., 1., len(centers)) if phase["phase"] == "loaded_rail_transfer"
                                    else 1.)
                        centers[:, 0] += float(offset_x) * fraction
                        phase["cube_centers_m"] = centers.tolist()
                record("front_center_alignment", "LANDING_X_FROM_D455_SLOT_EDGES", target_x_m=float(center_x))
            FIRST_LOADED = next(i for i, row in enumerate(GUIDE["phases"]) if row["carrying"])
            nominal = np.asarray(GUIDE["landing_m"])
            if abs(nominal[2] - floor - half) > .0001:
                raise ValueError("First-layer landing guide does not terminate at floor height")
            entry = report.setdefault("front_floor_landings", {}).setdefault(str(slot), {})
            entry.update(cube_id=current_task["cube_id"], landing_m=nominal.tolist(),
                         grasp_faces_world=GUIDE["grasp_faces_world"])
            saved_supports = LOWER_PATHS
            LOWER_PATHS = {"/World/Environment/floor"}
            try:
                run_cube()
            finally:
                LOWER_PATHS = saved_supports
            position, rotation = observe_resting_cube("front_floor_landing_pose", vision_camera, vision_estimator, floor)
            error = float(np.linalg.norm(position - nominal))
            if error > .015 or any(grippers.is_closed(side) for side in robots):
                raise RuntimeError(f"Front slot {slot} release invalid, landing error {error * 1000:.2f} mm")
            entry.update(observed_position_m=position.tolist(), landing_error_m=error)
            record("front_floor_landing", "REACHED_AFTER_SUCTION_RELEASE", slot=slot, landing_error_m=error)
            return position, rotation

        def run_cube():''')
