"""Five-cube single-row geometry and alternating packing order."""
from copy import deepcopy
from math import isclose


def row_config(base):
    config = deepcopy(base)
    size = config["cube_size_m"]
    inner_width = 5 * size + 0.090
    half_width = inner_width / 2
    z = config["floor_top_m"] + size / 2
    floor = next(box for box in config["static_boxes"] if box[0] == "floor")
    rear = next(box for box in config["static_boxes"] if box[0] == "wall_back")
    goal_y = rear[1][1] - rear[2][1] / 2 - size / 2
    config["scene_id"] = "dual_fr3_four_wall_150mm_five_cube_row"
    outer = half_width - size / 2
    inner = outer - size
    config["goal_cube_centers_m"] = [[x, goal_y, z] for x in (-outer, outer, -inner, inner, 0.)]
    config["task"]["scope"] = "five_cube_supply_A_B1_alternating_B2_row"
    config["scope"] = config["task"]["scope"]
    config["ab_runtime"] = "fr3_ab_visual_runner.py --five-cube-row"
    config["task"]["phase_B2"]["target_center_m"] = config["goal_cube_centers_m"][0]
    config["task"]["phase_B2"]["rail_assist"]["base_y_m"] = .80
    config["cube_sequence"] = [
        {"cube_id": index + 1, "marker_id": 23 + index,
         "landing_x_m": (0., .005, -.005, .145, 0.)[index],
         "active_contact_height_m": .020,
         "approach_standoff_m": .030 if index == 0 else .007,
         "goal_center_m": goal, "support_side": "left" if index in (0, 2, 4) else "right",
         "push_side": "right" if index in (0, 2) else "left" if index in (1, 3) else None,
         "push_direction": -1 if index in (0, 2) else 1 if index in (1, 3) else 0,
         "contact_target": "wall_left" if index == 0 else "wall_right" if index == 1 else
                           "Cube_1" if index == 2 else "Cube_2" if index == 3 else None}
        for index, goal in enumerate(config["goal_cube_centers_m"])
    ]
    config["row_colors_rgb"] = [(0.95, 0.36, 0.08), (0.22, 0.63, 0.93), (0.93, 0.74, 0.13),
                                (0.39, 0.74, 0.29), (0.67, 0.40, 0.88)]
    floor[2][0] = inner_width + 2 * rear[2][1]
    for name, center, extent in config["static_boxes"]:
        if name in ("wall_front", "wall_back"):
            extent[0] = inner_width
        if name in ("wall_left", "wall_right"):
            center[0] = (-1 if name == "wall_left" else 1) * (half_width + extent[0] / 2)
        if name == "table":
            extent[0] = 1.36
    for side, sign in (("left", -1), ("right", 1)):
        config["fr3"]["bases"][side][:2] = [sign * 0.60, 0.20]
    goals = config["goal_cube_centers_m"]
    ordered = sorted(goals)
    assert isclose(half_width - ordered[-1][0] - size / 2, 0., abs_tol=1e-10)
    assert isclose(ordered[2][0] - ordered[1][0] - size, 0.045)
    assert isclose(ordered[3][0] - ordered[2][0] - size, 0.045)
    return config
