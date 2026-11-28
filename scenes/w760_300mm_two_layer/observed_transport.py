"""Register the existing carry corridors to the observed scene before solving IK."""

from copy import deepcopy

import numpy as np

from front_row_skills import front_geometry
from source_hooks import replace_once


def wall_plane(config, wall, axis, sign):
    center, extent = next((center, extent) for name, center, extent in config["static_boxes"] if name == wall)
    return float(center[axis] - sign * extent[axis] / 2)


def front_placement_goal(config, slot, observations):
    size = config["cube_size_m"]
    wall_x = None
    if slot in (1, 2):
        sign = -1 if slot == 1 else 1
        wall = "wall_left" if slot == 1 else "wall_right"
        wall_x = wall_plane(config, wall, 0, sign) - sign * size / 2
    # A centers the landing in the open slot. B checks its actual yaw and both side gaps.
    return front_geometry(slot, observations, size, config["floor_top_m"], wall_x, total_clearance_m=0.)


def front_landing(guide, config, slot, observations):
    nominal_goal = np.asarray(config["goal_cube_centers_m"][slot - 1], dtype=float)
    nominal_goal[1] -= config["cube_size_m"]
    observed_goal = front_placement_goal(config, slot, observations)
    # Work in the accepted approach corridor relative to the actual seating target.
    landing = np.asarray(guide["landing_m"]) + observed_goal - nominal_goal
    if slot == 5:
        landing[0] = observed_goal[0]
    return landing


def upper_landing(guide, config, slot, seating_goal):
    nominal_goal = np.asarray(config["goal_cube_centers_m"][slot - 1], dtype=float)
    nominal_goal[2] += config["cube_size_m"]
    landing = np.asarray(guide["landing_m"]) + np.asarray(seating_goal) - nominal_goal
    if slot == 5:
        landing[0] = seating_goal[0]
    return landing


def register_transport(guide, pickup, landing):
    """Generate a Cartesian path from measured pickup and relative landing geometry."""
    guide = deepcopy(guide)
    loaded = [phase for phase in guide["phases"] if phase["carrying"]]
    pickup_shift = np.asarray(pickup) - loaded[0]["cube_centers_m"][0]
    landing_shift = np.asarray(landing) - guide["landing_m"]
    supply_rails = guide["rail_supply_y_m"].copy()
    landing_rails = guide["rail_landing_y_m"].copy()
    for phase in guide["phases"]:
        centers = np.asarray(phase["cube_centers_m"], dtype=float)
        name = phase["phase"]
        if name == "loaded_rail_transfer":
            blend = np.linspace(0., 1., len(centers))[:, None]
            shift = (1. - blend) * pickup_shift + blend * landing_shift
        else:
            shift = pickup_shift if name in ("lift_and_align", "high_align_x") else landing_shift
        phase["cube_centers_m"] = (centers + shift).tolist()
        for arm, path in phase["rail_paths_m"].items():
            travel = landing_rails[arm] - supply_rails[arm]
            if travel:
                progress = np.clip((np.asarray(path) - supply_rails[arm]) / travel, 0., 1.)
                offset = (1. - progress) * pickup_shift[1] + progress * landing_shift[1]
                phase["rail_paths_m"][arm] = (np.asarray(path) + offset).tolist()
    for arm, start in supply_rails.items():
        if start != landing_rails[arm]:
            guide["rail_supply_y_m"][arm] += float(pickup_shift[1])
            guide["rail_landing_y_m"][arm] += float(landing_shift[1])
    guide["landing_m"] = np.asarray(landing, dtype=float).tolist()
    return guide


def install(source):
    source = replace_once(source, 'from fr3_contact_control import NormalForceController, oriented_wall_gap\n',
                          '''from functools import partial
from fr3_contact_control import NormalForceController, oriented_wall_gap

NormalForceController = partial(NormalForceController, max_force_N=None)
''')
    start = source.index('                if contact_monitor.phase.startswith("A"):\n')
    end = source.index('                if force_label is not None', start)
    source = source[:start] + source[end:]
    source = replace_once(source, 'max_offset_m=max_travel_m, max_force_N=A_FORCE_LIMITS["max_force_N"])',
                          'max_offset_m=max_travel_m)')
    source = replace_once(source, '''                if compression > 10.0:
                    raise RuntimeError(f"{phase}: unexpected contact force {compression:.2f} N")
''', '')
    source = replace_once(source, '''                if abs(last_force_sample[side]["tcp_gravity_corrected_world"][1]) > 10.:
                    raise RuntimeError(f"{phase}: contact overloaded during plane alignment")
''', '')
    source = replace_once(source, '''                if abs(force) > 10 or ((follow_side is not None or probe_override is not None) and force_norm > 10):
                    raise RuntimeError(f"{side} wall acquisition overload: "
                                       f"normal {force:.3f} N, resultant {force_norm:.3f} N")
''', '')
    source = replace_once(source, 'max_offset_m=.002, max_force_N=10.)', 'max_offset_m=.002)')
    source = replace_once(source, '''                if any(abs(force) > 10. for force in current.values()):
                    raise RuntimeError(f"{phase}: dual-contact force exceeds 10 N")
''', '')
    source = replace_once(source, '''                    if abs(active_force) > 10.:
                        raise RuntimeError(f"{push_side} B2 push overload: {active_force:.3f} N")
''', '')
    source = replace_once(source, '''                if wrist_load < -load_limit:
                    raise RuntimeError(f"Upper seating downward compression {-wrist_load:.3f} N exceeds "
                                       f"{load_limit:.3f} N at {descent*1000:.3f} mm approach")
''', '')
    source = replace_once(source, 'maximum_compression_N=load_limit)',
                          'wrist_load_target_N=target_wrist_load)')

    anchor = '            GUIDE = floor_guide(slot)\n'
    start = source.index(anchor, source.index('        def run_front_a(')) + len(anchor)
    end = source.index('            FIRST_LOADED = ', start)
    source = source[:start] + '''            from observed_transport import front_landing
            current_task["live_landing_m"] = front_landing(GUIDE, CONFIG, slot, observed_neighbors)
''' + source[end:]
    source = replace_once(source, '''            placement_target = front_geometry(slot, observed_neighbors, CONFIG["cube_size_m"], CONFIG["floor_top_m"],
                                    CONFIG["goal_cube_centers_m"][slot-1][0])''',
                          '''            from observed_transport import front_placement_goal
            placement_target = front_placement_goal(CONFIG, slot, observed_neighbors)''')
    source = replace_once(
        source, '            nominal = np.asarray(GUIDE["landing_m"])\n',
        '            nominal = np.asarray(current_task["live_landing_m"])\n')
    source = replace_once(source, '        def run_cube():\n            nonlocal half',
                          '        def run_cube():\n            nonlocal half, GUIDE')
    hook = '            supply, supply_rotation = observe_resting_cube("A_pickup_pose", vision_camera, rows, supply_top)\n'
    source = replace_once(source, hook, hook + '''            from observed_transport import register_transport
            GUIDE = register_transport(GUIDE, supply, current_task["live_landing_m"])
            record("A_observed_transport", "RETARGETED", source="rear_rgbd_square",
                   pickup_m=supply.tolist(), landing_m=GUIDE["landing_m"],
                   measured_tcp_m={arm: tcp_fk(arm, q)[0].tolist() for arm, q in joint_state().items()})
''')
    hook = '            initial = {side: applied_targets[side].copy() for side in robots}\n'
    source = replace_once(source, hook, '            initial = joint_state()\n')
    source = replace_once(source, '            return np.array([x, 1.076, SUPPORT_TOP + half])',
                          '''            from observed_transport import wall_plane
            if slot in (1, 2):
                sign = -1 if slot == 1 else 1
                wall = "wall_left" if slot == 1 else "wall_right"
                x = wall_plane(CONFIG, wall, 0, sign) - sign * half
            return np.array([x, wall_plane(CONFIG, "wall_back", 1, 1) - half, SUPPORT_TOP + half])''')
    start = source.index('            target = find_upper_target(slot, upper_seen)\n')
    end = source.index('            current_task["goal_center_m"]', start)
    return source[:start] + '''            from observed_transport import upper_landing
            target = find_upper_target(slot, upper_seen)
            current_task["live_landing_m"] = upper_landing(GUIDE, CONFIG, slot, target)
''' + source[end:]
