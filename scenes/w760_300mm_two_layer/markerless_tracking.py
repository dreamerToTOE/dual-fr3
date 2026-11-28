"""Associate visible box faces with the packing sequence using the fixed rear view."""

import math

import cv2
import numpy as np

from single_cube_square import SquareTop, replace_block
from source_hooks import replace_once


FLOOR_REAR_ORDER = (1, 3, 5, 4, 2)
FLOOR_FRONT_ORDER = (12, 13, 6, 14, 15)
UPPER_ORDER = (7, 9, 11, 10, 8)


class RowFaces:
    def __init__(self, intrinsics, optical_to_world, resolution, config):
        self.floor = config["floor_top_m"]
        self.size = config["cube_size_m"]
        self.supply = np.asarray(config["supply_cube_center_m"])
        self.y = config["task"]["phase_B1"]["target_center_m"][1]
        self.x = sorted(row[0] for row in config["goal_cube_centers_m"])
        self.side_wall = {
            sign: center[0] - sign * extent[0] / 2
            for sign, label in ((-1, "wall_left"), (1, "wall_right"))
            for name, center, extent in config["static_boxes"] if name == label
        }
        back, extent = next((center, size) for name, center, size in config["static_boxes"] if name == "wall_back")
        camera_y, camera_z = optical_to_world[1:, 3][:2]
        wall_top = back[2] + extent[2] / 2
        wall_front = back[1] - extent[1] / 2
        self.rear_wall = wall_front
        self.planes = {}
        self.color_y_max = {}
        for z in (self.floor, self.floor + self.size):
            self.planes[z] = SquareTop(intrinsics, optical_to_world, resolution, config, support_z=z)
            self.color_y_max[z] = camera_y + (wall_front - camera_y) * (
                (camera_z - z - self.size) / (camera_z - wall_top)) - (.013 if z == self.floor else .002)
        self.exposure = {}
        self.placed = {}

    def height(self, cube_id):
        return self.floor + (self.size if cube_id in UPPER_ORDER else 0.)

    def location_hint(self, cube_id):
        rows = (FLOOR_REAR_ORDER, FLOOR_FRONT_ORDER, UPPER_ORDER)
        row = next(ids for ids in rows if cube_id in ids)
        return np.array([self.x[row.index(cube_id)],
                         self.y - (self.size if row == FLOOR_FRONT_ORDER else 0.),
                         self.height(cube_id) + self.size / 2])

    def masks(self, frame, support):
        support = min(self.planes, key=lambda z: abs(support - z))
        view = self.planes[support]
        frame_id = int(frame.get("rendering_frame", 0))
        cached = self.exposure.get(support)
        if cached is not None and cached[0] == frame_id:
            return cached[1:]
        raw_depth = np.asarray(frame.get("distance_to_image_plane", [])).squeeze()
        raw_color = np.asarray(frame.get("rgba", []))
        if raw_depth.ndim != 2 or raw_color.ndim != 3 or raw_color.shape[:2] != raw_depth.shape:
            return None
        depth = raw_depth[view.roi]
        rgb = raw_color[view.roi][..., :3].astype(np.float32)
        chroma = rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1.)
        with np.errstate(invalid="ignore"):
            delta_z = (depth - view.expected_depth) * view.ray_z
        valid = view.inside & np.isfinite(depth) & (depth > 0)
        top = valid & (abs(delta_z) < .004)
        low = valid & (delta_z <= -.004)
        cached = (frame_id, view, top, low, chroma)
        self.exposure[support] = cached
        return cached[1:]

    def locate(self, frame, support, hint, rotation=None):
        masks = self.masks(frame, support)
        if masks is None:
            return None
        view, top, low, chroma = masks
        extent = self.size / 2
        if hint[1] < 0.:
            # The original isolated-box detector already handles offset and yaw at the table.
            fit = view.estimate(frame, region_center=self.supply)
            return view.resting_pose(fit, support) if fit["cube_position_m"] is not None and fit["cube_position_m"][1] < 0 else None
        # Prefer the depth contour when the box has edges clear of its neighbors.
        # The wrist's shadow can skew RGB edges even when the cube has not rotated.
        fit = view.estimate(frame, region_center=hint)
        isolated = view.resting_pose(fit, support)
        if isolated is not None and np.linalg.norm(isolated[0][:2] - hint[:2]) < .010:
            return isolated

        # Select a search window using the incoming box's last estimate or the slot index.
        local = view.xy - np.asarray(hint[:2], dtype=np.float32)
        seed = top & (abs(local) < .040).all(axis=-1)
        if seed.sum() < 12:
            return None
        color = np.median(chroma[seed], axis=0)
        similar = (abs(chroma - color).max(axis=-1) <= .014)
        window = (abs(local) < extent + .034).all(axis=-1)
        patch = top & similar & window
        if patch.sum() < 90:
            return None
        different = top & ~similar & (view.xy[..., 1] < self.color_y_max[support])
        # Adjacent farther points are exposed edges. Nearer points occlude the box.
        points, outward, depth_edges = [], [], []
        for axis in (0, 1):
            preceding = [slice(None), slice(None)]
            following = preceding.copy()
            preceding[axis] = slice(None, -1)
            following[axis] = slice(1, None)
            for origin, neighbor in ((tuple(preceding), tuple(following)), (tuple(following), tuple(preceding))):
                depth_edge = (top & window)[origin] & low[neighbor]
                edge = depth_edge | (patch[origin] & different[neighbor])
                a, b = view.xy[origin][edge], view.xy[neighbor][edge]
                points.append((a + b) / 2)
                outward.append(b - a)
                depth_edges.append(depth_edge[edge])
        points = np.concatenate(points)
        outward = np.concatenate(outward)
        depth_edges = np.concatenate(depth_edges)
        if not len(points):
            return None
        outward /= np.linalg.norm(outward, axis=1, keepdims=True)
        yaw = math.atan2(rotation[1][0], rotation[0][0]) if rotation is not None else 0.
        cos, sin = math.cos(yaw), math.sin(yaw)
        axes = np.array([[cos, -sin], [sin, cos]])
        projected = (points - hint[:2]) @ axes
        orientation = outward @ axes
        edges, from_depth = {}, set()
        for axis in (0, 1):
            normal = projected[:, axis]
            tangent = projected[:, 1 - axis]
            for sign in (-1, 1):
                if axis == 0 and abs(hint[0] + sign * extent - self.side_wall[sign]) < .020:
                    continue
                candidates = ((orientation[:, axis] * sign > .45) & (abs(normal - sign * extent) < .027)
                              & (abs(tangent) < .058))
                physical = candidates & depth_edges
                if physical.sum() >= 6 and np.ptp(tangent[physical]) >= .033:
                    candidates = physical
                    from_depth.add((axis, sign))
                offsets = normal[candidates]
                if len(offsets) < 6:
                    continue
                around = tangent[candidates]
                # Align the long repeated edge and ignore short lighting and cup boundaries.
                windows = abs(offsets[:, None] - offsets) < .0045
                index = int(np.argmax(windows.sum(axis=1)))
                selected = windows[index]
                if np.ptp(around[selected]) < .033:
                    continue
                edges[axis, sign] = points[candidates][selected]
        if any(not any(item[0] == axis for item in edges) for axis in (0, 1)):
            return None
        yaws, weights = [], []
        yaw_edges = {edge: positions for edge, positions in edges.items() if edge in from_depth} or edges
        for (axis, _), positions in yaw_edges.items():
            dx, dy = cv2.fitLine(positions, cv2.DIST_HUBER, 0., .001, .001).ravel()[:2]
            angle = math.atan2(dy, dx) + (math.pi / 2 if axis == 0 else 0.)
            angle = yaw + (angle - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            if abs(angle - yaw) <= math.radians(10):
                yaws.append(angle)
                weights.append(len(positions))
        if yaws:
            yaw = float(np.average(yaws, weights=weights))
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.array([[c, -s], [s, c]])
        coordinate = [np.mean([np.median(edge @ axes[:, axis]) - sign * extent
                               for (i, sign), edge in edges.items() if i == axis]) for axis in (0, 1)]
        center = np.r_[axes @ coordinate, support + extent]
        return center, np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def neighbor(self, frame, cube_id):
        latest = self.placed.get(cube_id)
        guess, rotation = latest if latest is not None else (self.location_hint(cube_id), None)
        pose = self.locate(frame, self.height(cube_id), guess, rotation)
        if pose is None:
            return None
        position, attitude = pose
        extent = self.size / 2 * np.abs(attitude[:2, :2]).sum(axis=1)
        if (position[0] - extent[0] < self.side_wall[-1] - .004 or
                position[0] + extent[0] > self.side_wall[1] + .004 or
                position[1] + extent[1] > self.rear_wall + .004):
            return None
        return pose

    def save_outline(self, frame, support, pose, path):
        support = min(self.planes, key=lambda z: abs(support - z))
        center, rotation = pose
        self.planes[support].save_view(frame, {"cube_position_m": center, "cube_rotation_world": rotation}, path)


def install(source, *, observed_skills=False):
    source = replace_once(source, 'from fr3_vision import ArucoCubeEstimator, PlanarPoseMedian',
                          'from fr3_vision import PlanarPoseMedian')
    source = replace_block(source, '        def make_vision_estimator(', '        comparison_directory = ',
                           '''        from markerless_tracking import RowFaces

        rows = RowFaces(camera_matrix, camera_to_world, vision_cfg["resolution"], CONFIG)
        rows.placed.update({int(key): tuple(np.asarray(value) for value in pose)
                            for key, pose in CHECKPOINT.get("markerless_placements", {}).items()})
        report["vision"]["observation"] = "rear RGB-D square edges; no fiducials"
        report["control_input_boundary"]["target_sources"][0] = "rear RGB-D box edges on calibrated support planes"
''')
    source = source.replace('supply_camera, supply_estimator, supply_top', 'vision_camera, rows, supply_top')
    source = source.replace('vision_camera, vision_estimator,', 'vision_camera, rows,')
    source = replace_once(source,
                          '            supply, supply_rotation = observe_resting_cube("A_contact_pose", vision_camera, rows, supply_top)',
                          '            # Use the clear table observation until the tools touch the stationary box.\n'
                          '            # The tactile seek below measures the remaining gap along the grasp normal.')
    source = replace_once(source, '                    for side in (("right", "left") if current_task.get("slot") == 4 else robots):\n'
                          '                        if current_task.get("slot") == 4:',
                          '''                    stationary_seals = (current_task.get("slot") == 4
                                        or current_task.get("front_slot") in (2, 4))
                    for side in (("right", "left") if stationary_seals else robots):
                        if stationary_seals:''')
    source = replace_once(source, 'raise RuntimeError("Upper 4 right seal broke during left docking")',
                          'raise RuntimeError("Right seal broke during left docking")')
    source = replace_block(source, '            motion_timing = report["side_transfer_motion"][-1]',
                           '                step()\n                check_current_collision(name)', '''            motion_timing = report["side_transfer_motion"][-1]
            velocities = {side: np.gradient(q, 1 / 120, axis=0) for side, q in samples.items()}
            # At a front/rear grasp both wrists share the weight and constrain the same roll axis.
            balance_carry = carrying and current_task.get("front_slot") in (2, 4)
            if balance_carry:
                from scipy.spatial.transform import Rotation

                axis = actual_tool["right"][1][:, 1]
                roll = vertical = load_error = 0.
                last_delta = {side: np.zeros(7) for side in robots}
                balance = motion_timing["cup_load_balance"] = {"peak_roll_rad": 0., "peak_torque_Nm": 0.,
                                                             "peak_vertical_m": 0.}
            for index in range(1, frames + 1):
                for side in robots:
                    base = rail_states[side].copy()
                    if base[1] != bases[side][index]:
                        base[1] = bases[side][index]
                        set_robot_base_pose(side, base)
                commands = {side: samples[side][index] for side in robots}
                speeds = {side: profile[index] for side, profile in velocities.items()}
                if balance_carry:
                    left, right = (np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"])
                                   for side in ("left", "right"))
                    torque = float(.5 * (right[3:] - left[3:]) @ axis)
                    # One wrist follows the other to unload twist while retaining the same contact plane.
                    roll = float(np.clip(roll + .0001 * torque / 120, -.00035, .00035))
                    load_error += .25 * (right[2] - left[2] - load_error)
                    imbalance = math.copysign(max(0., abs(load_error) - .35), load_error)
                    vertical = float(np.clip(vertical + .00010 * imbalance / 120, -.00045, .00045))
                    for side, sign in (("left", -1), ("right", 1)):
                        seed = commands[side]
                        tcp, wrist_rotation = tcp_fk(side, seed)
                        if side == "right":
                            wrist_rotation = Rotation.from_rotvec(roll * axis).as_matrix() @ wrist_rotation
                        commands[side] = cartesian_joint_target(side, seed, tcp + [0., 0., sign * vertical], wrist_rotation)
                        delta = commands[side] - seed
                        speeds[side] = speeds[side] + 120 * (delta - last_delta[side])
                        last_delta[side] = delta
                    balance["peak_roll_rad"] = max(balance["peak_roll_rad"], abs(roll))
                    balance["peak_torque_Nm"] = max(balance["peak_torque_Nm"], abs(torque))
                    balance["peak_vertical_m"] = max(balance["peak_vertical_m"], abs(vertical))
                    issues = checker.check(commands, cube_center=payload_path[index], cube_rotation=payload_rotation,
                                           cup_contacts=allowed_contacts, margin_m=.001, first_only=True)
                    if issues:
                        raise RuntimeError(f"{name} load-balancing correction collision: {issues[0]}")
                for side, q in commands.items():
                    command_positions(side, q, velocity=speeds[side] if index < frames else None)
''')
    source = "\n".join(line for line in source.split("\n")
                       if "vision_estimator.marker_id = supply_estimator.marker_id" not in line)
    source = replace_once(source, '        def sample_vision(render=True):', '''        def active_square_hint(phase, support):
            if phase == "A_pickup_pose":
                return rows.supply, None
            if control_center is not None and abs(control_center[2] - support - half) < .025:
                return control_center, control_rotation
            if phase in ("upper_landing_pose", "front_floor_landing_resume"):
                if phase == "front_floor_landing_resume":
                    if CHECKPOINT.get("front_b1"):
                        return np.asarray(CHECKPOINT["front_b1"]["rear_rgb"]), None
                    from front_row_a import floor_guide
                    return np.asarray(floor_guide(current_task["front_slot"])["landing_m"]), None
                receipt = (RELEASE_CHECKPOINT or {}).get("seam_entry_resume")
                if receipt and current_task.get("slot") == START_SLOT:
                    return np.asarray(receipt["grounded_evidence"]["reattained_RGB"]["position_m"]), None
                return np.asarray(GUIDE["landing_m"]), None
            return rows.supply, None

        def sample_vision(render=True):''')
    source = replace_block(source, '                    observation = vision_estimator.estimate(',
                           '            latest_camera_observation = {',
                           '''                    support = SUPPORT_TOP if push_faces or (
                        control_center is not None and control_center[1] > 0.) else rows.floor
                    if carry_offsets is None:
                        hint, attitude = active_square_hint(contact_monitor.phase, support)
                        pose = rows.locate(frame, support, hint, attitude)
                        # The tracked pose supplies the identity of the moving box.
                        # A jump onto the wrist edge is larger than one frame of measured push travel.
                        if (pose is not None and push_faces and
                                np.linalg.norm(pose[0][:2] - hint[:2]) > .012):
                            pose = None
                        if pose is not None:
                            position, rotation = pose
                            uv = (position + [0., 0., half] - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
                            pixel = camera_matrix @ uv
                            observation = {"status": "detected", "cube_position_m": position.tolist(),
                                           "cube_rotation_world": rotation.tolist(),
                                           "pixel_center": (pixel[:2] / pixel[2]).tolist(),
                                           "reprojection_error_px": None}
                    target = current_task.get("contact_target", "")
                    if (contact_monitor.phase in ("B2_push", "B2_push_seat", "B2_push_unload")
                            and target in ("Cube_1", "Cube_2", "Cube_12", "Cube_15")
                            and control_center is not None
                            and abs(control_center[0] - current_task["goal_center_m"][0]) < .025):
                        neighbor_pose = rows.neighbor(frame, int(target.split("_")[-1]))
                        if neighbor_pose is not None:
                            if target not in neighbor_filters:
                                neighbor_filters[target] = PlanarPoseMedian()
                            filtered = neighbor_filters[target].update(frame_id, capture_time, *neighbor_pose)
                            if filtered is not None:
                                live_neighbors[target] = filtered
''')
    source = replace_block(source, '                raw_rotation = np.asarray(observation.get("cube_rotation_world")',
                           '                if supported_pose is not None:\n                    filtered_pose',
                           '''                supported_pose = pose if support == SUPPORT_TOP else None
''')
    source = replace_once(source, '                supported_pose = pose if support == SUPPORT_TOP else None',
                          '                supported_pose = pose if observation["status"] == "detected" and support == SUPPORT_TOP else None')
    source = replace_once(source, '                            if "front_slot" not in current_task:\n'
                          '                                raise RuntimeError("RGB observed more than 3 degrees of Cube rotation; contact replan required")',
                          '''                            alarm_root = output_directory / f"Cube_{current_task['cube_id']}_yaw_alarm.npz"
                            if not alarm_root.exists():
                                np.savez_compressed(alarm_root, rgba=frame["rgba"], depth_m=frame["distance_to_image_plane"],
                                                    hint=hint, support_z=support, attitude=attitude)
                            if "front_slot" not in current_task:
                                raise RuntimeError("RGB observed more than 3 degrees of Cube rotation; contact replan required")''')
    source = replace_once(source, '                record("front_B_" + label, "NATIVE_CONTACT_CONFIRMED", **evidence[label])',
                          '''                if label == "rear_seat":
                    current_task["rear_seat_verified"] = True
                record("front_B_" + label, "NATIVE_CONTACT_CONFIRMED", **evidence[label])''')
    source = replace_once(source, '                travel = 0.\n                stable = 0\n                for _ in range(2400):',
                          '                travel = 0.\n                stable = 0\n                rear_hold = None\n                for _ in range(2400):')
    source = replace_once(source, '                    else:\n                        stable = 0\n                        travel += .00045 / 120',
                          '''                        if verify_path is not None:
                            # Relieve the settling transient before confirming sustained rear contact.
                            if rear_hold is None:
                                rear_hold = NormalForceController(tip[1] + travel, target_force_N=6.,
                                    gain_m_per_Ns=.0006, max_speed_m_s=.0009, max_force_N=None)
                            travel = rear_hold.update(-wrench[1]) - tip[1]
                            q = cartesian_joint_target(arm, applied_targets[arm], tip + [0., travel, 0.], orientation)
                            command_positions(arm, q)
                    else:
                        stable = 0
                        rear_hold = None
                        travel += .00045 / 120''')
    source = replace_once(source, '            if issues and "front_slot" in current_task and push_faces and issues[0]["kind"] == "cube_obstacle":',
                          '''            seated_retreat = phase == "B2_support_retreat" and current_task.get("rear_seat_verified")
            if (issues and "front_slot" in current_task and (push_faces or seated_retreat)
                    and issues[0]["kind"] == "cube_obstacle"):''')
    source = replace_once(source, '"previous_rgb_age_s": float(world.current_time) - stored["capture_time_s"],',
                          '"previous_rgb_age_s": (float(world.current_time) - stored["capture_time_s"]\n'
                          '                                                              if "capture_time_s" in stored else None),')
    # The straight insertion still checks both projected edges along the path.
    # On the lower row each edge must be clear; the full slot width check still applies.
    source = source.replace('if min(gaps) < .001:', 'if min(gaps) <= 0.:', 3)
    source = replace_once(source, 'Center slot clearance remains below 1 mm:', 'Center slot still overlaps a fitted edge:')
    source = replace_once(source, '                        cartesian_move(push_side, target, 5., f"{prefix}_lateral_descent")',
                          '''                        # Plan to the measured standoff on the lower elbow branch before tactile seeking.
                        record(f"{prefix}_lateral_descent", "PLAN_COLLISION_CHECKED_STANDOFF")
                        seed = np.array([-.635907, .550997, .694341, -2.386040, -.854957, 2.681559, 2.352209])
                        move({push_side: cartesian_joint_target(push_side, seed, target, side_tool, retarget_guide=True)})''')
    source = replace_once(source, 'preload = (1.0 if final_u2 else 3.25) if slot == 2 else 1.25',
                          'preload = (1.0 if final_u2 else 3.25) if slot in (1, 2) else 1.25')
    if observed_skills:
        source = replace_once(source, '                span = .012',
                              '''                # Extend U1's lift to one third of a box width over the outer support.
                span = 2. * half / 3. if slot == 1 else .012''')
        source = replace_once(source, '                entry_x, exit_x = seam_x - push_sign * span, seam_x + push_sign * span',
                              '''                entry_x, exit_x = seam_x - push_sign * span, seam_x + push_sign * span
                if slot == 1 and push_sign * (exit_x - center[0]) <= .004:
                    return''')
        source = replace_once(source, '                    if not 0. < push_sign * (exit_x - entry_x) <= .14:',
                              '''                    max_transfer_m = .18 if slot == 1 else .14
                    if not 0. < push_sign * (exit_x - entry_x) <= max_transfer_m:''')
        source = replace_once(source, '                if slot in (1, 2, 3, 4) and completed_transport not in (1, 2):',
                              '                if slot == 1 or (slot in (2, 3, 4) and completed_transport not in (1, 2)):')
        # U1 uses the same terminal wall confirmation before and after the transfer.
        source = replace_once(source, 'if slot in (1, 2) and phase in ("upper_B_seam_rear_reseat", "upper_B_extra_seam_rear_reseat"):',
                              'if slot == 2 and phase in ("upper_B_seam_rear_reseat", "upper_B_extra_seam_rear_reseat"):')
    u1_slide_preload_N = 2.0 if observed_skills else 3.25
    source = replace_once(source, '                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)',
                          f'''                if slot == 1:
                    # Set the wall preload for lateral sliding.
                    left_hold.target_force_N = {u1_slide_preload_N}
                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)''')
    source = replace_once(source, 'force_paced_samples if force_paced_lateral or slot4_boundary else None,',
                          '''force_paced_samples if (force_paced_lateral or slot4_boundary
                                                        or (slot in (2, 5) and phase == "upper_B_rear")) else None,''')
    source = replace_once(source, 'adjust_targets=relieve_downward_load if force_paced_lateral or slot4_boundary else None)',
                          '''adjust_targets=relieve_downward_load if (force_paced_lateral or slot4_boundary
                                                        or (slot == 5 and phase == "upper_B_rear")) else None)''')
    source = replace_once(source, '                            elif lead > .0016 or pending > .00020 or missing_rear_contact >= 12:',
                          '                            elif lead > max_lead or pending > .00020 or missing_rear_contact >= 12:')
    source = replace_once(source, '                        waiting = entry_stable = 0\n                        while index < count:',
                          '''                        waiting = entry_stable = 0
                        # The softer front wrist needs extra spring travel to overcome sliding resistance.
                        max_lead = .0024 if slot == 5 and phase == "upper_B_rear" else .0016
                        while index < count:''')
    source = replace_once(source, '                touch_face(support_side, front_normal, .018, "upper_B_front_touch")',
                          '''                if slot == 5:
                    # Use the same contact compliance already used by the lateral pusher.
                    controller = robots[support_side].get_articulation_controller()
                    stiffness, damping = (np.asarray(values).copy() for values in controller.get_gains())
                    arm_joints = joints[support_side]
                    stiffness[arm_joints] *= .35
                    damping[arm_joints] *= math.sqrt(.35)
                    controller.set_gains(kps=stiffness, kds=damping)
                touch_face(support_side, front_normal, .018, "upper_B_front_touch")''')
    source = replace_once(source, 'if terminal_reference_x is not None and headroom < -.15:',
                          'if (terminal_reference_x is not None or phase == "upper_B_side_pre_seam") and headroom < -.15:')
    source = replace_once(source, '                        probe_distance = .000095',
                          '''                        # A short force rise can occur while the last fraction of a millimetre is still on static friction.
                        probe_distance = .00030''')
    source = replace_once(source, '                                if advance >= probe_distance - .000004:',
                          '''                                # A seated box can stop the compliant command short of the full travel.
                                if advance >= .00020:''')
    if observed_skills:
        source = replace_once(source, '''                                if advance >= .00020:
                                    held.append((wrist_x, center[0], normal, gap2d))
                                else:
                                    held.clear()
                                if len(held) == settle_frame_count:''',
                              '''                                held.append((wrist_x, center[0], normal, gap2d))
                                if len(held) == settle_frame_count:''')
        source = replace_once(source, '''                                    if (increase >= .15 and np.min(loads) >= threshold and np.max(gaps) <= .00045''',
                              '''                                    # The force controller can reach its preload before the nominal stroke.
                                    probe_loaded = advance >= .00020 or (
                                        advance >= .00010 and float(np.mean(loads)) >= probe_axial_limit - .15)
                                    if (probe_loaded and increase >= .15
                                            and np.min(loads) >= threshold and np.max(gaps) <= .00045''')
    source = replace_once(source, '                wall_impulses = deque(maxlen=settle_frame_count)\n                for _ in range(6000):',
                          '''                wall_impulses = deque(maxlen=settle_frame_count)
                side_probe_start = None
                for _ in range(6000):''')
    source = replace_once(source, '                    if front_engaged and measured_force >= contact_threshold and negative_gap_limit <= near <= near_limit:',
                          '''                    candidate = front_engaged and measured_force >= contact_threshold and negative_gap_limit <= near <= near_limit
                    if steer_front and candidate and side_probe_start is None:
                        side_probe_start = float(position[axis])
                    side_probe_travel = 0. if side_probe_start is None else sign * (position[axis] - side_probe_start)
                    # Let the wrist build a load above sliding resistance within a short position allowance.
                    probe_load = min(7.25, max(6.6, contact_threshold + .7))
                    probing = steer_front and side_probe_start is not None and (
                        side_probe_travel < .00035 or (side_probe_travel < .001 and measured_force < probe_load))
                    if candidate and not probing:''')
    source = replace_once(source, '                        if front_engaged:\n                            position[axis] += sign * increment',
                          '''                        force_headroom = (not probing or np.linalg.norm(
                            last_force_sample[side]["tcp_gravity_corrected_world"][:3]) < 7.8)
                        if front_engaged and force_headroom:
                            position[axis] += sign * increment''')
    if observed_skills:
        source = replace_once(source, '                side_probe_start = None',
                              '''                side_wrist_window = deque(maxlen=settle_frame_count)
                if steer_front:
                    terminal_height = NormalForceController(origin[2], target_force_N=0.,
                        gain_m_per_Ns=.00055, max_offset_m=.0014, max_speed_m_s=.001, max_force_N=None)''')
        start = source.index('                    negative_gap_limit = -.002 if slot4_boundary else -.003')
        end = source.index('                        forces.append(measured_force)', start)
        source = source[:start] + '''                    negative_gap_limit = -.002 if slot4_boundary and not steer_front else -.003
                    if steer_front:
                        side_wrist_window.append(float(last_force_sample[side]["tcp_position_m"][axis]))
                        stationary = (len(side_wrist_window) == settle_frame_count
                                      and np.ptp(side_wrist_window) <= .000012)
                        contact_threshold = min(7.25, max(6.6, contact_threshold + .7))
                    else:
                        stationary = True
                    if (front_engaged and stationary and measured_force >= contact_threshold
                            and negative_gap_limit <= near <= near_limit):
''' + source[end:]
        source = replace_once(source, '''                        force_headroom = (not probing or np.linalg.norm(
                            last_force_sample[side]["tcp_gravity_corrected_world"][:3]) < 7.8)
                        if front_engaged and force_headroom:
                            position[axis] += sign * increment''',
                              '''                        if steer_front:
                            resultant = float(np.linalg.norm(last_force_sample[side]["tcp_gravity_corrected_world"][:3]))
                            if resultant > 7.4:
                                # Remove stored drive compression instead of holding the high-force command.
                                relief = min(.0012, .0012 * (resultant - 7.4)) / 120
                                position[axis] -= sign * relief
                            elif front_engaged:
                                position[axis] += sign * increment
                        elif front_engaged:
                            position[axis] += sign * increment''')
        source = replace_once(source, '                    target_rotation = slot4_side_rotation(steer_rotation) if steer_front else tool_rotation',
                              '''                    if steer_front:
                        # Continue the lateral travel's upward shear in the terminal stroke.
                        vertical_force = float(last_force_sample[side]["tcp_gravity_corrected_world"][2])
                        vertical_error = vertical_force + 1.5
                        excess = math.copysign(max(0., abs(vertical_error) - .20), vertical_error)
                        position[2] = terminal_height.update(-excess)
                    target_rotation = slot4_side_rotation(steer_rotation) if steer_front else tool_rotation''')
    source = replace_once(source, '        def save_upper_checkpoint(next_slot, placement=None):',
                          '        def save_upper_checkpoint(next_slot, placement=None, *, filename=None):')
    source = replace_once(source, '            name = "upper_landing_checkpoint.json" if placement else "upper_row_checkpoint.json"',
                          '            name = filename or ("upper_landing_checkpoint.json" if placement else "upper_row_checkpoint.json")')
    source = replace_once(source, '                retouch_after_support_seam(center, rotation, local_side_offset,',
                          '''                transport_index = 2 if extra else 1
                placement = upper_pose_records({str(current_task["cube_id"]): cube})[0]
                placement["seam_entry_resume"] = {
                    "slot": slot, "cube_id": current_task["cube_id"], "completed_transport": transport_index,
                    "source_step": tick, "source_event": report["phases"][-1],
                    "grounded_evidence": {"reattained_RGB": {"position_m": center.tolist(),
                                           "rotation_world": rotation.tolist(), "source": "rear_rgbd_square"},
                                          "opened_grippers": {arm: grippers.is_closed(arm) for arm in arms}},
                }
                save_upper_checkpoint(slot, placement, filename=f"upper{slot}_after_seam{transport_index}.json")
                retouch_after_support_seam(center, rotation, local_side_offset,''')
    source = replace_block(source, '            camera.resume()\n            diagnostics = []', '            if len(estimates) < 3:',
                           '''            diagnostics = []
            hint, attitude = active_square_hint(phase, support_z)
            for _ in range(30):
                step(vision_period)
                frame = camera.get_current_frame(clone=True)
                frame_id = int(frame.get("rendering_frame", 0))
                stamp = float(frame.get("rendering_time", -1.))
                age = float(world.current_time) - stamp
                if frame_id <= 0 or frame_id in frames or stamp < started or not 0 <= age <= .20:
                    continue
                frames.add(frame_id)
                pose = rows.locate(frame, support_z, hint, attitude)
                diagnostics.append({"id": frame_id, "age": age, "fit": pose is not None})
                if pose is not None:
                    estimates.append(pose)
                    exposure_ids.append((frame_id, stamp))
                if len(estimates) >= 3:
                    if phase in ("A_pickup_pose", "front_floor_landing_pose", "upper_landing_pose",
                                 "front_final_pose", "upper_final_pose", "upper_B_straight_entry_pose"):
                        path = output_directory / f"Cube_{current_task['cube_id']}_{phase}.png"
                        rows.save_outline(frame, support_z, pose, path)
                    break
''')
    source = replace_once(source, '                report["failed_observation"] = diagnostics',
                          '''                report["failed_observation"] = diagnostics
                failed = camera.get_current_frame(clone=True)
                np.savez_compressed(output_directory / f"Cube_{current_task['cube_id']}_{phase}_unobserved.npz",
                                    rgba=failed.get("rgba", []), depth_m=failed.get("distance_to_image_plane", []),
                                    hint=hint, support_z=support_z)''')
    source = replace_once(source, '                "phase": phase, "source": "supply_rgb" if camera is supply_camera else "rear_rgb",',
                          '                "phase": phase, "source": "rear_rgbd_square",')
    source = replace_once(source, '            return control_center.copy(), control_rotation.copy()\n\n        def step',
                          '''            if phase in ("front_final_pose", "upper_final_pose"):
                rows.placed[current_task["cube_id"]] = (control_center.copy(), control_rotation.copy())
            return control_center.copy(), control_rotation.copy()

        def step''')

    source = replace_once(source, '            active_id = vision_estimator.marker_id\n', '')
    source = replace_block(source, '                from two_row_physx import frame_marker_corners',
                           '                if all(index in observations',
                           '''                for index in range(count):
                    if str(index + 1) not in lower_bodies and str(index + 1) not in upper_bodies:
                        continue
                    pose = rows.neighbor(frame, index + 1)
                    if pose is None:
                        continue
                    position, rotation = pose
                    exposure_windows[index].append({"frame_id": frame_id, "capture_time_s": stamp,
                        "position_m": position.tolist(), "yaw_rad": float(np.arctan2(rotation[1, 0], rotation[0, 0]))})
                    filtered = filters[index].update(frame_id, stamp, position, rotation)
                    if filtered is not None:
                        observations[index] = filtered
''')
    source = replace_once(source, '            if missing_required:\n                raise RuntimeError("Fresh median neighbor pose unavailable: " + ", ".join(missing_required))',
                          '''            unavailable = [name for name in missing_required if int(name[5:]) not in rows.placed]
            if unavailable:
                failed = vision_camera.get_current_frame(clone=True)
                np.savez_compressed(output_directory / f"Cube_{current_task['cube_id']}_neighbor_miss.npz",
                                    rgba=failed.get("rgba", []), depth_m=failed.get("distance_to_image_plane", []))
                raise RuntimeError("Fresh square edges unavailable: " + ", ".join(unavailable))''')
    source = replace_once(source, '''                if f"Cube_{index + 1}" not in available:
                    continue
                position, rotation, capture_time = observations[index]''',
                          '''                if f"Cube_{index + 1}" not in available:
                    if f"Cube_{index + 1}" in required and index + 1 in rows.placed:
                        position, rotation = rows.placed[index + 1]
                        observed.append((f"Cube_{index + 1}", position, rotation))
                        rows_log.append({"name": f"Cube_{index + 1}", "source": "last_observed_square",
                                         "position_m": position.tolist(), "rotation_world": rotation.tolist()})
                    continue
                position, rotation, capture_time = observations[index]
                rows.placed[index + 1] = position, rotation''')
    # Keep the shared pose tracker distinct from the per-call list of reported neighbors.
    start = source.index('        def observe_placed_cubes(')
    end = source.index('        def lower_poses(', start)
    body = source[start:end].replace('observed, rows = [], []', 'observed, rows_log = [], []')
    body = body.replace('                rows.append(', '                rows_log.append(')
    body = body.replace('"neighbors": rows,', '"neighbors": rows_log,')
    source = source[:start] + body + source[end:]
    source = replace_once(source, '\n                    "initial_boxes": upper_pose_records(lower_bodies),',
                          '''
                    "markerless_placements": {key: [p.tolist(), r.tolist()] for key, (p, r) in rows.placed.items()},
                    "initial_boxes": upper_pose_records(lower_bodies),''')
    source = replace_once(source, '\n                             "initial_boxes": upper_pose_records(lower_bodies),',
                          '''
                             "markerless_placements": {key: [p.tolist(), r.tolist()] for key, (p, r) in rows.placed.items()},
                             "initial_boxes": upper_pose_records(lower_bodies),''')
    source = replace_once(source, '                "initial_boxes": upper_pose_records(lower_bodies) + [',
                          '''                "markerless_placements": {key: [p.tolist(), r.tolist()] for key, (p, r) in rows.placed.items()},
                "initial_boxes": upper_pose_records(lower_bodies) + [''')
    source = source.replace('no stable fresh ArUco observation', 'no stable fresh square observation')
    source = source.replace('rear_RGB_marker_support_plane', 'rear_RGBD_square_edges')
    source = source.replace('READY_FOR_RGB_LOCALIZATION', 'READY_FOR_DEPTH_LOCALIZATION')
    source = source.replace('vision_estimator,', 'rows,')
    source = replace_once(source, '            "camera_source": "RGB + ArUco + solvePnP + fixed camera calibration only",',
                          '            "camera_source": "rear_rgbd_square",\n'
                          '            "camera_stream_source": "rear_rgbd_square",')
    return source
