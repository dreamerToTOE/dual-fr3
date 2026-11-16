"""Reproduce a geometry-checked B2 approach and synchronized dual-arm push.

Run with Isaac Sim's python.sh. This CPU planner does not execute PhysX.
The initial right configuration is the A_landing preflight configuration;
the cube and the left arm must already be at the B1 back-wall endpoint.
"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fr3_collision import FR3CollisionChecker
from fr3_trajectory import plan_joint_trajectory


ROOT = Path(__file__).resolve().parent
ISAAC = Path('/home/ubuntu2004/isaacsim-4.5.0/exts')
sys.path.insert(0, str(ISAAC / 'isaacsim.robot_motion.lula/pip_prebundle'))
import lula


def build_plan():
    cfg = json.loads((ROOT / 'fr3_four_wall_scene_config.json').read_text())
    pre = json.loads((ROOT / 'validation/fr3_ab_ik_preflight.json').read_text())
    directory = ISAAC / 'isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3'
    description = yaml.safe_load((directory / 'rmpflow/fr3_robot_description.yaml').read_text())
    kinematics = lula.load_robot(str(directory / 'rmpflow/fr3_robot_description.yaml'),
                                 str(directory / 'fr3.urdf')).kinematics()
    xml = ET.parse(directory / 'fr3.urdf')
    limits = [xml.find(f"joint[@name='fr3_joint{i}']/limit") for i in range(1, 8)]
    lower = np.array([float(limit.get('lower')) for limit in limits]) + 1e-6
    upper = np.array([float(limit.get('upper')) for limit in limits]) - 1e-6
    right_rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.]])
    left_rotation = np.diag([1., -1., -1.])
    half = float(cfg['cube_size_m']) / 2
    phase_a = cfg['task']['phase_A']
    phase_b1 = cfg['task']['phase_B1']
    phase_b2 = cfg['task']['phase_B2']

    class Solver:
        def __init__(self, base):
            self.base = np.asarray(base[:3], dtype=float)
            self.rotation = Rotation.from_euler('z', base[3]).as_matrix()

        def compute_forward_kinematics(self, frame, q):
            pose = kinematics.pose(q, frame)
            return (self.rotation @ pose.translation.reshape(3) + self.base,
                    self.rotation @ pose.rotation.matrix())

    solvers = {side: Solver(base) for side, base in cfg['fr3']['bases'].items()}
    right = solvers['right']
    checker = FR3CollisionChecker(cfg, solvers, description['collision_spheres'])

    def follow(side, seed, targets, rotation):
        path = []
        for target in targets:
            seed = np.asarray(seed)
            solver = solvers[side]
            offset = np.array([0, -.155 if side == 'left' else .155, .08])

            def residual(q):
                flange, actual_rotation = solver.compute_forward_kinematics('fr3_link8', q)
                return np.r_[flange + actual_rotation @ offset - target,
                             .2 * Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec(),
                             .00001 * (q - seed)]

            fit = least_squares(residual, np.clip(seed, lower, upper),
                                bounds=(np.maximum(lower, seed - .25), np.minimum(upper, seed + .25)),
                                max_nfev=70, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            error = residual(fit.x)
            if np.linalg.norm(error[:3]) >= .0001 or np.linalg.norm(error[3:6]) >= .0002:
                raise RuntimeError(f'{side} continuous IK failed at {np.asarray(target).tolist()}')
            seed = fit.x
            path.append(seed)
        return np.asarray(path)

    def waypoint(side, name):
        row = next(w for w in pre['waypoints'] if w['arm'] == side and w['waypoint'] == name)
        if not row['success']:
            raise RuntimeError(f'{side} {name} seed is not a successful IK result')
        return np.asarray(row['joint_positions_rad'])

    left_hold = waypoint('left', 'B1_back_wall')
    initial_right = waypoint('right', 'A_landing')
    cube_initial = np.asarray(phase_b1['target_center_m'], dtype=float)
    cube_final = np.asarray(phase_b2['target_center_m'], dtype=float)
    base_initial = np.asarray(cfg['fr3']['bases']['right'], dtype=float)
    rail_y = float(phase_b2['rail_assist']['base_y_m'])
    transit_z = 1.05
    contact_tcp_x = cube_initial[0] + half
    # Leave the right cup outside the cube while descending to the suction
    # plane, then approach the right face horizontally along -X.
    approach_standoff_m = 0.01
    approach_tcp_x = contact_tcp_x + approach_standoff_m
    high_y = cube_initial[1] - (rail_y - base_initial[1])
    cube_knots = np.linspace(cube_initial, cube_final, 43)
    left_contact = np.vstack((left_hold, follow('left', left_hold, cube_knots[1:] - [0, half, 0], left_rotation)))
    right.base[1] = rail_y
    target = cube_initial + [half, 0, 0]
    flange = target - right_rotation @ [0, .155, .08]
    pose = lula.Pose3(lula.Rotation3(right.rotation.T @ right_rotation),
                      right.rotation.T @ (flange - right.base))
    ik_cfg = lula.CyclicCoordDescentIkConfig()
    ik_cfg.position_tolerance, ik_cfg.orientation_tolerance = .0001, .001
    ik_cfg.cspace_seeds = [np.array(cfg['fr3']['official_start_q_rad'])]
    ik_cfg.sampling_seed, ik_cfg.max_num_descents = 0, 10
    result = lula.compute_ik_ccd(kinematics, pose, 'fr3_link8', ik_cfg)
    if not result.success:
        raise RuntimeError('B2 contact IK failed')
    q0 = np.asarray(result.cspace_position).reshape(7)
    right_contact = np.vstack((q0, follow('right', q0, cube_knots[1:] + [half, 0, 0], right_rotation)))
    alignment_q = follow('right', q0, [[approach_tcp_x, cube_initial[1], cube_initial[2]]], right_rotation)[-1]
    raised = np.vstack((alignment_q, follow('right', alignment_q,
                        [[approach_tcp_x, cube_initial[1], z]
                         for z in np.linspace(cube_initial[2], transit_z, 80)[1:]], right_rotation)))
    right.base[1] = base_initial[1]
    lift = np.vstack((initial_right, follow('right', initial_right,
                      [[approach_tcp_x, phase_a['landing_center_m'][1], z]
                       for z in np.linspace(cube_initial[2], transit_z, 80)[1:]], right_rotation)))
    shift = follow('right', lift[-1], [[approach_tcp_x, y, transit_z]
                                       for y in np.linspace(phase_a['landing_center_m'][1], high_y, 25)[1:]], right_rotation)
    high_transit = np.vstack((lift, shift, raised[-1]))
    phases = {}
    for name, path, base_y, contacts in (
            ('high_transit', high_transit, base_initial[1], ('left',)),
            ('descent', raised[::-1], rail_y, ('left', 'right'))):
        right.base[1] = base_y
        times, samples = plan_joint_trajectory(path, max_joint_speed=.65, sample_hz=120)
        if np.any(samples < lower) or np.any(samples > upper):
            raise RuntimeError(f'{name} trajectory exceeds FR3 joint limits')
        validation = checker.assert_clear_path({'right': samples, 'left': np.tile(left_hold, (len(samples), 1))},
                                               cube_centers=cube_initial, cup_contacts=contacts,
                                               max_joint_step_rad=.005)
        phases[name] = dict(joint_path_rad=path.tolist(), duration_s=float(times[-1]),
                            samples_120hz=len(samples), validation=validation)
    right.base[1] = rail_y
    approach_targets = [[x, cube_initial[1], cube_initial[2]]
                        for x in np.linspace(approach_tcp_x, contact_tcp_x, 21)[1:]]
    approach = follow('right', alignment_q, approach_targets, right_rotation)
    approach[-1] = q0
    times, samples = plan_joint_trajectory(approach, max_joint_speed=.65, sample_hz=120)
    validation = checker.assert_clear_path(
        {'right': samples, 'left': np.tile(left_hold, (len(samples), 1))},
        cube_centers=cube_initial, cup_contacts=('left', 'right'), max_joint_step_rad=.005)
    phases['approach'] = dict(joint_path_rad=approach.tolist(), duration_s=float(times[-1]),
                              samples_120hz=len(samples), validation=validation,
                              start_tcp_m=[approach_tcp_x, float(cube_initial[1]), float(cube_initial[2])],
                              contact_tcp_m=[contact_tcp_x, float(cube_initial[1]), float(cube_initial[2])],
                              standoff_m=approach_standoff_m, direction='-X')
    # Both arms share knot times; independent retiming loses their contact synchronization.
    times, samples = plan_joint_trajectory(np.hstack((left_contact, right_contact)), max_joint_speed=.65, sample_hz=120)
    cube_samples, sync_error = [], []
    for q in samples:
        flange, rotation = right.compute_forward_kinematics('fr3_link8', q[7:])
        center = flange + rotation @ [0, .155, .08] - [half, 0, 0]
        cube_samples.append(center)
        flange, rotation = solvers['left'].compute_forward_kinematics('fr3_link8', q[:7])
        sync_error.append(np.linalg.norm(flange + rotation @ [0, -.155, .08] - (center - [0, half, 0])))
    validation = checker.assert_clear_path({'left': samples[:, :7], 'right': samples[:, 7:]},
                                           cube_centers=cube_samples, cup_contacts=('left', 'right'),
                                           max_joint_step_rad=.005)
    if np.max(sync_error) > .001 or np.any(samples < np.tile(lower, 2)) or np.any(samples > np.tile(upper, 2)):
        raise RuntimeError('B2 synchronization or joint-limit validation failed')
    phases['contact'] = dict(left_joint_path_rad=left_contact.tolist(), right_joint_path_rad=right_contact.tolist(),
                             duration_s=float(times[-1]), samples_120hz=len(samples), validation=validation,
                             left_sync_error_max_m=float(np.max(sync_error)),
                             max_joint_speed_rad_s=float(np.max(np.abs(np.diff(samples, axis=0) / np.diff(times)[:, None]))))
    for y in np.linspace(base_initial[1], rail_y, 181):
        right.base[1] = y
        issues = checker.check({'left': left_hold, 'right': raised[-1]}, cube_center=cube_initial, cup_contacts=('left',))
        if issues:
            raise RuntimeError(f'Rail motion blocked at Y={y}: {issues[0]}')
    return dict(status='PASS_COLLISION_CHECKED_B2_PLAN',
                scope='nominal_B1_end_to_B2_suction_plane_alignment_horizontal_approach_rail_and_synchronized_push_official_Lula_spheres_full_tool_OBBs',
                limitations='Planning geometry only; PhysX mesh execution requires validation with every collision enabled. Initial right configuration is A_landing preflight, with Cube already at B1 back wall.',
                margin_m=.002, max_validation_joint_step_rad=.005, right_rotation_world=right_rotation.tolist(),
                right_base_initial_m=base_initial.tolist(),
                right_base_contact_m=[float(base_initial[0]), rail_y, float(base_initial[2]), float(base_initial[3])],
                left_base_m=solvers['left'].base.tolist(), cube_initial_m=cube_initial.tolist(), cube_final_m=cube_final.tolist(),
                transit_z_m=transit_z, approach_standoff_m=approach_standoff_m,
                initial_right_q_rad=initial_right.tolist(), left_hold_q_rad=left_hold.tolist(), phases=phases,
                rail=dict(hold_right_q_rad=raised[-1].tolist(), start_y_m=float(base_initial[1]), end_y_m=rail_y,
                          samples_checked=181, max_y_step_m=.0025, validation=dict(status='CLEAR', issues=[])),
                timing='Retime hstack(left_path,right_path) together; split the resulting 14-joint samples.')


if __name__ == '__main__':
    try:
        report = build_plan()
    except Exception as exc:
        (ROOT / 'validation/fr3_b2_collision_plan.json').write_text(json.dumps(dict(status='BLOCKED', error=str(exc)), indent=2) + '\n')
        raise
    (ROOT / 'validation/fr3_b2_collision_plan.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'scope', 'margin_m')}), flush=True)
