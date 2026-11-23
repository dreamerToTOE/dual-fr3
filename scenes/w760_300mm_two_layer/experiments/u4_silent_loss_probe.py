"""Inject a post-rear-confirmation physical side-loss perturbation for U4."""

from textwrap import indent


PROBE = r'''
U4_SILENT_LOSS_PROBE_ACTIVE = False

def run_u4_weak_reload(recovery):
    # Move only the side neighbor. Its current contact impulse feeds the fixture, not the observer.
    # The fixture deliberately changes only this experiment.  It gives the
    # neighbour a thin, visible low-friction pad and a low-friction body
    # material, so a bounded sub-newton drive can test whether the 6 mm release
    # gap can close without hiding the contact from PhysX.
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    neighbor = upper_bodies["8"]
    origin, neighbor_orientation = neighbor.get_world_pose()
    origin = [float(value) for value in origin]
    origin_x = float(origin[0])
    active_center, _ = cube.get_world_pose()
    cube_size_m = float(CONFIG["cube_size_m"])
    pad_thickness_m = .001
    pad_size_m = (.001, .120, .120)
    compliant_stiffness_N_m = 1500.0
    target_x = (float(active_center[0]) + cube_size_m + pad_thickness_m
                - float(U4_WEAK_RELOAD_N) / compliant_stiffness_N_m)
    low_friction = .005
    body_material = UsdShade.Material.Define(stage, "/World/U4WeakSideLowFrictionBodyMaterial")
    body_api = UsdPhysics.MaterialAPI.Apply(body_material.GetPrim())
    body_api.CreateStaticFrictionAttr().Set(low_friction)
    body_api.CreateDynamicFrictionAttr().Set(.003)
    body_api.CreateRestitutionAttr().Set(0.0)
    body_physx = PhysxSchema.PhysxMaterialAPI.Apply(body_material.GetPrim())
    body_physx.CreateFrictionCombineModeAttr().Set("min")
    body_physx.CreateRestitutionCombineModeAttr().Set("min")
    body_physx.CreateDampingCombineModeAttr().Set("average")
    body_physx.CreateCompliantContactStiffnessAttr().Set(0.0)
    pad_material = UsdShade.Material.Define(stage, "/World/U4WeakSideCompliantMaterial")
    pad_api = UsdPhysics.MaterialAPI.Apply(pad_material.GetPrim())
    pad_api.CreateStaticFrictionAttr().Set(low_friction)
    pad_api.CreateDynamicFrictionAttr().Set(.003)
    pad_api.CreateRestitutionAttr().Set(0.0)
    pad_physx = PhysxSchema.PhysxMaterialAPI.Apply(pad_material.GetPrim())
    pad_physx.CreateFrictionCombineModeAttr().Set("min")
    pad_physx.CreateRestitutionCombineModeAttr().Set("min")
    pad_physx.CreateDampingCombineModeAttr().Set("average")
    pad_physx.CreateCompliantContactStiffnessAttr().Set(1500.0)
    pad_physx.CreateCompliantContactDampingAttr().Set(15.0)
    pad_physx.CreateCompliantContactAccelerationSpringAttr().Set(False)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Cube_8")).Bind(
        body_material, UsdShade.Tokens.strongerThanDescendants, "physics")
    support_paths = [path for path in ("/World/Cube_15", "/World/Cube_14")
                     if stage.GetPrimAtPath(path).IsValid()]
    for support_path in support_paths:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(support_path)).Bind(
            body_material, UsdShade.Tokens.strongerThanDescendants, "physics")
        filtering_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/Cube_8"))
        filtering_api.CreateFilteredPairsRel().AddTarget(Sdf.Path(support_path))

    rail_path = "/World/U4WeakSlideRail"
    rail_size = (.18, .18, .012)
    rail = UsdGeom.Cube.Define(stage, rail_path)
    rail.CreateSizeAttr(1.0)
    rail_xform = UsdGeom.Xformable(rail.GetPrim())
    rail_xform.AddTranslateOp().Set(
        Gf.Vec3d(origin_x, float(origin[1]), float(origin[2]) - cube_size_m / 2.0 - .010 - .006))
    rail_xform.AddScaleOp().Set(Gf.Vec3f(*rail_size))
    rail.CreateDisplayColorAttr([Gf.Vec3f(.28, .32, .38)])
    UsdShade.MaterialBindingAPI.Apply(rail.GetPrim()).Bind(
        body_material, UsdShade.Tokens.strongerThanDescendants, "physics")

    sled_path = "/World/U4WeakSlideSled"
    sled_size = (.145, .145, .010)
    sled = UsdGeom.Cube.Define(stage, sled_path)
    sled.CreateSizeAttr(1.0)
    sled_xform = UsdGeom.Xformable(sled.GetPrim())
    sled_xform.AddTranslateOp().Set(
        Gf.Vec3d(origin_x, float(origin[1]), float(origin[2]) - cube_size_m / 2.0 - sled_size[2] / 2.0))
    sled_xform.AddScaleOp().Set(Gf.Vec3f(*sled_size))
    sled.CreateDisplayColorAttr([Gf.Vec3f(.16, .20, .25)])
    UsdShade.MaterialBindingAPI.Apply(sled.GetPrim()).Bind(
        body_material, UsdShade.Tokens.strongerThanDescendants, "physics")

    pad_path = "/World/Cube_8/U4WeakSideCompliantPad"
    pad = UsdGeom.Cube.Define(stage, pad_path)
    pad.CreateSizeAttr(1.0)
    pad_xform = UsdGeom.Xformable(pad.GetPrim())
    pad_xform.AddTranslateOp().Set(Gf.Vec3d(-.5 - pad_thickness_m / (cube_size_m * 2.), 0.0, 0.0))
    pad_xform.AddScaleOp().Set(Gf.Vec3f(*(size / cube_size_m for size in pad_size_m)))
    pad.CreateDisplayColorAttr([Gf.Vec3f(.12, .35, .65)])
    UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim()).CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(
        pad_material, UsdShade.Tokens.strongerThanDescendants, "physics")

    recovery["weak_fixture"] = {
        "target_N": U4_WEAK_RELOAD_N, "initial_neighbor_x_m": origin_x,
        "near_step_limit_m": 4.e-6, "target_neighbor_x_m": target_x,
        "fixture": "Cube_8_low_friction_compliant_pad_with_transX_drive",
        "pad_path": pad_path, "pad_size_m": list(pad_size_m),
        "pad_thickness_m": pad_thickness_m, "static_friction": low_friction,
        "dynamic_friction": .003, "restitution": 0.0,
        "low_friction_support_paths": support_paths,
        "support_model": "independent_pose_controlled_prismatic_rail",
        "rail_path": rail_path, "rail_size_m": list(rail_size),
        "sled_path": sled_path, "sled_size_m": list(sled_size),
        "filtered_support_paths": support_paths,
        "rail_top_m": float(origin[2]) - cube_size_m / 2.0 - .010,
        "support_clearance_m": .002,
        "compliant_contact_stiffness_N_m": compliant_stiffness_N_m,
        "compliant_contact_damping_Ns_m": 15.0,
    }
    recovery["weak_fixture"].update({"fixture": "Cube_8_low_friction_compliant_pad_with_pose_controlled_sled",
                                      "kinematic_target_position_m": target_x,
                                      "approach_displacement_m": target_x - origin_x})
    neighbor.set_linear_velocity(np.zeros(3, dtype=np.float32))
    neighbor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    contact_frozen = False
    frozen_x = None
    for reload_index in range(recovery["reload_observation_steps"]):
        alpha = min(1.0, (reload_index + 1) / float(recovery["reload_observation_steps"]))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        position = np.asarray(origin, dtype=np.float32)
        position[0] = (frozen_x if contact_frozen else
                       origin_x + (target_x - origin_x) * smooth)
        neighbor.set_world_pose(position, neighbor_orientation)
        neighbor.set_linear_velocity(np.zeros(3, dtype=np.float32))
        neighbor.set_angular_velocity(np.zeros(3, dtype=np.float32))
        step()
        check_current_collision(contact_monitor.phase)
        if not contact_frozen:
            snapshot = contact_monitor.cube_contact_snapshot()
            pad_touch = any(
                "/World/Cube_10" in {pair["collider0"], pair["collider1"]}
                and "/World/Cube_8/U4WeakSideCompliantPad" in {
                    pair["collider0"], pair["collider1"]}
                and pair["current_step_observed"]
                and pair["current_separation_observed"]
                and pair["separation_m"] <= 0.0
                for pair in snapshot["pairs"])
            if pad_touch:
                contact_frozen = True
                frozen_x = float(position[0])
    recovery["weak_fixture"].update({"contact_position_frozen": contact_frozen,
                                      "frozen_neighbor_x_m": frozen_x})
    recovery["weak_fixture"]["final_neighbor_x_m"] = float(neighbor.get_world_pose()[0][0])

def run_u4_silent_loss_probe():
    """Separate the active box from Cube 8 while preserving rear support."""
    import numpy as np
    global U4_SILENT_LOSS_PROBE_ACTIVE
    import u4_contact_observer

    probe_force_N = 0.0
    probe_force_steps = 0
    previous_phase = contact_monitor.phase
    info = report["u4_silent_loss_probe"] = {
        "start_step": tick,
        "arm": "left",
        "external_force_world_N": [probe_force_N, 0.0, 0.0],
        "duration_steps": 60,
        "force_steps": probe_force_steps,
        "right_arm_command_held": True,
        "physical_perturbation": "cube_pose_x_minus_6mm",
    }
    contact_monitor.phase = "upper_B_u4_silent_loss_probe"
    U4_SILENT_LOSS_PROBE_ACTIVE = True
    u4_contact_observer.U4_SILENT_LOSS_PROBE_ACTIVE = True
    left_q = applied_targets["left"].copy()
    left_tcp, left_rotation = tcp_fk("left", left_q)
    left_target = left_tcp.copy()
    left_target[0] -= .0040
    left_goal = cartesian_joint_target("left", left_q, left_target, left_rotation)
    try:
        record(contact_monitor.phase, "STARTED", cube_id=current_task["cube_id"], source="post_rear_force_confirmed")
        center, orientation = cube.get_world_pose()
        center = np.array(center, dtype=np.float32)
        initial_x = float(center[0])
        center[0] -= .006
        cube.set_world_pose(center, orientation)
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for index in range(60):
            alpha = (index + 1) / 60.
            smooth = alpha * alpha * (3. - 2. * alpha)
            left_command = left_q + smooth * (left_goal - left_q)
            command_positions("left", left_command)
            command_positions("right", applied_targets["right"])
            step()
            check_current_collision(contact_monitor.phase)
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        center, _ = cube.get_world_pose()
        info.update({"end_step": tick, "cube_position_m": np.asarray(center).tolist()})
        record(contact_monitor.phase, "COMPLETED", **info)
        if U4_SIDE_RECOVERY_PROBE:
            u4_contact_observer.U4_SILENT_LOSS_PROBE_ACTIVE = False
            recovery = report["u4_side_recovery_probe"] = {
                "start_step": info["start_step"], "unmask_step": tick,
                "unmasked_dwell_steps": 90, "reload_observation_steps": 120,
                "restored_x_m": initial_x + .00015,
            }
            record("upper_B_u4_reload", "UNMASKED_DWELL_STARTED", step=tick)
            for _ in range(recovery["unmasked_dwell_steps"]):
                step()
                check_current_collision(contact_monitor.phase)
            recovery["reload_command_step"] = tick
            record(contact_monitor.phase, "RELOAD_COMMANDED", step=tick)
            if U4_WEAK_RELOAD_N is not None:
                recovery.pop("restored_x_m")
                recovery["reload_observation_steps"] = 1200
                run_u4_weak_reload(recovery)
            else:
                center, orientation = cube.get_world_pose()
                center = np.array(center, dtype=np.float32)
                center[0] = recovery["restored_x_m"]
                cube.set_world_pose(center, orientation)
                cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
                for _ in range(recovery["reload_observation_steps"]):
                    step()
                    check_current_collision(contact_monitor.phase)
            recovery["end_step"] = tick
            record(contact_monitor.phase, "COMPLETED", **recovery)
            raise RuntimeError("U4 side-recovery experiment completed by request")
        if STOP_AFTER_U4_SILENT_LOSS:
            raise RuntimeError("U4 silent-loss probe completed by request")
    finally:
        U4_SILENT_LOSS_PROBE_ACTIVE = False
        u4_contact_observer.U4_SILENT_LOSS_PROBE_ACTIVE = False
        contact_monitor.phase = previous_phase
'''


def inject(source):
    from source_hooks import replace_once

    # The probe is nested in execute_upper_push, while the observer is a
    # separate imported module. Define the latch in the generated runner and
    # mirror it into the observer module only during this experiment.
    source = source.replace(
        'from __future__ import annotations\n',
        'from __future__ import annotations\n\nU4_SILENT_LOSS_PROBE_ACTIVE = False\n',
        1,
    )

    source = replace_once(
        source,
        '            def push_u4_to_rear(side, phase, target):\n',
        indent(PROBE.strip() + "\n\n", "            ")
        + '            def push_u4_to_rear(side, phase, target):\n',
    )
    marker = '''                                record(phase, "FORCE_CONFIRMED", arm=side, contact_target=target,
                                       held_samples=len(confirmations), normal_force_N=float(loads.mean()),
                                       normal_force_samples_N=loads.tolist(), estimated_gap_m=float(near),
                                       witnessed_target_contact_ticks=int(np.count_nonzero(witness)),
                                       sliding_baseline_N=resistance, force_threshold_N=threshold)
                                return'''
    replacement = marker.replace(
        "                                return",
        "                                if U4_SILENT_LOSS_PROBE:\n"
        "                                    run_u4_silent_loss_probe()\n"
        "                                return",
    )
    return replace_once(source, marker, replacement)
