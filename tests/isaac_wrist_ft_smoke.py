"""Run with Isaac Sim's python.sh to check actual fixed-joint wrench loads."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

from fr3_wrist_ft import add_wrist_sensor, wrench_feedback


def main():
    output = ROOT / "validation/idea_one_tcp_wrist_ft_load_check.json"
    config = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
    tcp_offsets = {side: np.array([0.0, sign * config["side_suction_tool"]["tcp_side_offset_m"],
                                   config["side_suction_tool"]["vertical_drop_m"]])
                   for side, sign in (("left", -1), ("right", 1))}
    results = {
        "status": "RUNNING",
        "source": "PhysX articulation incoming fixed-joint reaction wrench",
        "raw_convention": "parent_on_tool_at_child_joint_anchor_in_child_joint_frame",
        "units": {"force": "N", "torque": "N*m"},
        "force_tolerance_N": 0.03,
        "torque_tolerance_Nm": 0.005,
        "tcp_offset_sensor_m": {side: offset.tolist() for side, offset in tcp_offsets.items()},
        "tcp_local_axes": "parallel to sensor axes; reference point shifted to TCP",
        "load_application": "force at COM plus applied couple and (point-COM) cross force",
        "conversion_source": "same wrench_feedback helper used by actual per-step recorder",
        "cases": [],
    }
    world = None
    try:
        omni.usd.get_context().new_stage()
        app.update()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0,
                      rendering_dt=1.0 / 60.0, backend="numpy", device="cpu")
        scene = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        sensors = {}
        for side, angle in (("left", 0.0), ("right", 67.0)):
            robot_path = f"/World/{side}_fr3"
            root = UsdGeom.Xform.Define(stage, robot_path)
            root.AddTranslateOp().Set(Gf.Vec3d(-0.4 if side == "left" else 0.4, 0, 1))
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
            PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateSleepThresholdAttr(0.0)
            flange_path = f"{robot_path}/fr3_link8"
            flange = UsdGeom.Xform.Define(stage, flange_path)
            rotation = Gf.Rotation(Gf.Vec3d(1, 2, 3), angle)
            flange.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
            UsdPhysics.RigidBodyAPI.Apply(flange.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(flange.GetPrim())
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
            fixed_base = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/fixed_base")
            fixed_base.CreateBody1Rel().SetTargets([Sdf.Path(flange_path)])
            fixed_base.CreateLocalRot0Attr(Gf.Quatf(rotation.GetQuat()))
            fixed_base.CreateLocalPos0Attr(Gf.Vec3f(-0.4 if side == "left" else 0.4, 0, 1))
            cfg = {
                "mass_kg": 0.3,
                "center_of_mass_m": [0.0, -0.06 if side == "left" else 0.06, 0.055],
                "diagonal_inertia_kg_m2": [0.0015, 0.0007, 0.0012],
            }
            tool_path = add_wrist_sensor(stage, side, cfg)
            force = PhysxSchema.PhysxForceAPI.Apply(stage.GetPrimAtPath(tool_path))
            force.CreateForceAttr(Gf.Vec3f(0))
            force.CreateTorqueAttr(Gf.Vec3f(0))
            force.CreateModeAttr("force")
            force.CreateForceEnabledAttr(True)
            force.CreateWorldFrameEnabledAttr(True)
            robot = world.scene.add(SingleArticulation(robot_path, name=f"{side}_test"))
            # Gf matrix operations use row vectors; these columns are world axes.
            axes_world = np.column_stack([
                np.asarray(rotation.TransformDir(Gf.Vec3d(*axis)), dtype=float)
                for axis in np.eye(3)
            ])
            sensors[side] = {"robot": robot, "force_api": force,
                             "rotation": axes_world, "cfg": cfg, "tool_path": tool_path}
        world.reset()
        for side, sensor in sensors.items():
            view = sensor["robot"]._articulation_view
            sensor["index"] = view.get_link_index("ABSideTool")
            assert view.body_names[sensor["index"]] == "ABSideTool", (side, view.body_names)
            assert view.num_bodies == 2, (side, view.body_names)
        for _ in range(60):
            world.step(render=False)

        loads = [("static_gravity", np.zeros(3), np.zeros(3), np.zeros(3))]
        for index, axis in enumerate("xyz"):
            for sign in (-1.0, 1.0):
                force = np.eye(3)[index] * sign * 5.0
                torque = np.eye(3)[index] * sign * 0.7
                loads.append((f"force_{axis}_{sign:+g}", force, np.zeros(3), np.zeros(3)))
                loads.append((f"torque_{axis}_{sign:+g}", np.zeros(3), torque, np.zeros(3)))
        loads.append(("offset_force", np.array([3.0, -4.0, 2.0]),
                      np.zeros(3), np.array([0.04, 0.11, 0.08])))
        # None selects each tool's actual mirrored TCP instead of a common point.
        for index, axis in enumerate("xyz"):
            for sign in (-1.0, 1.0):
                loads.append((f"tcp_force_{axis}_{sign:+g}", np.eye(3)[index] * sign * 5.0,
                              np.zeros(3), None))
                loads.append((f"tcp_couple_{axis}_{sign:+g}", np.zeros(3),
                              np.eye(3)[index] * sign * 0.7, None))

        for name, applied_force, applied_torque, point in loads:
            samples = {side: [] for side in sensors}
            for side, sensor in sensors.items():
                com = np.asarray(sensor["cfg"]["center_of_mass_m"])
                rotation = sensor["rotation"]
                sensor["force_api"].GetForceAttr().Set(Gf.Vec3f(*(rotation @ applied_force).tolist()))
                application_point = tcp_offsets[side] if point is None else point
                torque_com = applied_torque + np.cross(application_point - com, applied_force)
                sensor["force_api"].GetTorqueAttr().Set(Gf.Vec3f(*(rotation @ torque_com).tolist()))
            for step in range(18):
                world.step(render=False)
                if step >= 12:
                    for side, sensor in sensors.items():
                        wrench = sensor["robot"].get_measured_joint_forces()[sensor["index"]]
                        samples[side].append(np.asarray(wrench, dtype=float).copy())
            for side, sensor in sensors.items():
                measured = np.mean(samples[side], axis=0)
                assert np.isfinite(measured).all(), (side, name, measured)
                gravity = sensor["rotation"].T @ np.array([0.0, 0.0, -9.81])
                gravity_force = sensor["cfg"]["mass_kg"] * gravity
                gravity_torque = np.cross(sensor["cfg"]["center_of_mass_m"], gravity_force)
                application_point = tcp_offsets[side] if point is None else point
                expected = -np.r_[gravity_force + applied_force,
                                  gravity_torque + np.cross(application_point, applied_force) + applied_torque]
                error = measured - expected
                feedback = wrench_feedback(
                    measured, sensor["rotation"], sensor["cfg"]["mass_kg"],
                    sensor["cfg"]["center_of_mass_m"], [0.0, 0.0, -9.81], tcp_offsets[side])
                tcp_local = np.asarray(feedback["tcp_gravity_corrected_local"], dtype=float)
                sensor_local = np.asarray(feedback["gravity_corrected_local"], dtype=float)
                expected_sensor = np.r_[applied_force,
                                        applied_torque + np.cross(application_point, applied_force)]
                expected_tcp = np.r_[applied_force, applied_torque
                                     + np.cross(application_point - tcp_offsets[side], applied_force)]
                expected_sensor_world = np.r_[sensor["rotation"] @ expected_sensor[:3],
                                              sensor["rotation"] @ expected_sensor[3:]]
                expected_tcp_world = np.r_[sensor["rotation"] @ expected_tcp[:3],
                                           sensor["rotation"] @ expected_tcp[3:]]
                converted_errors = np.array([
                    sensor_local - expected_sensor,
                    np.asarray(feedback["gravity_corrected_world"]) - expected_sensor_world,
                    tcp_local - expected_tcp,
                    np.asarray(feedback["tcp_gravity_corrected_world"]) - expected_tcp_world,
                ])
                assert converted_errors.shape == (4, 6) and np.isfinite(converted_errors).all()
                force_error = float(np.max(np.abs(converted_errors[:, :3])))
                torque_error = float(np.max(np.abs(converted_errors[:, 3:])))
                raw_passed = bool(np.max(np.abs(error[:3])) <= results["force_tolerance_N"] and
                                  np.max(np.abs(error[3:])) <= results["torque_tolerance_Nm"])
                passed = bool(raw_passed and force_error <= results["force_tolerance_N"]
                              and torque_error <= results["torque_tolerance_Nm"])
                if name.startswith("tcp_force_"):
                    assert np.linalg.norm(expected_sensor[3:]) > 0.1
                    assert np.max(np.abs(tcp_local[3:])) <= results["torque_tolerance_Nm"], (side, name, tcp_local)
                np.testing.assert_allclose(tcp_local[:3], sensor_local[:3], atol=1e-12, rtol=0)
                results["cases"].append({
                    "case": name, "side": side, "passed": passed,
                    "raw_passed": raw_passed,
                    "sensor_link_index": sensor["index"],
                    "measured_wrench": measured.tolist(), "expected_wrench": expected.tolist(),
                    "max_force_error_N": float(np.max(np.abs(error[:3]))),
                    "max_torque_error_Nm": float(np.max(np.abs(error[3:]))),
                    "applied_point_sensor_m": application_point.tolist(),
                    "sensor_corrected_local": sensor_local.tolist(),
                    "tcp_corrected_local": tcp_local.tolist(),
                    "expected_tcp_corrected_local": expected_tcp.tolist(),
                    "tcp_corrected_world": np.asarray(feedback["tcp_gravity_corrected_world"]).tolist(),
                    "expected_tcp_corrected_world": expected_tcp_world.tolist(),
                    "max_converted_force_error_N": force_error,
                    "max_converted_torque_error_Nm": torque_error,
                    "point_force_at_tcp_has_zero_moment": name.startswith("tcp_force_"),
                })
        assert all(case["passed"] for case in results["cases"]), results["cases"]
        results["status"] = "PASS"
        results["max_force_error_N"] = max(case["max_force_error_N"] for case in results["cases"])
        results["max_torque_error_Nm"] = max(case["max_torque_error_Nm"] for case in results["cases"])
        results["case_count"] = len(results["cases"])
        results["max_converted_force_error_N"] = max(case["max_converted_force_error_N"] for case in results["cases"])
        results["max_converted_torque_error_Nm"] = max(case["max_converted_torque_error_Nm"] for case in results["cases"])
    except Exception as exc:
        results["status"] = "FAIL"
        results["error"] = str(exc)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n")
        print("FR3_WRIST_FT_LOAD_CHECK=" + json.dumps({key: value for key, value in results.items()
                                                     if key != "cases"}), flush=True)
        if world is not None:
            world.stop()


try:
    main()
finally:
    app.close()
