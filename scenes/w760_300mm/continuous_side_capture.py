"""Capture render-only Isaac Sim viewport evidence at acceptance checkpoints."""


CAPTURE_HOOK = '''        def capture_acceptance_view(event):
            if vision_viewport is None or not app.is_running():
                return
            cube_id = event["cube_id"]
            phase, state = event["phase"], event["status"]
            if cube_id in (2, 4) and (phase, state) == ("A_attach", "SURFACE_GRIPPERS_CLOSED"):
                label = f"cube{cube_id}_a_front_rear_grasp"
            elif cube_id in (2, 4) and (phase, state) == ("A_lift_and_align", "CLEAR_PHYSX_EXECUTION"):
                label = f"cube{cube_id}_a_front_rear_lift"
            elif cube_id == 4 and phase in ("B2_contact_sync", "B2_push_seat") and state == "FORCE_CONFIRMED":
                label = "cube4_b2_dual_contact" if phase == "B2_contact_sync" else "cube4_b2_seated"
            elif phase == "packing_complete" and state.startswith("PASS"):
                label = "five_cube_final_row"
            else:
                return

            from omni.kit.viewport.utility import capture_viewport_to_file

            screenshot_directory = output_directory / "screenshots"
            screenshot_directory.mkdir(exist_ok=True)
            path = screenshot_directory / (label + ".png")
            pending = screenshot_directory / (label + "_pending.png")
            pending.unlink(missing_ok=True)
            capture_info = {"file": str(path.relative_to(output_directory)), "cube_id": cube_id,
                            "phase": phase, "status": state, "sim_time_s": event["sim_time_s"],
                            "attachment_closed": {side: grippers.is_closed(side) for side in robots}}
            if phase.startswith("B2"):
                support_side = current_task["support_side"]
                capture_info["support_arm"] = support_side
                capture_info["snapshot_support_normal_N"] = float(
                    -last_force_sample[support_side]["tcp_gravity_corrected_world"][1])

            shot_path = "/World/AcceptanceCamera"
            shot = UsdGeom.Camera.Define(stage, shot_path)
            ops = shot.GetOrderedXformOps()
            shot_transform = ops[0] if ops else shot.AddTransformOp()
            if label == "five_cube_final_row":
                target = Gf.Vec3d(0., .93, .84)
                eye = target + Gf.Vec3d(.52, -.82, 1.62)
                focal = 21.
            else:
                target = Gf.Vec3d(*event["cube_position_m"]) + Gf.Vec3d(0., 0., .04)
                eye = target + (Gf.Vec3d(.64, -.30, 1.20) if phase.startswith("A_")
                                else Gf.Vec3d(.71, -.50, 1.39))
                focal = 27.
            shot_transform.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
            shot.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
            shot.GetFocalLengthAttr().Set(focal)

            previous_camera = vision_viewport.camera_path
            vision_viewport.set_active_camera(shot_path)
            try:
                for _ in range(4):
                    render_only()
                capture = capture_viewport_to_file(vision_viewport, str(pending))
                for _ in range(24):
                    render_only()
                    if pending.is_file() and pending.stat().st_size > 1024:
                        with pending.open("rb") as stream:
                            signature = stream.read(8)
                            stream.seek(-8, 2)
                            complete = stream.read(8) == b"IEND\\xaeB\\x60\\x82"
                        if signature == b"\\x89PNG\\r\\n\\x1a\\n" and complete:
                            pending.replace(path)
                            capture_info["result"] = "CAPTURED"
                            break
                else:
                    capture_info["result"] = "CAPTURE_TIMEOUT"
            finally:
                vision_viewport.set_active_camera(previous_camera)
            report.setdefault("acceptance_screenshots", []).append(capture_info)
            print("FR3_SCREENSHOT " + json.dumps(capture_info), flush=True)

'''


def patch_capture(source):
    """Install evidence capture without adding simulation steps or sensor inputs."""
    record = "        def record(name, state, **extra):\n"
    log_event = '            print("FR3_AB " + json.dumps(item), flush=True)\n'
    if source.count(record) != 1 or source.count(log_event) != 1:
        raise ValueError("FR3 event recorder changed; review the screenshot hook")
    source = source.replace(record, CAPTURE_HOOK + record, 1)
    return source.replace(log_event, log_event + "            capture_acceptance_view(item)\n", 1)
