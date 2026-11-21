import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "scenes/w760_300mm_two_layer/experiments"))
from seven_observer import SevenObserver


CONFIG = {"cube_size_m": .15, "physics": {"cube_mass_kg": 1.25, "gravity_m_s2": [0, 0, -9.81]},
          "static_boxes": [["floor", [0, 0, -.005], [2, 2, .01]],
                           ["supply", [0, -2, -.005], [.3, .3, .01]],
                           ["wall_back", [0, 1, .14], [2, .01, .3]],
                           ["wall_front", [0, -1, .14], [2, .01, .3]],
                           ["wall_left", [-1, 0, .14], [.01, 2, .3]],
                           ["wall_right", [1, 0, .14], [.01, 2, .3]]]}


def sample(t, center=(0, 0, .075), left=(0., 0., 0.), right=(0., 0., 0.), separated=False):
    center = np.asarray(center)
    camera = {"status": "detected", "frame_id": int(round(t * 100)) + 1, "position_m": center,
              "cube_rotation_world": np.eye(3), "capture_time_s": t, "received_time_s": t}
    distance = .25 if separated else .075
    return {"time_s": t, "camera": camera, "cup_normals_world": {"left": [0, 1, 0], "right": [0, -1, 0]},
            "tcp_positions_m": {"left": center - [0, distance, 0], "right": center + [0, distance, 0]},
            "left_wrench_world": [*left, 0, 0, 0], "right_wrench_world": [*right, 0, 0, 0]}


def test_seven_and_missing_modes():
    observer = SevenObserver(CONFIG, "fusion_compensated")
    assert observer.update(sample(.1, separated=True))["state"] == "S0"
    assert observer.update(sample(.2, left=(0,-2,0)))["state"] == "S1"
    assert observer.update(sample(.3, left=(0,1,0)))["state"] == "S2"
    assert observer.update(sample(.4, left=(0,-1,-2.5), right=(0,1,-2.5)))["state"] == "S3"
    assert observer.update(sample(.5, center=(0,0,.4), left=(0,-1,-6), right=(0,1,-6)))["state"] == "S4"
    observer.update(sample(.6, center=(0,.920,.075), left=(0,-.5,0)))
    event = observer.update(sample(.86, center=(0,.920,.075), left=(0,-2,0)))
    assert (event["state"], event["boundaries"]) == ("S5", ["/World/Environment/wall_back"])
    neighbors = {"/World/Cube_8": {"position_m": [.15,.92,.075], "rotation_world": np.eye(3),
                                    "capture_time_s": .5, "received_time_s": 1.1}}
    assert observer.update(sample(1., center=(0,.92,.075), left=(0,-2,0)), neighbors)["state"] == "S5"
    assert observer.update(sample(1.1, center=(0,.92,.075), left=(-1,-2,0)), neighbors)["state"] == "S6"
    row = sample(1., separated=True)
    row["camera"] = {"status":"occluded", "frame_id":12, "supported_position_m":[0,0,.075]}
    assert SevenObserver(CONFIG, "vision_fk").update(row)["state"] == "UNKNOWN"
    assert SevenObserver(CONFIG, "wrench_fk").update(row)["state"] == "UNKNOWN"


def test_delayed_frame_no_force_or_projected_pose_leakage():
    observer = SevenObserver(CONFIG, "vision_fk", delay_ms=100)
    row = sample(.1, separated=True)
    row.pop("left_wrench_world")
    row.pop("right_wrench_world")
    assert observer.update(row)["camera_age_s"] is None
    row["time_s"] = .21
    result = observer.update(row)
    assert abs(result["camera_age_s"] - .11) < 1e-6
    assert result["state"] == "S0"
