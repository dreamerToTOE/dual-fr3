# dual-fr3

This repository contains the **FR3-only** four-wall scene package for the
100 mm cube study. It is intentionally separate from the earlier episode
robot project and does not reuse its six-axis robot, runner, or validation
files.

The package describes:

- two Isaac Sim Franka FR3 robots;
- fixed L-shaped side-suction preview tools mounted at `fr3_link8`;
- the current four-wall geometry with a 100 mm cube and five target positions;
- two visual Y-axis rails using the current 1.5 m travel range.

Run the static preview inside Isaac Sim 4.5 with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_four_wall_preview.py
```

The script writes `preview/fr3_four_wall_preview.usda` and two PNG captures.
The FR3 USD is resolved from the Isaac Sim asset root at runtime, so the
machine must have the official FR3 asset available.

This is a geometry preview only. It does not claim PhysX contact, suction,
ROS, IK, trajectory, or full B2 validation.

## Local A/B validation gates

The current A/B semantics are kept in `fr3_four_wall_scene_config.json`:

- A: dual-arm pickup at the supply position, lift over the front wall, and
  place at the landing position;
- B1: left side-face contact and `+Y` push to the back wall;
- B2: right side-face contact and `-X` push to the left wall, while preserving
  the back-wall contact.

Run the geometry/runtime preflight with:

```bash
python3 fr3_ab_preflight.py
```

Run the Isaac Sim headless dynamic loading smoke test with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_ab_dynamic_smoke.py --no-window
```

The smoke test creates a real `PhysicsScene`, dynamic `/World/Cube`, collision
walls/tools, and both FR3 articulations, then checks that the supply cube settles
under gravity. Its report is written to `validation/`; `ab_status` remains
`BLOCKED` until a dedicated FR3 IK/trajectory and suction/contact controller is
implemented.
