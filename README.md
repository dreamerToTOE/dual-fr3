# dual-fr3

## Five-cube row versions on `base`

| Version | Inner tray width | Nominal clearance around Cube 5 | Validation |
| --- | ---: | ---: | --- |
| **76 cm** (current) | 5 × 150 mm + 10 mm | 5 mm per side | Scoped Cube 3–5 checks. Full run pending. |
| [**84 cm**](https://github.com/dreamerToTOE/dual-fr3/commit/4a2c258499a2637e84159c13c75120eca49ec336) | 5 × 150 mm + 90 mm | 45 mm per side | Completed in three PhysX segments with two checkpoint restores. |

Check out the linked 84 cm commit to run that scene. Its
[checkpoint report](validation/fr3_five_cube_row_840mm_checkpoint_acceptance.json)
records the final order 1, 3, 5, 4, 2 and the 2.33 mm maximum RGB position error.

## Current five-cube scene · 76 cm

Each 150 mm, 1.25 kg cube arrives individually at the supply station and is
carried by both arms through A to an observed landing pose. Native breakable
Surface Grippers attach after wrist-force contact confirmation.

Packing order is rear then left wall, rear then right wall, beside Cube 1,
beside Cube 2, then a straight insertion into the measured center slot.
The inner tray width is 760 mm. Nominal final X coordinates in packing order
are −305, +305, −155, +155 and 0 mm. Earlier cubes remain dynamic collision obstacles. Cubes 2–4
use the observed neighboring faces to choose a landing lane with tool clearance.

Rail centers are X = ±600 mm. A starts at rail Y = 200 mm, the support arm
moves to 400 mm, and the lateral pushing arm moves to 800 mm. Lateral pushing
contacts are 20 mm above cube center.
After Cube 4 lands, both tools withdraw 5 mm, rise to TCP Z = 1.05 m,
then withdraw the remaining 20 mm so the right wrist clears the side wall.
The B1 support arm lifts only to an absolute safe height if it is already raised.

The rear D455 view and an additional supply view provide independent RGB
observations. Supported poses use measured marker corners and calibrated
support planes. The controller combines those observations with encoder FK
and wrist F/T. Cube truth is confined to independent evaluation and logs.
The optical identities are ArUco IDs 23 through 27.
The retained single-cube contact recognizer logs reference outputs; its
neighbor-contact vocabulary has not been validated for this five-cube task.

After the final insertion, fresh RGB observations check all five positions,
yaw angles and left-to-right order, including boxes that later contact may move.
Both articulations use 44 position and 4 velocity solver iterations, with SDK
readback recorded. Cube solver defaults remain 16/1. Contact-stage observations
refresh the active box and intended neighbor; occluded untouched boxes remain
collision obstacles. Initial and final parked scans still require all relevant markers.

Contact stages use 0.003 times the official USD joint drive gains; unloaded
transfers use the original gains. This simulated compliance and the vacuum
capacities have not been calibrated against hardware.
B2 uses different force checks for surveyed side walls and visually located
neighbors. At a side wall, a 6 N normal sample near the target pauses lateral
advance. That entry sample and the next two samples must average at least
4 N. At a neighbor, an 8 N normal sample starts the pause, and all 30 subsequent
physics samples must reach at least 6 N. The separate 10 N raw resultant-force
abort remains active throughout acquisition.
Loaded transitions retain the previous commanded joint reference. Contact seeking
advances 5 um per physics step, freezes above 0.2 N, and requires six consecutive
contact samples before synchronization.
The support arm maintains its 4 N target through the pushing arm's approach.
Once the pushing arm reaches 0.5 N filtered lateral force during synchronization,
the support target drops to 2 N. The approach and synchronized hold use
2e-5 m/(N s) admittance. Initial alignment and unloading use 1e-4 m/(N s).
The supported-pose model
assumes an upright box on a calibrated plane; sensor noise and rail dynamics
are not included.

Run the automatic visual experiment from this worktree.

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_ab_visual_runner.py --five-cube-row
```

The full-run checker requires all five A transports and the prescribed physical
wall or neighbor contact evidence.

```bash
python3 fr3_row_acceptance.py validation/five_cube_row
```

### 76 cm diagnostics

- **Cube 3.** The right arm reached the side face and pushed Cube 3 against
  Cube 1. PhysX recorded 34 consecutive positive impulse samples at a
  0.145 mm solver gap. The RGB-based geometry estimate reported a 3.008 mm
  overlap, so the guard stopped before force confirmation and tool withdrawal.
- **Cube 4.** The original rear-row side approach overlaps Cube 3 by 26 mm in
  the nominal tool-envelope model. With the required tool clearance the
  landing plane would need X = 179 mm for a +X push to a 155 mm target.
  A separate staged trial seated Cube 4 against Cube 2 (30/30 positive
  impulse samples) and withdrew the left tool. The follow-up rear-wall
  checkpoint ended about 1.26 mm short, with zero rear-wall impulses.
  The staged trial remains a diagnostic outside the automatic sequence.
- **Cube 5.** The nominal prepared scene passed straight insertion and
  withdrawal. The minimum physical side clearances during the push were
  4.46 and 4.08 mm, with zero loaded side contact. A geometric check with
  3° yaw and 2 mm lateral offset found a 0.82 mm overlap.

This repository contains the **FR3-only** four-wall scene package for the
150 mm cube study. It is intentionally separate from the earlier episode
robot project and does not reuse its six-axis robot, runner, or validation
files.

The package describes:

- two Isaac Sim Franka FR3 robots;
- fixed L-shaped side-suction preview tools mounted at `fr3_link8`;
- the expanded four-wall geometry with a 150 mm cube and five target positions;
- two visual Y-axis rails using the current 1.5 m travel range.

Run the static preview inside Isaac Sim 4.5 with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_four_wall_preview.py
```

The script writes `preview/fr3_four_wall_preview.usda` and two PNG captures.
The FR3 USD is resolved from the Isaac Sim asset root at runtime, so the
machine must have the official FR3 asset available.

## Historical single-cube base snapshots

The `base` history retains both single-cube experiment states.
Commit `9a5dd79` is the no-cache environment snapshot. It contains the scene,
validation code, preview assets, and reports while the runtime planner creates
the A/B1/B2 paths online. It is the baseline for studying planning-time
behavior and online anomalies.

Commit `d2cf2d4` is the stored-trajectory environment. It adds the complete
120 Hz A/B1/B2 runtime cache, cache-first replay, state checks, and fallback to
the online RRT/IK planner. Check out the corresponding commit to run either
single-cube snapshot. The five-cube version uses live sensing and ships without
a full five-cube trajectory cache.

`validation/fr3_b2_collision_plan.json` is the collision-constrained B2
preflight plan used by both states. The complete runtime cache generated by an
A/B1/B2 execution is stored separately in
`validation/fr3_trajectory_cache.json`.

The recorded scene parameters are `cube_size_m=0.15`, Cube supply center
`[0.0,-0.3,0.787]`, back-wall target center `[0.0,1.076,0.787]`, B2 left-wall
target `[-0.245,1.076,0.787]`, wall inner bounds `x=±0.320`, right rail contact
`base_y=0.65`, FR3 bases `x=±0.44`, and a 1.5 m Y-axis rail. The machine-readable
snapshot is `validation/fr3_environment_snapshot.json`.

After a complete run, the runner stores the 120 Hz phase trajectories in
`validation/fr3_trajectory_cache.json`. The cache includes raw joint
waypoints, time-sampled joint commands, expected Cube centers, collision-check
reports, and rail base samples. Regenerate it with the automatic runner after
changing the scene configuration.

The runner loads a `COMPLETE` cache for the matching scene before each A/B1/B2
phase. It checks the measured joint start, TCP/Cube pose, and rail endpoints
before replay. A mismatch disables replay for the remaining phases and uses
the existing online RRT/IK planner. PhysX contacts, joint limits, Cube
deviation, and B2 dual-contact checks remain active during replay. A measured
collision or mid-phase deviation stops the run for inspection rather than
silently continuing.

This is a geometry preview only. It does not claim PhysX contact, suction,
ROS, IK, trajectory, or full B2 validation.

## Historical single-cube A/B validation gates

The current A/B semantics are kept in `fr3_four_wall_scene_config.json`:

- A: dual-arm pickup at the supply position, lift over the front wall, and
  place at the landing position;
- B1: left side-face contact and `+Y` push to the back wall;
- B2: move the right tool to the Cube suction-height plane, align outside the
  right face, approach horizontally along `-X`, then push to the left wall
  while preserving the back-wall contact.

Run the geometry/runtime preflight with:

```bash
python3 fr3_ab_preflight.py
```

Run the Isaac Sim headless dynamic loading smoke test with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_ab_dynamic_smoke.py --no-window
```

The smoke test creates a real `PhysicsScene`, dynamic `/World/Cube`, collision
walls/tools/cups, and both FR3 articulations, then checks that the supply cube
settles under gravity. Its report is written to `validation/`; `ab_status`
remains `BLOCKED` until a dedicated FR3 trajectory and suction/contact
controller is implemented.

Run the Lula waypoint reachability gate with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_ab_ik_preflight.py --no-window
```

The IK preflight checks reachability only. The collision-constrained B2 plan
uses the right rail at `base_y=0.65 m`, an elevated wrist with the side
suction tool, a suction-plane alignment waypoint, and a separate horizontal
approach segment. Rebuild that plan with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_b2_collision_plan.py
```

Start the visible automatic task with:

```bash
/home/ubuntu2004/isaacsim-4.5.0/python.sh fr3_ab_visual_runner.py
```

Add `--headless` for the same task without a window. The viewer pauses at
completion or at the first failed check and retains the scene for inspection.
The runtime report is `validation/fr3_ab_visual_runner.json`.

The visible runner advances one 120 Hz physics step per control sample and
uses render-only updates for the 60 Hz viewport. Planning progress therefore
does not advance the Cube or the robots. The runtime report records both
control and physics step counts.

Free transfers use RRT-Connect with joint limits and full-arm collision checks.
The final 120 Hz trajectories are checked again after timing and blending.
Checks include official FR3 Lula collision spheres, every custom tool support
and cup, the Cube, walls, floor/table, interarm and nonadjacent self collisions.
Both arms share one 14-joint timeline during coordinated transfers and B2.
All retained FR3 arm and custom tool colliders remain enabled. The local runtime
deactivates the stock hand/finger bodies and their joints, removing the hidden
0.73 kg hand inertia per arm while retaining the seven arm DoFs and flange.

PhysX contact events stop execution on unintended robot/tool contact. Only
phase-authorized cup/Cube contact up to 3 mm penetration is allowed. The
planning geometry is conservative spheres/boxes, not a swept-mesh certificate.
The historical base snapshots use temporary PhysX fixed joints during A.
The five-cube experiment above replaces them with native Surface Grippers.
A successful Cube endpoint alone is insufficient to pass the task.
