# Track A Development Record: Nero7 Integration

This document records the Nero7 work completed so far for Track A of the
shape peg-insertion project. Track A integrates and validates the AgileX
Robotics Nero 7-DOF arm before it is used in the custom Track B environment.

The current Track A gate is satisfied for simulation development: Nero7 can be
created through Robosuite, controlled in joint and Cartesian space, and can
approach, grasp, and lift the cube in the existing `Lift` environment.

## 1. Current architecture

The integrated simulation is composed as follows:

```text
robosuite.make(env_name="Lift", robots="Nero7")
  -> RobotEnv._load_robots()
  -> FixedBaseRobot
  -> Nero7(ManipulatorModel)
       -> robots/nero/robot.xml
       -> PiperGripper
            -> grippers/piper_gripper.xml
       -> selected composite controller configuration
```

The robot model is registered when
`robosuite/models/robots/manipulators/__init__.py` imports
`robosuite.models.robots.manipulators.nero_robot.Nero7`.
`RobotModelMeta` in `robosuite/models/robots/robot_model.py` registers that
class in `REGISTERED_ROBOTS`. Runtime construction uses the
`FixedBaseRobot` mapping established for `Nero7`.

## 2. Robot model implementation

### Python model

`robosuite/models/robots/manipulators/nero_robot.py::Nero7` derives from
`ManipulatorModel` and defines the Robosuite-facing robot contract:

- `Nero7.arms` declares one arm named `right`.
- `Nero7.__init__()` loads
  `robosuite/models/assets/robots/nero/robot.xml`.
- `Nero7.default_base` selects `NeroTableMount`.
- `Nero7.default_gripper` attaches `PiperGripper`.
- `Nero7.default_controller_config` selects `default_nero7`.
- `Nero7.init_qpos` supplies the validated seven-joint initial pose.
- `Nero7.base_xpos_offset` supplies placement offsets for table, bin, and
  empty-workspace environments.
- `Nero7.arm_type` identifies the model as a single-arm robot.

The MJCF model is in
`robosuite/models/assets/robots/nero/robot.xml`. Its seven hinge joints and
actuators use the following physical ranges:

| Joint | Minimum (rad) | Maximum (rad) |
|---|---:|---:|
| `joint1` | -2.70526 | 2.70526 |
| `joint2` | -1.74 | 1.74 |
| `joint3` | -2.75 | 2.75 |
| `joint4` | -1.01 | 2.14 |
| `joint5` | -2.75 | 2.75 |
| `joint6` | -0.73 | 0.95 |
| `joint7` | -1.5707963 | 1.5707963 |

These ranges were reconciled with the supplied Nero URDF rather than leaving
generic or symmetric placeholder limits.

### Collision model

`robosuite/models/assets/robots/nero/robot.xml` now includes collision meshes
for the base and seven arm links. The model exposes eight arm contact geoms to
Robosuite. Adjacent-body exclusions prevent expected neighboring link contact
from being reported as self-collision.

Visual and collision meshes reside under:

```text
robosuite/models/assets/robots/nero/meshes/
```

The placeholder `right_hand` inertia was removed so the attached gripper mass
is not counted twice.

### Tabletop mount

`robosuite/models/bases/nero_table_mount.py::NeroTableMount` and
`robosuite/models/assets/bases/nero_table_mount.xml` define Nero7's fixed
tabletop flange. It contains a dark lower plate, raised metal collar, mounting
bolts, and a collar collision geom. Its `(0.2, 0.0, 0.85)` top offset
preserves the previously validated robot-root and end-effector world poses.
The lower plate is visual-only to avoid permanent table contact at the fixed
mounting interface.

## 3. Parallel gripper integration

`robosuite/models/grippers/piper_gripper.py::PiperGripperBase` derives from
`GripperModel`. `PiperGripper` converts one policy action into the opposed
motion of the two physical finger joints.

Important behavior:

- `PiperGripperBase.init_qpos` is `[0.05, -0.05]`, the fully open state.
- `PiperGripper.format_action()` sends opposite signed commands to the two
  finger joints.
- `PiperGripper.dof` exposes one gripper action to a policy.
- `PiperGripperBase._important_geoms` identifies the left and right finger
  pads, enabling `ManipulationEnv._check_grasp()` to evaluate a two-sided
  grasp.

The XML is
`robosuite/models/assets/grippers/piper_gripper.xml`, and its meshes are under
`robosuite/models/assets/grippers/meshes/piper_gripper/`.

The supplied URDF flange mass and inertia were retained, but the mount uses the
equivalent MJCF-frame transform. A literal copy of URDF transforms is unsafe
because URDF joint frames and the converted MJCF body frames are not the same.
The current transform produces a conventional symmetric parallel gripper.

## 4. Initial pose selection and validation

The selected initial joint pose is stored in both
`nero7_init_pose.json::qpos` and
`robosuite/models/robots/manipulators/nero_robot.py::Nero7.init_qpos`:

```text
[ 0.3590570800,
  0.1850950180,
 -0.3851502167,
  1.7581719380,
  0.0420627663,
 -0.0050082907,
  0.8511248906 ]
```

`robosuite/scripts/tune_nero7_init_pose.py` provides the interactive pose
tuner and simulator validation. Its main reusable functions are:

- `set_arm_state()` for applying a candidate configuration.
- `pose_metrics()` for end-effector pose, downward tilt, Jacobian condition,
  finite-state, and collision checks.
- `validate_dynamics()` for commanded hold validation.
- `Nero7PoseTuner` for the joint-slider interface and saving results.

Recorded metrics in `nero7_init_pose.json`:

- End-effector position: approximately
  `[0.00344, 0.00296, 1.04559]` metres.
- Downward tilt: `20.75` degrees.
- Geometric Jacobian condition number: `10.60`.
- Active collision pairs: none.
- Dynamic validation: passed for 200 steps.
- Maximum joint tracking error: approximately `8.58e-4` radians.

## 5. Workspace probing

`robosuite/scripts/nero7_workspace_probe.py::sample_workspace()` samples
joint configurations inside the physical limits and evaluates:

- end-effector position,
- Jacobian condition,
- end-effector downward alignment,
- self/environment collision state,
- user-selected Cartesian bounds.

The 10,000-sample probe with seed 7 accepted 7,885 configurations after the
specified filters. For accepted samples with `z >= 0.9 m`, the observed bounds
were approximately:

```text
minimum xyz: [-1.0744, -0.7201, 0.9000]
maximum xyz: [ 0.3492,  0.7082, 1.7249]
mean xyz:    [-0.3497, -0.0042, 1.3662]
```

The saved sample archive was `/tmp/nero7_workspace.npz`; `/tmp` is temporary,
so rerun the probe if that file is no longer present.

Example:

```bash
python -m robosuite.scripts.nero7_workspace_probe \
  --samples 10000 \
  --seed 7 \
  --joint-margin 0.1 \
  --minimum-z 0.9 \
  --maximum-condition 100 \
  --output /tmp/nero7_workspace.npz
```

## 6. Controller configurations

Three robot-specific configurations now exist.

### Absolute joint position

`robosuite/controllers/config/robots/default_nero7.json` is the default
configuration selected by `Nero7.default_controller_config`.

- Controller: `JOINT_POSITION`
- Input type: absolute
- Arm action: seven physical joint targets
- Gripper action: one `GRIP` command
- Total environment action dimension: eight
- Joint limits: the physical Nero7 ranges listed above

This mode is useful for exact scripted joint waypoints and for matching
absolute joint commands from a hardware-side source.

### Delta joint position

`robosuite/controllers/config/robots/nero7_joint_delta.json` provides
normalized joint increments.

- Controller: `JOINT_POSITION`
- Input range: `[-1, 1]`
- Maximum command increment: `0.05 rad` per arm action component
- Input type: delta
- Gripper action: one additional component

### Cartesian operational-space control

`robosuite/controllers/config/robots/nero7_osc_pose.json` provides
world-frame Cartesian control.

- Controller: `OSC_POSE`
- Input type: delta
- Reference frame: world
- Translation scale: `0.025 m`
- Rotation-vector scale: `0.25 rad`
- Action layout:
  `[dx, dy, dz, dRx, dRy, dRz, gripper]`
- Total environment action dimension: seven

`robosuite/scripts/validate_nero7_osc.py::measure_zero_hold()` checks
zero-command stability.
`measure_axis_response()` checks the sign and response of all six Cartesian
directions. The configured home pose passed the OSC direction, finite-state,
collision, and actuator-range checks.

## 7. Scripted policies and motion demonstrations

### Joint-space waypoint policy

`robosuite/scripts/nero7_waypoint_policy.py::Nero7JointWaypointPolicy`
implements deterministic absolute joint interpolation for data-collection
scripts. It:

- validates the eight-dimensional Nero7 joint-plus-gripper action contract,
- rejects targets outside the robot limits,
- limits motion using `max_joint_step`,
- reports whether the target was reached.

This remains useful for safe joint-space staging motions and deterministic
reset/recovery trajectories.

### OSC Cartesian motion

`robosuite/scripts/demo_nero7_osc_motion.py` is the current visual motion and
pick test.

- `current_eef_pose()` reads the right gripper site pose.
- `pose_action()` converts a world-frame pose error into a normalized OSC
  action.
- `run_to_pose()` executes closed-loop motion to an absolute Cartesian pose.
- `run_motion()` retains the earlier relative Cartesian movement mode.
- `hold_pose()` maintains an end-effector target while actuating the gripper.
- `run_pick_and_lift()` executes the complete scripted cube sequence.

The pick sequence is:

```text
validated init pose
  -> move above cube with gripper open
  -> descend to cube
  -> close both fingers
  -> verify two-sided grasp
  -> lift while maintaining pose and grasp
```

Run it visually:

```bash
python -m robosuite.scripts.demo_nero7_osc_motion
```

Run it headlessly:

```bash
python -m robosuite.scripts.demo_nero7_osc_motion --headless --mode pick
```

Relevant tuning options are `--approach-height`, `--grasp-offset-z`,
`--lift-height`, `--close-steps`, and `--max-steps`.

The validated default run:

- reached the approach and grasp poses,
- contacted the cube with both finger pads,
- passed `ManipulationEnv._check_grasp()`,
- lifted the cube approximately `0.138 m` for a `0.15 m` command,
- retained the grasp after lifting.

The earlier relative motion mode remains available:

```bash
python -m robosuite.scripts.demo_nero7_osc_motion \
  --mode move \
  --position-delta 0.10 0.0 0.05
```

## 8. Validation scripts

`robosuite/scripts/validate_nero7.py::main()` validates the base simulation
contract:

- robot/environment construction,
- action dimensions and bounds,
- initial arm and gripper positions,
- gripper positions against joint ranges,
- arm contact geom availability,
- finite simulation state,
- zero-action joint drift,
- optional Jacobian-condition threshold.

Examples:

```bash
python -m robosuite.scripts.validate_nero7 --controller-mode delta

python -m robosuite.scripts.validate_nero7 \
  --controller-mode absolute \
  --require-arm-collisions \
  --maximum-jacobian-condition 100

python -m robosuite.scripts.validate_nero7_osc
```

`tests/test_robots/test_nero7.py` contains the regression suite for:

- model and runtime registration,
- joint and actuator ordering,
- joint limits and initialization pose,
- arm collision geoms,
- gripper action and geometry contracts,
- supplied URDF flange/finger transformations,
- joint waypoint control,
- workspace filtering,
- candidate-pose stability,
- OSC hold and axis directions,
- closed-loop OSC motion,
- scripted cube pick-and-lift.

Current result:

```text
14 passed
```

Run the tests with the active checkout's Python:

```bash
python -m pytest -q tests/test_robots/test_nero7.py
```

Using a bare `pytest` executable may select a different installed Robosuite
checkout, depending on shell configuration.

## 9. Track A completion status

| Track A item | Status |
|---|---|
| Seven-joint MJCF loads | Complete |
| Physical joint ranges reconciled with URDF | Complete |
| Python model registered | Complete |
| Fixed-base runtime wrapper resolves | Complete |
| Parallel gripper registered and aligned | Complete |
| Arm and gripper collision geoms available | Complete |
| Collision-free initial pose selected | Complete |
| Dynamic hold validation | Complete |
| Workspace probe | Complete |
| Absolute joint controller | Complete |
| Delta joint controller | Complete |
| World-frame OSC controller | Complete |
| Existing `Lift` environment construction | Complete |
| Closed-loop Cartesian movement | Complete |
| Scripted cube grasp and lift | Complete |
| Regression tests | Complete |
| Real Nero7 hardware parity | Not yet validated |
| Camera calibration for data collection | Not yet validated |
| Track B shape-sorter integration | Not started in this track |

## 10. Safe extension points

The following are the intended places for subsequent work:

- Adjust robot metadata and defaults in
  `robosuite/models/robots/manipulators/nero_robot.py::Nero7`.
- Improve physical arm geometry and dynamics in
  `robosuite/models/assets/robots/nero/robot.xml`.
- Improve gripper physics in
  `robosuite/models/assets/grippers/piper_gripper.xml`, while preserving the
  action contract in `PiperGripper`.
- Add controller variants as new JSON files under
  `robosuite/controllers/config/robots/`.
- Build reusable scripted behavior on `run_to_pose()` or
  `Nero7JointWaypointPolicy` rather than writing direct actuator controls.
- Add each new robot behavior to
  `tests/test_robots/test_nero7.py`.

Avoid modifying generic upstream machinery unless a proven framework defect
requires it:

- `robosuite/environments/base.py`
- `robosuite/environments/robot_env.py`
- `robosuite/robots/robot.py`
- `robosuite/robots/fixed_base_robot.py`
- generic controller implementations under `robosuite/controllers/parts/`
- registration metaclasses in
  `robosuite/models/robots/robot_model.py`

Robot-specific behavior should remain in Nero7 model, asset, controller,
script, and test files.

## 11. Remaining work before dataset collection

Track A is simulation-ready, but dataset production should wait until the
following are defined and tested:

1. Decide the dataset observation schema, including proprioception, camera
   images, object state, actions, rewards, success flags, and timestamps.
2. Select and calibrate the fixed and wrist cameras, resolution, depth usage,
   segmentation usage, and frame rate.
3. Add reproducible environment and object randomization with an explicit
   recorded seed.
4. Convert the cube demonstration into a phase-based policy with timeouts,
   failure reasons, recovery, and per-phase logging suitable for many
   rollouts.
5. Add episode acceptance checks for grasp, lift, collisions, controller
   saturation, and finite observations/actions.
6. Validate joint signs, zero offsets, limits, gripper convention, base frame,
   tool frame, and controller timing against the physical Nero7 before using
   simulation commands on hardware.
7. Begin Track B only after the custom board and pieces have validated
   collision clearances and success predicates.

The next implementation handoff is therefore Track B object/arena prototyping
or, if dataset collection must begin first, a reusable Nero7 rollout recorder
around the existing `Lift` scripted policy.
