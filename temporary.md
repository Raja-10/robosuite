# Shape Peg Insertion Task and Custom 7-DOF Robot Plan

This plan separates the work into two independent tracks:

- **Track A:** integrate and validate the custom 7-DOF robot.
- **Track B:** build and validate the four-shape peg-insertion task.

The tracks should be combined only after the robot works in an existing
Robosuite task and the shape-sorter mechanics work independently.

Track A implementation and validation results are recorded in
`docs/track_a_development.md`.

The working Track B implementation plan is recorded in
`docs/track_b_development_plan.md`.

## Working task interpretation

A fixed board has four target stations:

- Circle with one peg.
- Square with two pegs.
- Triangle with three pegs.
- Rectangle with four pegs.

Each movable piece contains the corresponding hole pattern and must be grasped,
aligned with its matching station, and lowered over the pegs until properly
seated.

## Stage 0 — Freeze the task and robot specifications

Before implementation, define:

- Board dimensions and workspace height.
- Piece dimensions, thickness, mass, friction, and clearance around each peg.
- Peg locations, diameters, heights, and counts.
- Starting regions for the four pieces.
- Whether pieces may initially have random yaw.
- Whether completion requires all pieces to remain simultaneously seated.
- Robot name, base type, end-effector body, joint order, actuator order,
  limits, and initial pose.
- Gripper choice:
  - Existing Robosuite gripper.
  - Custom gripper.
  - Gripper already embedded in the robot MJCF.
- Initial controller:
  - Begin with `JOINT_POSITION`.
  - Add `OSC_POSE` after kinematics and actuator behavior are verified.

### Important collision-model decision

Do not use a single collision mesh for a piece containing holes. MuJoCo
generally collision-tests mesh convex hulls, which can effectively close the
holes.

Use:

- A visual mesh for appearance.
- Compound primitive collision geoms arranged around each hole, or explicitly
  decomposed convex collision meshes.
- Primitive cylinder geoms for the pegs.

### Stage 0 gate

Board geometry, piece geometry, coordinate conventions, grasp strategy,
insertion clearance, and success tolerances are documented before model
implementation begins.

---

# Track A — Add the custom 7-DOF robot

## Stage 1 — Audit and normalize the robot MJCF

Place the robot package provisionally under:

```text
robosuite/models/assets/robots/<robot_name>/
  robot.xml
  meshes/
  textures/
```

Audit `robot.xml` for:

- Exactly seven arm joints.
- Joint order from base to wrist.
- Matching actuator order.
- Unique names for bodies, joints, actuators, sites, and cameras.
- Correct joint axes and ranges.
- Suitable damping and armature values.
- Correct actuator ranges and gear values.
- A named terminal body to which Robosuite can attach a gripper.
- A suitable end-effector coordinate frame.
- Collision and visual geom groups.
- Valid relative asset paths.
- Optional `robotview` and `eye_in_hand` cameras.
- No free joint when the arm is intended to use a fixed base.

Run the XML-only checker:

```bash
python -m robosuite.scripts.check_custom_robot_model \
  --robot-xml-file /path/to/robot.xml
```

The checker validates naming-based body-part detection through
`robosuite/scripts/check_custom_robot_model.py::check_xml_definition`. It does
not verify that actuator ordering matches joint ordering, so that must also be
checked manually.

### Stage 1 gate

The standalone MJCF:

- Loads successfully in MuJoCo.
- Rests without exploding or drifting unexpectedly.
- Moves all seven joints correctly under direct actuator commands.
- Has verified joint and actuator ordering.

## Stage 2 — Implement the Python robot model

Create:

```text
robosuite/models/robots/manipulators/<robot_name>_robot.py
```

Implement a class deriving from
`robosuite.models.robots.manipulators.manipulator_model.ManipulatorModel`,
following `robosuite/models/robots/manipulators/panda_robot.py::Panda`.

Required class members and properties:

- `arms = ["right"]`
- `__init__()` pointing to the robot MJCF
- `default_base`
- `default_gripper`
- `default_controller_config`
- `init_qpos`
- `base_xpos_offset`
- `top_offset`
- `_horizontal_radius`
- `arm_type = "single"`
- `_eef_name` if the terminal XML body is not named `right_hand`
- Optional gripper mount position and quaternion offsets

Import the class from:

```text
robosuite/models/robots/manipulators/__init__.py
```

`robosuite/models/robots/robot_model.py::RobotModelMeta` automatically places
the imported model class in `REGISTERED_ROBOTS`.

### Stage 2 gate

`create_robot("<RobotName>")` returns the new model, and the following
prefix-adjusted values are correct:

- Joint names.
- Actuator names.
- Camera names.
- End-effector body name.
- Important sites.

## Stage 3 — Register the runtime robot wrapper

Model registration alone is insufficient. `RobotEnv._load_robots()` also
looks up the runtime robot wrapper in
`robosuite/robots/__init__.py::ROBOT_CLASS_MAPPING`.

For an ordinary fixed-base arm, register it as a `FixedBaseRobot`, preferably
using:

```python
@register_robot_class("FixedBaseRobot")
class MyRobot(ManipulatorModel):
    ...
```

The decorator is defined by
`robosuite/robots/__init__.py::register_robot_class`. If import ordering makes
decorator registration unsuitable, add the minimal explicit entry to
`ROBOT_CLASS_MAPPING`.

Run the registered-robot checker:

```bash
python -m robosuite.scripts.check_custom_robot_model --robot MyRobot
```

### Stage 3 gate

- The registered robot can be constructed through `RobotEnv`.
- Robot-to-simulation references resolve.
- The arm and gripper are recognized correctly.
- `robot.action_dim` and `robot.action_limits` are valid.

## Stage 4 — Add a conservative controller configuration

Create:

```text
robosuite/controllers/config/robots/default_<normalized_robot_name>.json
```

Begin with a `BASIC` composite controller containing:

- One `right` arm controller.
- One `right_gripper` controller if a gripper is attached.
- A `JOINT_POSITION` arm controller for initial bring-up.
- Robot-specific joint count, limits, gains, damping, and interpolation
  settings.

After joint-space control is stable, add and test `OSC_POSE`.

Verify:

- Jacobian dimensions.
- End-effector site and frame orientation.
- Mass matrix and torque compensation.
- Cartesian position and orientation action signs.
- Controller outputs against MuJoCo actuator ranges.
- Gripper action dimensions and ordering.

Relevant loading path:

```text
Robot.__init__()
  -> load_composite_controller_config(robot=<name>)
  -> FixedBaseRobot._load_controller()
  -> CompositeController.load_controller_config()
  -> controller_factory()
```

### Stage 4 gate

- Every action dimension moves the expected joint or gripper component.
- Zero actions remain stable.
- Commands do not exceed actuator ranges.
- OSC Cartesian directions match the intended coordinate conventions.

## Stage 5 — Validate the robot in an existing task

Before introducing the custom task, instantiate:

```python
robosuite.make("Lift", robots="<RobotName>", ...)
```

Validate:

- Reset stability over many episodes.
- Reachability of the table center and corners.
- Gripper mounting pose.
- Gripper opening and closing direction.
- Proprioceptive observations.
- `agentview`.
- `robotview` and `eye_in_hand`, if provided.
- Joint-space control.
- OSC control.
- Self-collision behavior.
- Table collision and base placement.

### Stage 5 gate

The robot completes, or can be reliably teleoperated through, the existing
`Lift` environment. This isolates robot integration failures from custom task
failures.

---

# Track B — Build the shape-and-peg task

## Stage 6 — Implement the board, pegs, and pieces

Recommended model structure:

```text
robosuite/models/objects/shape_sorter.py
robosuite/models/assets/objects/shape_sorter/
  board_visual_mesh.*
  circle_visual_mesh.*
  square_visual_mesh.*
  triangle_visual_mesh.*
  rectangle_visual_mesh.*
```

Suggested model classes:

- `ShapeSorterBoard`
- `PeggedTarget`, unless the pegs are integrated into the board model
- `CirclePiece`
- `SquarePiece`
- `TrianglePiece`
- `RectanglePiece`

The models should derive from
`robosuite/models/objects/objects.py::MujocoObject`, normally through
`MujocoGeneratedObject` or `MujocoXMLObject`.

Design requirements:

- The board and pegs are fixed.
- Each movable piece has one free joint.
- Each piece has:
  - A graspable outer boundary.
  - Visual geometry.
  - Compound collision geometry that preserves its holes.
  - A named center site.
  - A named grasp site.
  - A named bottom or seating-plane site.
  - Optional named sites for each hole center.
- Each target has:
  - A named target-pose site.
  - Named sites for each peg axis or peg top.
  - A named final seating-height site.
- Each piece and target pair has a distinct color.
- Task identity is not inferred from color alone.

### Stage 6 gate

Load an `EmptyArena` model containing only the board and pieces. Manually place
each piece above its target and confirm that it:

- Physically slides over the pegs.
- Does not collide with an artificial convex hull covering its holes.
- Reaches the intended final seating height.
- Rests stably on the board.

## Stage 7 — Create the environment skeleton

Create:

```text
robosuite/environments/manipulation/shape_peg_insertion.py
```

Proposed class:

```python
class ShapePegInsertion(ManipulationEnv):
    ...
```

Initially implement:

- `__init__`
- `_check_robot_configuration`
- `_load_model`
- `_setup_references`
- `_reset_internal`
- `reward`
- `_check_success`

`_load_model()` should:

1. Call `super()._load_model()`.
2. Position the robot relative to the table.
3. Create a `TableArena`.
4. Create the board and four movable pieces.
5. Create placement samplers for the pieces.
6. Assign a `ManipulationTask` containing:
   - Arena.
   - Robot model.
   - Board.
   - Four movable pieces.

The concrete environment will be automatically registered by
`robosuite/environments/base.py::EnvMeta` when its module is imported.

### Stage 7 gate

Both of the following construct, reset, and render successfully:

```python
robosuite.make("ShapePegInsertion", robots="Panda")
robosuite.make("ShapePegInsertion", robots="<RobotName>")
```

At this stage, observations and shaped rewards may remain minimal.

## Stage 8 — Implement safe randomized placement

Use `SequentialCompositeSampler` or constrained `UniformRandomSampler`
instances from `robosuite/utils/placement_samplers.py`.

Randomize:

- Piece XY positions within reachable pickup regions.
- Piece yaw, initially within a small range.
- Optionally board position within a much smaller range.

Constrain:

- Pieces do not overlap.
- No piece starts already seated.
- Pieces remain inside the robot's reachable workspace.
- Pieces have adequate clearance from the board.
- Pieces have adequate clearance from table edges.
- Deterministic resets remain reproducible.

Begin with fixed positions and introduce randomization only after the physical
insertion behavior works.

### Stage 8 gate

Hundreds of resets produce scenes that are:

- Nonpenetrating.
- Reachable.
- Free of initial success states.
- Reproducible when seeded.

## Stage 9 — Add references and observations

In `_setup_references()`, cache:

- Body IDs for all four pieces.
- Target site IDs.
- Piece center and bottom site IDs.
- Peg site or geom IDs.
- Board body ID.
- Contact geom IDs if contact is part of success validation.

In `_setup_observables()`, call `super()` and add, for every piece:

- Piece position.
- Piece quaternion.
- Matching target position.
- Matching target quaternion.
- Piece-to-target relative position.
- Relative orientation or yaw error.
- End-effector-to-piece position.
- Per-piece insertion or seated flag.
- Optional hole-to-peg alignment errors.

Also add:

- Number of completed pieces.
- Overall task-complete flag.
- Camera observations through existing environment camera arguments.

Prefer relative task coordinates over exposing every individual peg position
unless a learning policy specifically requires them.

### Stage 9 gate

- Observation keys and shapes are constant across resets.
- All observation values are finite.
- Piece and target values update correctly.
- Relative pose values approach zero at valid insertion poses.

## Stage 10 — Define insertion and success robustly

A piece counts as inserted only when all required conditions hold:

- It is the correct piece for the target.
- XY translation error is below tolerance.
- Yaw or full orientation error is below tolerance.
- Its bottom height is close to the target seating height.
- It is no longer held by the gripper.
- Optionally its linear and angular speeds are below thresholds.
- Optionally the condition persists for several simulation steps.

For multi-peg pieces, use the maximum projected hole-to-peg alignment error,
not only center-position error. This prevents a centered but incorrectly
rotated square, triangle, or rectangle from counting as complete.

Suggested helpers:

```text
_piece_alignment_error(piece_name)
_is_piece_seated(piece_name)
_completed_piece_count()
_check_success()
```

`_check_success()` should return true only when all four
`_is_piece_seated(...)` checks are true.

### Stage 10 gate

Success is false for:

- A hovering piece.
- Incorrect rotation.
- A piece over the wrong target.
- Partial insertion.
- A piece still held by the gripper.
- Only a subset of completed pieces.

Success is true when all four matching pieces are properly seated.

## Stage 11 — Add reward in layers

Implement and test reward components incrementally.

### 11.1 Sparse completion

- Reward only when all four pieces are seated.

### 11.2 Per-piece completion

- Reward each correctly seated piece.
- Decide whether the reward is state-based each step or emitted once.

### 11.3 Reach

- Reward the end effector approaching an unfinished piece.

### 11.4 Grasp

- Reward a valid grasp of an unfinished piece.

### 11.5 Transport

- Reward a grasped piece approaching its matching target.

### 11.6 Alignment

- Reward piece-center alignment.
- Reward yaw or hole-to-peg alignment above the target.

### 11.7 Insertion

- Reward downward progress only while the piece is sufficiently aligned.

### 11.8 Optional small penalties

- Wrong target.
- Dropped piece.
- Excessive collision.
- Excessive action magnitude.

Keep `_check_success()` independent from reward shaping. Normalize the maximum
task reward through `reward_scale`, following
`robosuite/environments/manipulation/lift.py::Lift.reward`.

Do not reward unrestricted downward motion. Without an alignment condition,
the policy may learn to force pieces into the board.

### Stage 11 gate

Scripted states produce monotonically sensible rewards through:

```text
reach -> grasp -> transport -> align -> insert
```

Maximum reward occurs only at full task completion.

## Stage 12 — Registration and configuration exposure

After the environment works:

- Import `ShapePegInsertion` from `robosuite/__init__.py`.
- Confirm it appears in `robosuite.ALL_ENVIRONMENTS`.
- Confirm the custom robot appears in `robosuite.ALL_ROBOTS`.
- Provide a project-owned controller JSON when modification of upstream
  defaults is undesirable.
- Expose task constructor parameters for:
  - Geometry tolerances.
  - Placement ranges.
  - Reward shaping.
  - Camera selection.
  - Piece randomization.
  - Board randomization.
  - Success persistence duration.

### Stage 12 gate

The integrated environment constructs successfully:

```python
env = robosuite.make(
    "ShapePegInsertion",
    robots="<RobotName>",
    controller_configs=...,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
)
```

The resulting action and observation spaces match the custom robot,
controller, gripper, and task configuration.

---

# Stage 13 — Test strategy

Suggested test files:

```text
tests/test_models/test_custom_7dof_robot.py
tests/test_models/test_shape_sorter_objects.py
tests/test_environments/test_shape_peg_insertion.py
```

## Robot tests

- Robot registration and construction.
- Exactly seven arm joints.
- Matching arm actuator count.
- Verified joint and actuator order.
- Default controller loading.
- Action dimensions and limits.
- Gripper attachment.
- End-effector reference resolution.
- Camera reference resolution.

## Object and board tests

- Board and piece MJCF compilation.
- Piece free-joint configuration.
- Collision geometry leaves holes usable.
- Each piece can be manually seated.
- Board and pegs remain fixed.
- No unstable contacts at the seated pose.

## Environment tests

- Construction with Panda.
- Construction with the custom robot.
- Reset validity over many seeds.
- Observation keys, shapes, and finite values.
- Per-piece success predicates at boundary cases.
- Wrong-piece and wrong-target rejection.
- Partial-insertion rejection.
- Full-task success.
- Reward scaling.
- Shaped-reward ordering.
- Camera rendering.
- Short random-action rollouts without NaNs or simulation errors.

Run existing environment, robot, controller, and rendering tests after
integration to catch registry or import regressions.

---

# Recommended implementation order

```text
Robot MJCF audit
  -> Python robot model
  -> runtime wrapper registration
  -> joint-position controller
  -> existing Lift validation
  -> board and piece collision prototypes
  -> environment skeleton with fixed placements
  -> insertion predicates
  -> observations
  -> sparse reward
  -> randomized reset
  -> shaped reward
  -> camera and controller tuning
  -> integrated task
  -> complete test suite
```

# Inputs needed before implementation

## Robot inputs

- Complete robot MJCF package.
- Intended public robot class/name.
- Mesh and texture files.
- Joint order.
- Actuator order and actuator type.
- End-effector body name.
- Initial joint configuration.
- Base mounting requirements.
- Intended gripper and its attachment transform.
- Available kinematic or dynamic parameters.

## Task inputs

- Board dimensions.
- Piece dimensions and thickness.
- Peg positions, diameter, and height.
- Hole positions and intended clearance.
- Piece masses and material assumptions.
- Desired initial placement ranges.
- Required camera views.
- Success tolerances.
- Whether the task is ordered or allows insertion in any sequence.
- Whether all pieces must remain seated simultaneously.

Peg diameter, peg height, and piece-hole clearance are especially important:
they determine whether insertion is physically tractable rather than merely
visually correct.
