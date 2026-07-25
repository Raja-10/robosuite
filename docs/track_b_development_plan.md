# Track B Temporary Development Plan

This is the working implementation plan for the four-shape peg-insertion
task. It should be updated as dimensions, tolerances, and task behavior are
validated.

Current status: Stage B1 is implemented for the fixed board, one-peg circle
piece, standalone drop validator, and model regression tests. The aligned
circle drop seats successfully in headless MuJoCo simulation. Visual
inspection succeeded after reducing the piece thickness to 12 mm. Deliberate
misalignment checks reject a laterally offset circle.

Stage B2 is implemented for the two-hole square, generalized object validator,
per-hole error reporting, yaw reporting, and aligned/incorrect-yaw regression
tests. Its aligned headless drop succeeds; visual inspection remains before
the Stage B2 gate is closed.

Stage B3 is implemented for the three-hole triangle and four-hole rectangle.
Both use horizontal primitive bands with explicit collision openings.
Individual and simultaneous aligned headless drops succeed, and incorrect-yaw
configurations are rejected. Visual inspection remains before the Stage B3
gate is closed.

The triangle uses collision-only primitive bands plus a separate triangular
prism visual mesh. This preserves reliable three-hole collision mechanics
without exposing the stepped band approximation in rendered observations.

Stage B4 validation infrastructure is implemented. The standalone scene now
has a larger support surface and offers deterministic shuffled off-board
spawns through `--layout shuffled --seed <n>`. Seating additionally requires
low linear and angular velocity and at least 50 consecutive valid simulation
steps. Seed reproducibility, spawn separation, non-successful shuffled resets,
and stability-aware success are covered by regression tests.

Stage B5 is implemented as
`robosuite.environments.manipulation.shape_peg_insertion.ShapePegInsertion`.
It uses `TableArena`, the complete fixed board, separated randomized piece
samplers, an additional fill light, task/agent/top/front cameras, object-state
observables, sparse and shaped rewards, and control-step stability-aware
success. Registration, reset placement, finite stepping, and exact all-piece
success are regression-tested with Panda. Nero7 plus its world-frame OSC
configuration also passes a headless construction and 100-step hold smoke
test.

Nero7 is integrated through its default
`robosuite.models.bases.nero_table_mount.NeroTableMount`. The mount preserves
the validated root pose `[-0.36, 0.0, 0.85]` in this 0.8 m table setup, places
the collar at `[-0.36, 0.0, 0.8375]`, and does not contact-penetrate the table.
The mounted robot passes a world-frame OSC zero-action hold in the task.

The task observation contract is expanded for scripted policies. Each piece
now reports world pose and velocity, EEF-to-piece and piece-to-target vectors,
flattened per-hole `peg_xy - hole_xy` residuals, maximum hole error, height,
tilt and yaw errors, linear and angular speeds, grasped and lifted flags,
stability progress, and the final seated flag. With Panda proprioception
disabled from cameras, the concatenated object-state vector has 148 values.

Nero7 task teleoperation is implemented in
`robosuite/scripts/teleop_nero7_shape_peg_insertion.py`. It uses the validated
world-frame OSC configuration, supports keyboard and SpaceMouse inputs,
maintains gripper state, reports nearest-piece and seating status, resets on
`q`, exits on window close or Ctrl+C, and can optionally record raw Robosuite
states and actions through `DataCollectionWrapper`.

Track B must initially be developed independently of Nero7. The board and
pieces should work in a minimal MuJoCo simulation before they are integrated
with the robot and scripted data-collection policy.

## 1. Task definition

The task contains four movable pieces and four matching target stations:

| Piece | Hole and peg pattern |
|---|---|
| Circle | One central hole and peg |
| Square | Two holes and pegs |
| Triangle | Three holes and pegs |
| Rectangle | Four holes and pegs |

A piece is complete only when it is correctly matched, aligned over every peg,
lowered to the seated height, and left in a stable pose.

## 2. Proposed files

Initial implementation:

```text
robosuite/models/objects/shape_sorter.py
robosuite/scripts/validate_shape_sorter_objects.py
tests/test_models/test_shape_sorter.py
```

Optional visual assets, added only after collision mechanics work:

```text
robosuite/models/assets/objects/shape_sorter/
  meshes/
    board.obj
    circle_piece.obj
    square_piece.obj
    triangle_piece.obj
    rectangle_piece.obj
  textures/
```

Future task environment:

```text
robosuite/environments/manipulation/shape_peg_insertion.py
tests/test_environments/test_shape_peg_insertion.py
```

## 3. Object representation

Use `CompositeObject` from
`robosuite/models/objects/generated_objects.py::CompositeObject` for the
mechanical prototypes.

Relevant upstream references:

- `robosuite/models/objects/composite/hollow_cylinder.py::HollowCylinderObject`
  for circular shells with a real collision opening.
- `robosuite/models/assets/objects/square-nut.xml` for constructing a plate
  around a hole from multiple box geoms.
- `robosuite/models/objects/xml_objects.py::SquareNutObject` for free-joint
  object construction and important sites.
- `robosuite/environments/manipulation/nut_assembly.py` for peg placement,
  reward shaping, and success-check architecture.

Do not represent a piece containing holes with one solid collision mesh.
MuJoCo may collision-test the convex hull, effectively filling the holes.

Use:

- Compound primitive collision geoms around each hole.
- Cylinder geoms for pegs.
- Optional visual-only meshes for appearance.

## 4. Coordinate convention

All dimensions use metres.

Piece-local frame:

```text
origin: geometric center
+x: piece width
+y: piece depth
+z: upward through the top face
```

Board-local frame:

```text
origin: board center
+x: board width
+y: board depth
+z: upward
```

All hole positions are expressed in the corresponding piece-local frame.
Board peg positions are generated from the same hole-pattern specification,
translated by the target station position.

This shared specification prevents drift between peg and hole layouts.

## 5. Initial dimensions

The following are prototype values and must be tuned in simulation:

```python
BOARD_SIZE = (0.32, 0.24, 0.012)
PIECE_THICKNESS = 0.012

PEG_RADIUS = 0.006
PEG_HEIGHT = 0.035

HOLE_RADIUS = 0.0075
RADIAL_CLEARANCE = 0.0015
```

Candidate piece dimensions:

| Piece | Dimensions |
|---|---|
| Circle | Radius `0.040 m` |
| Square | `0.070 × 0.070 m` |
| Triangle | Bounding size approximately `0.080 × 0.070 m` |
| Rectangle | `0.095 × 0.055 m` |

Start with `1.5–2.0 mm` radial clearance. Reduce clearance only after
insertion is stable.

Candidate target stations:

```python
TARGET_STATIONS = {
    "circle":    (-0.09,  0.065),
    "square":    ( 0.09,  0.065),
    "triangle":  (-0.09, -0.065),
    "rectangle": ( 0.09, -0.065),
}
```

Candidate local hole patterns:

```python
SHAPE_SPECS = {
    "circle": {
        "hole_positions": [(0.0, 0.0)],
    },
    "square": {
        "hole_positions": [
            (-0.018, 0.0),
            ( 0.018, 0.0),
        ],
    },
    "triangle": {
        "hole_positions": [
            ( 0.000,  0.012),
            (-0.020, -0.015),
            ( 0.020, -0.015),
        ],
    },
    "rectangle": {
        "hole_positions": [
            (-0.028, -0.014),
            (-0.028,  0.014),
            ( 0.028, -0.014),
            ( 0.028,  0.014),
        ],
    },
}
```

These layouts are provisional and should ultimately match the intended
physical board.

## 6. Board model

Implement `ShapeSorterBoard` as a fixed `CompositeObject` with `joints=None`.

It should contain:

- One board base.
- One cylinder geom per peg.
- Optional visual-only target-region geoms.
- One target site per station.
- One site per peg axis.
- One seated-height site per station.

Naming convention:

```text
board_base
circle_peg_0
square_peg_0
square_peg_1
triangle_peg_0
triangle_peg_1
triangle_peg_2
rectangle_peg_0
rectangle_peg_1
rectangle_peg_2
rectangle_peg_3
```

Target sites:

```text
circle_target
square_target
triangle_target
rectangle_target

circle_seated
square_seated
triangle_seated
rectangle_seated
```

For a MuJoCo cylinder, `size` is `(radius, half_height)`. The peg center must
therefore be located at the board top plus half the peg height.

## 7. Piece models

Movable pieces use a free joint:

```python
joints=[dict(type="free", damping="0.0005")]
```

Each piece must expose:

```text
center
top
bottom
grasp
orientation_x
hole_0
hole_1
...
```

These sites will support observations, scripted grasp targets, alignment
errors, rewards, and success evaluation.

### Circle

Build the circle collision shell from 16–24 tangential box geoms around the
central hole, following
`HollowCylinderObject._get_geom_attrs()`.

### Square

Build the plate from rectangular rails surrounding its two holes:

- Top and bottom rails.
- Left and right rails.
- A central rail between the holes.

### Triangle

Build the triangle from:

- Three rotated edge rails.
- Connector regions between the holes.
- Optional capsule or box geoms near the corners.

Add a visual-only triangle mesh later if the primitive appearance is
insufficient.

### Rectangle

Build the four-hole rectangle from:

- Outer top and bottom rails.
- Outer left and right rails.
- Internal rails between the four holes.

Avoid unnecessary geom overlap because overlapping collision geoms duplicate
mass and contacts.

## 8. Collision and visual separation

Use:

```text
collision geoms: group 0
visual geoms: group 1, contype 0, conaffinity 0
```

During the first prototype, collision primitives may also provide the visible
colors. Add decorative meshes only after all insertion tests pass.

If a mesh is added later:

- Keep it visual-only.
- Retain primitive or explicitly convex-decomposed collision geometry.
- Confirm that the visual mesh uses the same origin and scale as the collision
  model.

## 9. Initial physical parameters

Candidate starting values:

```python
density = 300.0
friction = (1.0, 0.005, 0.0001)
solref = (0.02, 1.0)
solimp = (0.9, 0.95, 0.001)
condim = 4
```

Inspect compiled values through:

```python
env.sim.model.body_mass[body_id]
env.sim.model.body_inertia[body_id]
```

Target a plausible prototype piece mass, initially approximately
`0.05–0.15 kg`.

Do not tune contact parameters to hide incorrect geometry. Verify dimensions,
hole clearance, body frames, and geom overlap first.

## 10. Success definition

Success must be based on geometry, not visual appearance alone.

For every corresponding hole and peg:

```text
horizontal hole-to-peg error < xy tolerance
```

For the complete piece:

```text
bottom height near seated height
roll and pitch near zero
yaw aligned with target
piece stable for a minimum number of steps
optionally, robot no longer grasping the piece
```

Conceptual condition:

```python
holes_aligned = all(
    np.linalg.norm(hole_xy[i] - peg_xy[i]) < xy_tolerance
    for i in range(number_of_holes)
)

piece_success = (
    holes_aligned
    and seated
    and upright
    and yaw_correct
    and stable
)
```

The final task succeeds when all four pieces satisfy their own conditions
simultaneously.

`NutAssembly.on_peg()` and `NutAssembly._check_success()` are architectural
references, but their center-distance test is not strict enough for this
multi-hole task.

## 11. Staged implementation

### Stage B1 — One-peg circle prototype

Implement only:

- Fixed board base.
- Circle target station and one peg.
- Circular piece with one real collision hole.
- Standalone object validator.

Validate:

- XML compilation.
- Correct object mass and inertia.
- Piece free-fall stability.
- Peg entry through the hole.
- Board prevents further downward movement.
- Stable seated pose.
- Gripper-width compatibility.

Gate: the circle can be dropped over the peg repeatedly without tunneling,
exploding, or resting above the peg on false collision geometry.

### Stage B2 — Two-hole square

Add:

- Square piece.
- Two matching pegs.
- Hole-to-peg alignment reporting.
- Yaw-error reporting.

Gate: both pegs enter simultaneously, and incorrect yaw prevents seating.

### Stage B3 — Triangle and rectangle

Add the three-hole triangle and four-hole rectangle using compound primitive
collision geometry.

Gate: each piece independently inserts and remains seated.

### Stage B4 — Complete board

Combine all stations and test every piece:

- Independently.
- In shuffled initial positions.
- With random initial yaw.
- With all pieces present simultaneously.

Gate: no unwanted collision between unrelated stations, pieces, or pegs.

### Stage B5 — Robosuite task environment

Implement:

```text
robosuite/environments/manipulation/shape_peg_insertion.py
```

The environment should:

- Derive from `ManipulationEnv`.
- Load `TableArena`.
- Instantiate the board and four pieces.
- Use a placement sampler for initial piece positions.
- Cache body, geom, joint, and site IDs in `_setup_references()`.
- Register object and target observables in `_setup_observables()`.
- Compute shaped and sparse rewards in `reward()`.
- Evaluate per-piece and overall success in `_check_success()`.

Gate: the task works first with Panda or another established robot before
Nero7-specific testing.

### Stage B6 — Nero7 integration

Instantiate the completed task with `robots="Nero7"` and validate:

- Reachability of every spawn region and station.
- Camera visibility.
- Collision-free approach paths.
- Grasp width and grasp sites.
- OSC stability near the board.
- Insertion force/contact behavior.

Gate: each shape can be inserted manually or by a diagnostic policy.

### Stage B7 — Scripted policy and dataset collection

Implement a phase-based policy:

```text
select piece
-> approach grasp
-> descend
-> close gripper
-> verify grasp
-> lift
-> move above target
-> align yaw and holes
-> descend/insertion
-> verify seating
-> release
-> retreat
```

Record phase, failure reason, controller saturation, collisions, grasp state,
per-hole errors, and success state in every rollout.

Gate: repeated rollouts meet an agreed success threshold before bulk dataset
collection.

## 12. Standalone validation script

Create:

```text
robosuite/scripts/validate_shape_sorter_objects.py
```

Initial interface:

```bash
python -m robosuite.scripts.validate_shape_sorter_objects \
  --shape circle
```

It should support:

- Visual and headless modes.
- Selection of one piece.
- Reset above the matching target.
- Drop test.
- Exact seated-pose test.
- Collision-pair reporting.
- Body mass and inertia reporting.
- Hole-to-peg XY errors.
- Piece-bottom height error.
- Roll, pitch, and yaw error.
- Finite-state validation.

This script must not require Nero7 or the final environment.

## 13. Regression tests

Create:

```text
tests/test_models/test_shape_sorter.py
```

Test:

- Expected peg and hole counts.
- Unique geom and site names.
- Board has no free joint.
- Each piece has one free joint.
- Piece mass and inertia are finite and positive.
- Exact aligned configurations compile without false collision blocking.
- Deliberately incorrect positions and orientations do not pass success
  predicates.
- Hole and peg patterns are generated from the same specification.

Later environment tests should cover reset randomization, observations,
rewards, termination, and all-four-piece success.

## 14. Safe extension points

Track B-specific work should remain in:

- `robosuite/models/objects/shape_sorter.py`
- Optional `robosuite/models/assets/objects/shape_sorter/`
- `robosuite/environments/manipulation/shape_peg_insertion.py`
- Track B scripts and tests

Normally do not modify:

- `robosuite/models/objects/generated_objects.py`
- `robosuite/models/objects/objects.py`
- `robosuite/environments/base.py`
- `robosuite/environments/robot_env.py`
- `robosuite/environments/manipulation/manipulation_env.py`
- Generic placement sampler implementation
- Generic reward or observable infrastructure

Extend these upstream abstractions through subclasses and task-specific helper
functions.

## 15. Immediate next implementation

The first development increment should contain only:

```text
robosuite/models/objects/shape_sorter.py
  shared dimensions
  shared target and hole patterns
  ShapeSorterBoard with the circle station
  CirclePiece

robosuite/scripts/validate_shape_sorter_objects.py
  circle-only standalone drop and seating validation

tests/test_models/test_shape_sorter.py
  circle and board structural tests
```

Do not add the remaining pieces or the Robosuite task environment until the
circle collision hole and one-peg insertion are mechanically validated.
