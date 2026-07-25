# Robosuite codebase map

This map describes the task path in the checked-out Robosuite 1.5.1 code. In
this codebase, “task” has two related meanings:

- A concrete environment such as `Lift` implements episode behavior, including
  reset placement, observations, reward, and success
  (`robosuite/environments/manipulation/lift.py::Lift`).
- `robosuite.models.tasks.task.Task` is the MJCF composition object that merges
  an arena, robot models, and object models. `ManipulationTask` currently adds
  no behavior to `Task`
  (`robosuite/models/tasks/task.py::Task.__init__`,
  `robosuite/models/tasks/manipulation_task.py::ManipulationTask`).

## 1. Environment registration and instantiation

Important files and symbols:

- `robosuite/environments/base.py::REGISTERED_ENVS` is the name-to-class
  registry.
- `robosuite/environments/base.py::EnvMeta.__new__` registers every subclass
  whose class name is not one of `MujocoEnv`, `RobotEnv`, `ManipulationEnv`, or
  `TwoArmEnv`. Registration uses the concrete Python class name through
  `register_env()`.
- `robosuite/environments/base.py::make` validates `env_name` and invokes
  `REGISTERED_ENVS[env_name](*args, **kwargs)`.
- `robosuite/__init__.py` re-exports `make` and imports the built-in concrete
  manipulation environments. Those imports execute their class definitions
  and therefore trigger `EnvMeta` registration.
- `robosuite/environments/__init__.py::ALL_ENVIRONMENTS` is a view of the
  resulting registry keys.

Instantiation call path:

```text
import robosuite
  -> robosuite/__init__.py imports concrete environment modules
  -> EnvMeta.__new__()
  -> register_env(concrete_class)

robosuite.make("Lift", robots="Panda", ...)
  -> environments/base.py::make()
  -> REGISTERED_ENVS["Lift"](...)
  -> Lift.__init__()
  -> ManipulationEnv.__init__()
  -> RobotEnv.__init__()
  -> MujocoEnv.__init__()
  -> _load_model()                 # virtual dispatch to Lift
  -> _initialize_sim()
  -> initialize_renderer()
  -> _reset_internal()
  -> _setup_observables()
```

The constructor ordering above is explicit in
`robosuite/environments/base.py::MujocoEnv.__init__`: it calls `_load_model`,
`_initialize_sim`, renderer initialization, `_reset_internal`, and finally
`_setup_observables`. `_initialize_sim` serializes `self.model` with
`get_xml()`, applies XML processors, constructs
`robosuite.utils.binding_utils.MjSim`, calls `sim.forward()`, and initializes
the timing values (`robosuite/environments/base.py::MujocoEnv._initialize_sim`).

An external custom environment does not have to modify the global registry
directly. Defining a concrete `MujocoEnv` descendant registers it through
`EnvMeta`; its defining module must be imported before calling `make`. A
built-in environment is made discoverable on `import robosuite` by adding its
import to `robosuite/__init__.py`, but an out-of-tree extension should import
its own module instead of patching that upstream file
(`robosuite/environments/base.py::EnvMeta.__new__`,
`robosuite/__init__.py`).

## 2. Task inheritance

The usual single-arm manipulation inheritance chain is:

```text
MujocoEnv
  └── RobotEnv
        └── ManipulationEnv
              └── Lift / Stack / Door / PickPlace / ...
```

- `MujocoEnv` owns simulation lifecycle, stepping, observation polling,
  horizon termination, rendering, and abstract `reward()` /
  `_check_success()` hooks
  (`robosuite/environments/base.py::MujocoEnv`).
- `RobotEnv` adds robot selection and construction, robot and camera
  observables, combined action space, robot reset, and action dispatch
  (`robosuite/environments/robot_env.py::RobotEnv`).
- `ManipulationEnv` adds gripper configuration and common manipulation helpers
  such as `_get_obj_eef_sensor`, `_check_grasp`, and
  `_gripper_to_target`
  (`robosuite/environments/manipulation/manipulation_env.py::ManipulationEnv`).
- A concrete environment supplies task-specific `_load_model`,
  `_setup_references`, `_setup_observables`, `_reset_internal`, `reward`, and
  `_check_success` as needed. `Lift` is the smallest representative example
  (`robosuite/environments/manipulation/lift.py::Lift`).
- Two-arm environments use `TwoArmEnv`, which itself derives from
  `ManipulationEnv`
  (`robosuite/environments/manipulation/two_arm_env.py::TwoArmEnv`).

The separate model inheritance chain is:

```text
MujocoWorldBase
  └── Task
        └── ManipulationTask
```

`Task.__init__` merges models; it does not implement environment stepping,
reward, or success (`robosuite/models/tasks/task.py::Task`,
`robosuite/models/tasks/manipulation_task.py::ManipulationTask`).

## 3. Loading robots, objects, and arenas

### Robots

`RobotEnv._load_model()` calls `RobotEnv._load_robots()`. For each requested
name, `_load_robots()`:

1. Selects a runtime wrapper from
   `robosuite/robots/__init__.py::ROBOT_CLASS_MAPPING` (for example, `Panda`
   maps to `FixedBaseRobot`).
2. Instantiates that wrapper and calls
   `robosuite.robots.robot.Robot.load_model()`.
3. `Robot.load_model()` calls
   `robosuite.models.robots.robot_model.create_robot()`, whose model-class
   registry is `REGISTERED_ROBOTS`.
4. `RobotModelMeta.__new__` populates that model registry when concrete robot
   model classes are imported
   (`robosuite/models/robots/robot_model.py::RobotModelMeta`).
5. The concrete model loads its MJCF. For example,
   `robosuite/models/robots/manipulators/panda_robot.py::Panda.__init__`
   resolves `robots/panda/robot.xml`, located at
   `robosuite/models/assets/robots/panda/robot.xml`.
6. `Robot.load_model()` attaches the selected/default base using
   `robot_base_factory()`, creates the selected/default gripper using
   `gripper_factory()`, and attaches it through
   `RobotModel.add_gripper()`
   (`robosuite/robots/robot.py::Robot.load_model`).

There are therefore two robot layers:

- Runtime/control wrappers:
  `robosuite/robots/robot.py::Robot` and subclasses
  `FixedBaseRobot`, `WheeledRobot`, `MobileRobot`, and `LeggedRobot`.
- MJCF descriptions:
  `robosuite/models/robots/robot_model.py::RobotModel` and concrete model
  classes under `robosuite/models/robots/`.

### Arenas and objects

A concrete environment creates these in its `_load_model()`. In `Lift`:

- `TableArena(...)` loads
  `robosuite/models/assets/arenas/table_arena.xml` through
  `robosuite/models/arenas/table_arena.py::TableArena.__init__`.
- `BoxObject(...)` creates the cube through
  `robosuite/models/objects/primitive/box.py::BoxObject`.
- Other reusable XML-backed objects are defined in
  `robosuite/models/objects/xml_objects.py` and load files from
  `robosuite/models/assets/objects/`.
- `Lift._load_model()` assigns `self.model = ManipulationTask(...)`, passing
  the arena, `[robot.robot_model for robot in self.robots]`, and the cube
  (`robosuite/environments/manipulation/lift.py::Lift._load_model`).

`Task.__init__()` then calls `merge_arena()`, `merge_robot()` for every robot,
and `merge_objects()`. Object merging adds assets and appends each
`MujocoObject.get_obj()` body
(`robosuite/models/tasks/task.py::Task.__init__`,
`Task.merge_objects`). The lower-level XML merge implementation combines
world bodies, assets, actuators, sensors, tendons, equality constraints, and
contacts (`robosuite/models/base.py::MujocoXML.merge`).

On simulation setup, `MujocoEnv._setup_references()` calls
`Task.generate_id_mappings()`. `RobotEnv._setup_references()` additionally
binds each runtime robot to the current `MjSim` and calls
`Robot.setup_references()` to resolve MuJoCo joint, actuator, body, and site
indices (`robosuite/environments/base.py::MujocoEnv._setup_references`,
`robosuite/environments/robot_env.py::RobotEnv._setup_references`,
`robosuite/robots/robot.py::Robot.setup_references`).

## 4. Observation creation

Observations are procedural `Observable` objects rather than a static schema:

1. A `@sensor(modality=...)` function accepts `obs_cache` and returns a scalar
   or array (`robosuite/utils/observables.py::sensor`).
2. The sensor is wrapped in
   `robosuite.utils.observables.Observable`, which controls sampling rate,
   corruption, filtering, delay, enabled state, and active state
   (`robosuite/utils/observables.py::Observable`).
3. `MujocoEnv.__init__()` calls the virtual `_setup_observables()` and stores
   its ordered mapping in `self._observables`
   (`robosuite/environments/base.py::MujocoEnv.__init__`).
4. `RobotEnv._setup_observables()` first gathers each
   `Robot.setup_observables()` result, then creates RGB/depth/segmentation
   camera observables when `use_camera_obs` is true
   (`robosuite/environments/robot_env.py::RobotEnv._setup_observables`,
   `RobotEnv._create_camera_sensors`).
5. A concrete task calls `super()._setup_observables()` and adds its object
   sensors. For example, `Lift._setup_observables()` adds `cube_pos`,
   `cube_quat`, and gripper-to-cube position sensors
   (`robosuite/environments/manipulation/lift.py::Lift._setup_observables`).
6. Every internal physics step calls
   `MujocoEnv._update_observables()`. This calls `Observable.update()`, which
   evaluates the sensor and updates `obs_cache` at the configured sample time
   (`robosuite/environments/base.py::MujocoEnv.step`,
   `robosuite/environments/base.py::MujocoEnv._update_observables`,
   `robosuite/utils/observables.py::Observable.update`).
7. `MujocoEnv._get_observations()` returns enabled and active values by name
   and also concatenates values into `<modality>-state` entries (except images
   unless `macros.CONCATENATE_IMAGES` is enabled)
   (`robosuite/environments/base.py::MujocoEnv._get_observations`,
   `robosuite/macros.py::CONCATENATE_IMAGES`).

Robot proprioception originates in
`robosuite/robots/robot.py::Robot.setup_observables` and
`Robot._create_arm_sensors`; task object state belongs in the concrete
environment's `_setup_observables`; camera images belong to
`RobotEnv._create_camera_sensors`.

## 5. Action-to-controller call path

For a fixed-base robot, the main path is:

```text
MujocoEnv.step(action)
  -> physics-rate loop
  -> RobotEnv._pre_action(action, policy_step)
  -> split flat action per robot
  -> FixedBaseRobot.control(robot_action, policy_step)
  -> CompositeController.update_state()
  -> CompositeController.set_goal(action)       # policy steps only
  -> split action per configured body part
  -> part Controller.set_goal(part_action)
  -> CompositeController.run_controller()
  -> each enabled part Controller.run_controller()
  -> clip against MuJoCo actuator_ctrlrange
  -> sim.data.ctrl[part actuator indexes] = output
  -> sim.step2() / sim.step()
```

The environment-level split is implemented by
`robosuite/environments/robot_env.py::RobotEnv._pre_action`. The fixed-base
control path and final write to `sim.data.ctrl` are in
`robosuite/robots/fixed_base_robot.py::FixedBaseRobot.control`. Mobile,
wheeled, and legged wrappers override `control()` in their corresponding files
under `robosuite/robots/`.

`robosuite/controllers/composite/composite_controller.py::CompositeController.load_controller_config`
creates body-part controllers through
`CompositeController._init_controllers()` and
`robosuite/controllers/parts/controller_factory.py::controller_factory`.
`CompositeController.setup_action_split_idx()` determines each part's action
slice; `set_goal()` handles the split (and formats gripper commands); and
`run_controller()` obtains actuator-space outputs. Concrete arm controllers
are selected in
`robosuite/controllers/parts/controller_factory.py::arm_controller_factory`.

`policy_step` is true only for the first physics step corresponding to a new
environment action. Thus goals are set once, while controllers can produce
multiple actuator commands at the MuJoCo timestep before the next policy
action (`robosuite/environments/base.py::MujocoEnv.step`,
`robosuite/robots/fixed_base_robot.py::FixedBaseRobot.control`).

## 6. Reward, success, and termination

After the physics-rate loop, `MujocoEnv.step()` calls `_post_action(action)`.
`_post_action()` calls the virtual `reward(action)` and independently marks
`done` when `timestep >= horizon` unless `ignore_done` is true
(`robosuite/environments/base.py::MujocoEnv.step`,
`MujocoEnv._post_action`). In this API, task success does **not**
automatically terminate an episode; a concrete environment would have to
override the post-action/termination behavior to do that.

Concrete tasks define their own relationship between reward and success. For
example:

- `Lift._check_success()` compares cube height with arena table height plus
  `0.04`
  (`robosuite/environments/manipulation/lift.py::Lift._check_success`).
- `Lift.reward()` returns the completion reward when `_check_success()` is
  true, otherwise optionally adds reaching and grasping shaping, and finally
  applies `reward_scale`
  (`robosuite/environments/manipulation/lift.py::Lift.reward`).

The base `MujocoEnv.reward()` and `_check_success()` are deliberately
unimplemented, and `RobotEnv` only delegates to them. A new concrete task must
implement both semantics itself
(`robosuite/environments/base.py::MujocoEnv.reward`,
`MujocoEnv._check_success`,
`robosuite/environments/robot_env.py::RobotEnv.reward`,
`RobotEnv._check_success`).

## 7. Where to implement a custom task

For an in-tree manipulation benchmark, the conventional location is a new
module under `robosuite/environments/manipulation/`, alongside
`lift.py`, `stack.py`, and the other concrete tasks. Derive from
`ManipulationEnv` for ordinary manipulation or `TwoArmEnv` when its
configuration behavior applies
(`robosuite/environments/manipulation/manipulation_env.py::ManipulationEnv`,
`robosuite/environments/manipulation/two_arm_env.py::TwoArmEnv`).

The safe task-level extension hooks are:

- `__init__`: store task parameters before calling `super().__init__`, because
  `MujocoEnv.__init__` immediately invokes virtual `_load_model()`
  (`robosuite/environments/base.py::MujocoEnv.__init__`,
  `robosuite/environments/manipulation/lift.py::Lift.__init__`).
- `_check_robot_configuration`: reject incompatible embodiments
  (`robosuite/environments/robot_env.py::RobotEnv._check_robot_configuration`).
- `_load_model`: call `super()`, position robot model bases, create the arena
  and objects, configure placement samplers, and assign a
  `ManipulationTask`
  (`robosuite/environments/manipulation/lift.py::Lift._load_model`).
- `_setup_references`: call `super()` and cache task-specific MuJoCo IDs
  (`robosuite/environments/manipulation/lift.py::Lift._setup_references`).
- `_setup_observables`: call `super()` and append task-specific `Observable`
  instances
  (`robosuite/environments/manipulation/lift.py::Lift._setup_observables`).
- `_reset_internal`: call `super()` and randomize object joints through a
  placement sampler
  (`robosuite/environments/manipulation/lift.py::Lift._reset_internal`,
  `robosuite/utils/placement_samplers.py::ObjectPositionSampler`).
- `reward` and `_check_success`: define task semantics
  (`robosuite/environments/manipulation/lift.py::Lift.reward`,
  `Lift._check_success`).

Reusable physical components should be added at their own layer rather than
embedded into environment plumbing:

- Arena class: `robosuite/models/arenas/`, deriving from
  `robosuite/models/arenas/arena.py::Arena`.
- Generated object: `robosuite/models/objects/generated_objects.py` or
  `robosuite/models/objects/primitive/`, deriving from
  `robosuite/models/objects/objects.py::MujocoObject`.
- XML-backed object: class in `robosuite/models/objects/xml_objects.py` plus an
  MJCF asset under `robosuite/models/assets/objects/`.

For an out-of-tree project, the safer upgrade-friendly form is the same class
hierarchy in the project's own package. Import that module before
`robosuite.make()` so `EnvMeta` sees the class; do not directly mutate
`REGISTERED_ENVS`
(`robosuite/environments/base.py::EnvMeta`,
`robosuite/environments/base.py::register_env`).

## 8. Camera, robot, and controller configuration

### Cameras

There is no single camera configuration file.

- Observation selection and output format are environment constructor
  arguments handled by `RobotEnv.__init__`: `use_camera_obs`,
  `camera_names`, `camera_heights`, `camera_widths`, `camera_depths`, and
  `camera_segmentations`
  (`robosuite/environments/robot_env.py::RobotEnv.__init__`).
- On-screen camera selection is the separate `render_camera` constructor
  argument owned by `MujocoEnv.__init__`
  (`robosuite/environments/base.py::MujocoEnv.__init__`).
- Camera pose, field of view, and name are MJCF `<camera>` elements. Arena
  cameras such as `agentview`, `frontview`, and `topview` are in
  `robosuite/models/assets/arenas/table_arena.xml`; Panda's `robotview` and
  `eye_in_hand` cameras are in
  `robosuite/models/assets/robots/panda/robot.xml`.
- `RobotModel.__init__()` discovers robot camera names from the MJCF, and
  `RobotEnv._reset_internal()` expands names of the form `all-<camera-key>`
  using those names
  (`robosuite/models/robots/robot_model.py::RobotModel.__init__`,
  `robosuite/environments/robot_env.py::RobotEnv._reset_internal`).
- Image orientation and concatenation defaults are in
  `robosuite/macros.py::IMAGE_CONVENTION` and
  `robosuite/macros.py::CONCATENATE_IMAGES`.
- Renderer-specific settings, which are distinct from task observation
  cameras, are loaded by
  `robosuite/renderers/base.py::load_renderer_config`; the checked-in
  non-default renderer configuration is
  `robosuite/renderers/config/nvisii_config.json`.

### Robots

- Robot kinematics, bodies, joints, actuators, sites, and attached cameras are
  defined in `robosuite/models/assets/robots/<robot>/robot.xml`; for example,
  `robosuite/models/assets/robots/panda/robot.xml`.
- The corresponding Python model class supplies semantic metadata and defaults
  such as `arms`, `default_base`, `default_gripper`, `init_qpos`, and
  `base_xpos_offset`; see
  `robosuite/models/robots/manipulators/panda_robot.py::Panda`.
- Model lookup and registration live in
  `robosuite/models/robots/robot_model.py::REGISTERED_ROBOTS`,
  `RobotModelMeta`, and `create_robot`.
- Runtime wrapper selection lives in
  `robosuite/robots/__init__.py::ROBOT_CLASS_MAPPING`.
- Bases and grippers have their own factories and assets under
  `robosuite/models/bases/`, `robosuite/models/grippers/`,
  `robosuite/models/assets/bases/`, and
  `robosuite/models/assets/grippers/`
  (`robosuite/robots/robot.py::Robot.load_model`).

### Controllers

- Per-robot composite defaults are JSON files under
  `robosuite/controllers/config/robots/`, for example
  `robosuite/controllers/config/robots/default_panda.json`.
- Generic composite templates are under
  `robosuite/controllers/config/default/composite/`, including `basic.json`,
  `hybrid_mobile_base.json`, `whole_body_ik.json`, and
  `whole_body_mink_ik.json`.
- Reusable part-controller defaults are under
  `robosuite/controllers/config/default/parts/`, including
  `osc_pose.json`, `osc_position.json`, `joint_position.json`,
  `joint_velocity.json`, `joint_torque.json`, and `ik_pose.json`.
- `load_composite_controller_config()` resolves a robot default, named
  composite template, or explicit JSON path, validates it, and normalizes
  nested arm entries
  (`robosuite/controllers/composite/composite_controller_factory.py::load_composite_controller_config`).
- `load_part_controller_config()` resolves an explicit or default part JSON
  (`robosuite/controllers/parts/controller_factory.py::load_part_controller_config`).
- `Robot.__init__()` uses the supplied `controller_configs` dictionary or
  loads the selected robot's default composite config
  (`robosuite/robots/robot.py::Robot.__init__`).

Passing a loaded configuration through the environment's
`controller_configs` argument is the normal customization point; modifying
controller factory internals is only appropriate when adding a genuinely new
controller type
(`robosuite/environments/robot_env.py::RobotEnv.__init__`,
`robosuite/controllers/parts/controller_factory.py::controller_factory`,
`robosuite/controllers/composite/composite_controller.py::register_composite_controller`).

## Safe extension boundary

Normally extend:

- New concrete environment subclasses and their task hooks under
  `robosuite/environments/manipulation/`
  (`ManipulationEnv`, `TwoArmEnv`).
- New arena/object model subclasses and new MJCF assets through the public
  model composition APIs (`Arena`, `MujocoObject`, `Task`).
- New observations by returning additional `Observable` entries from a
  subclass `_setup_observables()`
  (`MujocoEnv._setup_observables`, `Observable`).
- New part controllers through `controller_factory` and new composite
  controllers by subclassing and registering
  `CompositeController`
  (`robosuite/controllers/parts/controller_factory.py::controller_factory`,
  `robosuite/controllers/composite/composite_controller.py::CompositeController`,
  `register_composite_controller`).
- New robot model classes through `RobotModel` / `ManipulatorModel`, with
  runtime wrapper registration through
  `robosuite/robots/__init__.py::register_robot_class` when needed.
- Custom controller settings via a project-owned JSON path passed to
  `load_composite_controller_config()`.

Normally do not modify:

- `MujocoEnv.step`, reset lifecycle, or `MjSim` binding code:
  `robosuite/environments/base.py` and
  `robosuite/utils/binding_utils.py`. These define the framework-wide timing,
  simulation, observation, and termination contracts.
- Global registries by hand:
  `REGISTERED_ENVS`, `REGISTERED_ROBOTS`, and
  `REGISTERED_COMPOSITE_CONTROLLERS_DICT`. Use their metaclass, decorator, or
  factory extension mechanisms
  (`EnvMeta`, `RobotModelMeta`, `register_composite_controller`).
- `RobotEnv._pre_action`,
  `FixedBaseRobot.control`, or
  `CompositeController` action splitting merely to change a task. A task
  should consume the existing action contract; controller changes belong in a
  controller subclass/config
  (`RobotEnv._pre_action`, `FixedBaseRobot.control`,
  `CompositeController.setup_action_split_idx`).
- Existing upstream robot/arena/object MJCF assets merely to specialize one
  experiment. Add a new model/asset or parameterize/subclass the model so
  other built-in environments retain their behavior
  (`robosuite/models/assets/`,
  `robosuite/models/base.py::MujocoXML.merge`).
- Built-in default controller JSON merely for one task. Supply a copied,
  project-owned config through `controller_configs`
  (`load_composite_controller_config`,
  `RobotEnv.__init__`).
- `robosuite/__init__.py` for an out-of-tree extension. Import the extension
  module in the application before calling `make`; reserve the upstream import
  list for built-in environments
  (`robosuite/__init__.py`,
  `robosuite/environments/base.py::EnvMeta`).

