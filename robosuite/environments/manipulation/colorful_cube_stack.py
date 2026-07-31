"""Multi-cube colorful pyramid task built on robosuite's Stack environment."""

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


DEFAULT_CUBE_COLORS = (
    ("cyan", (0.00, 0.72, 0.75, 1.0)),
    ("pink", (0.95, 0.25, 0.50, 1.0)),
    ("navy", (0.02, 0.25, 0.70, 1.0)),
    ("lime", (0.62, 0.86, 0.05, 1.0)),
    ("yellow", (1.00, 0.72, 0.00, 1.0)),
    ("orange", (1.00, 0.42, 0.00, 1.0)),
    ("red", (0.90, 0.04, 0.05, 1.0)),
    ("blue", (0.00, 0.40, 0.85, 1.0)),
    ("green", (0.00, 0.62, 0.20, 1.0)),
    ("purple", (0.48, 0.16, 0.68, 1.0)),
)


class ColorfulCubeStack(Stack):
    """Stack ten assigned colored cubes into a four-level pyramid."""

    def __init__(
        self,
        robots,
        cube_half_size=0.025,
        pyramid_center=(0.0, 0.10),
        position_tolerance=0.012,
        orientation_tolerance=0.15,
        success_stability_steps=10,
        **kwargs,
    ):
        if cube_half_size <= 0:
            raise ValueError("cube_half_size must be positive")
        if success_stability_steps <= 0:
            raise ValueError("success_stability_steps must be positive")
        self.cube_half_size = float(cube_half_size)
        self.pyramid_center = np.asarray(pyramid_center, dtype=float)
        if self.pyramid_center.shape != (2,):
            raise ValueError("pyramid_center must contain exactly x and y")
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.success_stability_steps = int(success_stability_steps)
        self.cube_specs = DEFAULT_CUBE_COLORS
        self.cube_names = tuple(name for name, _ in self.cube_specs)
        self._stable_steps = 0
        super().__init__(robots=robots, **kwargs)

    def _target_positions(self):
        """Returns assigned world-frame centers for the 4-3-2-1 pyramid."""
        side = 2.0 * self.cube_half_size
        positions = OrderedDict()
        index = 0
        for level, count in enumerate((4, 3, 2, 1)):
            z = self.table_offset[2] + self.cube_half_size + level * side
            for column in range(count):
                x = self.pyramid_center[0] + (
                    column - (count - 1) / 2.0
                ) * side
                positions[self.cube_names[index]] = np.array(
                    [x, self.pyramid_center[1], z], dtype=float
                )
                index += 1
        return positions

    def _load_model(self):
        """Loads Stack's table and robot with ten colored cube objects."""
        # Stack._load_model hard-codes cubeA and cubeB, so call its parent.
        super(Stack, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        arena.set_camera(
            camera_name="agentview",
            pos=[0.70, 0.0, 1.45],
            quat=[0.638, 0.305, 0.307, 0.638],
        )
        arena.set_camera(
            camera_name="stackview",
            pos=[0.72, -0.72, 1.25],
            quat=[0.590, 0.245, 0.325, 0.700],
        )

        self.cubes = OrderedDict()
        size = [self.cube_half_size] * 3
        for name, rgba in self.cube_specs:
            self.cubes[name] = BoxObject(
                name=f"{name}_cube",
                size_min=size,
                size_max=size,
                rgba=rgba,
                density=250.0,
                friction=(1.0, 0.005, 0.0001),
            )

        cube_models = list(self.cubes.values())
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cube_models)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ColorfulCubeSampler",
                mujoco_objects=cube_models,
                x_range=(-0.30, 0.30),
                y_range=(-0.30, 0.02),
                rotation=(-np.pi, np.pi),
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.002,
            )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cube_models,
        )

    def _setup_references(self):
        # Bypass Stack's cubeA / cubeB references.
        super(Stack, self)._setup_references()
        self.cube_body_ids = {
            name: self.sim.model.body_name2id(cube.root_body)
            for name, cube in self.cubes.items()
        }
        self.target_positions = self._target_positions()

    def _reset_internal(self):
        # Bypass Stack's two-cube reset.
        super(Stack, self)._reset_internal()
        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.asarray(obj_pos), np.asarray(obj_quat)]),
                )
        self._stable_steps = 0

    def _cube_metrics(self, name):
        body_id = self.cube_body_ids[name]
        position = self.sim.data.body_xpos[body_id]
        rotation = self.sim.data.body_xmat[body_id].reshape(3, 3)
        qvel_address = self.sim.model.get_joint_qvel_addr(
            self.cubes[name].joints[0]
        )
        qvel = self.sim.data.qvel[slice(*qvel_address)]
        target = self.target_positions[name]
        return {
            "position_error": float(np.linalg.norm(position - target)),
            "xy_error": float(np.linalg.norm(position[:2] - target[:2])),
            "z_error": float(abs(position[2] - target[2])),
            "tilt": float(
                np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            ),
            "linear_speed": float(np.linalg.norm(qvel[:3])),
            "angular_speed": float(np.linalg.norm(qvel[3:])),
        }

    def _cube_is_placed(self, name, require_stable=True):
        metrics = self._cube_metrics(name)
        placed = (
            metrics["xy_error"] <= self.position_tolerance
            and metrics["z_error"] <= self.position_tolerance
            and metrics["tilt"] <= self.orientation_tolerance
        )
        if require_stable:
            placed = (
                placed
                and metrics["linear_speed"] <= 0.02
                and metrics["angular_speed"] <= 0.10
            )
        return bool(placed)

    def _update_stability(self):
        if all(self._cube_is_placed(name) for name in self.cube_names):
            self._stable_steps += 1
        else:
            self._stable_steps = 0

    def _post_action(self, action):
        self._update_stability()
        reward, done, info = super(Stack, self)._post_action(action)
        info["cubes_placed"] = {
            name: self._cube_is_placed(name) for name in self.cube_names
        }
        info["stack_progress"] = sum(info["cubes_placed"].values()) / len(
            self.cube_names
        )
        info["success"] = self._check_success()
        return reward, done, info

    def reward(self, action=None):
        placed = sum(
            self._cube_is_placed(name, require_stable=False)
            for name in self.cube_names
        )
        if self.reward_shaping:
            pose_progress = np.mean(
                [
                    1.0
                    - np.tanh(8.0 * self._cube_metrics(name)["position_error"])
                    for name in self.cube_names
                ]
            )
            reward = 0.5 * placed / len(self.cube_names) + 0.5 * pose_progress
        else:
            reward = float(self._check_success())
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return float(reward)

    def _check_success(self):
        return self._stable_steps >= self.success_stability_steps

    def _setup_observables(self):
        # Bypass Stack's cubeA / cubeB observables.
        observables = super(Stack, self)._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"
        arm_prefixes = self._get_arm_prefixes(
            self.robots[0], include_robot_name=False
        )
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        for name in self.cube_names:
            body_id = self.cube_body_ids[name]
            target = self.target_positions[name].copy()

            @sensor(modality=modality)
            def cube_pos(obs_cache, body_id=body_id):
                return np.array(self.sim.data.body_xpos[body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache, body_id=body_id):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def cube_target(obs_cache, target=target):
                return target

            @sensor(modality=modality)
            def cube_to_target(obs_cache, name=name, target=target):
                return target - np.array(
                    self.sim.data.body_xpos[self.cube_body_ids[name]]
                )

            @sensor(modality=modality)
            def cube_grasped(obs_cache, name=name):
                return np.array(
                    [
                        float(
                            self._check_grasp(
                                self.robots[0].gripper,
                                self.cubes[name],
                            )
                        )
                    ]
                )

            @sensor(modality=modality)
            def cube_placed(obs_cache, name=name):
                return np.array(
                    [float(self._cube_is_placed(name, require_stable=False))]
                )

            named_sensors = (
                (f"{name}_cube_pos", cube_pos),
                (f"{name}_cube_quat", cube_quat),
                (f"{name}_target_pos", cube_target),
                (f"{name}_cube_to_target", cube_to_target),
                (f"{name}_grasped", cube_grasped),
                (f"{name}_placed", cube_placed),
            )
            for observable_name, observable_sensor in named_sensors:
                observables[observable_name] = Observable(
                    name=observable_name,
                    sensor=observable_sensor,
                    sampling_rate=self.control_freq,
                )

            for arm_prefix, full_prefix in zip(arm_prefixes, full_prefixes):
                observable_name = f"{arm_prefix}gripper_to_{name}_cube"
                observable_sensor = self._get_obj_eef_sensor(
                    full_prefix,
                    f"{name}_cube_pos",
                    observable_name,
                    modality,
                )
                observables[observable_name] = Observable(
                    name=observable_name,
                    sensor=observable_sensor,
                    sampling_rate=self.control_freq,
                )

        return observables

    def visualize(self, vis_settings):
        # Bypass Stack's hard-coded cubeA visualization.
        super(Stack, self).visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            candidates = [
                name
                for name in self.cube_names
                if not self._cube_is_placed(name, require_stable=False)
            ]
            if candidates:
                arm = self.robots[0].arms[0]
                eef = self.sim.data.site_xpos[
                    self.robots[0].eef_site_id[arm]
                ]
                nearest = min(
                    candidates,
                    key=lambda name: np.linalg.norm(
                        self.sim.data.body_xpos[self.cube_body_ids[name]] - eef
                    ),
                )
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=self.cubes[nearest],
                )
