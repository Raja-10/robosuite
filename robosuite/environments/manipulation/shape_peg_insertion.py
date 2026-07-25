"""Four-piece shape sorting and multi-peg insertion environment."""

from collections import OrderedDict
from itertools import permutations

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    CirclePiece,
    RectanglePiece,
    ShapeSorterBoard,
    SquarePiece,
    TrianglePiece,
)
from robosuite.models.objects.shape_sorter import (
    BOARD_HALF_SIZE,
    PIECE_HALF_THICKNESS,
    SHAPE_SPECS,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_element
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.transform_utils import convert_quat


SHAPES = ("circle", "square", "triangle", "rectangle")
PIECE_CLASSES = (CirclePiece, SquarePiece, TrianglePiece, RectanglePiece)
DEFAULT_SPAWN_X = (-0.20, -0.07, 0.07, 0.17)


class ShapePegInsertion(ManipulationEnv):
    """Insert four pieces over their matching one-to-four-peg stations."""

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        board_offset=(0.0, 0.10),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        success_stability_steps=5,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=2000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.table_full_size = np.asarray(table_full_size, dtype=float)
        self.table_friction = table_friction
        self.table_offset = np.array([0.0, 0.0, 0.8])
        self.board_offset = np.asarray(board_offset, dtype=float)
        if self.board_offset.shape != (2,):
            raise ValueError("board_offset must contain x and y")

        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer
        self.success_stability_steps = int(success_stability_steps)
        if self.success_stability_steps <= 0:
            raise ValueError("success_stability_steps must be positive")

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _load_model(self):
        super()._load_model()

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
            pos=[0.62, -0.62, 1.48],
            quat=[0.579, 0.393, 0.403, 0.590],
        )
        arena.set_camera(
            camera_name="taskview",
            pos=[0.15, -0.72, 1.55],
            quat=[0.653, 0.271, 0.271, 0.653],
        )
        arena.worldbody.append(
            new_element(
                tag="light",
                name="shape_sorter_fill",
                pos=(-0.8, -0.6, 1.8),
                dir=(0.35, 0.25, -1.0),
                directional="true",
                castshadow="false",
                diffuse=(0.55, 0.55, 0.55),
                specular=(0.1, 0.1, 0.1),
            )
        )

        self.board = ShapeSorterBoard(stations=SHAPES)
        board_body = self.board.get_obj()
        board_body.set(
            "pos",
            array_to_string(
                [
                    self.board_offset[0],
                    self.board_offset[1],
                    self.table_offset[2],
                ]
            ),
        )

        self.pieces = OrderedDict(
            (shape, piece_cls(name=f"{shape}_piece"))
            for shape, piece_cls in zip(SHAPES, PIECE_CLASSES)
        )
        self._configure_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.board, *self.pieces.values()],
        )

    def _configure_placement_initializer(self):
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(
                name="ShapePieceSampler"
            )
            for shape, x_center in zip(SHAPES, DEFAULT_SPAWN_X):
                if shape == "circle":
                    y_range = (-0.215, -0.19)
                elif shape == "rectangle":
                    y_range = (-0.11, -0.10)
                else:
                    y_range = (-0.23, -0.19)
                self.placement_initializer.append_sampler(
                    UniformRandomSampler(
                        name=f"{shape}Sampler",
                        x_range=(x_center - 0.01, x_center + 0.01),
                        y_range=y_range,
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.002,
                    )
                )
        else:
            self.placement_initializer.reset()

        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            for shape, piece in self.pieces.items():
                self.placement_initializer.add_objects_to_sampler(
                    sampler_name=f"{shape}Sampler",
                    mujoco_objects=piece,
                )
        else:
            self.placement_initializer.add_objects(list(self.pieces.values()))

    def _setup_references(self):
        super()._setup_references()
        self.board_body_id = self.sim.model.body_name2id(self.board.root_body)
        self.piece_body_ids = {
            shape: self.sim.model.body_name2id(piece.root_body)
            for shape, piece in self.pieces.items()
        }
        self.piece_joint_qvel_addresses = {
            shape: self.sim.model.get_joint_qvel_addr(piece.joints[0])
            for shape, piece in self.pieces.items()
        }
        self.hole_site_ids = {
            shape: [
                self.sim.model.site_name2id(piece.important_sites[f"hole_{index}"])
                for index in range(len(SHAPE_SPECS[shape]["hole_positions"]))
            ]
            for shape, piece in self.pieces.items()
        }
        self.peg_site_ids = {
            shape: [
                self.sim.model.site_name2id(
                    self.board.important_sites[f"{shape}_peg_{index}"]
                )
                for index in range(len(SHAPE_SPECS[shape]["hole_positions"]))
            ]
            for shape in SHAPES
        }
        self.target_site_ids = {
            shape: self.sim.model.site_name2id(
                self.board.important_sites[f"{shape}_seated"]
            )
            for shape in SHAPES
        }
        self.bottom_site_ids = {
            shape: self.sim.model.site_name2id(piece.important_sites["bottom"])
            for shape, piece in self.pieces.items()
        }
        self.eef_site_id = next(iter(self.robots[0].eef_site_id.values()))
        self._stable_steps = {shape: 0 for shape in SHAPES}

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"
        for shape in SHAPES:
            sensors = self._create_shape_sensors(shape, modality)
            for shape_sensor in sensors:
                observables[shape_sensor.__name__] = Observable(
                    name=shape_sensor.__name__,
                    sensor=shape_sensor,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _create_shape_sensors(self, shape, modality):
        @sensor(modality=modality)
        def piece_pos(obs_cache):
            return self.sim.data.body_xpos[self.piece_body_ids[shape]].copy()

        @sensor(modality=modality)
        def piece_quat(obs_cache):
            return convert_quat(
                self.sim.data.body_xquat[self.piece_body_ids[shape]].copy(),
                to="xyzw",
            )

        @sensor(modality=modality)
        def target_pos(obs_cache):
            return self.sim.data.site_xpos[self.target_site_ids[shape]].copy()

        @sensor(modality=modality)
        def piece_linear_vel(obs_cache):
            return self._piece_qvel(shape)[:3]

        @sensor(modality=modality)
        def piece_angular_vel(obs_cache):
            return self._piece_qvel(shape)[3:6]

        @sensor(modality=modality)
        def eef_to_piece_pos(obs_cache):
            return (
                self.sim.data.body_xpos[self.piece_body_ids[shape]]
                - self.sim.data.site_xpos[self.eef_site_id]
            )

        @sensor(modality=modality)
        def piece_to_target_pos(obs_cache):
            return (
                self.sim.data.site_xpos[self.target_site_ids[shape]]
                - self.sim.data.body_xpos[self.piece_body_ids[shape]]
            )

        @sensor(modality=modality)
        def hole_to_peg_xy(obs_cache):
            return self._hole_to_peg_xy_residuals(shape).reshape(-1)

        @sensor(modality=modality)
        def maximum_hole_error(obs_cache):
            return np.array(
                [self._piece_metrics(shape)["maximum_hole_to_peg_xy_error"]]
            )

        @sensor(modality=modality)
        def height_error(obs_cache):
            return np.array(
                [self._piece_metrics(shape)["center_to_seated_z_error"]]
            )

        @sensor(modality=modality)
        def tilt_error(obs_cache):
            return np.array([self._piece_metrics(shape)["tilt_rad"]])

        @sensor(modality=modality)
        def yaw_error(obs_cache):
            return np.array([self._piece_metrics(shape)["yaw_error_rad"]])

        @sensor(modality=modality)
        def linear_speed(obs_cache):
            return np.array([self._piece_metrics(shape)["linear_speed"]])

        @sensor(modality=modality)
        def angular_speed(obs_cache):
            return np.array([self._piece_metrics(shape)["angular_speed"]])

        @sensor(modality=modality)
        def grasped(obs_cache):
            return np.array(
                [
                    float(
                        self._check_grasp(
                            gripper=self.robots[0].gripper,
                            object_geoms=self.pieces[shape],
                        )
                    )
                ]
            )

        @sensor(modality=modality)
        def lifted(obs_cache):
            piece_z = self.sim.data.body_xpos[self.piece_body_ids[shape]][2]
            threshold = self.table_offset[2] + PIECE_HALF_THICKNESS + 0.03
            return np.array([float(piece_z > threshold)])

        @sensor(modality=modality)
        def stability_progress(obs_cache):
            return np.array(
                [
                    min(
                        self._stable_steps[shape] / self.success_stability_steps,
                        1.0,
                    )
                ]
            )

        @sensor(modality=modality)
        def seated(obs_cache):
            return np.array([float(self._piece_is_stably_seated(shape))])

        sensors = [
            piece_pos,
            piece_quat,
            target_pos,
            piece_linear_vel,
            piece_angular_vel,
            eef_to_piece_pos,
            piece_to_target_pos,
            hole_to_peg_xy,
            maximum_hole_error,
            height_error,
            tilt_error,
            yaw_error,
            linear_speed,
            angular_speed,
            grasped,
            lifted,
            stability_progress,
            seated,
        ]
        for shape_sensor in sensors:
            shape_sensor.__name__ = f"{shape}_{shape_sensor.__name__}"
        return sensors

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for piece_pos, piece_quat, piece in placements.values():
                self.sim.data.set_joint_qpos(
                    piece.joints[0],
                    np.concatenate([piece_pos, piece_quat]),
                )
        self._stable_steps = {shape: 0 for shape in SHAPES}

    def _piece_qvel(self, shape):
        address = self.piece_joint_qvel_addresses[shape]
        if isinstance(address, tuple):
            return self.sim.data.qvel[slice(*address)].copy()
        return np.atleast_1d(self.sim.data.qvel[address]).copy()

    def _hole_to_peg_xy_residuals(self, shape):
        holes = np.array(
            [self.sim.data.site_xpos[site_id][:2] for site_id in self.hole_site_ids[shape]]
        )
        pegs = np.array(
            [self.sim.data.site_xpos[site_id][:2] for site_id in self.peg_site_ids[shape]]
        )
        # Pegs are physically indistinguishable. Match the two point sets
        # instead of requiring a fixed site-index pairing, which incorrectly
        # rejects the pi-rotated square and rectangle poses accepted by yaw.
        best_residuals = None
        best_cost = np.inf
        for ordering in permutations(range(len(pegs))):
            residuals = pegs[list(ordering)] - holes
            cost = float(np.sum(residuals**2))
            if cost < best_cost:
                best_cost = cost
                best_residuals = residuals
        return best_residuals

    def _piece_metrics(self, shape):
        hole_residuals = self._hole_to_peg_xy_residuals(shape)
        hole_errors = np.linalg.norm(hole_residuals, axis=1)
        body_id = self.piece_body_ids[shape]
        rotation = self.sim.data.body_xmat[body_id].reshape(3, 3)
        tilt = np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
        yaw_error = abs(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if shape == "circle":
            yaw_error = 0.0
        elif shape in {"square", "rectangle"}:
            yaw_error = min(yaw_error, abs(np.pi - yaw_error))
        qvel = self._piece_qvel(shape)
        target_z = self.sim.data.site_xpos[self.target_site_ids[shape]][2]
        return {
            "maximum_hole_to_peg_xy_error": float(max(hole_errors)),
            "center_to_seated_z_error": float(
                abs(self.sim.data.body_xpos[body_id][2] - target_z)
            ),
            "tilt_rad": float(tilt),
            "yaw_error_rad": float(yaw_error),
            "linear_speed": float(np.linalg.norm(qvel[:3])),
            "angular_speed": float(np.linalg.norm(qvel[3:6])),
        }

    @staticmethod
    def _piece_pose_is_seated(metrics, shape=None):
        # The triangle's smaller pegs provide 3.5 mm radial hole clearance.
        # Accept a conservative 3.2 mm residual once it is flat and stationary;
        # the other stations retain the tighter generic threshold.
        xy_tolerance = 0.0032 if shape == "triangle" else 0.002
        return bool(
            metrics["maximum_hole_to_peg_xy_error"] <= xy_tolerance
            and metrics["center_to_seated_z_error"] <= 0.002
            and metrics["tilt_rad"] <= 0.05
            and metrics["yaw_error_rad"] <= 0.05
            and metrics["linear_speed"] <= 0.02
            and metrics["angular_speed"] <= 0.10
        )

    def _piece_is_stably_seated(self, shape):
        return self._stable_steps[shape] >= self.success_stability_steps

    def _update_piece_stability(self):
        for shape in SHAPES:
            if self._piece_pose_is_seated(self._piece_metrics(shape), shape):
                self._stable_steps[shape] += 1
            else:
                self._stable_steps[shape] = 0

    def _post_action(self, action):
        self._update_piece_stability()
        reward, done, info = super()._post_action(action)
        info.update(
            {
                "pieces_seated": {
                    shape: self._piece_is_stably_seated(shape)
                    for shape in SHAPES
                },
                "success": self._check_success(),
            }
        )
        return reward, done, info

    def reward(self, action=None):
        seated = np.array(
            [self._piece_is_stably_seated(shape) for shape in SHAPES],
            dtype=float,
        )
        if self.reward_shaping:
            progress = []
            for shape in SHAPES:
                metrics = self._piece_metrics(shape)
                alignment = 1.0 - np.tanh(
                    10.0 * metrics["maximum_hole_to_peg_xy_error"]
                )
                height = 1.0 - np.tanh(
                    10.0 * metrics["center_to_seated_z_error"]
                )
                progress.append(
                    1.0
                    if self._piece_is_stably_seated(shape)
                    else 0.25 * alignment * height
                )
            reward = float(np.mean(progress))
        else:
            reward = float(np.mean(seated))
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _check_success(self):
        return all(self._piece_is_stably_seated(shape) for shape in SHAPES)
