import numpy as np

import robosuite as suite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.environments.manipulation.shape_peg_insertion import (
    SHAPES,
    ShapePegInsertion,
)
from robosuite.scripts.demo_shape_peg_insertion import create_env as create_demo_env


def make_env(**kwargs):
    return suite.make(
        env_name="ShapePegInsertion",
        robots="Panda",
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        **kwargs,
    )


def test_shape_peg_insertion_is_registered_and_resets():
    assert "ShapePegInsertion" in REGISTERED_ENVS
    env = make_env(use_object_obs=True)
    try:
        observations = env.reset()
        assert set(env.pieces) == set(SHAPES)
        assert "rectangle" not in env.pieces
        for peg_index in range(4):
            assert (
                env.sim.model.geom_name2id(
                    f"shape_sorter_board_rectangle_peg_{peg_index}"
                )
                >= 0
            )
        assert env.sim.model.camera_name2id("taskview") >= 0
        assert env.sim.model.light_name2id("shape_sorter_fill") >= 0
        assert not env._check_success()
        for shape in SHAPES:
            assert f"{shape}_piece_pos" in observations
            assert f"{shape}_piece_quat" in observations
            assert f"{shape}_target_pos" in observations
            assert observations[f"{shape}_piece_linear_vel"].shape == (3,)
            assert observations[f"{shape}_piece_angular_vel"].shape == (3,)
            assert observations[f"{shape}_eef_to_piece_pos"].shape == (3,)
            assert observations[f"{shape}_piece_to_target_pos"].shape == (3,)
            assert observations[f"{shape}_hole_to_peg_xy"].shape == (
                2 * len(env.hole_site_ids[shape]),
            )
            assert f"{shape}_maximum_hole_error" in observations
            assert f"{shape}_height_error" in observations
            assert f"{shape}_tilt_error" in observations
            assert f"{shape}_yaw_error" in observations
            assert f"{shape}_linear_speed" in observations
            assert f"{shape}_angular_speed" in observations
            assert f"{shape}_grasped" in observations
            assert f"{shape}_lifted" in observations
            assert f"{shape}_stability_progress" in observations
            assert f"{shape}_seated" in observations
            assert not env._piece_is_stably_seated(shape)
        assert observations["object-state"].shape == (108,)
    finally:
        env.close()


def test_shape_peg_insertion_zero_action_step_is_finite():
    env = make_env(use_object_obs=False)
    try:
        env.reset()
        for _ in range(5):
            _, reward, _, info = env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert np.all(np.isfinite(env.sim.data.qvel))
        assert reward == 0.0
        assert not info["success"]
    finally:
        env.close()


def test_scripted_policy_observation_vector_conventions():
    env = make_env(use_object_obs=True)
    try:
        observations = env.reset()
        for shape in SHAPES:
            piece_pos = observations[f"{shape}_piece_pos"]
            target_pos = observations[f"{shape}_target_pos"]
            eef_pos = env.sim.data.site_xpos[env.eef_site_id]
            np.testing.assert_allclose(
                observations[f"{shape}_eef_to_piece_pos"],
                piece_pos - eef_pos,
            )
            np.testing.assert_allclose(
                observations[f"{shape}_piece_to_target_pos"],
                target_pos - piece_pos,
            )

            expected_residuals = env._hole_to_peg_xy_residuals(shape).reshape(-1)
            np.testing.assert_allclose(
                observations[f"{shape}_hole_to_peg_xy"],
                expected_residuals,
            )
            assert observations[f"{shape}_grasped"] == 0.0
            assert observations[f"{shape}_lifted"] == 0.0
            assert observations[f"{shape}_stability_progress"] == 0.0
    finally:
        env.close()


def test_shape_peg_insertion_reset_placements_are_separated():
    env = make_env(use_object_obs=False)
    try:
        env.reset()
        positions = np.array(
            [
                env.sim.data.body_xpos[env.piece_body_ids[shape]].copy()
                for shape in SHAPES
            ]
        )
        pairwise = np.linalg.norm(
            positions[:, None, :2] - positions[None, :, :2],
            axis=-1,
        )
        assert np.min(pairwise[np.nonzero(pairwise)]) > 0.08
        assert np.all(positions[:, 1] < -0.12)
        assert np.all(positions[:, 2] > env.table_offset[2])
    finally:
        env.close()


def test_shape_peg_insertion_exact_seated_state_and_reward():
    env = make_env(
        use_object_obs=False,
        success_stability_steps=3,
        reward_shaping=False,
    )
    try:
        env.reset()
        for shape, piece in env.pieces.items():
            target = env.sim.data.site_xpos[env.target_site_ids[shape]].copy()
            env.sim.data.set_joint_qpos(
                piece.joints[0],
                np.concatenate([target, [1.0, 0.0, 0.0, 0.0]]),
            )
            address = env.piece_joint_qvel_addresses[shape]
            env.sim.data.qvel[slice(*address)] = 0.0
        env.sim.forward()

        for _ in range(env.success_stability_steps):
            env._update_piece_stability()

        assert env._check_success()
        assert env.reward() == 1.0
        assert all(
            env._piece_metrics(shape)["center_to_seated_z_error"] <= 0.002
            for shape in SHAPES
        )
    finally:
        env.close()


def test_triangle_seated_threshold_uses_available_peg_clearance():
    metrics = {
        "maximum_hole_to_peg_xy_error": 0.0022,
        "center_to_seated_z_error": 0.0,
        "tilt_rad": 0.0,
        "yaw_error_rad": 0.0,
        "linear_speed": 0.0,
        "angular_speed": 0.0,
    }

    assert ShapePegInsertion._piece_pose_is_seated(metrics, "triangle")
    assert not ShapePegInsertion._piece_pose_is_seated(metrics, "square")


def test_shape_peg_insertion_nero7_mount_and_osc_hold():
    env = create_demo_env(robot="Nero7", render=False)
    try:
        env.reset()
        robot_model = env.robots[0].robot_model
        assert type(robot_model.base).__name__ == "NeroTableMount"

        root_id = env.sim.model.body_name2id(robot_model.root_body)
        collar_id = env.sim.model.body_name2id(
            robot_model.base.naming_prefix + "mount_collar"
        )
        np.testing.assert_allclose(
            env.sim.data.body_xpos[root_id],
            [-0.36, 0.0, 0.85],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            env.sim.data.body_xpos[collar_id],
            [-0.36, 0.0, 0.8375],
            atol=1e-8,
        )
        assert not env.check_contact(
            robot_model.base.contact_geoms,
            "table_collision",
        )

        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert np.all(np.isfinite(env.sim.data.qvel))
    finally:
        env.close()
