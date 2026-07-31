import numpy as np

import robosuite as suite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.environments.manipulation.colorful_cube_stack import (
    DEFAULT_CUBE_COLORS,
)
from robosuite.scripts.demo_colorful_cube_stack import (
    arrange_completed_pyramid,
)


def make_env(**kwargs):
    return suite.make(
        env_name="ColorfulCubeStack",
        robots="Panda",
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        **kwargs,
    )


def test_colorful_cube_stack_registers_and_resets():
    assert "ColorfulCubeStack" in REGISTERED_ENVS
    env = make_env(use_object_obs=True)
    try:
        observations = env.reset()
        assert tuple(env.cubes) == tuple(name for name, _ in DEFAULT_CUBE_COLORS)
        assert len(env.cubes) == 10
        assert env.sim.model.camera_name2id("stackview") >= 0
        assert not env._check_success()
        for name in env.cube_names:
            assert observations[f"{name}_cube_pos"].shape == (3,)
            assert observations[f"{name}_cube_quat"].shape == (4,)
            assert observations[f"{name}_target_pos"].shape == (3,)
            assert observations[f"{name}_cube_to_target"].shape == (3,)
            assert observations[f"{name}_grasped"].shape == ()
            assert observations[f"{name}_placed"].shape == ()
            assert observations[f"gripper_to_{name}_cube"].shape == (3,)
    finally:
        env.close()


def test_colorful_cube_stack_target_layout_is_4_3_2_1():
    env = make_env(use_object_obs=False)
    try:
        env.reset()
        targets = np.array(list(env.target_positions.values()))
        levels, counts = np.unique(np.round(targets[:, 2], 6), return_counts=True)
        assert counts.tolist() == [4, 3, 2, 1]
        np.testing.assert_allclose(np.diff(levels), 2 * env.cube_half_size)
        for level in levels:
            row = targets[np.isclose(targets[:, 2], level)]
            assert np.allclose(row[:, 1], env.pyramid_center[1])
            assert np.isclose(np.mean(row[:, 0]), env.pyramid_center[0])
    finally:
        env.close()


def test_colorful_cube_stack_completed_layout_reaches_success():
    env = make_env(
        use_object_obs=False,
        success_stability_steps=3,
        reward_shaping=False,
    )
    try:
        env.reset()
        arrange_completed_pyramid(env)
        for _ in range(env.success_stability_steps):
            env._update_stability()
        assert env._check_success()
        assert env.reward() == 1.0
        assert all(env._cube_is_placed(name) for name in env.cube_names)
    finally:
        env.close()


def test_colorful_cube_stack_zero_action_is_finite():
    env = make_env(use_object_obs=False)
    try:
        env.reset()
        for _ in range(5):
            _, reward, _, info = env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert np.all(np.isfinite(env.sim.data.qvel))
        assert 0.0 <= reward <= 1.0
        assert 0.0 <= info["stack_progress"] <= 1.0
    finally:
        env.close()
