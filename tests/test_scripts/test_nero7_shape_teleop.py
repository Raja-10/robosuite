import numpy as np

from robosuite.scripts.teleop_nero7_shape_peg_insertion import (
    create_env,
    device_action_to_env_action,
)


def test_nero7_shape_teleop_action_contract():
    env = create_env(camera="free", render=False)
    try:
        env.reset()
        device_action = {
            "right_delta": np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2]),
            "right_gripper": np.array([1.0]),
        }
        action = device_action_to_env_action(env.robots[0], device_action)

        assert action.shape == (7,)
        np.testing.assert_allclose(action[:6], device_action["right_delta"])
        assert action[6] == 1.0

        observations, _, _, _ = env.step(action)
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert observations["object-state"].shape == (108,)
    finally:
        env.close()


def test_nero7_shape_teleop_reset_signal():
    env = create_env(camera="free", render=False)
    try:
        env.reset()
        assert device_action_to_env_action(env.robots[0], None) is None
    finally:
        env.close()
