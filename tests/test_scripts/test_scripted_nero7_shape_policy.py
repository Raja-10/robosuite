import numpy as np

from robosuite.scripts.scripted_nero7_shape_peg_insertion import (
    CLOSE_GRIPPER,
    ScriptedShapeInsertionPolicy,
    create_env,
    pose_action,
    top_down_orientation,
)


def test_scripted_policy_action_is_bounded_and_finite():
    env = create_env(render=False)
    try:
        env.reset()
        eef_site = env.robots[0].eef_site_id["right"]
        target = env.sim.data.site_xpos[eef_site] + np.array([1.0, -1.0, 1.0])
        action, _, _ = pose_action(
            env,
            target_position=target,
            target_orientation=top_down_orientation(),
            gripper_command=CLOSE_GRIPPER,
        )
        assert action.shape == (7,)
        assert np.all(np.isfinite(action))
        assert np.max(np.abs(action[:3])) <= 0.70
        assert np.max(np.abs(action[3:6])) <= 0.40
        assert action[6] == CLOSE_GRIPPER
    finally:
        env.close()


def test_scripted_policy_short_hold_preserves_finite_simulation():
    env = create_env(render=False)
    try:
        observations = env.reset()
        policy = ScriptedShapeInsertionPolicy(
            env,
            render=False,
            realtime=False,
            max_motion_steps=2,
        )
        policy.last_observation = observations
        position = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()
        orientation = env.sim.data.site_xmat[
            env.robots[0].eef_site_id["right"]
        ].reshape(3, 3)
        policy.hold(position, orientation, CLOSE_GRIPPER, steps=2)
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert policy.last_observation["object-state"].shape == (108,)
    finally:
        env.close()


def test_top_down_orientation_points_tool_down():
    orientation = top_down_orientation()
    np.testing.assert_allclose(orientation[:, 2], [0.0, 0.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(orientation.T @ orientation, np.eye(3), atol=1e-7)
