import pathlib

import numpy as np
import pytest

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.bases import BASE_MAPPING, NeroTableMount
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot
from robosuite.scripts.nero7_waypoint_policy import Nero7JointWaypointPolicy
from robosuite.scripts.nero7_workspace_probe import sample_workspace, validate_range
from robosuite.scripts.tune_nero7_init_pose import (
    DEFAULT_CANDIDATE,
    pose_metrics,
    set_arm_state,
    validate_dynamics,
)
from robosuite.scripts.validate_nero7_osc import (
    load_osc_config,
    measure_axis_response,
    measure_zero_hold,
)
from robosuite.scripts.demo_nero7_osc_motion import create_env as create_osc_demo_env
from robosuite.scripts.demo_nero7_osc_motion import run_motion, run_pick_and_lift


NERO7_JOINT_LIMITS = np.array(
    [
        [-2.70526, 2.70526],
        [-1.74, 1.74],
        [-2.75, 2.75],
        [-1.01, 2.14],
        [-2.75, 2.75],
        [-0.73, 0.95],
        [-1.5707963, 1.5707963],
    ]
)

NERO7_INIT_QPOS = np.array(
    [
        0.3590570800436392,
        0.1850950179686991,
        -0.38515021674470995,
        1.758171938003951,
        0.04206276630372575,
        -0.005008290725163654,
        0.8511248905816553,
    ]
)


def make_nero7_env(controller_config=None):
    return suite.make(
        env_name="Lift",
        robots="Nero7",
        controller_configs=controller_config,
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        ignore_done=True,
    )


def test_nero7_is_registered_as_fixed_base_robot():
    assert "Nero7" in REGISTERED_ROBOTS
    assert ROBOT_CLASS_MAPPING["Nero7"] is FixedBaseRobot
    assert BASE_MAPPING["NeroTableMount"] is NeroTableMount


def test_nero7_model_joint_and_actuator_contract():
    robot_model = REGISTERED_ROBOTS["Nero7"]()

    assert robot_model.default_base == "NeroTableMount"
    assert robot_model.dof == 7
    assert [name.removeprefix("robot0_") for name in robot_model.joints] == [
        f"joint{i}" for i in range(1, 8)
    ]
    assert [name.removeprefix("robot0_") for name in robot_model.actuators] == [
        f"torq_j{i}" for i in range(1, 8)
    ]
    assert robot_model.eef_name == {"right": "robot0_right_hand"}
    assert robot_model.cameras == ["robot0_eye_in_hand"]
    assert len(robot_model.contact_geoms) == 8
    np.testing.assert_allclose(robot_model.init_qpos, NERO7_INIT_QPOS)


@pytest.mark.parametrize("controller_kind", ["absolute", "delta"])
def test_nero7_controller_and_reset_contract(controller_kind):
    controller_config = None
    if controller_kind == "delta":
        config_path = (
            pathlib.Path(suite.__file__).parent
            / "controllers"
            / "config"
            / "robots"
            / "nero7_joint_delta.json"
        )
        controller_config = load_composite_controller_config(controller=str(config_path))

    env = make_nero7_env(controller_config)
    try:
        observations = env.reset()
        robot = env.robots[0]

        assert env.action_dim == 8
        assert len(robot.robot_joints) == 7
        assert len(robot.robot_model.arm_actuators) == 7
        assert np.all(np.isfinite(robot._joint_positions))
        np.testing.assert_allclose(robot._joint_positions, NERO7_INIT_QPOS)
        assert np.all(robot._joint_positions >= NERO7_JOINT_LIMITS[:, 0])
        assert np.all(robot._joint_positions <= NERO7_JOINT_LIMITS[:, 1])

        gripper_qpos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
        gripper_ranges = env.sim.model.jnt_range[robot._ref_joints_indexes_dict["right_gripper"]]
        assert np.all(gripper_qpos >= gripper_ranges[:, 0])
        assert np.all(gripper_qpos <= gripper_ranges[:, 1])

        assert observations["robot0_joint_pos"].shape == (7,)
        assert observations["robot0_gripper_qpos"].shape == (2,)

        low, high = env.action_spec
        if controller_kind == "absolute":
            np.testing.assert_allclose(low[:7], NERO7_JOINT_LIMITS[:, 0])
            np.testing.assert_allclose(high[:7], NERO7_JOINT_LIMITS[:, 1])
        else:
            np.testing.assert_allclose(low, -np.ones(8))
            np.testing.assert_allclose(high, np.ones(8))
    finally:
        env.close()


def test_nero7_delta_zero_action_holds_reset_pose():
    config_path = (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_joint_delta.json"
    )
    controller_config = load_composite_controller_config(controller=str(config_path))
    env = make_nero7_env(controller_config)
    try:
        env.reset()
        initial_qpos = env.robots[0]._joint_positions.copy()

        for _ in range(100):
            env.step(np.zeros(env.action_dim))

        np.testing.assert_allclose(env.robots[0]._joint_positions, initial_qpos, atol=1e-3)
        assert np.all(np.isfinite(env.sim.data.qpos))
        assert np.all(np.isfinite(env.sim.data.qvel))
    finally:
        env.close()


def test_piper_gripper_matches_supplied_urdf_transforms():
    env = make_nero7_env()
    try:
        env.reset()
        gripper = env.robots[0].gripper["right"]
        gripper_root = gripper.worldbody.find("./body")
        gripper_base = gripper_root.find("./body[@name='gripper0_right_gripper_base']")
        finger1 = gripper_base.find("./body[@name='gripper0_right_gripper_link1']")
        finger2 = gripper_base.find("./body[@name='gripper0_right_gripper_link2']")

        np.testing.assert_allclose(
            np.fromstring(gripper_root.get("pos"), sep=" "),
            [0.032, 0.0, -0.0235],
        )
        assert gripper_root.get("euler") == "1.5707963 1.5707963 0"
        flange_inertial = gripper_root.find("./inertial")
        assert float(flange_inertial.get("mass")) == pytest.approx(0.04771096)
        np.testing.assert_allclose(
            np.fromstring(flange_inertial.get("pos"), sep=" "),
            [0.000253201, -0.000322696, -0.010082901],
        )
        np.testing.assert_allclose(
            np.fromstring(flange_inertial.get("fullinertia"), sep=" "),
            [3.614107e-05, 1.7824164e-05, 2.695944e-05, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            np.fromstring(gripper_base.get("pos"), sep=" "),
            [0.0, 0.0, 0.0055],
        )
        np.testing.assert_allclose(
            np.fromstring(finger1.get("pos"), sep=" "),
            [0.0, 0.005, 0.1358],
        )
        np.testing.assert_allclose(
            np.fromstring(finger2.get("pos"), sep=" "),
            [0.0, -0.005, 0.1358],
        )
        assert finger1.get("euler") == "-1.5707963 0 3.1415926"
        assert finger2.get("euler") == "1.5707963 0 0"
    finally:
        env.close()


def test_nero7_absolute_waypoint_policy_reaches_target():
    env = make_nero7_env()
    try:
        env.reset()
        policy = Nero7JointWaypointPolicy(
            env,
            max_joint_step=0.02,
            position_tolerance=0.01,
        )
        target = NERO7_INIT_QPOS + np.array([0.0, 0.05, 0.0, -0.05, 0.0, 0.05, 0.0])
        result = policy.move_to(target, gripper_command=-1.0, max_steps=250)

        assert result.reached
        assert result.steps <= 250
        np.testing.assert_allclose(policy.current_qpos, target, atol=0.01)
    finally:
        env.close()


def test_nero7_workspace_probe_records_collision_and_orientation_filters():
    env = make_nero7_env()
    try:
        env.reset()
        (
            _,
            eef_positions,
            conditions,
            alignments,
            collision_free,
            accepted,
            _,
        ) = sample_workspace(
            env=env,
            samples=100,
            seed=3,
            joint_margin=0.1,
            minimum_z=0.9,
            maximum_condition=100.0,
            maximum_tilt_deg=45.0,
            x_range=(-0.5, 0.5),
            y_range=(-0.5, 0.5),
            z_range=(0.9, 1.5),
            reject_collisions=True,
        )

        assert eef_positions.shape == (100, 3)
        assert conditions.shape == (100,)
        assert alignments.shape == (100,)
        assert collision_free.shape == (100,)
        assert accepted.shape == (100,)
        assert np.all(collision_free[accepted])
        assert np.all(conditions[accepted] <= 100.0)
        assert np.all(alignments[accepted] >= np.cos(np.deg2rad(45.0)))
    finally:
        env.close()


def test_workspace_probe_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        validate_range("--x-range", (1.0, -1.0))


def test_nero7_candidate_pose_is_collision_free_and_dynamically_stable():
    env = make_nero7_env()
    try:
        env.reset()
        set_arm_state(env, DEFAULT_CANDIDATE)
        metrics = pose_metrics(env)
        validation = validate_dynamics(
            env,
            target_qpos=DEFAULT_CANDIDATE,
            steps=50,
            tracking_tolerance=0.02,
        )

        assert metrics.finite
        assert metrics.collision_pairs == []
        assert metrics.downward_tilt_deg <= 30.0
        assert metrics.jacobian_condition <= 100.0
        assert validation.passed
    finally:
        env.close()


def test_nero7_osc_action_contract_and_zero_hold():
    env = make_nero7_env(load_osc_config())
    try:
        env.reset()
        assert env.action_dim == 7
        np.testing.assert_allclose(env.action_spec[0], -np.ones(7))
        np.testing.assert_allclose(env.action_spec[1], np.ones(7))
    finally:
        env.close()

    hold = measure_zero_hold(steps=50)
    assert hold["finite"]
    assert hold["collision_pairs"] == []
    assert hold["joint_drift"] < 1e-3
    assert hold["position_drift"] < 1e-3
    assert hold["orientation_drift"] < 1e-3
    assert hold["maximum_control_fraction"] <= 1.0


def test_nero7_osc_world_frame_axis_directions():
    for action_index in range(6):
        response = measure_axis_response(
            action_index=action_index,
            pulse=0.2,
            settle_steps=10,
        )
        assert response.finite
        assert response.collision_pairs == []
        assert response.primary_response > 1e-4
        assert response.maximum_control_fraction <= 1.0


def test_nero7_closed_loop_osc_motion_reaches_command():
    env = create_osc_demo_env(render=False)
    try:
        env.reset()
        result = run_motion(
            env,
            position_delta=[0.03, 0.0, 0.02],
            rotation_delta=[0.0, 0.0, 0.05],
            max_steps=200,
            position_tolerance=0.005,
            orientation_tolerance=0.03,
            render=False,
        )

        assert result["reached"]
        assert result["collision_pairs"] == []
        assert result["position_error"] <= 0.005
        assert result["orientation_error"] <= 0.03
    finally:
        env.close()


def test_nero7_scripted_pick_and_lift():
    env = create_osc_demo_env(render=False)
    try:
        env.reset()
        result = run_pick_and_lift(
            env,
            approach_height=0.12,
            lift_height=0.15,
            close_steps=40,
            max_motion_steps=200,
            render=False,
        )

        assert result["success"]
        assert result["failed_phase"] is None
        assert result["phases"]["approach"]["reached"]
        assert result["phases"]["descend"]["reached"]
        assert result["phases"]["grasp"]["grasped"]
        assert result["phases"]["lift"]["reached"]
        assert result["still_grasped"]
        assert result["cube_lift"] >= 0.075
    finally:
        env.close()
