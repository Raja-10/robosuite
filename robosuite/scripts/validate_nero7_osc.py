"""Validate Nero7 world-frame OSC pose actions at the configured home pose."""

import argparse
import pathlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.tune_nero7_init_pose import active_robot_collision_pairs


@dataclass
class AxisResponse:
    action_index: int
    commanded_value: float
    response: list
    primary_response: float
    maximum_control_fraction: float
    collision_pairs: list
    finite: bool


def osc_config_path():
    return (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_osc_pose.json"
    )


def load_osc_config():
    return load_composite_controller_config(controller=str(osc_config_path()))


def create_env():
    return suite.make(
        env_name="Lift",
        robots="Nero7",
        controller_configs=load_osc_config(),
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        ignore_done=True,
    )


def maximum_control_fraction(env):
    ranges = env.sim.model.actuator_ctrlrange
    scale = np.maximum(np.abs(ranges[:, 0]), np.abs(ranges[:, 1]))
    nonzero = scale > 0
    fractions = np.zeros_like(scale)
    fractions[nonzero] = np.abs(env.sim.data.ctrl[nonzero]) / scale[nonzero]
    return float(np.max(fractions))


def measure_axis_response(action_index, pulse=0.2, settle_steps=20):
    if action_index not in range(6):
        raise ValueError("OSC action index must be in [0, 5]")
    if not 0 < abs(pulse) <= 1:
        raise ValueError("pulse magnitude must be in (0, 1]")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")

    env = create_env()
    try:
        env.reset()
        if env.action_dim != 7:
            raise RuntimeError(f"Expected six OSC actions plus gripper, got {env.action_dim}")

        robot = env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        initial_position = env.sim.data.site_xpos[eef_site_id].copy()
        initial_rotation = env.sim.data.site_xmat[eef_site_id].reshape(3, 3).copy()

        action = np.zeros(env.action_dim)
        action[action_index] = pulse
        env.step(action)
        max_control = maximum_control_fraction(env)
        collisions = {tuple(pair) for pair in active_robot_collision_pairs(env)}

        for _ in range(settle_steps):
            env.step(np.zeros(env.action_dim))
            max_control = max(max_control, maximum_control_fraction(env))
            collisions.update(tuple(pair) for pair in active_robot_collision_pairs(env))

        final_position = env.sim.data.site_xpos[eef_site_id].copy()
        final_rotation = env.sim.data.site_xmat[eef_site_id].reshape(3, 3).copy()
        if action_index < 3:
            response = final_position - initial_position
            primary_response = response[action_index]
        else:
            response = Rotation.from_matrix(final_rotation @ initial_rotation.T).as_rotvec()
            primary_response = response[action_index - 3]

        finite = bool(
            np.all(np.isfinite(env.sim.data.qpos))
            and np.all(np.isfinite(env.sim.data.qvel))
            and np.all(np.isfinite(env.sim.data.ctrl))
        )
        return AxisResponse(
            action_index=action_index,
            commanded_value=pulse,
            response=response.tolist(),
            primary_response=float(primary_response),
            maximum_control_fraction=max_control,
            collision_pairs=[list(pair) for pair in sorted(collisions)],
            finite=finite,
        )
    finally:
        env.close()


def measure_zero_hold(steps=200):
    if steps <= 0:
        raise ValueError("hold steps must be positive")
    env = create_env()
    try:
        env.reset()
        robot = env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        initial_qpos = robot._joint_positions.copy()
        initial_position = env.sim.data.site_xpos[eef_site_id].copy()
        initial_rotation = env.sim.data.site_xmat[eef_site_id].reshape(3, 3).copy()
        max_control = 0.0
        collisions = set()

        for _ in range(steps):
            env.step(np.zeros(env.action_dim))
            max_control = max(max_control, maximum_control_fraction(env))
            collisions.update(tuple(pair) for pair in active_robot_collision_pairs(env))

        rotation_error = Rotation.from_matrix(
            env.sim.data.site_xmat[eef_site_id].reshape(3, 3) @ initial_rotation.T
        ).magnitude()
        return {
            "joint_drift": float(np.max(np.abs(robot._joint_positions - initial_qpos))),
            "position_drift": float(np.linalg.norm(env.sim.data.site_xpos[eef_site_id] - initial_position)),
            "orientation_drift": float(rotation_error),
            "maximum_control_fraction": max_control,
            "collision_pairs": [list(pair) for pair in sorted(collisions)],
            "finite": bool(
                np.all(np.isfinite(env.sim.data.qpos))
                and np.all(np.isfinite(env.sim.data.qvel))
                and np.all(np.isfinite(env.sim.data.ctrl))
            ),
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse", type=float, default=0.2)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--hold-steps", type=int, default=200)
    parser.add_argument("--minimum-translation", type=float, default=1e-4)
    parser.add_argument("--minimum-rotation", type=float, default=1e-4)
    parser.add_argument("--maximum-control-fraction", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    hold = measure_zero_hold(args.hold_steps)
    print(f"zero_hold: {hold}")
    if (
        not hold["finite"]
        or hold["collision_pairs"]
        or hold["maximum_control_fraction"] > args.maximum_control_fraction
    ):
        raise RuntimeError("Nero7 OSC zero-action hold failed")

    for action_index in range(6):
        response = measure_axis_response(
            action_index=action_index,
            pulse=args.pulse,
            settle_steps=args.settle_steps,
        )
        print(f"axis_{action_index}: {asdict(response)}")
        minimum = args.minimum_translation if action_index < 3 else args.minimum_rotation
        if not response.finite:
            raise RuntimeError(f"OSC axis {action_index} produced a non-finite state")
        if response.collision_pairs:
            raise RuntimeError(f"OSC axis {action_index} produced robot contacts")
        if np.sign(response.primary_response) != np.sign(args.pulse):
            raise RuntimeError(f"OSC axis {action_index} moved in the wrong direction")
        if abs(response.primary_response) < minimum:
            raise RuntimeError(f"OSC axis {action_index} response was too small")
        if response.maximum_control_fraction > args.maximum_control_fraction:
            raise RuntimeError(f"OSC axis {action_index} saturated an actuator")

    print("Nero7 OSC validation passed")


if __name__ == "__main__":
    main()
