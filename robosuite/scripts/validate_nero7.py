"""Headless validation for the Nero7 simulation and controller contract."""

import argparse
import pathlib

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-mode",
        choices=("absolute", "delta"),
        default="delta",
        help="Use the hardware-compatible absolute config or normalized delta config.",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of zero-action stability steps.")
    parser.add_argument(
        "--hold-tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed joint drift in delta mode.",
    )
    parser.add_argument(
        "--require-arm-collisions",
        action="store_true",
        help="Fail instead of warning when the arm model has no contact geoms.",
    )
    parser.add_argument(
        "--maximum-jacobian-condition",
        type=float,
        default=None,
        help="Optional upper bound for the home-pose geometric Jacobian condition number.",
    )
    return parser.parse_args()


def load_controller(mode):
    if mode == "absolute":
        return load_composite_controller_config(robot="Nero7")

    config_path = (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_joint_delta.json"
    )
    return load_composite_controller_config(controller=str(config_path))


def main():
    args = parse_args()
    controller_config = load_controller(args.controller_mode)
    env = suite.make(
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

    try:
        observations = env.reset()
        robot = env.robots[0]
        initial_qpos = robot._joint_positions.copy()
        gripper_indexes = robot._ref_gripper_joint_pos_indexes["right"]
        gripper_joint_ids = robot._ref_joints_indexes_dict["right_gripper"]
        gripper_qpos = env.sim.data.qpos[gripper_indexes].copy()
        gripper_ranges = env.sim.model.jnt_range[gripper_joint_ids].copy()

        print(f"controller_mode: {args.controller_mode}")
        print(f"action_dim: {env.action_dim}")
        print(f"action_low: {env.action_spec[0]}")
        print(f"action_high: {env.action_spec[1]}")
        print(f"initial_arm_qpos: {initial_qpos}")
        print(f"initial_gripper_qpos: {gripper_qpos}")
        print(f"gripper_joint_ranges: {gripper_ranges}")
        print(f"eef_position: {observations['robot0_eef_pos']}")
        print(f"arm_contact_geom_count: {len(robot.robot_model.contact_geoms)}")

        eef_name = robot.gripper["right"].important_sites["grip_site"]
        qvel_indexes = robot._ref_joint_vel_indexes
        jacobian = np.vstack(
            (
                env.sim.data.get_site_jacp(eef_name)[:, qvel_indexes],
                env.sim.data.get_site_jacr(eef_name)[:, qvel_indexes],
            )
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        jacobian_condition = (
            float("inf")
            if singular_values[-1] <= np.finfo(float).eps
            else float(singular_values[0] / singular_values[-1])
        )
        print(f"home_jacobian_singular_values: {singular_values}")
        print(f"home_jacobian_condition: {jacobian_condition}")

        if not np.all(gripper_qpos >= gripper_ranges[:, 0]) or not np.all(
            gripper_qpos <= gripper_ranges[:, 1]
        ):
            raise RuntimeError("Piper gripper reset position lies outside its joint ranges")

        max_joint_drift = 0.0
        for _ in range(args.steps):
            env.step(np.zeros(env.action_dim))
            max_joint_drift = max(
                max_joint_drift,
                float(np.max(np.abs(robot._joint_positions - initial_qpos))),
            )
            if not np.all(np.isfinite(env.sim.data.qpos)) or not np.all(np.isfinite(env.sim.data.qvel)):
                raise RuntimeError("Nero7 simulation produced a non-finite state")

        print(f"max_joint_drift: {max_joint_drift}")
        if args.controller_mode == "delta" and max_joint_drift > args.hold_tolerance:
            raise RuntimeError(
                f"Delta zero-action drift {max_joint_drift} exceeds tolerance {args.hold_tolerance}"
            )

        if not robot.robot_model.contact_geoms:
            message = (
                "Nero7 has no arm contact geoms; add validated collision primitives "
                "before collecting contact-rich task data"
            )
            if args.require_arm_collisions:
                raise RuntimeError(message)
            print(f"warning: {message}")

        if (
            args.maximum_jacobian_condition is not None
            and jacobian_condition > args.maximum_jacobian_condition
        ):
            raise RuntimeError(
                f"Home Jacobian condition {jacobian_condition} exceeds "
                f"{args.maximum_jacobian_condition}"
            )

        print("Nero7 headless validation passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
