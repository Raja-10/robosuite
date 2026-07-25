"""Deterministic joint-space waypoint policy for Nero7 scripted rollouts."""

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


@dataclass(frozen=True)
class WaypointResult:
    reached: bool
    steps: int
    final_error: float


class Nero7JointWaypointPolicy:
    """Produces absolute Nero7 joint targets with deterministic interpolation."""

    def __init__(
        self,
        env,
        max_joint_step: float = 0.02,
        position_tolerance: float = 0.01,
    ):
        if env.robots[0].name != "Nero7":
            raise ValueError("Nero7JointWaypointPolicy requires a Nero7 environment")
        if max_joint_step <= 0:
            raise ValueError("max_joint_step must be positive")
        if position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")

        self.env = env
        self.robot = env.robots[0]
        self.max_joint_step = float(max_joint_step)
        self.position_tolerance = float(position_tolerance)
        self.joint_limits = env.sim.model.jnt_range[self.robot.arm_joint_indexes].copy()

        if env.action_dim != 8:
            raise ValueError(f"Expected Nero7 action dimension 8, got {env.action_dim}")

    @property
    def current_qpos(self):
        return self.robot._joint_positions.copy()

    def validate_target(self, target_qpos: Iterable[float]):
        target = np.asarray(target_qpos, dtype=float)
        if target.shape != (7,):
            raise ValueError(f"Expected seven joint targets, got shape {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("Joint target contains a non-finite value")
        if np.any(target < self.joint_limits[:, 0]) or np.any(target > self.joint_limits[:, 1]):
            raise ValueError(f"Joint target lies outside Nero7 limits: {target}")
        return target

    def action_toward(self, target_qpos: Iterable[float], gripper_command: float = 0.0):
        target = self.validate_target(target_qpos)
        error = target - self.current_qpos
        largest_error = float(np.max(np.abs(error)))
        if largest_error > self.max_joint_step:
            command = self.current_qpos + error * (self.max_joint_step / largest_error)
        else:
            command = target

        action = np.empty(8, dtype=float)
        action[:7] = np.clip(command, self.joint_limits[:, 0], self.joint_limits[:, 1])
        action[7] = np.clip(gripper_command, -1.0, 1.0)
        return action

    def move_to(
        self,
        target_qpos: Iterable[float],
        gripper_command: float = 0.0,
        max_steps: int = 500,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        target = self.validate_target(target_qpos)

        for step in range(1, max_steps + 1):
            error = float(np.max(np.abs(target - self.current_qpos)))
            if error <= self.position_tolerance:
                return WaypointResult(reached=True, steps=step - 1, final_error=error)
            self.env.step(self.action_toward(target, gripper_command))

        final_error = float(np.max(np.abs(target - self.current_qpos)))
        return WaypointResult(
            reached=final_error <= self.position_tolerance,
            steps=max_steps,
            final_error=final_error,
        )


def create_env():
    controller_config = load_composite_controller_config(robot="Nero7")
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        nargs=7,
        type=float,
        default=[0.0, 0.2, 0.0, 0.2, 0.0, 0.1, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--gripper",
        type=float,
        default=-1.0,
        help="Negative opens, positive closes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = create_env()
    try:
        env.reset()
        policy = Nero7JointWaypointPolicy(
            env,
            max_joint_step=args.max_joint_step,
            position_tolerance=args.tolerance,
        )
        result = policy.move_to(
            args.target,
            gripper_command=args.gripper,
            max_steps=args.max_steps,
        )
        print(f"reached: {result.reached}")
        print(f"steps: {result.steps}")
        print(f"final_error: {result.final_error}")
        print(f"final_qpos: {policy.current_qpos}")
        if not result.reached:
            raise RuntimeError("Nero7 failed to reach the requested waypoint")
    finally:
        env.close()


if __name__ == "__main__":
    main()
