"""Visualize Nero7 moving from its initial pose to pick and lift a cube."""

import argparse
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.tune_nero7_init_pose import active_robot_collision_pairs


POSITION_ACTION_SCALE = 0.025
ORIENTATION_ACTION_SCALE = 0.25


def osc_config_path():
    return (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_osc_pose.json"
    )


def create_env(render):
    controller_config = load_composite_controller_config(controller=str(osc_config_path()))
    return suite.make(
        env_name="Lift",
        robots="Nero7",
        controller_configs=controller_config,
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        hard_reset=False,
        ignore_done=True,
    )


def current_eef_pose(env):
    site_id = env.robots[0].eef_site_id["right"]
    position = env.sim.data.site_xpos[site_id].copy()
    orientation = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    return position, orientation


def pose_action(env, target_position, target_orientation, gripper_command):
    current_position, current_orientation = current_eef_pose(env)
    position_error = target_position - current_position
    orientation_error = Rotation.from_matrix(target_orientation @ current_orientation.T).as_rotvec()

    action = np.empty(7)
    action[:3] = np.clip(position_error / POSITION_ACTION_SCALE, -1.0, 1.0)
    action[3:6] = np.clip(orientation_error / ORIENTATION_ACTION_SCALE, -1.0, 1.0)
    action[6] = np.clip(gripper_command, -1.0, 1.0)
    return action, position_error, orientation_error


def run_to_pose(
    env,
    target_position,
    target_orientation,
    gripper_command=-1.0,
    max_steps=500,
    position_tolerance=0.005,
    orientation_tolerance=0.03,
    render=False,
):
    target_position = np.asarray(target_position, dtype=float)
    target_orientation = np.asarray(target_orientation, dtype=float)
    if target_position.shape != (3,) or target_orientation.shape != (3, 3):
        raise ValueError("Target position must be length 3 and orientation must be 3x3")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    initial_position, initial_orientation = current_eef_pose(env)
    observed_collisions = set()

    for step in range(1, max_steps + 1):
        action, position_error, orientation_error = pose_action(
            env,
            target_position,
            target_orientation,
            gripper_command,
        )
        if (
            np.linalg.norm(position_error) <= position_tolerance
            and np.linalg.norm(orientation_error) <= orientation_tolerance
        ):
            return {
                "reached": True,
                "steps": step - 1,
                "initial_position": initial_position,
                "target_position": target_position,
                "final_position": current_eef_pose(env)[0],
                "position_error": float(np.linalg.norm(position_error)),
                "orientation_error": float(np.linalg.norm(orientation_error)),
                "collision_pairs": [list(pair) for pair in sorted(observed_collisions)],
            }

        env.step(action)
        observed_collisions.update(tuple(pair) for pair in active_robot_collision_pairs(env))
        if render:
            env.render()
        if not (
            np.all(np.isfinite(env.sim.data.qpos))
            and np.all(np.isfinite(env.sim.data.qvel))
            and np.all(np.isfinite(env.sim.data.ctrl))
        ):
            raise RuntimeError("Nero7 OSC motion produced a non-finite simulation state")

    _, position_error, orientation_error = pose_action(
        env,
        target_position,
        target_orientation,
        gripper_command,
    )
    return {
        "reached": False,
        "steps": max_steps,
        "initial_position": initial_position,
        "target_position": target_position,
        "final_position": current_eef_pose(env)[0],
        "position_error": float(np.linalg.norm(position_error)),
        "orientation_error": float(np.linalg.norm(orientation_error)),
        "collision_pairs": [list(pair) for pair in sorted(observed_collisions)],
    }


def run_motion(
    env,
    position_delta,
    rotation_delta,
    gripper_command=-1.0,
    max_steps=500,
    position_tolerance=0.005,
    orientation_tolerance=0.03,
    render=False,
):
    position_delta = np.asarray(position_delta, dtype=float)
    rotation_delta = np.asarray(rotation_delta, dtype=float)
    if position_delta.shape != (3,) or rotation_delta.shape != (3,):
        raise ValueError("Position and rotation deltas must each contain three values")

    initial_position, initial_orientation = current_eef_pose(env)
    return run_to_pose(
        env=env,
        target_position=initial_position + position_delta,
        target_orientation=Rotation.from_rotvec(rotation_delta).as_matrix() @ initial_orientation,
        gripper_command=gripper_command,
        max_steps=max_steps,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        render=render,
    )


def hold_pose(env, target_position, target_orientation, gripper_command, steps, render=False):
    observed_collisions = set()
    for _ in range(steps):
        action, _, _ = pose_action(
            env,
            target_position=target_position,
            target_orientation=target_orientation,
            gripper_command=gripper_command,
        )
        env.step(action)
        observed_collisions.update(tuple(pair) for pair in active_robot_collision_pairs(env))
        if render:
            env.render()
    return [list(pair) for pair in sorted(observed_collisions)]


def run_pick_and_lift(
    env,
    approach_height=0.12,
    grasp_offset_z=0.0,
    lift_height=0.15,
    close_steps=40,
    max_motion_steps=500,
    render=False,
):
    if min(approach_height, lift_height) <= 0:
        raise ValueError("Approach and lift heights must be positive")
    if close_steps <= 0:
        raise ValueError("close_steps must be positive")

    cube_start = env.sim.data.body_xpos[env.cube_body_id].copy()
    _, grasp_orientation = current_eef_pose(env)
    grasp_position = cube_start + np.array([0.0, 0.0, grasp_offset_z])
    approach_position = grasp_position + np.array([0.0, 0.0, approach_height])
    phases = {}

    phases["approach"] = run_to_pose(
        env,
        target_position=approach_position,
        target_orientation=grasp_orientation,
        gripper_command=-1.0,
        max_steps=max_motion_steps,
        render=render,
    )
    if not phases["approach"]["reached"]:
        return {"success": False, "failed_phase": "approach", "phases": phases}

    phases["descend"] = run_to_pose(
        env,
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        gripper_command=-1.0,
        max_steps=max_motion_steps,
        position_tolerance=0.003,
        render=render,
    )
    if not phases["descend"]["reached"]:
        return {"success": False, "failed_phase": "descend", "phases": phases}

    close_collisions = hold_pose(
        env,
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        gripper_command=1.0,
        steps=close_steps,
        render=render,
    )
    grasped = env._check_grasp(
        gripper=env.robots[0].gripper["right"],
        object_geoms=env.cube,
    )
    phases["grasp"] = {
        "grasped": bool(grasped),
        "steps": close_steps,
        "collision_pairs": close_collisions,
    }
    if not grasped:
        return {"success": False, "failed_phase": "grasp", "phases": phases}

    lift_position = grasp_position + np.array([0.0, 0.0, lift_height])
    phases["lift"] = run_to_pose(
        env,
        target_position=lift_position,
        target_orientation=grasp_orientation,
        gripper_command=1.0,
        max_steps=max_motion_steps,
        render=render,
    )
    cube_final = env.sim.data.body_xpos[env.cube_body_id].copy()
    cube_lift = float(cube_final[2] - cube_start[2])
    still_grasped = env._check_grasp(
        gripper=env.robots[0].gripper["right"],
        object_geoms=env.cube,
    )
    success = bool(
        phases["lift"]["reached"]
        and still_grasped
        and cube_lift >= 0.5 * lift_height
    )
    return {
        "success": success,
        "failed_phase": None if success else "lift",
        "cube_start": cube_start,
        "cube_final": cube_final,
        "cube_lift": cube_lift,
        "still_grasped": bool(still_grasped),
        "phases": phases,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pick", "move"),
        default="pick",
        help="Pick and lift the cube, or run the original relative Cartesian move.",
    )
    parser.add_argument(
        "--position-delta",
        nargs=3,
        type=float,
        default=[0.10, 0.0, 0.05],
        metavar=("DX", "DY", "DZ"),
        help="Desired world-frame translation in metres.",
    )
    parser.add_argument(
        "--rotation-delta",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Desired world-frame rotation vector in radians.",
    )
    parser.add_argument("--gripper", type=float, default=-1.0)
    parser.add_argument("--approach-height", type=float, default=0.12)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.15)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--initial-hold-steps", type=int, default=50)
    parser.add_argument("--final-hold-steps", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.headless
    env = create_env(render=render)
    try:
        env.reset()
        if render:
            for _ in range(args.initial_hold_steps):
                env.step(np.zeros(env.action_dim))
                env.render()

        if args.mode == "pick":
            result = run_pick_and_lift(
                env=env,
                approach_height=args.approach_height,
                grasp_offset_z=args.grasp_offset_z,
                lift_height=args.lift_height,
                close_steps=args.close_steps,
                max_motion_steps=args.max_steps,
                render=render,
            )
            print(f"success: {result['success']}")
            print(f"failed_phase: {result['failed_phase']}")
            if "cube_start" in result:
                print(f"cube_start: {result['cube_start']}")
                print(f"cube_final: {result['cube_final']}")
                print(f"cube_lift: {result['cube_lift']}")
                print(f"still_grasped: {result['still_grasped']}")
            for phase, phase_result in result["phases"].items():
                print(f"{phase}: {phase_result}")
        else:
            result = run_motion(
                env=env,
                position_delta=args.position_delta,
                rotation_delta=args.rotation_delta,
                gripper_command=args.gripper,
                max_steps=args.max_steps,
                position_tolerance=args.position_tolerance,
                orientation_tolerance=args.orientation_tolerance,
                render=render,
            )

            print(f"reached: {result['reached']}")
            print(f"steps: {result['steps']}")
            print(f"initial_position: {result['initial_position']}")
            print(f"target_position: {result['target_position']}")
            print(f"final_position: {result['final_position']}")
            print(f"position_error: {result['position_error']}")
            print(f"orientation_error: {result['orientation_error']}")
            print(f"collision_pairs: {result['collision_pairs']}")

        if render:
            for _ in range(args.final_hold_steps):
                env.step(np.zeros(env.action_dim))
                env.render()

        if args.mode == "pick":
            if not result["success"]:
                raise RuntimeError(f"Nero7 pick failed during {result['failed_phase']}")
        else:
            if not result["reached"]:
                raise RuntimeError("Nero7 did not reach the requested Cartesian command")
            if result["collision_pairs"]:
                raise RuntimeError("Nero7 encountered a collision during Cartesian motion")
    finally:
        env.close()


if __name__ == "__main__":
    main()
