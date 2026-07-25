"""Teleoperate mounted Nero7 in ShapePegInsertion with keyboard or SpaceMouse.

Keyboard controls:
    Arrow keys       translate in world X / Y
    ; and .          translate up / down
    e / r            roll
    y / h            pitch
    o / p            yaw
    Space            toggle gripper open / closed
    q                reset the task
    Ctrl+C           exit
"""

import argparse
import pathlib
import time

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper


def osc_config():
    path = (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_osc_pose.json"
    )
    return load_composite_controller_config(controller=str(path))


def create_env(camera="free", record_dir=None, render=True):
    env = suite.make(
        env_name="ShapePegInsertion",
        robots="Nero7",
        controller_configs=osc_config(),
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        render_camera=None if camera == "free" else camera,
        control_freq=20,
        hard_reset=False,
        ignore_done=True,
    )
    if record_dir is not None:
        env = DataCollectionWrapper(
            env,
            directory=str(pathlib.Path(record_dir).expanduser()),
            collect_freq=1,
            flush_freq=100,
        )
    return env


def create_device(env, name, pos_sensitivity, rot_sensitivity):
    if name == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
    if name == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
    raise ValueError(f"Unsupported teleoperation device: {name}")


def device_action_to_env_action(robot, device_action):
    """Converts the standard device dictionary to Nero7's OSC action vector."""
    if device_action is None:
        return None
    action_dict = {
        "right": device_action["right_delta"],
        "right_gripper": device_action["right_gripper"],
    }
    action = robot.create_action_vector(action_dict)
    if action.shape != (7,):
        raise RuntimeError(f"Expected Nero7 OSC action shape (7,), got {action.shape}")
    return np.clip(action, -1.0, 1.0)


def viewer_is_running(env):
    renderer = getattr(env, "viewer", None)
    native_viewer = getattr(renderer, "viewer", None)
    return native_viewer is None or native_viewer.is_running()


def print_policy_status(observations, info, reward):
    distances = {
        shape: np.linalg.norm(observations[f"{shape}_eef_to_piece_pos"])
        for shape in ("circle", "square", "triangle", "rectangle")
    }
    nearest = min(distances, key=distances.get)
    seated = [shape for shape, value in info["pieces_seated"].items() if value]
    print(
        f"nearest={nearest:<9} distance={distances[nearest]:.3f} m "
        f"reward={reward:.3f} seated={seated}",
        end="\r",
        flush=True,
    )


def run_episode(env, device, max_frequency, status_frequency):
    observations = env.reset()
    env.render()
    device.start_control()
    print("\nTeleoperation active. Press q to reset; close the window or press Ctrl+C to exit.")

    step = 0
    success_announced = False
    while viewer_is_running(env):
        start = time.monotonic()
        device_action = device.input2action()
        action = device_action_to_env_action(env.robots[0], device_action)
        if action is None:
            print("\nReset requested.")
            return True

        observations, reward, _, info = env.step(action)
        if not (
            np.all(np.isfinite(env.sim.data.qpos))
            and np.all(np.isfinite(env.sim.data.qvel))
            and np.all(np.isfinite(action))
        ):
            raise RuntimeError("Nero7 teleoperation produced a non-finite state")

        if step % status_frequency == 0:
            print_policy_status(observations, info, reward)
        if info["success"] and not success_announced:
            print("\nTask complete: all four pieces are stably seated.")
            success_announced = True
        step += 1

        elapsed = time.monotonic() - start
        remaining = 1.0 / max_frequency - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("keyboard", "spacemouse"),
        default="keyboard",
    )
    parser.add_argument(
        "--camera",
        choices=("free", "agentview", "taskview", "topview", "frontview"),
        default="free",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=0.1,
        help="Translation gain; 0.03-0.1 is recommended for insertion.",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=0.5,
        help="Rotation gain.",
    )
    parser.add_argument("--max-frequency", type=float, default=20.0)
    parser.add_argument("--status-frequency", type=int, default=20)
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Optional directory for raw states, actions, model XML, and success metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_frequency <= 0:
        raise ValueError("--max-frequency must be positive")
    if args.status_frequency <= 0:
        raise ValueError("--status-frequency must be positive")

    env = create_env(camera=args.camera, record_dir=args.record_dir)
    device = create_device(
        env,
        name=args.device,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    try:
        reset_requested = True
        while reset_requested and viewer_is_running(env):
            reset_requested = run_episode(
                env,
                device,
                max_frequency=args.max_frequency,
                status_frequency=args.status_frequency,
            )
    except KeyboardInterrupt:
        print("\nTeleoperation interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
