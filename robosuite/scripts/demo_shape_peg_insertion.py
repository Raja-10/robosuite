"""Launch and inspect the registered ShapePegInsertion environment."""

import argparse
import itertools
import pathlib
import time

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def controller_config(robot):
    if robot != "Nero7":
        return None
    path = (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_osc_pose.json"
    )
    return load_composite_controller_config(controller=str(path))


def create_env(robot="Panda", camera="taskview", render=True):
    return suite.make(
        env_name="ShapePegInsertion",
        robots=robot,
        controller_configs=controller_config(robot),
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera=None if camera == "free" else camera,
        hard_reset=False,
        ignore_done=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("Panda", "Nero7"), default="Panda")
    parser.add_argument(
        "--camera",
        choices=("free", "agentview", "taskview", "topview", "frontview"),
        default="taskview",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps in headless mode; visual mode runs until its window closes.",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    env = create_env(
        robot=args.robot,
        camera=args.camera,
        render=not args.headless,
    )
    try:
        observations = env.reset()
        print(f"environment: {type(env).__name__}")
        print(f"robot: {args.robot}")
        print(f"action_dim: {env.action_dim}")
        for shape in env.pieces:
            print(f"{shape}_position: {observations[f'{shape}_piece_pos']}")

        action = np.zeros(env.action_dim)
        reward = 0.0
        info = {
            "success": False,
            "pieces_seated": {shape: False for shape in env.pieces},
        }
        if args.headless:
            step_iterator = range(args.steps)
        else:
            step_iterator = itertools.count()
            print("Interactive viewer running. Close the window or press Ctrl+C to exit.")

        try:
            for _ in step_iterator:
                observations, reward, _, info = env.step(action)
                if not args.headless:
                    native_viewer = getattr(env.viewer, "viewer", None)
                    if native_viewer is not None and not native_viewer.is_running():
                        break
                    time.sleep(0.01)
                if not (
                    np.all(np.isfinite(env.sim.data.qpos))
                    and np.all(np.isfinite(env.sim.data.qvel))
                ):
                    raise RuntimeError("ShapePegInsertion produced a non-finite state")
        except KeyboardInterrupt:
            print("\nInteractive viewer interrupted.")

        print(f"reward: {reward}")
        print(f"success: {info['success']}")
        print(f"pieces_seated: {info['pieces_seated']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
