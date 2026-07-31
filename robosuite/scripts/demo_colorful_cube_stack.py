"""Inspect the ColorfulCubeStack task with scattered or completed cubes."""

import argparse
import time

import numpy as np

import robosuite as suite


def create_env(
    robot="Panda",
    render=True,
    camera="stackview",
    reward_shaping=False,
):
    return suite.make(
        env_name="ColorfulCubeStack",
        robots=robot,
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=reward_shaping,
        render_camera=None if camera == "free" else camera,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
        hard_reset=False,
    )


def arrange_completed_pyramid(env):
    """Places every cube at its assigned target for visual inspection."""
    for name, cube in env.cubes.items():
        env.sim.data.set_joint_qpos(
            cube.joints[0],
            np.concatenate(
                [env.target_positions[name], [1.0, 0.0, 0.0, 0.0]]
            ),
        )
        address = env.sim.model.get_joint_qvel_addr(cube.joints[0])
        env.sim.data.qvel[slice(*address)] = 0.0
    env.sim.forward()


def viewer_is_running(env):
    renderer = getattr(env, "viewer", None)
    viewer = getattr(renderer, "viewer", None)
    return viewer is None or viewer.is_running()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="Panda")
    parser.add_argument(
        "--camera",
        choices=("free", "agentview", "stackview", "frontview"),
        default="stackview",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--start-stacked",
        action="store_true",
        help="Start from the completed 4-3-2-1 pyramid.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Zero-action settling steps; ignored by the interactive viewer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    env = create_env(
        robot=args.robot,
        render=not args.headless,
        camera=args.camera,
    )
    try:
        observations = env.reset()
        if args.start_stacked:
            arrange_completed_pyramid(env)

        print("environment: ColorfulCubeStack")
        print(f"robot: {args.robot}")
        print(f"cube_count: {len(env.cubes)}")
        print(f"cube_names: {list(env.cube_names)}")
        print(f"object_state_dim: {observations['object-state'].shape[0]}")
        for name in env.cube_names:
            position = env.sim.data.body_xpos[env.cube_body_ids[name]]
            print(
                f"{name:>6} position={np.round(position, 4)} "
                f"target={np.round(env.target_positions[name], 4)}"
            )

        if args.headless:
            action = np.zeros(env.action_dim)
            info = {}
            reward = env.reward()
            for _ in range(args.steps):
                _, reward, _, info = env.step(action)
            print(f"reward: {reward:.3f}")
            print(f"success: {env._check_success()}")
            if info:
                print(f"stack_progress: {info['stack_progress']:.2f}")
            return

        action = np.zeros(env.action_dim)
        # The native mjviewer is initialized from the first environment step;
        # render() alone does not create a window on every renderer backend.
        env.step(action)
        env.render()
        print("Inspection mode: close the MuJoCo window or press Ctrl+C.")
        while viewer_is_running(env):
            env.step(action)
            env.render()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInspection interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
