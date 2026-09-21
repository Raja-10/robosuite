"""
Collects demonstrations of the scripted Lift policy (see scripted_lift_policy.py) into
the standard robosuite hdf5 dataset format -- the same "data/demo_N -> {states, actions,
model_file}" layout produced by collect_human_demonstrations.py, and playable with
robosuite/scripts/playback_demonstrations_from_hdf5.py. Only successful episodes (the
cube ends up lifted) are kept, exactly as in collect_human_demonstrations.py's
gather_demonstrations_as_hdf5.

Usage:
    python robosuite/scripts/collect_lift_data.py --num-episodes 20
"""
import argparse
import json
import os
import tempfile

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.scripts.scripted_lift_policy import run_episode
from robosuite.wrappers import DataCollectionWrapper


def main(args):
    controller_config = load_composite_controller_config(controller=None, robot="Nero7")
    config = {
        "env_name": "LiftLabSetup1",
        "robots": "Nero7",
        "controller_configs": controller_config,
    }

    env = robosuite.make(
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
        horizon=1000,
    )
    env_info = json.dumps(config)

    tmp_directory = tempfile.mkdtemp()
    env = DataCollectionWrapper(env, tmp_directory)

    os.makedirs(args.directory, exist_ok=True)

    num_successful = 0
    for ep in range(args.num_episodes):
        env.reset()
        success = run_episode(env, render=args.render)
        num_successful += int(success)
        print(f"Episode {ep + 1}/{args.num_episodes}: {'success' if success else 'incomplete'}")
    env.close()  # flushes the final episode's data -- no later reset() will do it

    print(f"{num_successful}/{args.num_episodes} episodes were successful and will be saved.")
    # Note: the count above is run_episode's single post-lift _check_success() call, while
    # DataCollectionWrapper's own "successful" flag (which actually decides what
    # gather_demonstrations_as_hdf5 keeps) is OR'd across every step of the episode. They
    # agree in virtually all runs -- if they ever differ, the hdf5's demo_N group count
    # below is the authoritative one.
    gather_demonstrations_as_hdf5(tmp_directory, args.directory, env_info)
    print(f"Saved to {os.path.join(args.directory, 'demo.hdf5')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--directory", type=str, default="nero_datasets/lift", help="Where to write the combined demo.hdf5.")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering while collecting.")
    args = parser.parse_args()

    main(args)
