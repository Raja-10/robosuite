"""
Collects demonstrations of the scripted Stack policy (see scripted_stack_policy.py) into
the standard robosuite hdf5 dataset format -- the same "data/demo_N -> {states, actions,
model_file}" layout produced by collect_human_demonstrations.py, and playable with
robosuite/scripts/playback_demonstrations_from_hdf5.py, or convertible to robomimic's
obs/image format via robomimic/scripts/dataset_states_to_obs.py. Only successful episodes
(cubeA ends up stacked on cubeB) are kept, exactly as in collect_lift_data.py.

Domain randomization is on by default: camera pose (d435 only), lighting, table color
tint, and arm actuator dynamics (joint stiffness/frictionloss/damping/armature) each get
one fresh random draw per episode (--no-dr disables all of it), via the small custom
wrappers defined below (EpisodeDRWrapper, TableTintWrapper) rather than robosuite's own
DomainRandomizationWrapper -- see EpisodeDRWrapper's docstring for why (a real bug when
used against an env whose reset() recompiles the whole model, confirmed directly, not a
style choice). Both wrappers must sit *inside* DataCollectionWrapper, not outside:
DataCollectionWrapper captures each episode's model.xml via sim.model.get_xml(), which
reflects live sim.model values, so the randomized-this-episode camera/light/tint/joint
values only end up in the saved model.xml if they were already applied to sim.model by the
time DataCollectionWrapper reads it (confirmed directly -- mutating sim.model.cam_pos and
checking get_xml() picked up the new value). That saved model.xml is what
dataset_states_to_obs.py rebuilds the sim from per-episode (via reset_from_xml_string) when
converting to images, so this ordering is what makes the randomization actually show up in
the extracted image observations, not just in the live collection run.

Usage:
    python robosuite/scripts/collect_stack_data.py --num-episodes 100
"""
import argparse
import json
import os
import tempfile

import numpy as np

import robosuite
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.scripts.scripted_stack_policy import run_episode
from robosuite.utils.mjmod import CameraModder, DynamicsModder, LightingModder
from robosuite.wrappers import DataCollectionWrapper, Wrapper

DR_CAMERA_NAMES = ["d435"]  # only the camera a real deployed policy would actually see from
DR_ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]  # arm only -- not gripper/cube joints
TABLE_TINT_GEOM = "table_visual"
TABLE_TINT_RANGE = (0.85, 1.0)  # per-channel multiplier on the real photo's own colors

CAMERA_DR_ARGS = {
    "camera_names": DR_CAMERA_NAMES,
    "randomize_position": True,
    "randomize_rotation": True,
    "randomize_fovy": False,
    "position_perturbation_size": 0.01,  # +/- 1cm, matches a real remount's translational error
    "rotation_perturbation_size": 0.087,  # +/- ~5deg, matches a real remount's angular error
}
# Lighting defaults (position/direction/specular/ambient/diffuse/active on the arena's one
# light) are fine as-is -- no need to override.
LIGHTING_DR_ARGS = {}
DYNAMICS_DR_ARGS = {
    "randomize_density": False,
    "randomize_viscosity": False,
    "randomize_position": False,
    "randomize_quaternion": False,
    "randomize_inertia": False,
    "randomize_mass": False,
    "randomize_friction": False,
    "randomize_solref": False,
    "randomize_solimp": False,
    "joint_names": DR_ARM_JOINTS,
    "randomize_stiffness": True,
    "randomize_frictionloss": True,
    "randomize_damping": True,
    "randomize_armature": True,
}


class EpisodeDRWrapper(Wrapper):
    """
    Randomizes camera pose, lighting, and joint dynamics once per episode by
    constructing fresh Modder instances bound to the current sim inside reset() itself.

    robosuite's own DomainRandomizationWrapper instead constructs its Modders once (in
    __init__) and only rebinds them to a new sim via an explicit update_sim() call at
    the *end* of its reset(). Its reset() calls restore_default_domain() *first*, using
    whatever sim the modders were bound to as of the end of the *previous* reset. That's
    fine for envs that mutate one persistent sim in place -- but StackLabSetup1 (like
    most robosuite envs) recompiles a brand new MjModel/MjSim on every reset(), not just
    the first, so by the second episode that cached sim reference is already stale.
    Confirmed directly: this crashes with `AttributeError: 'MjSim' object has no
    attribute 'model'` inside restore_default_domain() on the second episode.

    Sidestepping this by re-fetching self.env.sim fresh inside reset() (same approach
    TableTintWrapper below uses) and skipping the restore-before-reset step entirely --
    there's nothing to restore *from*, since the env's own reset() already produces a
    pristine, default-valued model.
    """

    def __init__(self, env, seed=None, camera_args=None, lighting_args=None, dynamics_args=None):
        super().__init__(env)
        self.random_state = np.random.RandomState(seed)
        self.camera_args = camera_args or {}
        self.lighting_args = lighting_args or {}
        self.dynamics_args = dynamics_args or {}

    def reset(self):
        ret = self.env.reset()
        CameraModder(self.env.sim, random_state=self.random_state, **self.camera_args).randomize()
        LightingModder(self.env.sim, random_state=self.random_state, **self.lighting_args).randomize()
        DynamicsModder(self.env.sim, random_state=self.random_state, **self.dynamics_args).randomize()
        return ret


class TableTintWrapper(Wrapper):
    """
    Jitters the table's material rgba (a per-channel multiplier on the real photo
    texture) once per episode, as a stand-in for TextureModder-based table texture
    randomization.

    robosuite's own TextureModder can't be used here: it indexes model.tex_rgb directly
    (robosuite/utils/mjmod.py's Texture class), a field MuJoCo renamed to tex_data (and
    added tex_nchannel/tex_colorspace alongside) in a later texture-pipeline rewrite --
    confirmed directly (model.tex_rgb doesn't exist on this repo's mujoco 3.3.7;
    model.tex_data does). DomainRandomizationWrapper hard-asserts mujoco==3.1.1 before
    constructing a TextureModder for exactly this reason. Reimplementing TextureModder
    for the new layout is real framework work, out of scope here -- multiplying the
    material's rgba tint achieves the same practical goal (table looks slightly
    different each episode) through a much simpler, version-independent mechanism.
    """

    def __init__(self, env, seed=None):
        super().__init__(env)
        self.random_state = np.random.RandomState(seed)
        geom_id = self.env.sim.model.geom_name2id(TABLE_TINT_GEOM)
        self.mat_id = self.env.sim.model.geom_matid[geom_id]

    def reset(self):
        ret = self.env.reset()
        self.env.sim.model.mat_rgba[self.mat_id][:3] = self.random_state.uniform(*TABLE_TINT_RANGE, size=3)
        return ret


def main(args):
    config = {
        "env_name": "StackLabSetup1",
        "robots": args.robot,
    }

    env = robosuite.make(
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
        horizon=args.horizon,
    )
    env_info = json.dumps(config)

    if not args.no_dr:
        env = EpisodeDRWrapper(
            env,
            seed=args.seed,
            camera_args=CAMERA_DR_ARGS,
            lighting_args=LIGHTING_DR_ARGS,
            dynamics_args=DYNAMICS_DR_ARGS,
        )
        env = TableTintWrapper(env, seed=args.seed)

    tmp_directory = tempfile.mkdtemp()
    env = DataCollectionWrapper(env, tmp_directory)  # must wrap DR, not the other way around -- see module docstring

    os.makedirs(args.directory, exist_ok=True)

    num_successful = 0
    attempts = 0
    while num_successful < args.num_episodes and attempts < args.max_attempts:
        attempts += 1
        env.reset()
        success = run_episode(env, args, verbose=False)
        num_successful += int(success)
        print(f"Attempt {attempts} (successes so far: {num_successful}/{args.num_episodes}): {'success' if success else 'failed'}")
    env.close()  # flushes the final episode's data -- no later reset() will do it

    if num_successful < args.num_episodes:
        print(
            f"Only got {num_successful}/{args.num_episodes} successes in {attempts} attempts "
            f"(--max-attempts {args.max_attempts}). Saving what succeeded."
        )
    gather_demonstrations_as_hdf5(tmp_directory, args.directory, env_info)
    print(f"Saved to {os.path.join(args.directory, 'demo.hdf5')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of *successful* episodes to collect.")
    parser.add_argument("--max-attempts", type=int, default=200, help="Give up after this many attempts regardless of successes.")
    parser.add_argument("--directory", type=str, default="nero_datasets/stack", help="Where to write the combined demo.hdf5.")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering while collecting.")
    parser.add_argument("--no-dr", action="store_true", help="Disable domain randomization (camera/lighting/table texture/actuator dynamics).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for domain randomization sampling (reproducible batches).")
    parser.add_argument("--robot", type=str, default="Nero7")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--speed-scale", type=float, default=0.6, help="Fraction of max controller translation output used per step.")
    parser.add_argument("--rot-speed-scale", type=float, default=0.6, help="Fraction of max controller rotation output used per step.")
    parser.add_argument("--pos-tolerance", type=float, default=0.005, help="Position tolerance (m) to consider a waypoint reached.")
    parser.add_argument("--yaw-tolerance", type=float, default=np.deg2rad(3.0), help="Yaw tolerance (rad) to consider a waypoint reached.")
    parser.add_argument("--hold-steps", type=int, default=5, help="Consecutive in-tolerance steps required before advancing to the next waypoint.")
    parser.add_argument("--max-phase-steps", type=int, default=300, help="Max steps allowed per movement phase before giving up and moving on.")
    parser.add_argument("--grasp-steps", type=int, default=20, help="Steps spent closing the gripper before checking/using the grasp.")
    parser.add_argument("--release-steps", type=int, default=10, help="Steps spent opening the gripper to release a cube.")
    args = parser.parse_args()

    main(args)
