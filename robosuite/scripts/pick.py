"""
Scripted pick: Nero7 approaches, aligns to the cube's nearest face, grasps, and
lifts the cube in LiftLabSetup1.

The cube is placed with a random yaw (Lift's default placement sampler). Since a
cube has 4-fold rotational symmetry about its vertical axis, the gripper only ever
needs to align to within +/-45 degrees of a face pair -- so before descending, the
script rotates the gripper by the minimal correction that lines it up with the
nearest pair of opposing faces, rather than assuming a fixed grasp orientation.

Usage:
    python robosuite/scripts/pick.py                          # on-screen viewer, default camera
    python robosuite/scripts/pick.py --headless                # no display; prints success only
    python robosuite/scripts/pick.py --camera d435              # on-screen viewer, view from the d435 camera
    python robosuite/scripts/pick.py --view --camera d435       # just look around (no pick), on-screen
    python robosuite/scripts/pick.py --view --camera d435 --save frame.png   # headless, save one frame, no pick
"""
import argparse
import time

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config


# The controller maps a clipped [-1, 1] action straight to output_max (near-top speed),
# so with no distance-dependent easing the arm reaches most waypoints in a handful of
# steps, then just holds still doing nothing for the rest of that phase's step budget
# (measured: the 300-step above_cube approach was actually converging by step 16) --
# that idle hold, not the phases themselves, is what reads as a long pause. Scaling the
# commanded action down spreads the same travel over most of the budgeted steps instead,
# so the phase duration is spent actually moving, slowly, rather than sprinting then
# waiting.
SPEED_SCALE = 0.3


def go_to(env, site_id, output_max, target_pos, gripper_action, n_steps, render, target_mat=None):
    for _ in range(n_steps):
        step_start = time.perf_counter()
        cur_pos = env.sim.data.site_xpos[site_id]
        action = np.zeros(env.action_dim)
        action[:3] = np.clip((target_pos - cur_pos) / output_max[:3], -1, 1) * SPEED_SCALE
        if target_mat is not None:
            cur_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
            ori_error = T.quat2axisangle(T.mat2quat(target_mat @ cur_mat.T))
            action[3:6] = np.clip(ori_error / output_max[3:6], -1, 1) * SPEED_SCALE
        action[-1] = gripper_action
        env.step(action)
        if render:
            env.render()
            # env.step()/env.render() run as fast as the CPU allows with no built-in
            # real-time pacing, so a 300-step phase can flash by in a fraction of a
            # second on screen. Sleep out whatever's left of this step's real-time
            # budget (env.control_timestep = 1/control_freq seconds) so watching it
            # actually takes that many real seconds, not however long the compute took.
            remaining = env.control_timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def cube_yaw(env):
    """World-frame yaw (rad) of the cube, assuming it rests flat on the table."""
    w, x, y, z = env.sim.data.body_xquat[env.cube_body_id]
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return 2 * np.arctan2(z, w)


def nearest_face_correction(yaw):
    """Minimal rotation (rad, in [-45, 45] deg) to align with the nearest face pair
    of a 4-fold-symmetric cube."""
    return ((yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4


# Occlusion-avoidance offset removed: with it (targeting the face pair 90deg from
# nearest, so the gripper's narrow profile faces the d435 camera instead of its flat
# body), joint 6 was repeatedly running into its actual angle limit (-44/+54deg range)
# during the approach phase -- confirmed directly by checking qpos against jnt_range,
# not just inferred. Without the offset (plain nearest-face alignment, <=45deg swing),
# joint 6 no longer hits its limit there and success held at 10/10 in testing. Tradeoff:
# the gripper's flat body faces the d435 camera again, so the cube may be
# partially/fully occluded during the grasp. The lift phase still shows tilt afterward
# (18-38deg in testing, occasionally joint 4 nearing its own limit) -- a separate,
# still-unresolved issue from the lift motion itself, not this offset.
GRIPPER_YAW_OFFSET = 0.0


def view_live(camera):
    """Open an on-screen viewer on LiftLabSetup1, optionally fixed to one camera, no pick."""
    env = robosuite.make(
        env_name="LiftLabSetup1",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.reset()
    if camera is not None:
        env.viewer.set_camera(camera_id=env.sim.model.camera_name2id(camera))
    while True:
        env.step(np.zeros(env.action_dim))
        env.render()


def save_frame(camera, out_path, width, height):
    """Save a single frame from the given camera on LiftLabSetup1, no pick."""
    import imageio

    env = robosuite.make(
        env_name="LiftLabSetup1",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera,
        camera_heights=height,
        camera_widths=width,
    )
    obs = env.reset()
    imageio.imwrite(out_path, obs[f"{camera}_image"][::-1])  # mujoco images are vertically flipped
    print(f"Saved {camera} view ({width}x{height}) to {out_path}")
    env.close()


def pick(render=True, camera=None):
    controller_config = load_composite_controller_config(controller=None, robot="Nero7")
    output_max = np.array(controller_config["body_parts"]["right"]["output_max"])

    env = robosuite.make(
        env_name="LiftLabSetup1",
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
    )
    env.reset()
    if render and camera is not None:
        env.viewer.set_camera(camera_id=env.sim.model.camera_name2id(camera))

    site_id = env.robots[0].eef_site_id["right"]
    # Reference pose captured right after reset, before any motion -- the arm's known-good
    # vertical pose (~0-2deg off world -Z). This doubles as the episode's "home" OSC pose
    # (matching scripted_lift_policy.py): explicitly moved to at the start (a real
    # waypoint, not an assumption) and returned to at the end, still holding the cube, so
    # every episode both starts and ends at the same well-defined Cartesian pose -- whatever
    # angle the robot happens to be in at reset, not a hardcoded one. Building the grasp
    # target from this (rather than the arm's current, possibly-drifted orientation) also
    # avoids baking pre-existing tilt into the "aligned" target.
    home_pos = env.sim.data.site_xpos[site_id].copy()
    vertical_mat = env.sim.data.site_xmat[site_id].copy().reshape(3, 3)
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    above_cube = cube_pos + np.array([0, 0, 0.15])

    # Explicitly move to (rather than assume we're already at) the home pose, so it's a
    # real, recorded first waypoint of the episode, not just an implicit assumption.
    go_to(env, site_id, output_max, home_pos, -1, 60, render, target_mat=vertical_mat)

    # align gripper yaw to the nearest face pair of the cube, computed against the
    # clean vertical reference (not the arm's current, possibly-drifted orientation)
    correction = nearest_face_correction(cube_yaw(env)) + GRIPPER_YAW_OFFSET
    target_mat = T.quat2mat(T.axisangle2quat(np.array([0, 0, correction]))) @ vertical_mat

    # move above cube, gripper open -- hold vertical+aligned orientation from the start.
    # A handful of cube placements need substantially more than 150 steps to fully
    # converge here (combined position+yaw move), so this phase gets extra budget --
    # safe since the target doesn't change, just more patience.
    go_to(env, site_id, output_max, above_cube, -1, 300, render, target_mat=target_mat)

    # descend to grasp height, holding the aligned orientation. Lowered from +5mm above
    # cube center to the center itself, for a firmer/more centered grip on the fingers.
    grasp_pos = cube_pos + np.array([0, 0, 0.0])
    go_to(env, site_id, output_max, grasp_pos, -1, 60, render, target_mat=target_mat)
    # close gripper
    go_to(env, site_id, output_max, grasp_pos, 1, 40, render, target_mat=target_mat)
    # lift
    go_to(env, site_id, output_max, cube_pos + np.array([0, 0, 0.2]), 1, 60, render, target_mat=target_mat)

    success = env._check_success()

    # Return to the same home pose/orientation the episode started at, still holding the
    # cube, so every episode ends at a known, consistent Cartesian pose. Double budget:
    # this reach-back is comparable in size to the initial approach, but done under load.
    go_to(env, site_id, output_max, home_pos, 1, 120, render, target_mat=vertical_mat)

    print(f"Cube yaw: {np.degrees(cube_yaw(env)):.1f} deg, alignment correction applied: {np.degrees(correction):.1f} deg")
    print("Pick succeeded:", success)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without an on-screen viewer")
    parser.add_argument("--camera", default=None, help="camera to view from (e.g. d435, agentview)")
    parser.add_argument("--view", action="store_true", help="just look around (no pick); combine with --camera")
    parser.add_argument("--save", default=None, help="with --view: headless, save one frame to this path instead of a live viewer")
    parser.add_argument("--width", type=int, default=640, help="with --view --save: image width")
    parser.add_argument("--height", type=int, default=480, help="with --view --save: image height")
    args = parser.parse_args()

    if args.view:
        if args.save:
            save_frame(args.camera or "d435", args.save, args.width, args.height)
        else:
            view_live(args.camera or "d435")
    else:
        pick(render=not args.headless, camera=args.camera)
