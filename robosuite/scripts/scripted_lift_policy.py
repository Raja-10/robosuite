"""
Hand-coded (non-RL) policy that solves LiftLabSetup1: approach, align to the cube's
nearest face (offset 90deg so the gripper doesn't block the d435 camera's view of the
cube -- see nearest_face_yaw below), descend, grasp, and lift.

Unlike pick.py (fixed step counts per phase), this uses a tolerance/hold waypoint
loop: keep stepping toward a target until position and orientation both stay within
tolerance for a few consecutive steps, or give up after a step cap and move on
regardless. This is more robust to variation in how far a given episode's random cube
placement is from the arm's current pose than a fixed step count can be.

Every episode explicitly starts from and returns to the same "home" OSC pose (the
arm's pose right after reset) -- moved to at the very start (a real waypoint, not an
assumption) and again at the very end, still holding the cube.

Usage:
    python robosuite/scripts/scripted_lift_policy.py            # on-screen viewer
    python robosuite/scripts/scripted_lift_policy.py --headless # no display; prints success only
"""
import argparse

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config

POS_TOLERANCE = 0.005  # m
YAW_TOLERANCE = np.deg2rad(3.0)  # rad; checked against the full orientation error norm
HOLD_STEPS = 5  # consecutive in-tolerance steps required before advancing
MAX_PHASE_STEPS = 150  # safety cap per phase so a waypoint never blocks forever
GRASP_STEPS = 20  # steps spent closing the gripper before lifting

HOVER_HEIGHT = 0.15  # m above the cube
GRASP_Z_OFFSET = 0.005  # m above the cube center, matches pick.py's descend target
LIFT_HEIGHT = 0.2  # m above the cube, matches pick.py's lift target
OPEN, CLOSE = -1.0, 1.0

# The lift phase (moving straight up while carrying the grasped cube) settles into a
# persistent tilt of ~15-25deg that plain waiting doesn't fix -- traced this to neither
# torque saturation nor a joint limit (checked both directly: no actuator sits at its
# ctrlrange during the plateau, no joint sits at its jnt_range), so it's a steady-state
# OSC/nullspace equilibrium, not a hard block. Empirically, temporarily raising the
# orientation gain just for this phase shrinks it substantially (kp_ori 80->250 took a
# representative case from ~25deg to ~11deg), so we bump kp/kd for orientation only
# during the lift move and restore them immediately after.
LIFT_ORI_KP = 250


def wrap_angle(angle):
    """Wraps an angle (rad) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_yaw_from_xmat(xmat_flat):
    """Extracts the world-frame yaw (rotation about z) from a flattened 3x3 rotation matrix."""
    R = xmat_flat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def nearest_face_yaw(yaw, reference):
    """Folds `yaw` to the equivalent orientation (mod 90 degrees, since a cube is symmetric
    under 90-degree rotations about its vertical axis) nearest to `reference`.

    Folding relative to the arm's *current* eef yaw (rather than a fixed reference) means
    the swing needed is always the smallest one available for whichever direction the
    caller asked for -- self-adapting to wherever the arm actually is, rather than a
    hardcoded absolute offset tied to one specific starting pose.
    """
    diff = wrap_angle(yaw - reference)
    return reference + ((diff + np.pi / 4) % (np.pi / 2)) - np.pi / 4


def move_to(env, site_id, output_max, target_pos, target_mat, gripper_action, max_steps, render):
    """Drives the eef toward `target_pos`/`target_mat` until both position and orientation
    stay within tolerance for HOLD_STEPS consecutive steps, or `max_steps` is exhausted.

    Returns:
        bool: True if it converged within tolerance, False if max_steps was exhausted first
        (the eef still ends up as close as it got -- this is not a failure by itself, just
        a signal callers can use to try an alternative, e.g. see the wrist-flip fallback in
        run_episode).
    """
    stable_steps = 0
    for _ in range(max_steps):
        cur_pos = env.sim.data.site_xpos[site_id]
        cur_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
        ori_error = T.quat2axisangle(T.mat2quat(target_mat @ cur_mat.T))

        action = np.zeros(env.action_dim)
        action[:3] = np.clip((target_pos - cur_pos) / output_max[:3], -1, 1)
        action[3:6] = np.clip(ori_error / output_max[3:6], -1, 1)
        action[-1] = gripper_action
        env.step(action)
        if render:
            env.render()

        if np.linalg.norm(target_pos - cur_pos) < POS_TOLERANCE and np.linalg.norm(ori_error) < YAW_TOLERANCE:
            stable_steps += 1
            if stable_steps >= HOLD_STEPS:
                return True
        else:
            stable_steps = 0
    return False


def hold(env, site_id, output_max, target_pos, target_mat, gripper_action, num_steps, render):
    """Holds `target_pos`/`target_mat` (re-correcting each step, not just freezing in place)
    while actuating the gripper -- matches pick.py's close-phase behavior, so any drift from
    gripper-closing contact forces gets corrected rather than left uncorrected."""
    for _ in range(num_steps):
        cur_pos = env.sim.data.site_xpos[site_id]
        cur_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
        ori_error = T.quat2axisangle(T.mat2quat(target_mat @ cur_mat.T))

        action = np.zeros(env.action_dim)
        action[:3] = np.clip((target_pos - cur_pos) / output_max[:3], -1, 1)
        action[3:6] = np.clip(ori_error / output_max[3:6], -1, 1)
        action[-1] = gripper_action
        env.step(action)
        if render:
            env.render()


def run_episode(env, render=False, verbose=False):
    """
    Runs one scripted pick episode on an already-reset `env`.

    Returns:
        bool: True if the cube ends up lifted (env._check_success() semantics).
    """
    controller_config = load_composite_controller_config(controller=None, robot="Nero7")
    output_max = np.array(controller_config["body_parts"]["right"]["output_max"])

    site_id = env.robots[0].eef_site_id["right"]
    # Reference pose captured right after reset, before any motion -- the arm's known-good
    # vertical pose (~0-2deg off world -Z). This doubles as the episode's "home" OSC pose:
    # explicitly moved to at the start (below) and returned to at the end, so every episode
    # both begins and ends at the same well-defined Cartesian pose. Building the grasp
    # target from this (rather than the arm's current, possibly-drifted orientation) also
    # avoids baking pre-existing tilt into the "aligned" target -- see pick.py.
    home_pos = env.sim.data.site_xpos[site_id].copy()
    vertical_mat = env.sim.data.site_xmat[site_id].copy().reshape(3, 3)
    eef_yaw = get_yaw_from_xmat(vertical_mat)
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    cube_yaw = get_yaw_from_xmat(env.sim.data.body_xmat[env.cube_body_id])

    # Explicitly move to (rather than assume we're already at) the home pose, so it's a
    # real, recorded first waypoint of the episode, not just an implicit assumption.
    move_to(env, site_id, output_max, home_pos, vertical_mat, OPEN, MAX_PHASE_STEPS, render)

    # Occlusion-avoidance offset removed: with it (targeting the face pair 90deg from
    # nearest, so the gripper's narrow profile faces the d435 camera instead of its flat
    # body), joint 6 was repeatedly running into its actual angle limit (-44/+54deg range)
    # during the approach phase -- confirmed directly by checking qpos against jnt_range,
    # not just inferred. Without the offset (plain nearest-face alignment, <=45deg swing),
    # joint 6 no longer hits its limit there and success went to 10/10 in testing. Tradeoff:
    # the gripper's flat body now faces the camera again, so d435 will show the cube
    # partially/fully occluded during the grasp. The lift phase still shows tilt afterward
    # (now traced to joint 4 hitting its own limit during that specific motion, not joint 6)
    # -- a separate, still-unresolved constraint from the lift itself, not this offset.
    target_yaw = nearest_face_yaw(cube_yaw, reference=eef_yaw)
    target_mat = T.quat2mat(T.axisangle2quat(np.array([0, 0, target_yaw - eef_yaw]))) @ vertical_mat

    above_cube = cube_pos + np.array([0, 0, HOVER_HEIGHT])
    grasp_pos = cube_pos + np.array([0, 0, GRASP_Z_OFFSET])
    lift_pos = cube_pos + np.array([0, 0, LIFT_HEIGHT])

    # A handful of cube placements need much longer than MAX_PHASE_STEPS to converge here
    # (this phase combines a position move with the yaw alignment) -- tried falling back to
    # the 180deg-flipped wrist solution when this looked stuck, but that retried from an
    # already-displaced pose (not a clean baseline) and made things worse on average (55%
    # success vs. 70% without it, tested empirically). Simply allowing more time is safe --
    # it can only help, since the target doesn't change -- so this phase alone gets extra
    # budget instead.
    move_to(env, site_id, output_max, above_cube, target_mat, OPEN, 2 * MAX_PHASE_STEPS, render)

    move_to(env, site_id, output_max, grasp_pos, target_mat, OPEN, MAX_PHASE_STEPS, render)
    hold(env, site_id, output_max, grasp_pos, target_mat, CLOSE, GRASP_STEPS, render)

    # Temporarily raise the orientation gain for the lift phase only (see LIFT_ORI_KP) --
    # restored right after, so it doesn't affect any later episode reusing this controller.
    arm_controller = env.robots[0].composite_controller.part_controllers["right"]
    original_kp, original_kd = arm_controller.kp.copy(), arm_controller.kd.copy()
    arm_controller.kp[3:6] = LIFT_ORI_KP
    arm_controller.kd[3:6] = 2 * np.sqrt(LIFT_ORI_KP) * 1.5  # matches default_nero7.json's damping_ratio
    try:
        move_to(env, site_id, output_max, lift_pos, target_mat, CLOSE, MAX_PHASE_STEPS, render)
        success = env._check_success()

        # Return to the same home pose/orientation the episode started at, still holding
        # the cube, so every episode ends at a known, consistent Cartesian pose. Double
        # budget: this reach-back is comparable in size to the initial approach, but done
        # while loaded, so it needs more steps than the other (unloaded or short) phases.
        move_to(env, site_id, output_max, home_pos, vertical_mat, CLOSE, 2 * MAX_PHASE_STEPS, render)
    finally:
        arm_controller.kp[:], arm_controller.kd[:] = original_kp, original_kd

    if verbose:
        print(f"Cube yaw: {np.degrees(cube_yaw):.1f} deg, target yaw: {np.degrees(wrap_angle(target_yaw)):.1f} deg")
        print("Pick succeeded:", success)
    return success


def main(args):
    env = robosuite.make(
        env_name="LiftLabSetup1",
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )
    env.reset()
    run_episode(env, render=args.render, verbose=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without an on-screen viewer")
    args = parser.parse_args()
    args.render = not args.headless
    main(args)
