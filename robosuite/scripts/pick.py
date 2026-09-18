"""
Scripted pick: Nero7 approaches, aligns to the cube's nearest face, grasps, and
lifts the cube in LiftLabSetup1.

The cube is placed with a random yaw (Lift's default placement sampler). Since a
cube has 4-fold rotational symmetry about its vertical axis, the gripper only ever
needs to align to within +/-45 degrees of a face pair -- so before descending, the
script rotates the gripper by the minimal correction that lines it up with the
nearest pair of opposing faces, rather than assuming a fixed grasp orientation.

Usage:
    python robosuite/scripts/pick.py                 # on-screen viewer
    python robosuite/scripts/pick.py --headless       # no display; prints success only
"""
import argparse

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config


def go_to(env, site_id, output_max, target_pos, gripper_action, n_steps, render, target_mat=None):
    for _ in range(n_steps):
        cur_pos = env.sim.data.site_xpos[site_id]
        action = np.zeros(env.action_dim)
        action[:3] = np.clip((target_pos - cur_pos) / output_max[:3], -1, 1)
        if target_mat is not None:
            cur_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
            ori_error = T.quat2axisangle(T.mat2quat(target_mat @ cur_mat.T))
            action[3:6] = np.clip(ori_error / output_max[3:6], -1, 1)
        action[-1] = gripper_action
        env.step(action)
        if render:
            env.render()


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


def pick(render=True):
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

    site_id = env.robots[0].eef_site_id["right"]
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    above_cube = cube_pos + np.array([0, 0, 0.15])

    # move above cube, gripper open
    go_to(env, site_id, output_max, above_cube, -1, 60, render)

    # align gripper yaw to the nearest face pair of the cube
    correction = nearest_face_correction(cube_yaw(env))
    start_mat = env.sim.data.site_xmat[site_id].copy().reshape(3, 3)
    target_mat = T.quat2mat(T.axisangle2quat(np.array([0, 0, correction]))) @ start_mat
    go_to(env, site_id, output_max, above_cube, -1, 40, render, target_mat=target_mat)

    # descend to grasp height, holding the aligned orientation
    go_to(env, site_id, output_max, cube_pos + np.array([0, 0, 0.005]), -1, 60, render)
    # close gripper
    go_to(env, site_id, output_max, cube_pos + np.array([0, 0, 0.005]), 1, 40, render)
    # lift
    go_to(env, site_id, output_max, cube_pos + np.array([0, 0, 0.2]), 1, 60, render)

    print(f"Cube yaw: {np.degrees(cube_yaw(env)):.1f} deg, alignment correction applied: {np.degrees(correction):.1f} deg")
    print("Pick succeeded:", env._check_success())
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without an on-screen viewer")
    args = parser.parse_args()
    pick(render=not args.headless)
