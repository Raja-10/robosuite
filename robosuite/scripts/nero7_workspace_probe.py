"""Sample Nero7 joint space and summarize its reachable end-effector workspace."""

import argparse
import pathlib

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--joint-margin",
        type=float,
        default=0.05,
        help="Fraction of each joint range excluded at both limits.",
    )
    parser.add_argument(
        "--minimum-z",
        type=float,
        default=0.0,
        help="Legacy lower world-frame z bound; combined with --z-range when supplied.",
    )
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Optional accepted world-frame end-effector x range.",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Optional accepted world-frame end-effector y range.",
    )
    parser.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Optional accepted world-frame end-effector z range.",
    )
    parser.add_argument(
        "--maximum-condition",
        type=float,
        default=1000.0,
        help="Only accept samples below this geometric Jacobian condition number.",
    )
    parser.add_argument(
        "--maximum-tilt-deg",
        type=float,
        default=30.0,
        help="Maximum angle between the gripper approach axis and world-down.",
    )
    parser.add_argument(
        "--allow-collisions",
        action="store_true",
        help="Do not reject samples in which the robot has active contacts.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional .npz output containing sampled joint and end-effector positions.",
    )
    return parser.parse_args()


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


def validate_range(name, bounds):
    if bounds is None:
        return None
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f"{name} must contain finite MIN MAX values with MIN <= MAX")
    return bounds


def value_in_range(value, bounds):
    return bounds is None or bounds[0] <= value <= bounds[1]


def robot_has_active_contact(env, robot_geom_ids):
    for contact in env.sim.data.contact:
        if contact.dist <= 0 and (contact.geom1 in robot_geom_ids or contact.geom2 in robot_geom_ids):
            return True
    return False


def sample_workspace(
    env,
    samples,
    seed,
    joint_margin,
    minimum_z,
    maximum_condition,
    maximum_tilt_deg,
    x_range=None,
    y_range=None,
    z_range=None,
    reject_collisions=True,
):
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if not 0 <= joint_margin < 0.5:
        raise ValueError("--joint-margin must be in [0, 0.5)")
    if maximum_condition <= 1:
        raise ValueError("--maximum-condition must be greater than 1")
    if not 0 <= maximum_tilt_deg <= 180:
        raise ValueError("--maximum-tilt-deg must be in [0, 180]")

    x_range = validate_range("--x-range", x_range)
    y_range = validate_range("--y-range", y_range)
    z_range = validate_range("--z-range", z_range)

    robot = env.robots[0]
    joint_ids = robot.arm_joint_indexes
    limits = env.sim.model.jnt_range[joint_ids].copy()
    span = limits[:, 1] - limits[:, 0]
    lower = limits[:, 0] + joint_margin * span
    upper = limits[:, 1] - joint_margin * span
    rng = np.random.default_rng(seed)

    original_qpos = robot._joint_positions.copy()
    sampled_qpos = np.empty((samples, len(joint_ids)))
    eef_positions = np.empty((samples, 3))
    jacobian_conditions = np.empty(samples)
    approach_alignments = np.empty(samples)
    collision_free = np.empty(samples, dtype=bool)
    accepted = np.zeros(samples, dtype=bool)
    eef_name = robot.gripper["right"].important_sites["grip_site"]
    qvel_indexes = robot._ref_joint_vel_indexes
    eef_site_id = robot.eef_site_id["right"]
    robot_geom_ids = {
        geom_id
        for geom_id in range(env.sim.model.ngeom)
        if (name := env.sim.model.geom_id2name(geom_id)) is not None
        and (name.startswith("robot0_") or name.startswith("gripper0_right_"))
    }
    minimum_alignment = np.cos(np.deg2rad(maximum_tilt_deg))

    try:
        for index in range(samples):
            qpos = rng.uniform(lower, upper)
            robot.set_robot_joint_positions(qpos)
            eef_pos = env.sim.data.site_xpos[eef_site_id].copy()
            eef_rotation = env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
            approach_axis = eef_rotation[:, 2]
            approach_alignment = float(np.dot(approach_axis, np.array([0.0, 0.0, -1.0])))
            jacobian = np.vstack(
                (
                    env.sim.data.get_site_jacp(eef_name)[:, qvel_indexes],
                    env.sim.data.get_site_jacr(eef_name)[:, qvel_indexes],
                )
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            condition = (
                float("inf")
                if singular_values[-1] <= np.finfo(float).eps
                else float(singular_values[0] / singular_values[-1])
            )

            sampled_qpos[index] = qpos
            eef_positions[index] = eef_pos
            jacobian_conditions[index] = condition
            approach_alignments[index] = approach_alignment
            collision_free[index] = not robot_has_active_contact(env, robot_geom_ids)
            accepted[index] = (
                eef_pos[2] >= minimum_z
                and value_in_range(eef_pos[0], x_range)
                and value_in_range(eef_pos[1], y_range)
                and value_in_range(eef_pos[2], z_range)
                and condition <= maximum_condition
                and approach_alignment >= minimum_alignment
                and (collision_free[index] or not reject_collisions)
            )
    finally:
        robot.set_robot_joint_positions(original_qpos)

    return (
        sampled_qpos,
        eef_positions,
        jacobian_conditions,
        approach_alignments,
        collision_free,
        accepted,
        limits,
    )


def main():
    args = parse_args()
    env = create_env()
    try:
        env.reset()
        (
            qpos,
            eef_positions,
            conditions,
            approach_alignments,
            collision_free,
            accepted,
            limits,
        ) = sample_workspace(
            env=env,
            samples=args.samples,
            seed=args.seed,
            joint_margin=args.joint_margin,
            minimum_z=args.minimum_z,
            maximum_condition=args.maximum_condition,
            maximum_tilt_deg=args.maximum_tilt_deg,
            x_range=args.x_range,
            y_range=args.y_range,
            z_range=args.z_range,
            reject_collisions=not args.allow_collisions,
        )
        accepted_positions = eef_positions[accepted]
        if len(accepted_positions) == 0:
            raise RuntimeError("No workspace samples passed the requested minimum z")

        print(f"seed: {args.seed}")
        print(f"samples: {args.samples}")
        print(f"accepted_samples: {len(accepted_positions)}")
        print(f"collision_free_samples: {int(collision_free.sum())}")
        print(f"joint_limits:\n{limits}")
        print(f"eef_min_xyz: {accepted_positions.min(axis=0)}")
        print(f"eef_max_xyz: {accepted_positions.max(axis=0)}")
        print(f"eef_mean_xyz: {accepted_positions.mean(axis=0)}")
        best_index = int(np.argmin(np.where(accepted, conditions, np.inf)))
        print(f"best_condition: {conditions[best_index]}")
        print(
            "best_approach_tilt_deg: "
            f"{np.rad2deg(np.arccos(np.clip(approach_alignments[best_index], -1.0, 1.0)))}"
        )
        print(f"best_qpos: {qpos[best_index]}")
        print(f"best_eef_xyz: {eef_positions[best_index]}")

        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                joint_positions=qpos,
                eef_positions=eef_positions,
                jacobian_conditions=conditions,
                approach_alignments=approach_alignments,
                collision_free=collision_free,
                accepted=accepted,
                joint_limits=limits,
                seed=args.seed,
                joint_margin=args.joint_margin,
                minimum_z=args.minimum_z,
                maximum_condition=args.maximum_condition,
                maximum_tilt_deg=args.maximum_tilt_deg,
                x_range=np.asarray(args.x_range) if args.x_range is not None else np.array([]),
                y_range=np.asarray(args.y_range) if args.y_range is not None else np.array([]),
                z_range=np.asarray(args.z_range) if args.z_range is not None else np.array([]),
                reject_collisions=not args.allow_collisions,
            )
            print(f"saved: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
