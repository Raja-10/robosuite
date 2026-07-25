"""Interactively tune and validate a Nero7 initial pose.

The GUI uses Tkinter sliders and a passive MuJoCo viewer. It also supports a
headless validation mode for CI and remote machines without a display.
"""

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

import mujoco.viewer
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


DEFAULT_CANDIDATE = np.array(
    [
        0.3590570800436392,
        0.1850950179686991,
        -0.38515021674470995,
        1.758171938003951,
        0.04206276630372575,
        -0.005008290725163654,
        0.8511248905816553,
    ]
)


@dataclass
class PoseMetrics:
    qpos: list
    eef_position: list
    downward_tilt_deg: float
    jacobian_condition: float
    collision_pairs: list
    finite: bool


@dataclass
class DynamicValidation:
    passed: bool
    steps: int
    maximum_tracking_error: float
    final_tracking_error: float
    maximum_velocity: float
    finite: bool
    collision_pairs: list


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs=7,
        type=float,
        default=DEFAULT_CANDIDATE,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("nero7_init_pose.json"),
        help="JSON path used by the Save button.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--tracking-tolerance", type=float, default=0.02)
    return parser.parse_args()


def create_env():
    return suite.make(
        env_name="Lift",
        robots="Nero7",
        controller_configs=load_composite_controller_config(robot="Nero7"),
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        ignore_done=True,
    )


def validate_qpos(robot, qpos):
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape != (7,):
        raise ValueError(f"Expected seven joint values, got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Candidate contains a non-finite joint value")
    limits = robot.sim.model.jnt_range[robot.arm_joint_indexes]
    if np.any(qpos < limits[:, 0]) or np.any(qpos > limits[:, 1]):
        raise ValueError(f"Candidate lies outside Nero7 joint limits: {qpos}")
    return qpos


def set_arm_state(env, qpos):
    robot = env.robots[0]
    qpos = validate_qpos(robot, qpos)
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = qpos
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    env.sim.forward()


def robot_geom_ids(env):
    return {
        geom_id
        for geom_id in range(env.sim.model.ngeom)
        if (name := env.sim.model.geom_id2name(geom_id)) is not None
        and (name.startswith("robot0_") or name.startswith("gripper0_right_"))
    }


def active_robot_collision_pairs(env):
    ids = robot_geom_ids(env)
    pairs = set()
    for contact in env.sim.data.contact:
        if contact.dist > 0 or (contact.geom1 not in ids and contact.geom2 not in ids):
            continue
        name1 = env.sim.model.geom_id2name(contact.geom1) or f"geom_{contact.geom1}"
        name2 = env.sim.model.geom_id2name(contact.geom2) or f"geom_{contact.geom2}"
        pairs.add(tuple(sorted((name1, name2))))
    return [list(pair) for pair in sorted(pairs)]


def pose_metrics(env):
    robot = env.robots[0]
    eef_site_id = robot.eef_site_id["right"]
    eef_name = robot.gripper["right"].important_sites["grip_site"]
    qvel_indexes = robot._ref_joint_vel_indexes

    rotation = env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
    approach_axis = rotation[:, 2]
    downward_alignment = np.clip(np.dot(approach_axis, [0.0, 0.0, -1.0]), -1.0, 1.0)
    tilt = float(np.rad2deg(np.arccos(downward_alignment)))

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

    return PoseMetrics(
        qpos=robot._joint_positions.tolist(),
        eef_position=env.sim.data.site_xpos[eef_site_id].tolist(),
        downward_tilt_deg=tilt,
        jacobian_condition=condition,
        collision_pairs=active_robot_collision_pairs(env),
        finite=bool(np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel))),
    )


def validate_dynamics(env, target_qpos, steps, tracking_tolerance, sync_callback=None):
    if steps <= 0:
        raise ValueError("settle steps must be positive")
    if tracking_tolerance <= 0:
        raise ValueError("tracking tolerance must be positive")

    target = validate_qpos(env.robots[0], target_qpos)
    maximum_error = 0.0
    maximum_velocity = 0.0
    observed_collisions = set()
    finite = True

    for _ in range(steps):
        action = np.empty(8)
        action[:7] = target
        action[7] = -1.0
        env.step(action)

        error = float(np.max(np.abs(env.robots[0]._joint_positions - target)))
        velocity = float(np.max(np.abs(env.robots[0]._joint_velocities)))
        maximum_error = max(maximum_error, error)
        maximum_velocity = max(maximum_velocity, velocity)
        observed_collisions.update(tuple(pair) for pair in active_robot_collision_pairs(env))
        finite = finite and bool(
            np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel))
        )
        if sync_callback is not None:
            sync_callback()
        if not finite:
            break

    final_error = float(np.max(np.abs(env.robots[0]._joint_positions - target)))
    collision_pairs = [list(pair) for pair in sorted(observed_collisions)]
    return DynamicValidation(
        passed=finite and final_error <= tracking_tolerance and not collision_pairs,
        steps=steps,
        maximum_tracking_error=maximum_error,
        final_tracking_error=final_error,
        maximum_velocity=maximum_velocity,
        finite=finite,
        collision_pairs=collision_pairs,
    )


def save_pose(path, metrics, dynamic_validation=None):
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot": "Nero7",
        "qpos": metrics.qpos,
        "pose_metrics": asdict(metrics),
        "dynamic_validation": asdict(dynamic_validation) if dynamic_validation is not None else None,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def print_metrics(metrics):
    print(f"qpos: {np.asarray(metrics.qpos)}")
    print(f"eef_position: {np.asarray(metrics.eef_position)}")
    print(f"downward_tilt_deg: {metrics.downward_tilt_deg}")
    print(f"jacobian_condition: {metrics.jacobian_condition}")
    print(f"collision_pairs: {metrics.collision_pairs}")
    print(f"finite: {metrics.finite}")


def run_headless(args):
    env = create_env()
    try:
        env.reset()
        set_arm_state(env, args.candidate)
        metrics = pose_metrics(env)
        validation = validate_dynamics(
            env,
            target_qpos=args.candidate,
            steps=args.settle_steps,
            tracking_tolerance=args.tracking_tolerance,
        )
        print_metrics(metrics)
        print(f"dynamic_validation: {asdict(validation)}")
        if not metrics.finite or metrics.collision_pairs or not validation.passed:
            raise RuntimeError("Candidate failed Nero7 simulation validation")
    finally:
        env.close()


class PoseTuner:
    def __init__(self, env, candidate, output, settle_steps, tracking_tolerance):
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.env = env
        self.robot = env.robots[0]
        self.output = output
        self.settle_steps = settle_steps
        self.tracking_tolerance = tracking_tolerance
        self.viewer = None
        self.closed = False
        self.last_validation = None
        self.candidate = validate_qpos(self.robot, candidate).copy()

        self.root = tk.Tk()
        self.root.title("Nero7 Initial Pose Tuner")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar()
        self.variables = []
        self.scales = []

        limits = env.sim.model.jnt_range[self.robot.arm_joint_indexes]
        for index, (lower, upper) in enumerate(limits):
            variable = tk.DoubleVar(value=float(self.candidate[index]))
            scale = tk.Scale(
                self.root,
                label=f"Joint {index + 1} [rad] ({lower:.3f}, {upper:.3f})",
                variable=variable,
                from_=float(lower),
                to=float(upper),
                resolution=0.001,
                orient=tk.HORIZONTAL,
                length=560,
                command=self.on_slider,
            )
            scale.pack(fill=tk.X, padx=10)
            self.variables.append(variable)
            self.scales.append(scale)

        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=10, pady=8)
        tk.Button(buttons, text="Reset candidate", command=self.reset_candidate).pack(side=tk.LEFT)
        tk.Button(buttons, text="Print pose", command=self.print_pose).pack(side=tk.LEFT)
        tk.Button(buttons, text="Test dynamics", command=self.test_dynamics).pack(side=tk.LEFT)
        tk.Button(buttons, text="Save pose", command=self.save).pack(side=tk.LEFT)
        tk.Button(buttons, text="Close", command=self.close).pack(side=tk.RIGHT)

        tk.Label(
            self.root,
            textvariable=self.status,
            justify=tk.LEFT,
            anchor="w",
            font=("TkFixedFont", 10),
        ).pack(fill=tk.X, padx=10, pady=8)

    def current_slider_qpos(self):
        return np.array([variable.get() for variable in self.variables])

    def on_slider(self, _value=None):
        if self.viewer is None:
            return
        with self.viewer.lock():
            set_arm_state(self.env, self.current_slider_qpos())
        self.viewer.sync()
        self.update_status()

    def update_status(self):
        metrics = pose_metrics(self.env)
        collision_text = "none" if not metrics.collision_pairs else str(metrics.collision_pairs)
        self.status.set(
            f"EEF xyz: {np.asarray(metrics.eef_position).round(4)}\n"
            f"Downward tilt: {metrics.downward_tilt_deg:.2f} deg\n"
            f"Jacobian condition: {metrics.jacobian_condition:.2f}\n"
            f"Active collisions: {collision_text}\n"
            f"Finite state: {metrics.finite}"
        )

    def reset_candidate(self):
        for variable, value in zip(self.variables, self.candidate):
            variable.set(float(value))
        self.on_slider()

    def print_pose(self):
        print_metrics(pose_metrics(self.env))

    def sync_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()
            self.root.update_idletasks()

    def test_dynamics(self):
        target = self.current_slider_qpos()
        self.last_validation = validate_dynamics(
            self.env,
            target_qpos=target,
            steps=self.settle_steps,
            tracking_tolerance=self.tracking_tolerance,
            sync_callback=self.sync_viewer,
        )
        for variable, value in zip(self.variables, self.robot._joint_positions):
            variable.set(float(value))
        self.update_status()
        print(f"dynamic_validation: {asdict(self.last_validation)}")
        if self.last_validation.passed:
            self.messagebox.showinfo("Nero7 validation", "Dynamic hold validation passed.")
        else:
            self.messagebox.showwarning(
                "Nero7 validation",
                "Validation failed. Check tracking error and collisions in the terminal.",
            )

    def save(self):
        metrics = pose_metrics(self.env)
        saved = save_pose(self.output, metrics, self.last_validation)
        print(f"saved: {saved}")
        self.messagebox.showinfo("Nero7 pose saved", str(saved))

    def poll(self):
        if self.closed:
            return
        if self.viewer is None or not self.viewer.is_running():
            self.close()
            return
        self.viewer.sync()
        self.root.after(30, self.poll)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.viewer is not None:
            self.viewer.close()
        self.root.destroy()

    def run(self):
        set_arm_state(self.env, self.candidate)
        self.viewer = mujoco.viewer.launch_passive(
            self.env.sim.model._model,
            self.env.sim.data._data,
            show_left_ui=True,
            show_right_ui=False,
        )
        self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -25
        self.viewer.sync()
        self.update_status()
        self.root.after(30, self.poll)
        self.root.mainloop()


def run_gui(args):
    env = create_env()
    try:
        env.reset()
        tuner = PoseTuner(
            env=env,
            candidate=args.candidate,
            output=args.output,
            settle_steps=args.settle_steps,
            tracking_tolerance=args.tracking_tolerance,
        )
        tuner.run()
    finally:
        env.close()


def main():
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
