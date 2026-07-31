"""Solve ShapePegInsertion with a smooth, observation-driven Nero7 policy.

The policy uses privileged object poses, making it suitable for simulation
validation and scripted demonstration collection. It executes, for every piece:

    hover -> descend -> grasp -> lift -> yaw-align -> transfer
    -> pre-insert -> insert -> release -> retreat

Run visually:

    python -m robosuite.scripts.scripted_nero7_shape_peg_insertion

Collect scripted demonstrations:

    python -m robosuite.scripts.scripted_nero7_shape_peg_insertion \
        --episodes 20 --headless --record-dir ~/nero7_scripted_demos
"""

import argparse
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.objects.shape_sorter import PEG_HEIGHT, PIECE_HALF_THICKNESS
from robosuite.wrappers import DataCollectionWrapper


SHAPES = ("circle", "square", "triangle")
POSITION_ACTION_SCALE = 0.025
ORIENTATION_ACTION_SCALE = 0.25
OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0


def osc_config():
    path = (
        pathlib.Path(suite.__file__).parent
        / "controllers"
        / "config"
        / "robots"
        / "nero7_osc_pose.json"
    )
    return load_composite_controller_config(controller=str(path))


def create_env(render=True, camera="free", record_dir=None):
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
        horizon=10000,
        hard_reset=False,
        ignore_done=True,
    )
    if record_dir:
        env = DataCollectionWrapper(
            env,
            directory=str(pathlib.Path(record_dir).expanduser()),
            collect_freq=1,
            flush_freq=100,
        )
    return env


def current_eef_pose(env):
    site_id = env.robots[0].eef_site_id["right"]
    position = env.sim.data.site_xpos[site_id].copy()
    orientation = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    return position, orientation


def object_pose(env, shape):
    body_id = env.piece_body_ids[shape]
    return (
        env.sim.data.body_xpos[body_id].copy(),
        env.sim.data.body_xmat[body_id].reshape(3, 3).copy(),
    )


def object_yaw(env, shape):
    return float(Rotation.from_matrix(object_pose(env, shape)[1]).as_euler("xyz")[2])


def canonical_yaw_error(env, shape):
    """Returns the shortest yaw correction accepted by the target pattern."""
    yaw = object_yaw(env, shape)
    if shape == "circle":
        return 0.0
    period = np.pi if shape in {"square", "rectangle"} else 2.0 * np.pi
    return float((yaw + period / 2.0) % period - period / 2.0)


def top_down_orientation(yaw=np.pi):
    """Returns a vertical tool pose on Nero7's saved-pose wrist branch."""
    return (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", np.pi).as_matrix()
    )


def shape_grasp_orientation(env, shape):
    """Aligns the parallel jaws with a stable axis of the current piece."""
    if shape == "circle":
        grasp_yaw = np.pi
    else:
        yaw = canonical_yaw_error(env, shape)
        # Piper closes along tool Y. For the triangle this runs one pad toward
        # the apex and the opposing pad toward the centre of the opposite base.
        # Rectangle is π-symmetric, so use the equivalent wrist branch that is
        # easier for Nero7 to reach after servicing the triangle station.
        grasp_yaw = yaw if shape == "rectangle" else np.pi + yaw
    return top_down_orientation(grasp_yaw)


def pose_action(
    env,
    target_position,
    target_orientation,
    gripper_command,
    max_translation_action=0.70,
    max_rotation_action=0.40,
):
    """Creates a bounded OSC action; bounds make waypoint motion visibly smooth."""
    position, orientation = current_eef_pose(env)
    position_error = np.asarray(target_position) - position
    orientation_error = Rotation.from_matrix(
        np.asarray(target_orientation) @ orientation.T
    ).as_rotvec()
    action = np.empty(7, dtype=float)
    action[:3] = np.clip(
        position_error / POSITION_ACTION_SCALE,
        -max_translation_action,
        max_translation_action,
    )
    action[3:6] = np.clip(
        orientation_error / ORIENTATION_ACTION_SCALE,
        -max_rotation_action,
        max_rotation_action,
    )
    action[6] = gripper_command
    return action, position_error, orientation_error


def simulation_is_finite(env, action):
    return bool(
        np.all(np.isfinite(action))
        and np.all(np.isfinite(env.sim.data.qpos))
        and np.all(np.isfinite(env.sim.data.qvel))
        and np.all(np.isfinite(env.sim.data.ctrl))
    )


class ScriptedShapeInsertionPolicy:
    """Waypoint state machine for privileged scripted data generation."""

    def __init__(
        self,
        env,
        render=True,
        realtime=True,
        approach_height=0.08,
        transfer_height=0.14,
        grasp_offset_z=0.0,
        close_steps=12,
        release_steps=10,
        settle_steps=20,
        max_motion_steps=700,
        grasp_attempts=3,
    ):
        self.env = env
        self.render = render
        self.realtime = realtime and render
        self.approach_height = approach_height
        self.transfer_height = transfer_height
        self.grasp_offset_z = grasp_offset_z
        self.close_steps = close_steps
        self.release_steps = release_steps
        self.settle_steps = settle_steps
        self.max_motion_steps = max_motion_steps
        self.grasp_attempts = grasp_attempts
        self.control_period = 1.0 / env.control_freq
        self.last_observation = None
        self.last_info = None
        self.last_reward = 0.0

    def step(self, action):
        start = time.monotonic()
        self.last_observation, self.last_reward, _, self.last_info = self.env.step(
            action
        )
        if not simulation_is_finite(self.env, action):
            raise RuntimeError("Scripted Nero7 policy produced a non-finite state")
        if self.render:
            self.env.render()
        if self.realtime:
            delay = self.control_period - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)

    def move_to(
        self,
        phase,
        position,
        orientation,
        gripper,
        position_tolerance=0.008,
        orientation_tolerance=0.05,
        max_steps=None,
        max_translation_action=0.70,
        max_rotation_action=0.40,
    ):
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        max_steps = max_steps or self.max_motion_steps
        for step in range(1, max_steps + 1):
            action, position_error, orientation_error = pose_action(
                self.env,
                position,
                orientation,
                gripper,
                max_translation_action=max_translation_action,
                max_rotation_action=max_rotation_action,
            )
            if (
                np.linalg.norm(position_error) <= position_tolerance
                and np.linalg.norm(orientation_error) <= orientation_tolerance
            ):
                return {
                    "phase": phase,
                    "reached": True,
                    "steps": step - 1,
                    "position_error": float(np.linalg.norm(position_error)),
                    "orientation_error": float(np.linalg.norm(orientation_error)),
                }
            self.step(action)
        _, position_error, orientation_error = pose_action(
            self.env,
            position,
            orientation,
            gripper,
            max_translation_action=max_translation_action,
            max_rotation_action=max_rotation_action,
        )
        return {
            "phase": phase,
            "reached": False,
            "steps": max_steps,
            "position_error": float(np.linalg.norm(position_error)),
            "orientation_error": float(np.linalg.norm(orientation_error)),
        }

    def hold(self, position, orientation, gripper, steps):
        for _ in range(steps):
            action, _, _ = pose_action(
                self.env, position, orientation, gripper
            )
            self.step(action)

    def hold_and_detect_grasp(
        self,
        shape,
        position,
        orientation,
        steps,
    ):
        """Holds closed and latches transient valid opposing-pad contact."""
        grasp_detected = False
        for _ in range(steps):
            action, _, _ = pose_action(
                self.env,
                position,
                orientation,
                CLOSE_GRIPPER,
            )
            self.step(action)
            grasp_detected = grasp_detected or self.is_grasped(shape)
        return grasp_detected

    def release_until_clear(self, shape, position, orientation, max_steps=25):
        """Opens for the actuator minimum, then stops once grasp contact clears."""
        for step in range(1, max_steps + 1):
            action, _, _ = pose_action(
                self.env, position, orientation, OPEN_GRIPPER
            )
            self.step(action)
            if step >= self.release_steps and not self.is_grasped(shape):
                return step
        return max_steps

    def move_piece_to(
        self,
        phase,
        shape,
        piece_target,
        orientation,
        position_tolerance=0.005,
        max_steps=None,
        max_translation_action=0.50,
        servo_gain=2.0,
        command_bias=None,
        integral_gain=0.0,
    ):
        """Servos on object error, compensating small slip inside the fingers."""
        piece_target = np.asarray(piece_target, dtype=float)
        if command_bias is None:
            command_bias = np.zeros(3, dtype=float)
        max_steps = max_steps or self.max_motion_steps
        for step in range(1, max_steps + 1):
            piece_position, _ = object_pose(self.env, shape)
            piece_error = piece_target - piece_position
            if np.linalg.norm(piece_error) <= position_tolerance:
                return {
                    "phase": phase,
                    "reached": True,
                    "steps": step - 1,
                    "position_error": float(np.linalg.norm(piece_error)),
                    "command_bias": command_bias.copy(),
                }
            if not self.is_grasped(shape):
                return {
                    "phase": phase,
                    "reached": False,
                    "lost_grasp": True,
                    "steps": step - 1,
                    "position_error": float(np.linalg.norm(piece_error)),
                    "command_bias": command_bias.copy(),
                }
            eef_position, _ = current_eef_pose(self.env)
            command_bias += np.asarray(integral_gain) * piece_error
            command_bias[:] = np.clip(
                command_bias,
                [-0.02, -0.02, -0.015],
                [0.02, 0.02, 0.015],
            )
            action, _, _ = pose_action(
                self.env,
                eef_position + servo_gain * piece_error + command_bias,
                orientation,
                CLOSE_GRIPPER,
                max_translation_action=max_translation_action,
            )
            self.step(action)
        piece_error = piece_target - object_pose(self.env, shape)[0]
        return {
            "phase": phase,
            "reached": False,
            "steps": max_steps,
            "position_error": float(np.linalg.norm(piece_error)),
            "command_bias": command_bias.copy(),
        }

    def align_piece_orientation(self, shape, max_steps=None, tolerance=0.03):
        """Levels the piece and aligns yaw while accounting for grasp slip."""
        max_steps = max_steps or self.max_motion_steps
        for step in range(1, max_steps + 1):
            _, piece_orientation = object_pose(self.env, shape)
            yaw = object_yaw(self.env, shape)
            if shape == "circle":
                target_yaw = yaw
            elif shape in {"square", "rectangle"}:
                target_yaw = np.round(yaw / np.pi) * np.pi
            else:
                target_yaw = 0.0
            target_piece_orientation = Rotation.from_euler(
                "z", target_yaw
            ).as_matrix()
            correction = target_piece_orientation @ piece_orientation.T
            orientation_error = Rotation.from_matrix(correction).as_rotvec()
            error_norm = float(np.linalg.norm(orientation_error))
            if error_norm <= tolerance:
                return {
                    "phase": "orientation_align",
                    "reached": True,
                    "steps": step - 1,
                    "orientation_error": error_norm,
                    "yaw_error": abs(canonical_yaw_error(self.env, shape)),
                }
            if not self.is_grasped(shape):
                return {
                    "phase": "orientation_align",
                    "reached": False,
                    "lost_grasp": True,
                    "steps": step - 1,
                    "orientation_error": error_norm,
                }
            eef_position, eef_orientation = current_eef_pose(self.env)
            target_orientation = correction @ eef_orientation
            action, _, _ = pose_action(
                self.env,
                eef_position,
                target_orientation,
                CLOSE_GRIPPER,
                max_translation_action=0.20,
                max_rotation_action=0.40,
            )
            self.step(action)
        _, piece_orientation = object_pose(self.env, shape)
        tilt_and_yaw = self.env._piece_metrics(shape)
        return {
            "phase": "orientation_align",
            "reached": False,
            "steps": max_steps,
            "orientation_error": float(
                max(tilt_and_yaw["tilt_rad"], tilt_and_yaw["yaw_error_rad"])
            ),
        }

    def is_grasped(self, shape):
        return bool(
            self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=self.env.pieces[shape],
            )
        )

    def acquire(self, shape, tool_orientation):
        """Grasps a piece, retrying around the measured EEF tracking residual."""
        phases = []
        command_correction = np.zeros(3)
        grasp_position_tolerance = {
            "circle": 0.015,
            "triangle": 0.006,
        }.get(shape, 0.004)
        close_steps = 12 if shape == "circle" else 18
        for attempt in range(1, self.grasp_attempts + 1):
            piece_position, _ = object_pose(self.env, shape)
            grasp_z = self.grasp_offset_z + {
                "circle": -0.004,
                # Place the pad centres slightly below the piece centre so
                # both fingers overlap the thin triangle before closing.
                "triangle": -0.002,
                # Ensure the inner pad faces overlap the thin rectangle before
                # the fingers close.
                "rectangle": -0.004,
            }.get(shape, 0.0)
            grasp_xy_offset = np.zeros(2)
            if shape == "triangle":
                yaw = object_yaw(self.env, shape)
                grasp_xy_offset = (
                    Rotation.from_euler("z", yaw).as_matrix()
                    # Centre the jaws between a finite-width point just below
                    # the apex and the opposite base. This realizes the
                    # vertex-to-base grasp without asking a pad to contact the
                    # zero-width mathematical tip.
                    @ np.array([0.0, -0.005, 0.0])
                )[:2]
            desired_grasp = piece_position + np.array(
                [
                    grasp_xy_offset[0],
                    grasp_xy_offset[1],
                    grasp_z,
                ]
            )
            commanded_grasp = desired_grasp + command_correction
            approach = commanded_grasp + np.array(
                [0.0, 0.0, self.approach_height]
            )
            approach_result = self.move_to(
                f"approach_{attempt}",
                approach,
                tool_orientation,
                OPEN_GRIPPER,
                # Hover is only a safe staging waypoint. Precise centering is
                # enforced below at grasp height before the fingers may close.
                position_tolerance=0.025,
                orientation_tolerance=0.30 if shape == "rectangle" else 0.18,
                max_steps=180,
                max_translation_action=0.50,
            )
            phases.append(approach_result)
            if not approach_result["reached"]:
                continue
            descend_result = self.move_to(
                f"descend_{attempt}",
                commanded_grasp,
                tool_orientation,
                OPEN_GRIPPER,
                position_tolerance=grasp_position_tolerance,
                orientation_tolerance=0.08,
                max_steps=120,
                max_translation_action=0.50 if shape == "circle" else 0.25,
            )
            phases.append(descend_result)

            # Never close at a timed-out Cartesian pose. Correct the measured
            # tracking residual in place with open fingers, avoiding a false
            # grasp followed by a full retreat-and-retry cycle.
            centered = shape == "circle" or bool(descend_result["reached"])
            for refinement in range(1, 3) if shape != "circle" else ():
                piece_position, _ = object_pose(self.env, shape)
                desired_grasp = piece_position + np.array(
                    [
                        grasp_xy_offset[0],
                        grasp_xy_offset[1],
                        grasp_z,
                    ]
                )
                eef_position, _ = current_eef_pose(self.env)
                physical_error = desired_grasp - eef_position
                if np.linalg.norm(physical_error) <= grasp_position_tolerance:
                    centered = True
                    break
                # Nero's delta OSC retains a workspace-dependent offset whose
                # command-space correction has the opposite sign here.
                command_correction -= physical_error
                command_correction[:] = np.clip(
                    command_correction,
                    [-0.025, -0.025, -0.015],
                    [0.025, 0.025, 0.015],
                )
                commanded_grasp = desired_grasp + command_correction
                phases.append(
                    self.move_to(
                        f"grasp_refine_{attempt}_{refinement}",
                        commanded_grasp,
                        tool_orientation,
                        OPEN_GRIPPER,
                        position_tolerance=grasp_position_tolerance,
                        orientation_tolerance=0.08,
                        max_steps=50,
                        max_translation_action=0.20,
                    )
                )

            if not centered:
                piece_position, _ = object_pose(self.env, shape)
                desired_grasp = piece_position + np.array(
                    [
                        grasp_xy_offset[0],
                        grasp_xy_offset[1],
                        grasp_z,
                    ]
                )
                final_grasp_tolerance = (
                    0.010 if shape == "triangle" else grasp_position_tolerance
                )
                centered = bool(
                    np.linalg.norm(
                        desired_grasp - current_eef_pose(self.env)[0]
                    )
                    <= final_grasp_tolerance
                )

            if not centered:
                phases.append(
                    {
                        "phase": f"grasp_gate_{attempt}",
                        "reached": False,
                        "position_error": float(
                            np.linalg.norm(
                                desired_grasp - current_eef_pose(self.env)[0]
                            )
                        ),
                        "position_tolerance": final_grasp_tolerance,
                    }
                )
                eef_position, _ = current_eef_pose(self.env)
                retry_hover = eef_position.copy()
                retry_hover[2] = max(
                    retry_hover[2] + 0.04,
                    piece_position[2] + self.approach_height,
                )
                self.move_to(
                    f"retry_retreat_{attempt}",
                    retry_hover,
                    tool_orientation,
                    OPEN_GRIPPER,
                    max_steps=60,
                )
                continue

            grasp_detected = self.hold_and_detect_grasp(
                shape,
                commanded_grasp,
                tool_orientation,
                close_steps,
            )
            if shape not in {"circle", "rectangle"} and (
                grasp_detected or self.is_grasped(shape)
            ):
                return True, phases

            # Contact can take a few simulation steps to settle on polygonal
            # pieces. Keep the fingers closed and stationary before declaring
            # the grasp failed; never combine opening with an upward motion.
            if shape != "circle":
                grasp_detected = self.hold_and_detect_grasp(
                    shape,
                    commanded_grasp,
                    tool_orientation,
                    6,
                )
                if shape != "rectangle" and (
                    grasp_detected or self.is_grasped(shape)
                ):
                    return True, phases
                if shape in {"triangle", "rectangle"}:
                    piece_z = object_pose(self.env, shape)[0][2]
                    probe_position = current_eef_pose(self.env)[0] + np.array(
                        [0.0, 0.0, 0.030]
                    )
                    probe_result = self.move_to(
                        f"grasp_probe_{attempt}",
                        probe_position,
                        tool_orientation,
                        CLOSE_GRIPPER,
                        position_tolerance=0.006,
                        orientation_tolerance=0.10,
                        max_steps=60,
                        max_translation_action=0.35,
                    )
                    piece_lift = (
                        object_pose(self.env, shape)[0][2] - piece_z
                    )
                    probe_result["object_lift"] = float(piece_lift)
                    phases.append(probe_result)
                    required_probe_lift = (
                        0.010 if shape == "rectangle" else 0.020
                    )
                    if piece_lift >= required_probe_lift:
                        return True, phases
            else:
                # Circle's segmented ring can make the instantaneous contact
                # predicate flicker. Verify the actual physical outcome with
                # a short closed-jaw lift before deciding to retry.
                circle_z = object_pose(self.env, shape)[0][2]
                probe_position = current_eef_pose(self.env)[0] + np.array(
                    [0.0, 0.0, 0.030]
                )
                phases.append(
                    self.move_to(
                        f"grasp_probe_{attempt}",
                        probe_position,
                        tool_orientation,
                        CLOSE_GRIPPER,
                        position_tolerance=0.006,
                        orientation_tolerance=0.10,
                        max_steps=60,
                        max_translation_action=0.35,
                    )
                )
                circle_lift = object_pose(self.env, shape)[0][2] - circle_z
                if circle_lift >= 0.020:
                    return True, phases

            self.hold(
                commanded_grasp,
                tool_orientation,
                OPEN_GRIPPER,
                self.release_steps,
            )

            eef_position, _ = current_eef_pose(self.env)
            piece_position, _ = object_pose(self.env, shape)
            command_correction -= piece_position - eef_position
            command_correction[:] = np.clip(
                command_correction,
                [-0.025, -0.025, -0.015],
                [0.025, 0.025, 0.015],
            )
            retry_hover = eef_position.copy()
            retry_hover[2] = max(
                retry_hover[2] + 0.06,
                piece_position[2] + self.approach_height,
            )
            self.move_to(
                f"retry_retreat_{attempt}",
                retry_hover,
                tool_orientation,
                OPEN_GRIPPER,
            )
        return False, phases

    def place_shape(self, shape):
        print(f"\n[{shape}] approaching piece")
        phases = []
        grasp_orientation = shape_grasp_orientation(self.env, shape)
        acquired, acquire_phases = self.acquire(shape, grasp_orientation)
        phases.extend(acquire_phases)
        if not acquired:
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "grasp",
                "phases": phases,
            }

        piece_position, _ = object_pose(self.env, shape)
        lift_eef = current_eef_pose(self.env)[0]
        lift_eef[2] = piece_position[2] + self.transfer_height
        lift = self.move_to(
            "lift",
            lift_eef,
            grasp_orientation,
            CLOSE_GRIPPER,
            position_tolerance=0.015,
            orientation_tolerance=0.12,
            max_steps=100,
            max_translation_action=1.0,
        )
        phases.append(lift)
        if not self.is_grasped(shape):
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "lift",
                "phases": phases,
            }

        # Servo on the piece itself rather than assuming a perfectly rigid
        # finger-to-object transform.
        yaw_tolerance = {
            "circle": 0.03,
            "square": 0.02,
            "triangle": 0.006,
            "rectangle": 0.006,
        }[shape]
        if shape == "circle":
            align = {
                "phase": "orientation_align",
                "reached": True,
                "steps": 0,
                "orientation_error": 0.0,
                "yaw_error": 0.0,
            }
        else:
            align = self.align_piece_orientation(
                shape,
                tolerance=yaw_tolerance,
            )
        phases.append(align)
        if not align["reached"] or not self.is_grasped(shape):
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "orientation_align",
                "phases": phases,
            }
        aligned_orientation = current_eef_pose(self.env)[1]
        if shape == "rectangle":
            # The π-equivalent branch makes the right-side pickup reachable,
            # but the opposite branch is better conditioned for transport to
            # the board. Rebranch at lift height while retaining the grasp.
            rebranched_orientation = (
                Rotation.from_euler("z", np.pi).as_matrix()
                @ aligned_orientation
            )
            rebranch = self.move_to(
                "rectangle_wrist_rebranch",
                current_eef_pose(self.env)[0],
                rebranched_orientation,
                CLOSE_GRIPPER,
                position_tolerance=0.015,
                orientation_tolerance=0.08,
                max_steps=250,
                max_translation_action=0.20,
                max_rotation_action=0.40,
            )
            phases.append(rebranch)
            if not rebranch["reached"] or not self.is_grasped(shape):
                return {
                    "shape": shape,
                    "success": False,
                    "failed_phase": "rectangle_wrist_rebranch",
                    "phases": phases,
                }
            aligned_orientation = current_eef_pose(self.env)[1]

        target = self.last_observation[f"{shape}_target_pos"].copy()

        transfer_piece = target + np.array([0.0, 0.0, self.transfer_height])
        preinsert_piece = target + np.array([0.0, 0.0, PEG_HEIGHT + 0.018])
        insertion_bias = np.zeros(3, dtype=float)
        transfer = self.move_piece_to(
            "transfer",
            shape,
            transfer_piece,
            aligned_orientation,
        )
        phases.append(transfer)
        # The high transit waypoint is deliberately approximate; precise
        # convergence is enforced at pre-insert after the arm has descended
        # into its better-conditioned workspace.
        if not self.is_grasped(shape):
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "transfer",
                "phases": phases,
            }
        preinsert_tolerance = 0.0012 if shape == "triangle" else 0.0008
        preinsert = self.move_piece_to(
            "preinsert",
            shape,
            preinsert_piece,
            aligned_orientation,
            position_tolerance=preinsert_tolerance,
            max_translation_action=0.25,
            command_bias=insertion_bias,
            integral_gain=0.01,
        )
        phases.append(preinsert)
        if not preinsert["reached"] or not self.is_grasped(shape):
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "preinsert",
                "phases": phases,
            }
        if shape == "rectangle":
            insertion_realign = self.align_piece_orientation(
                shape,
                max_steps=180,
                tolerance=yaw_tolerance,
            )
            insertion_realign["phase"] = "preinsert_orientation_realign"
            phases.append(insertion_realign)
            if (
                not insertion_realign["reached"]
                or not self.is_grasped(shape)
            ):
                return {
                    "shape": shape,
                    "success": False,
                    "failed_phase": "preinsert_orientation_realign",
                    "phases": phases,
                }
            aligned_orientation = current_eef_pose(self.env)[1]
        # Vertical steady-state bias is useful in free space but would become
        # excessive insertion force at contact. Preserve only XY calibration.
        insertion_bias[2] = 0.0
        contact_clearance = 0.001 if shape == "triangle" else 0.005
        contact_piece = target + np.array(
            [0.0, 0.0, PEG_HEIGHT + contact_clearance]
        )
        contact = self.move_piece_to(
            "contact_approach",
            shape,
            contact_piece,
            aligned_orientation,
            position_tolerance=preinsert_tolerance,
            max_translation_action=0.12,
            command_bias=insertion_bias,
            integral_gain=np.array([0.002, 0.002, 0.0]),
        )
        phases.append(contact)
        if not contact["reached"] or not self.is_grasped(shape):
            return {
                "shape": shape,
                "success": False,
                "failed_phase": "contact_approach",
                "phases": phases,
            }
        insertion_bias[2] = 0.0
        # Engage the holes by 4 mm, then let gravity complete the slide. This
        # avoids forcing the arm through the low-Z portion of the trajectory.
        engagement_depth = 0.004
        engaged_piece = target + np.array(
            [
                0.0,
                0.0,
                PEG_HEIGHT - PIECE_HALF_THICKNESS - engagement_depth,
            ]
        )
        insert = self.move_piece_to(
            "peg_engage",
            shape,
            engaged_piece,
            aligned_orientation,
            position_tolerance=0.0012,
            max_steps=self.max_motion_steps,
            max_translation_action=0.05,
            command_bias=insertion_bias,
            integral_gain=np.array([0.001, 0.001, 0.0]),
        )
        phases.append(insert)
        if not insert["reached"]:
            metrics = self.env._piece_metrics(shape)
            handoff_xy_tolerance = 0.0032 if shape == "triangle" else 0.002
            handoff_tilt_tolerance = 0.18 if shape == "triangle" else 0.05
            insert["handoff_xy_error"] = metrics[
                "maximum_hole_to_peg_xy_error"
            ]
            insert["handoff_tilt"] = metrics["tilt_rad"]
            insert["handoff_yaw_error"] = metrics["yaw_error_rad"]
            gravity_handoff = bool(
                insert.get("lost_grasp", False)
                and metrics["maximum_hole_to_peg_xy_error"]
                <= handoff_xy_tolerance
                and metrics["tilt_rad"] <= handoff_tilt_tolerance
                and metrics["yaw_error_rad"] <= 0.05
            )
            if not gravity_handoff:
                return {
                    "shape": shape,
                    "success": False,
                    "failed_phase": "peg_engage",
                    "phases": phases,
                }

        release_eef = current_eef_pose(self.env)[0]
        release_steps = self.release_until_clear(
            shape,
            release_eef,
            aligned_orientation,
        )
        phases.append(
            {
                "phase": "release",
                "reached": not self.is_grasped(shape),
                "steps": release_steps,
            }
        )
        # Keep the open gripper stationary while the engaged piece slides to
        # its seated pose, then retreat without an additional idle hold.
        self.hold(
            release_eef,
            aligned_orientation,
            OPEN_GRIPPER,
            self.settle_steps,
        )
        retreat = release_eef + np.array(
            [0.0, 0.0, self.approach_height]
        )
        phases.append(
            self.move_to(
                "retreat",
                retreat,
                aligned_orientation,
                OPEN_GRIPPER,
            )
        )
        metrics = self.env._piece_metrics(shape)
        seated = self.env._piece_is_stably_seated(shape)
        print(
            f"[{shape}] seated={seated} "
            f"xy_error={metrics['maximum_hole_to_peg_xy_error']:.4f} m "
            f"z_error={metrics['center_to_seated_z_error']:.4f} m "
            f"yaw_error={metrics['yaw_error_rad']:.3f} rad"
        )
        return {
            "shape": shape,
            "success": bool(seated),
            "failed_phase": None if seated else "insertion",
            "metrics": metrics,
            "phases": phases,
        }

    def solve(self, shape_order=SHAPES):
        results = []
        for shape in shape_order:
            result = self.place_shape(shape)
            results.append(result)
            if not result["success"]:
                break
        return {
            "success": bool(
                len(results) == len(shape_order)
                and all(result["success"] for result in results)
                and all(
                    self.env._piece_is_stably_seated(shape)
                    for shape in shape_order
                )
            ),
            "results": results,
            "reward": float(self.last_reward),
        }


def viewer_is_running(env):
    renderer = getattr(env, "viewer", None)
    viewer = getattr(renderer, "viewer", None)
    return viewer is None or viewer.is_running()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--shape-order",
        nargs="+",
        choices=SHAPES,
        default=list(SHAPES),
    )
    parser.add_argument(
        "--camera",
        choices=("free", "agentview", "taskview", "topview", "frontview"),
        default="free",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--approach-height", type=float, default=0.08)
    parser.add_argument("--transfer-height", type=float, default=0.14)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--grasp-attempts", type=int, default=3)
    parser.add_argument("--max-motion-steps", type=int, default=700)
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the interactive viewer open after the final episode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.grasp_attempts <= 0:
        raise ValueError("--grasp-attempts must be positive")
    if args.max_motion_steps <= 0:
        raise ValueError("--max-motion-steps must be positive")

    np.random.seed(args.seed)
    render = not args.headless
    env = create_env(
        render=render,
        camera=args.camera,
        record_dir=args.record_dir,
    )
    episode_results = []
    try:
        for episode in range(args.episodes):
            observations = env.reset()
            if render:
                env.render()
            policy = ScriptedShapeInsertionPolicy(
                env,
                render=render,
                realtime=not args.no_realtime,
                approach_height=args.approach_height,
                transfer_height=args.transfer_height,
                grasp_offset_z=args.grasp_offset_z,
                grasp_attempts=args.grasp_attempts,
                max_motion_steps=args.max_motion_steps,
            )
            policy.last_observation = observations
            print(f"\nEpisode {episode + 1}/{args.episodes}, seed={args.seed}")
            result = policy.solve(tuple(args.shape_order))
            episode_results.append(result)
            print(
                f"episode_success={result['success']} "
                f"final_reward={result['reward']:.3f}"
            )
            for shape_result in result["results"]:
                if shape_result["success"]:
                    continue
                print(
                    f"failed_shape={shape_result['shape']} "
                    f"failed_phase={shape_result['failed_phase']}"
                )
                for phase in shape_result["phases"]:
                    print(f"  {phase}")

        successes = sum(result["success"] for result in episode_results)
        print(f"\ncompleted: {successes}/{args.episodes} successful episodes")
        if render and args.keep_open:
            print("Inspection mode: close the MuJoCo window or press Ctrl+C.")
            while viewer_is_running(env):
                env.render()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nScripted simulation interrupted.")
    finally:
        env.close()

    if not all(result["success"] for result in episode_results):
        raise RuntimeError(
            "The scripted policy did not complete every requested episode; "
            "inspect the per-shape phase report above."
        )


if __name__ == "__main__":
    main()
