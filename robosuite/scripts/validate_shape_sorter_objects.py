"""Validate the primitive circle piece and one-peg shape-sorter board."""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.models import MujocoWorldBase
from robosuite.models.objects.shape_sorter import (
    BOARD_HALF_SIZE,
    PIECE_HALF_THICKNESS,
    SHAPE_SPECS,
    TARGET_STATIONS,
    CirclePiece,
    RectanglePiece,
    ShapeSorterBoard,
    SquarePiece,
    TrianglePiece,
)
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import array_to_string, new_geom


PIECE_FACTORIES = {
    "circle": CirclePiece,
    "square": SquarePiece,
    "triangle": TrianglePiece,
    "rectangle": RectanglePiece,
}
MAX_SAFE_DROP_HEIGHT = 0.20
REQUIRED_STABLE_STEPS = 50
SPAWN_SLOTS = np.array(
    [
        [-0.20, -0.19],
        [-0.07, -0.19],
        [0.07, -0.19],
        [0.20, -0.19],
    ]
)


def add_support_surface(world):
    """Adds a table-like surface around the sorter board for shuffled layouts."""
    world.worldbody.append(
        new_geom(
            name="shape_sorter_support",
            type="box",
            size=(0.30, 0.27, 0.01),
            pos=(0.0, 0.0, -0.01),
            rgba=(0.32, 0.30, 0.26, 1.0),
            friction=(1.0, 0.005, 0.0001),
            condim=4,
        )
    )


def build_multi_sim(
    shapes=tuple(PIECE_FACTORIES),
    drop_height=0.06,
    layout="aligned",
    seed=0,
):
    unknown = set(shapes) - set(PIECE_FACTORIES)
    if unknown:
        raise ValueError(f"Unsupported shapes: {sorted(unknown)}")
    if layout not in {"aligned", "shuffled"}:
        raise ValueError(f"Unsupported layout: {layout}")

    board = ShapeSorterBoard(stations=shapes)
    pieces = {shape: PIECE_FACTORIES[shape](name=f"{shape}_piece") for shape in shapes}
    world = MujocoWorldBase()
    add_support_surface(world)

    board_body = board.get_obj()
    board_body.set("pos", "0 0 0")
    world.worldbody.append(board_body)
    world.merge_assets(board)

    rng = np.random.default_rng(seed)
    shuffled_slots = rng.permutation(SPAWN_SLOTS)[: len(shapes)]
    for index, (shape, piece) in enumerate(pieces.items()):
        if layout == "aligned":
            piece_xy = TARGET_STATIONS[shape]
            piece_z = (
                2.0 * BOARD_HALF_SIZE[2]
                + PIECE_HALF_THICKNESS
                + drop_height
            )
            yaw = 0.0
        else:
            piece_xy = shuffled_slots[index] + rng.uniform(-0.005, 0.005, size=2)
            piece_z = PIECE_HALF_THICKNESS + drop_height
            yaw = rng.uniform(-np.pi, np.pi)
        piece_body = piece.get_obj()
        piece_body.set(
            "pos",
            array_to_string(
                np.array(
                    [
                        piece_xy[0],
                        piece_xy[1],
                        piece_z,
                    ]
                )
            ),
        )
        piece_body.set(
            "quat",
            array_to_string(
                Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
            ),
        )
        world.worldbody.append(piece_body)
        world.merge_assets(piece)

    return MjSim(world.get_model()), board, pieces


def build_sim(
    shape="circle",
    drop_height=0.06,
    lateral_offset=(0.0, 0.0),
    yaw=0.0,
):
    if shape not in PIECE_FACTORIES:
        raise ValueError(f"Unsupported shape: {shape}")
    board = ShapeSorterBoard(stations=(shape,))
    piece = PIECE_FACTORIES[shape]()
    world = MujocoWorldBase()
    add_support_surface(world)

    board_body = board.get_obj()
    board_body.set("pos", "0 0 0")
    world.worldbody.append(board_body)
    world.merge_assets(board)

    station_xy = TARGET_STATIONS[shape]
    piece_body = piece.get_obj()
    piece_body.set(
        "pos",
        array_to_string(
            np.array(
                [
                    station_xy[0] + lateral_offset[0],
                    station_xy[1] + lateral_offset[1],
                    2.0 * BOARD_HALF_SIZE[2]
                    + PIECE_HALF_THICKNESS
                    + drop_height,
                ]
            )
        ),
    )
    piece_body.set(
        "quat",
        array_to_string(Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]),
    )
    world.worldbody.append(piece_body)
    world.merge_assets(piece)
    model = world.get_model()
    return MjSim(model), board, piece


def body_rotation_matrix(sim, body_id):
    return sim.data.body_xmat[body_id].reshape(3, 3).copy()


def validation_metrics(sim, board, piece, shape):
    piece_body_id = sim.model.body_name2id(piece.root_body)
    piece_position = sim.data.body_xpos[piece_body_id].copy()
    piece_rotation = body_rotation_matrix(sim, piece_body_id)
    hole_to_peg_errors = []
    for index in range(len(SHAPE_SPECS[shape]["hole_positions"])):
        hole_position = sim.data.site_xpos[
            sim.model.site_name2id(piece.important_sites[f"hole_{index}"])
        ].copy()
        peg_position = sim.data.site_xpos[
            sim.model.site_name2id(board.important_sites[f"{shape}_peg_{index}"])
        ].copy()
        hole_to_peg_errors.append(
            float(np.linalg.norm(hole_position[:2] - peg_position[:2]))
        )
    seated_position = sim.data.site_xpos[
        sim.model.site_name2id(board.important_sites[f"{shape}_seated"])
    ].copy()
    bottom_position = sim.data.site_xpos[
        sim.model.site_name2id(piece.important_sites["bottom"])
    ].copy()

    z_axis = piece_rotation[:, 2]
    tilt = float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
    x_axis = piece_rotation[:, 0]
    yaw_error = float(abs(np.arctan2(x_axis[1], x_axis[0])))
    if shape == "circle":
        yaw_error = 0.0
    elif shape in {"square", "rectangle"}:
        yaw_error = min(yaw_error, abs(np.pi - yaw_error))
    qvel_address = sim.model.get_joint_qvel_addr(piece.joints[0])
    if isinstance(qvel_address, tuple):
        piece_qvel = sim.data.qvel[slice(*qvel_address)].copy()
    else:
        piece_qvel = np.atleast_1d(sim.data.qvel[qvel_address]).copy()
    return {
        "piece_position": piece_position,
        "hole_to_peg_xy_errors": np.array(hole_to_peg_errors),
        "maximum_hole_to_peg_xy_error": max(hole_to_peg_errors),
        "center_to_seated_z_error": float(
            abs(piece_position[2] - seated_position[2])
        ),
        "bottom_to_board_top_error": float(
            abs(bottom_position[2] - 2.0 * BOARD_HALF_SIZE[2])
        ),
        "tilt_rad": tilt,
        "yaw_error_rad": yaw_error,
        "linear_speed": float(np.linalg.norm(piece_qvel[:3])),
        "angular_speed": float(np.linalg.norm(piece_qvel[3:6])),
        "finite": bool(
            np.all(np.isfinite(sim.data.qpos))
            and np.all(np.isfinite(sim.data.qvel))
        ),
    }


def is_seated(
    metrics,
    xy_tolerance=0.002,
    z_tolerance=0.002,
    tilt_tolerance=0.05,
    yaw_tolerance=0.05,
    linear_speed_tolerance=0.02,
    angular_speed_tolerance=0.10,
):
    return bool(
        metrics["finite"]
        and metrics["maximum_hole_to_peg_xy_error"] <= xy_tolerance
        and metrics["center_to_seated_z_error"] <= z_tolerance
        and metrics["tilt_rad"] <= tilt_tolerance
        and metrics["yaw_error_rad"] <= yaw_tolerance
        and metrics["linear_speed"] <= linear_speed_tolerance
        and metrics["angular_speed"] <= angular_speed_tolerance
    )


def run_drop_test(
    shape="circle",
    steps=1000,
    drop_height=0.06,
    lateral_offset=(0.0, 0.0),
    yaw=0.0,
    render=False,
):
    sim, board, piece = build_sim(
        shape=shape,
        drop_height=drop_height,
        lateral_offset=lateral_offset,
        yaw=yaw,
    )
    viewer = None
    try:
        sim.forward()
        if render:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(
                sim.model._model,
                sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            viewer.cam.lookat[:] = [0.0, 0.0, 0.04]
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35

        stable_steps = 0
        for _ in range(steps):
            sim.step()
            if viewer is not None:
                viewer.sync()
                time.sleep(sim.model.opt.timestep)
            if not (
                np.all(np.isfinite(sim.data.qpos))
                and np.all(np.isfinite(sim.data.qvel))
            ):
                break
            current_metrics = validation_metrics(sim, board, piece, shape)
            stable_steps = stable_steps + 1 if is_seated(current_metrics) else 0
        metrics = validation_metrics(sim, board, piece, shape)
        metrics["stable_steps"] = stable_steps
        metrics["seated"] = bool(
            is_seated(metrics) and stable_steps >= REQUIRED_STABLE_STEPS
        )
        piece_body_id = sim.model.body_name2id(piece.root_body)
        metrics["mass"] = float(sim.model.body_mass[piece_body_id])
        metrics["inertia"] = sim.model.body_inertia[piece_body_id].copy()
        metrics["contact_count"] = int(sim.data.ncon)
        return metrics
    finally:
        if viewer is not None:
            viewer.close()


def run_multi_drop_test(
    steps=1000,
    drop_height=0.06,
    layout="aligned",
    seed=0,
    render=False,
):
    sim, board, pieces = build_multi_sim(
        drop_height=drop_height,
        layout=layout,
        seed=seed,
    )
    viewer = None
    try:
        sim.forward()
        if render:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(
                sim.model._model,
                sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            viewer.cam.lookat[:] = [0.0, 0.02, 0.04]
            viewer.cam.distance = 0.7
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35

        stable_steps = {shape: 0 for shape in pieces}
        for _ in range(steps):
            sim.step()
            if viewer is not None:
                viewer.sync()
                time.sleep(sim.model.opt.timestep)
            if not (
                np.all(np.isfinite(sim.data.qpos))
                and np.all(np.isfinite(sim.data.qvel))
            ):
                break
            for shape, piece in pieces.items():
                current_metrics = validation_metrics(sim, board, piece, shape)
                stable_steps[shape] = (
                    stable_steps[shape] + 1
                    if is_seated(current_metrics)
                    else 0
                )

        results = {}
        for shape, piece in pieces.items():
            metrics = validation_metrics(sim, board, piece, shape)
            metrics["stable_steps"] = stable_steps[shape]
            metrics["seated"] = bool(
                is_seated(metrics)
                and stable_steps[shape] >= REQUIRED_STABLE_STEPS
            )
            results[shape] = metrics
        return results
    finally:
        if viewer is not None:
            viewer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        choices=(*PIECE_FACTORIES, "all"),
        default="circle",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--drop-height", type=float, default=0.06)
    parser.add_argument(
        "--layout",
        choices=("aligned", "shuffled"),
        default="aligned",
        help="Drop over matching pegs or use randomized off-board spawn slots.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lateral-offset",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw in radians.")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.drop_height < 0:
        raise ValueError("--drop-height cannot be negative")
    if args.drop_height > MAX_SAFE_DROP_HEIGHT:
        raise ValueError(
            f"--drop-height must not exceed {MAX_SAFE_DROP_HEIGHT} m in this "
            "contact validator; high-speed impacts can tunnel through the thin board"
        )

    if args.shape == "all":
        results = run_multi_drop_test(
            steps=args.steps,
            drop_height=args.drop_height,
            layout=args.layout,
            seed=args.seed,
            render=args.render,
        )
        for shape, metrics in results.items():
            print(f"{shape}:")
            for name, value in metrics.items():
                print(f"  {name}: {value}")
        if args.layout == "aligned":
            failed = [
                shape for shape, metrics in results.items() if not metrics["seated"]
            ]
            if failed:
                raise RuntimeError(f"Pieces did not settle into seated poses: {failed}")
        elif not all(metrics["finite"] for metrics in results.values()):
            raise RuntimeError("Shuffled layout produced a non-finite state")
    else:
        metrics = run_drop_test(
            shape=args.shape,
            steps=args.steps,
            drop_height=args.drop_height,
            lateral_offset=args.lateral_offset,
            yaw=args.yaw,
            render=args.render,
        )
        for name, value in metrics.items():
            print(f"{name}: {value}")
        if not metrics["finite"]:
            raise RuntimeError("Shape-sorter simulation produced a non-finite state")
        if not metrics["seated"]:
            raise RuntimeError(f"{args.shape} piece did not settle into the seated pose")


if __name__ == "__main__":
    main()
