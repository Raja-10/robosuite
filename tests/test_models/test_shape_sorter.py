import numpy as np

from robosuite.models.objects.shape_sorter import (
    HOLE_RADIUS,
    PEG_RADIUS,
    SHAPE_SPECS,
    TRIANGLE_HALF_WIDTH,
    TRIANGLE_MAX_Y,
    TRIANGLE_MIN_Y,
    CirclePiece,
    RectanglePiece,
    ShapeSorterBoard,
    SquarePiece,
    TrianglePiece,
)
from robosuite.scripts.validate_shape_sorter_objects import (
    REQUIRED_STABLE_STEPS,
    build_multi_sim,
    build_sim,
    run_multi_drop_test,
    is_seated,
    run_drop_test,
    validation_metrics,
)


def test_circle_board_and_piece_contract():
    board = ShapeSorterBoard(stations=("circle",))
    piece = CirclePiece(segments=20)

    assert board.joints == []
    assert len(piece.joints) == 1
    assert len(SHAPE_SPECS["circle"]["hole_positions"]) == 1
    assert HOLE_RADIUS > PEG_RADIUS
    assert len([name for name in board.contact_geoms if "circle_peg" in name]) == 1
    assert len([name for name in piece.contact_geoms if "ring_" in name]) == 20
    assert "circle_peg_0" in board.important_sites
    assert "hole_0" in piece.important_sites
    board_visual = board.get_obj().find(
        ".//geom[@name='shape_sorter_board_board_base_vis']"
    )
    assert board_visual.get("material") == "shape_sorter_board_sorter_wood_mat"
    assert any(
        asset.get("name") == "shape_sorter_board_sorter_wood_mat"
        for asset in board.asset.findall("material")
    )


def test_circle_shape_sorter_compiles_with_positive_mass():
    sim, _, piece = build_sim(drop_height=0.0)
    sim.forward()
    piece_body_id = sim.model.body_name2id(piece.root_body)

    assert sim.model.body_mass[piece_body_id] > 0
    assert np.all(sim.model.body_inertia[piece_body_id] > 0)
    assert np.all(np.isfinite(sim.data.qpos))


def test_aligned_circle_drops_and_seats():
    metrics = run_drop_test(steps=1000, drop_height=0.04)

    assert metrics["finite"]
    assert metrics["seated"]
    assert metrics["maximum_hole_to_peg_xy_error"] <= 0.002
    assert metrics["center_to_seated_z_error"] <= 0.002


def test_offset_circle_is_not_reported_as_seated():
    sim, board, piece = build_sim(drop_height=0.0, lateral_offset=(0.02, 0.0))
    sim.forward()
    metrics = validation_metrics(sim, board, piece, "circle")

    assert not is_seated(metrics)


def test_circle_seating_is_yaw_invariant():
    sim, board, piece = build_sim(shape="circle", drop_height=0.0, yaw=0.7)
    sim.forward()
    metrics = validation_metrics(sim, board, piece, "circle")

    assert metrics["yaw_error_rad"] == 0.0
    assert is_seated(metrics)


def test_square_board_and_piece_contract():
    board = ShapeSorterBoard(stations=("square",))
    piece = SquarePiece()

    assert board.joints == []
    assert len(piece.joints) == 1
    assert len(SHAPE_SPECS["square"]["hole_positions"]) == 2
    assert len([name for name in board.contact_geoms if "square_peg" in name]) == 2
    assert len(piece.contact_geoms) == 5
    assert "square_peg_1" in board.important_sites
    assert "hole_1" in piece.important_sites


def test_aligned_square_drops_and_seats():
    metrics = run_drop_test(shape="square", steps=1000, drop_height=0.04)

    assert metrics["finite"]
    assert metrics["seated"]
    assert metrics["maximum_hole_to_peg_xy_error"] <= 0.002
    assert metrics["center_to_seated_z_error"] <= 0.002


def test_rotated_square_is_not_reported_as_seated():
    sim, board, piece = build_sim(shape="square", drop_height=0.0, yaw=0.3)
    sim.forward()
    metrics = validation_metrics(sim, board, piece, "square")

    assert metrics["maximum_hole_to_peg_xy_error"] > 0.002
    assert not is_seated(metrics)


def test_all_shapes_drop_together():
    results = run_multi_drop_test(steps=1000, drop_height=0.04)

    assert set(results) == {"circle", "square", "triangle", "rectangle"}
    assert all(metrics["finite"] for metrics in results.values())
    assert all(metrics["seated"] for metrics in results.values())
    assert all(
        metrics["stable_steps"] >= REQUIRED_STABLE_STEPS
        for metrics in results.values()
    )


def test_shuffled_layout_is_seeded_and_not_initially_successful():
    first_sim, _, first_pieces = build_multi_sim(
        layout="shuffled",
        seed=7,
        drop_height=0.04,
    )
    second_sim, _, second_pieces = build_multi_sim(
        layout="shuffled",
        seed=7,
        drop_height=0.04,
    )
    first_sim.forward()
    second_sim.forward()

    first_positions = np.array(
        [
            first_sim.data.body_xpos[
                first_sim.model.body_name2id(first_pieces[shape].root_body)
            ]
            for shape in first_pieces
        ]
    )
    second_positions = np.array(
        [
            second_sim.data.body_xpos[
                second_sim.model.body_name2id(second_pieces[shape].root_body)
            ]
            for shape in second_pieces
        ]
    )
    np.testing.assert_allclose(first_positions, second_positions)
    pairwise_xy = np.linalg.norm(
        first_positions[:, None, :2] - first_positions[None, :, :2],
        axis=-1,
    )
    assert np.min(pairwise_xy[np.nonzero(pairwise_xy)]) > 0.1

    results = run_multi_drop_test(
        layout="shuffled",
        seed=7,
        steps=1000,
        drop_height=0.04,
    )
    assert all(metrics["finite"] for metrics in results.values())
    assert not any(metrics["seated"] for metrics in results.values())
    assert all(metrics["linear_speed"] < 0.02 for metrics in results.values())


def test_success_requires_consecutive_stable_steps():
    metrics = run_drop_test(shape="circle", steps=10, drop_height=0.0)

    assert metrics["stable_steps"] < REQUIRED_STABLE_STEPS
    assert not metrics["seated"]


def test_triangle_and_rectangle_contracts():
    triangle = TrianglePiece()
    rectangle = RectanglePiece()

    assert len(SHAPE_SPECS["triangle"]["hole_positions"]) == 3
    assert len(SHAPE_SPECS["rectangle"]["hole_positions"]) == 4
    assert len(triangle.contact_geoms) > 3
    assert len(rectangle.contact_geoms) > 4
    assert "hole_2" in triangle.important_sites
    assert "hole_3" in rectangle.important_sites
    assert triangle.visual_geoms == ["triangle_piece_triangle_visual"]
    assert all("_vis" not in name for name in triangle.contact_geoms)
    triangle_mesh = triangle.asset.find(
        "./mesh[@name='triangle_piece_triangle_visual_mesh']"
    )
    assert triangle_mesh is not None
    assert len(triangle_mesh.get("vertex").split()) > 18
    upper_hole_y = SHAPE_SPECS["triangle"]["hole_positions"][0, 1]
    triangle_width_at_hole_top = TRIANGLE_HALF_WIDTH * (
        TRIANGLE_MAX_Y - (upper_hole_y + HOLE_RADIUS)
    ) / (TRIANGLE_MAX_Y - TRIANGLE_MIN_Y)
    assert triangle_width_at_hole_top - HOLE_RADIUS >= 0.004


def test_triangle_and_rectangle_drop_and_seat():
    for shape in ("triangle", "rectangle"):
        metrics = run_drop_test(shape=shape, steps=1000, drop_height=0.04)
        assert metrics["finite"]
        assert metrics["seated"]
        assert metrics["maximum_hole_to_peg_xy_error"] <= 0.002
        assert metrics["center_to_seated_z_error"] <= 0.002


def test_triangle_and_rectangle_reject_wrong_yaw():
    for shape in ("triangle", "rectangle"):
        sim, board, piece = build_sim(shape=shape, drop_height=0.0, yaw=0.25)
        sim.forward()
        metrics = validation_metrics(sim, board, piece, shape)
        assert metrics["maximum_hole_to_peg_xy_error"] > 0.002
        assert not is_seated(metrics)
