"""Primitive-based objects for the four-shape peg-insertion task."""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.objects.generated_objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


BOARD_HALF_SIZE = np.array([0.16, 0.12, 0.006])
PIECE_HALF_THICKNESS = 0.006
PEG_RADIUS = 0.006
PEG_HEIGHT = 0.035
HOLE_RADIUS = 0.0075
CIRCLE_RADIUS = 0.04
SQUARE_HALF_SIZE = 0.035
TRIANGLE_HALF_WIDTH = 0.045
TRIANGLE_MIN_Y = -0.035
TRIANGLE_MAX_Y = 0.04
RECTANGLE_HALF_SIZE = np.array([0.0475, 0.0275])

TARGET_STATIONS = {
    "circle": np.array([-0.09, 0.065]),
    "square": np.array([0.09, 0.065]),
    "triangle": np.array([-0.07, -0.065]),
    "rectangle": np.array([0.09, -0.065]),
}

SHAPE_SPECS = {
    "circle": {"hole_positions": np.array([[0.0, 0.0]])},
    "square": {"hole_positions": np.array([[-0.018, 0.0], [0.018, 0.0]])},
    "triangle": {
        "hole_positions": np.array(
            [[0.0, 0.012], [-0.020, -0.015], [0.020, -0.015]]
        )
    },
    "rectangle": {
        "hole_positions": np.array(
            [
                [-0.028, -0.014],
                [-0.028, 0.014],
                [0.028, -0.014],
                [0.028, 0.014],
            ]
        )
    },
}

CONTACT_FRICTION = (1.0, 0.005, 0.0001)
CONTACT_SOLREF = (0.02, 1.0)
CONTACT_SOLIMP = (0.9, 0.95, 0.001)
CIRCLE_RGBA = (0.90, 0.05, 0.05, 1.0)
SQUARE_RGBA = (0.05, 0.80, 0.10, 1.0)
TRIANGLE_RGBA = (0.10, 0.45, 0.90, 1.0)
RECTANGLE_RGBA = (0.95, 0.65, 0.05, 1.0)


def _add_piece_box(args, name, x_interval, y_interval, half_thickness, rgba):
    """Adds one axis-aligned box spanning the supplied closed XY intervals."""
    x0, x1 = x_interval
    y0, y1 = y_interval
    if x1 <= x0 or y1 <= y0:
        return
    add_to_dict(
        dic=args,
        geom_types="box",
        geom_locations=((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0),
        geom_sizes=((x1 - x0) / 2.0, (y1 - y0) / 2.0, half_thickness),
        geom_names=name,
        geom_rgbas=rgba,
        geom_frictions=CONTACT_FRICTION,
        geom_condims=4,
    )


def _piece_sites(half_thickness, orientation_x, hole_positions, hole_radius):
    sites = [
        {
            "name": "center",
            "pos": (0.0, 0.0, 0.0),
            "size": "0.002",
            "rgba": "0 0 1 0.5",
            "type": "sphere",
        },
        {
            "name": "top",
            "pos": (0.0, 0.0, half_thickness),
            "size": "0.002",
            "rgba": "0 1 0 0.5",
            "type": "sphere",
        },
        {
            "name": "bottom",
            "pos": (0.0, 0.0, -half_thickness),
            "size": "0.002",
            "rgba": "1 0 0 0.5",
            "type": "sphere",
        },
        {
            "name": "grasp",
            "pos": (0.0, 0.0, 0.0),
            "size": "0.003",
            "rgba": "1 1 0 0.5",
            "type": "sphere",
        },
        {
            "name": "orientation_x",
            "pos": (orientation_x, 0.0, 0.0),
            "size": "0.002",
            "rgba": "0 1 1 0.5",
            "type": "sphere",
        },
    ]
    for index, hole_xy in enumerate(hole_positions):
        sites.append(
            {
                "name": f"hole_{index}",
                "pos": (*hole_xy, 0.0),
                "size": f"{hole_radius} {half_thickness}",
                "rgba": "1 0 1 0.2",
                "type": "cylinder",
            }
        )
    return sites


class _MultiHolePiece(CompositeObject):
    """Shared site contract for primitive pieces with multiple holes."""

    shape = None

    @property
    def important_sites(self):
        sites = super().important_sites
        sites.update(
            {
                key: self.naming_prefix + key
                for key in ("center", "top", "bottom", "grasp", "orientation_x")
            }
        )
        for index in range(len(SHAPE_SPECS[self.shape]["hole_positions"])):
            sites[f"hole_{index}"] = self.naming_prefix + f"hole_{index}"
        return sites


class ShapeSorterBoard(CompositeObject):
    """Fixed board containing the target pegs.

    The first Track B increment creates only the circle station. Additional
    stations can be enabled after their matching pieces are mechanically
    validated.
    """

    def __init__(self, name="shape_sorter_board", stations=("circle",)):
        unknown = set(stations) - set(SHAPE_SPECS)
        if unknown:
            raise ValueError(f"Unknown shape-sorter stations: {sorted(unknown)}")
        self.stations = tuple(stations)

        wood = CustomMaterial(
            texture="WoodLight",
            tex_name="sorter_wood",
            mat_name="sorter_wood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={
                "texrepeat": "2 2",
                "specular": "0.15",
                "shininess": "0.05",
            },
        )
        args = {}
        add_to_dict(
            dic=args,
            geom_types="box",
            geom_locations=(0.0, 0.0, BOARD_HALF_SIZE[2]),
            geom_sizes=tuple(BOARD_HALF_SIZE),
            geom_names="board_base",
            geom_rgbas=(0.82, 0.78, 0.67, 1.0),
            geom_materials=wood.mat_attrib["name"],
            geom_frictions=CONTACT_FRICTION,
            geom_condims=4,
        )

        sites = []
        peg_center_z = 2.0 * BOARD_HALF_SIZE[2] + PEG_HEIGHT / 2.0
        seated_center_z = 2.0 * BOARD_HALF_SIZE[2] + PIECE_HALF_THICKNESS
        for shape in self.stations:
            station_xy = TARGET_STATIONS[shape]
            station_peg_radius = 0.004 if shape == "triangle" else PEG_RADIUS
            sites.extend(
                [
                    {
                        "name": f"{shape}_target",
                        "pos": (*station_xy, seated_center_z),
                        "size": "0.003",
                        "rgba": "0 0 1 0.4",
                        "type": "sphere",
                    },
                    {
                        "name": f"{shape}_seated",
                        "pos": (*station_xy, seated_center_z),
                        "size": "0.002",
                        "rgba": "0 1 0 0.4",
                        "type": "sphere",
                    },
                ]
            )
            for index, hole_xy in enumerate(SHAPE_SPECS[shape]["hole_positions"]):
                peg_xy = station_xy + hole_xy
                add_to_dict(
                    dic=args,
                    geom_types="cylinder",
                    geom_locations=(*peg_xy, peg_center_z),
                    geom_sizes=(station_peg_radius, PEG_HEIGHT / 2.0),
                    geom_names=f"{shape}_peg_{index}",
                    geom_rgbas=(0.72, 0.72, 0.72, 1.0),
                    geom_frictions=CONTACT_FRICTION,
                    geom_condims=4,
                )
                sites.append(
                    {
                        "name": f"{shape}_peg_{index}",
                        "pos": (*peg_xy, peg_center_z),
                        "size": f"{station_peg_radius} {PEG_HEIGHT / 2.0}",
                        "rgba": "1 0 0 0.25",
                        "type": "cylinder",
                    }
                )

        args.update(
            {
                "name": name,
                "total_size": (
                    BOARD_HALF_SIZE[0],
                    BOARD_HALF_SIZE[1],
                    2.0 * BOARD_HALF_SIZE[2] + PEG_HEIGHT,
                ),
                "locations_relative_to_center": True,
                "joints": None,
                "sites": sites,
                "obj_types": "all",
                "density": 500.0,
                "solref": CONTACT_SOLREF,
                "solimp": CONTACT_SOLIMP,
            }
        )
        super().__init__(**args)
        self.append_material(wood)

    @property
    def important_sites(self):
        sites = super().important_sites
        for shape in self.stations:
            sites[f"{shape}_target"] = self.naming_prefix + f"{shape}_target"
            sites[f"{shape}_seated"] = self.naming_prefix + f"{shape}_seated"
            for index in range(len(SHAPE_SPECS[shape]["hole_positions"])):
                sites[f"{shape}_peg_{index}"] = (
                    self.naming_prefix + f"{shape}_peg_{index}"
                )
        return sites


class CirclePiece(CompositeObject):
    """Circular piece with a genuine primitive-based collision hole."""

    def __init__(
        self,
        name="circle_piece",
        outer_radius=CIRCLE_RADIUS,
        hole_radius=HOLE_RADIUS,
        half_thickness=PIECE_HALF_THICKNESS,
        segments=20,
        density=300.0,
    ):
        if not 8 <= segments:
            raise ValueError("CirclePiece requires at least 8 ring segments")
        if not 0 < hole_radius < outer_radius:
            raise ValueError("Require 0 < hole_radius < outer_radius")
        if half_thickness <= 0:
            raise ValueError("half_thickness must be positive")

        self.outer_radius = float(outer_radius)
        self.hole_radius = float(hole_radius)
        self.half_thickness = float(half_thickness)
        self.segments = int(segments)

        args = {}
        radial_half_width = (
            (outer_radius - hole_radius) * np.cos(np.pi / segments) / 2.0
        )
        tangential_half_width = outer_radius * np.sin(np.pi / segments)
        center_radius = (
            hole_radius * np.cos(np.pi / segments) + radial_half_width
        )

        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            center = (
                center_radius * np.cos(angle),
                center_radius * np.sin(angle),
                0.0,
            )
            quat = (np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0))
            add_to_dict(
                dic=args,
                geom_types="box",
                geom_locations=center,
                geom_quats=quat,
                geom_sizes=(
                    radial_half_width,
                    tangential_half_width,
                    half_thickness,
                ),
                geom_names=f"ring_{index}",
                geom_rgbas=CIRCLE_RGBA,
                geom_frictions=CONTACT_FRICTION,
                geom_condims=4,
            )

        sites = [
            {
                "name": "center",
                "pos": (0.0, 0.0, 0.0),
                "size": "0.002",
                "rgba": "0 0 1 0.5",
                "type": "sphere",
            },
            {
                "name": "top",
                "pos": (0.0, 0.0, half_thickness),
                "size": "0.002",
                "rgba": "0 1 0 0.5",
                "type": "sphere",
            },
            {
                "name": "bottom",
                "pos": (0.0, 0.0, -half_thickness),
                "size": "0.002",
                "rgba": "1 0 0 0.5",
                "type": "sphere",
            },
            {
                "name": "grasp",
                "pos": (0.0, 0.0, 0.0),
                "size": "0.003",
                "rgba": "1 1 0 0.5",
                "type": "sphere",
            },
            {
                "name": "orientation_x",
                "pos": (outer_radius, 0.0, 0.0),
                "size": "0.002",
                "rgba": "0 1 1 0.5",
                "type": "sphere",
            },
            {
                "name": "hole_0",
                "pos": (0.0, 0.0, 0.0),
                "size": f"{hole_radius} {half_thickness}",
                "rgba": "1 0 1 0.2",
                "type": "cylinder",
            },
        ]
        super().__init__(
            name=name,
            total_size=(outer_radius, outer_radius, half_thickness),
            geom_types=args["geom_types"],
            geom_sizes=args["geom_sizes"],
            geom_locations=args["geom_locations"],
            geom_quats=args["geom_quats"],
            geom_names=args["geom_names"],
            geom_rgbas=args["geom_rgbas"],
            geom_frictions=args["geom_frictions"],
            geom_condims=args["geom_condims"],
            density=density,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            locations_relative_to_center=True,
            joints=[dict(type="free", damping="0.0005")],
            sites=sites,
            obj_types="all",
        )

    @property
    def important_sites(self):
        sites = super().important_sites
        sites.update(
            {
                key: self.naming_prefix + key
                for key in (
                    "center",
                    "top",
                    "bottom",
                    "grasp",
                    "orientation_x",
                    "hole_0",
                )
            }
        )
        return sites


class SquarePiece(_MultiHolePiece):
    """Square piece with two primitive-based collision holes."""

    shape = "square"

    def __init__(
        self,
        name="square_piece",
        half_size=SQUARE_HALF_SIZE,
        hole_radius=HOLE_RADIUS,
        half_thickness=PIECE_HALF_THICKNESS,
        density=300.0,
    ):
        hole_positions = SHAPE_SPECS["square"]["hole_positions"]
        hole_offset = float(abs(hole_positions[0, 0]))
        if not 0 < hole_radius < hole_offset:
            raise ValueError("Square holes must be smaller than their center offset")
        if hole_offset + hole_radius >= half_size:
            raise ValueError("Square holes must fit inside the piece boundary")
        if half_thickness <= 0:
            raise ValueError("half_thickness must be positive")

        self.half_size = float(half_size)
        self.hole_radius = float(hole_radius)
        self.half_thickness = float(half_thickness)

        args = {}

        def add_box(geom_name, center_x, center_y, half_x, half_y):
            add_to_dict(
                dic=args,
                geom_types="box",
                geom_locations=(center_x, center_y, 0.0),
                geom_sizes=(half_x, half_y, half_thickness),
                geom_names=geom_name,
                geom_rgbas=SQUARE_RGBA,
                geom_frictions=CONTACT_FRICTION,
                geom_condims=4,
            )

        outer_rail_half_width = (half_size - hole_radius) / 2.0
        outer_rail_center = (half_size + hole_radius) / 2.0
        add_box(
            "top_rail",
            0.0,
            outer_rail_center,
            half_size,
            outer_rail_half_width,
        )
        add_box(
            "bottom_rail",
            0.0,
            -outer_rail_center,
            half_size,
            outer_rail_half_width,
        )

        left_edge = -half_size
        left_hole_edge = -hole_offset - hole_radius
        left_half_width = (left_hole_edge - left_edge) / 2.0
        add_box(
            "left_rail",
            left_edge + left_half_width,
            0.0,
            left_half_width,
            hole_radius,
        )

        right_hole_edge = hole_offset + hole_radius
        right_edge = half_size
        right_half_width = (right_edge - right_hole_edge) / 2.0
        add_box(
            "right_rail",
            right_hole_edge + right_half_width,
            0.0,
            right_half_width,
            hole_radius,
        )

        inner_left_edge = -hole_offset + hole_radius
        inner_right_edge = hole_offset - hole_radius
        add_box(
            "center_rail",
            0.0,
            0.0,
            (inner_right_edge - inner_left_edge) / 2.0,
            hole_radius,
        )

        sites = [
            {
                "name": "center",
                "pos": (0.0, 0.0, 0.0),
                "size": "0.002",
                "rgba": "0 0 1 0.5",
                "type": "sphere",
            },
            {
                "name": "top",
                "pos": (0.0, 0.0, half_thickness),
                "size": "0.002",
                "rgba": "0 1 0 0.5",
                "type": "sphere",
            },
            {
                "name": "bottom",
                "pos": (0.0, 0.0, -half_thickness),
                "size": "0.002",
                "rgba": "1 0 0 0.5",
                "type": "sphere",
            },
            {
                "name": "grasp",
                "pos": (0.0, 0.0, 0.0),
                "size": "0.003",
                "rgba": "1 1 0 0.5",
                "type": "sphere",
            },
            {
                "name": "orientation_x",
                "pos": (half_size, 0.0, 0.0),
                "size": "0.002",
                "rgba": "0 1 1 0.5",
                "type": "sphere",
            },
        ]
        for index, hole_xy in enumerate(hole_positions):
            sites.append(
                {
                    "name": f"hole_{index}",
                    "pos": (*hole_xy, 0.0),
                    "size": f"{hole_radius} {half_thickness}",
                    "rgba": "1 0 1 0.2",
                    "type": "cylinder",
                }
            )

        super().__init__(
            name=name,
            total_size=(half_size, half_size, half_thickness),
            geom_types=args["geom_types"],
            geom_sizes=args["geom_sizes"],
            geom_locations=args["geom_locations"],
            geom_names=args["geom_names"],
            geom_rgbas=args["geom_rgbas"],
            geom_frictions=args["geom_frictions"],
            geom_condims=args["geom_condims"],
            density=density,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            locations_relative_to_center=True,
            joints=[dict(type="free", damping="0.0005")],
            sites=sites,
            obj_types="all",
        )

    @property
    def important_sites(self):
        return super().important_sites


class TrianglePiece(_MultiHolePiece):
    """Three-hole triangular piece built from conservative horizontal bands."""

    shape = "triangle"

    def __init__(
        self,
        name="triangle_piece",
        hole_radius=HOLE_RADIUS,
        half_thickness=PIECE_HALF_THICKNESS,
        density=300.0,
    ):
        holes = SHAPE_SPECS[self.shape]["hole_positions"]
        args = {}
        y_breaks = sorted(
            {
                TRIANGLE_MIN_Y,
                TRIANGLE_MAX_Y,
                *np.linspace(TRIANGLE_MIN_Y, TRIANGLE_MAX_Y, 16),
                *(float(y - hole_radius) for _, y in holes),
                *(float(y + hole_radius) for _, y in holes),
            }
        )

        geom_index = 0
        for y0, y1 in zip(y_breaks[:-1], y_breaks[1:]):
            if y1 <= TRIANGLE_MIN_Y or y0 >= TRIANGLE_MAX_Y:
                continue
            # The triangle narrows toward +Y. Using the upper band boundary
            # keeps every rectangular collision geom inside the outline.
            half_width = TRIANGLE_HALF_WIDTH * (
                TRIANGLE_MAX_Y - y1
            ) / (TRIANGLE_MAX_Y - TRIANGLE_MIN_Y)
            intervals = [(-half_width, half_width)]
            for hole_x, hole_y in holes:
                if y0 < hole_y + hole_radius and y1 > hole_y - hole_radius:
                    blocked = (hole_x - hole_radius, hole_x + hole_radius)
                    split = []
                    for x0, x1 in intervals:
                        if blocked[1] <= x0 or blocked[0] >= x1:
                            split.append((x0, x1))
                        else:
                            if x0 < blocked[0]:
                                split.append((x0, blocked[0]))
                            if blocked[1] < x1:
                                split.append((blocked[1], x1))
                    intervals = split
            for interval in intervals:
                _add_piece_box(
                    args,
                    f"triangle_band_{geom_index}",
                    interval,
                    (y0, y1),
                    half_thickness,
                    TRIANGLE_RGBA,
                )
                geom_index += 1

        super().__init__(
            name=name,
            total_size=(
                TRIANGLE_HALF_WIDTH,
                max(abs(TRIANGLE_MIN_Y), abs(TRIANGLE_MAX_Y)),
                half_thickness,
            ),
            geom_types=args["geom_types"],
            geom_sizes=args["geom_sizes"],
            geom_locations=args["geom_locations"],
            geom_names=args["geom_names"],
            geom_rgbas=args["geom_rgbas"],
            geom_frictions=args["geom_frictions"],
            geom_condims=args["geom_condims"],
            density=density,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            locations_relative_to_center=True,
            joints=[dict(type="free", damping="0.0005")],
            sites=_piece_sites(
                half_thickness,
                TRIANGLE_HALF_WIDTH,
                holes,
                hole_radius,
            ),
            obj_types="collision",
        )
        self._append_triangle_visual_mesh(half_thickness, hole_radius)

    def _append_triangle_visual_mesh(self, half_thickness, hole_radius):
        """Adds a straight-edged triangular prism with three visual apertures."""
        holes = SHAPE_SPECS[self.shape]["hole_positions"]
        y_breaks = sorted(
            {
                TRIANGLE_MIN_Y,
                TRIANGLE_MAX_Y,
                *(float(y - hole_radius) for _, y in holes),
                *(float(y + hole_radius) for _, y in holes),
            }
        )
        vertices = []
        faces = []

        for y0, y1 in zip(y_breaks[:-1], y_breaks[1:]):
            # Avoid a zero-width final edge while retaining a visually sharp apex.
            mesh_y1 = min(y1, TRIANGLE_MAX_Y - 1e-5)
            lower_half_width = TRIANGLE_HALF_WIDTH * (
                TRIANGLE_MAX_Y - y0
            ) / (TRIANGLE_MAX_Y - TRIANGLE_MIN_Y)
            upper_half_width = TRIANGLE_HALF_WIDTH * (
                TRIANGLE_MAX_Y - mesh_y1
            ) / (TRIANGLE_MAX_Y - TRIANGLE_MIN_Y)
            intervals = [(-upper_half_width, upper_half_width)]
            for hole_x, hole_y in holes:
                if y0 < hole_y + hole_radius and y1 > hole_y - hole_radius:
                    blocked = (hole_x - hole_radius, hole_x + hole_radius)
                    split = []
                    for x0, x1 in intervals:
                        if blocked[1] <= x0 or blocked[0] >= x1:
                            split.append((x0, x1))
                        else:
                            if x0 < blocked[0]:
                                split.append((x0, blocked[0]))
                            if blocked[1] < x1:
                                split.append((blocked[1], x1))
                    intervals = split

            for upper_x0, upper_x1 in intervals:
                if upper_x1 <= upper_x0:
                    continue
                lower_x0 = (
                    -lower_half_width
                    if np.isclose(upper_x0, -upper_half_width)
                    else upper_x0
                )
                lower_x1 = (
                    lower_half_width
                    if np.isclose(upper_x1, upper_half_width)
                    else upper_x1
                )
                base = len(vertices)
                vertices.extend(
                    [
                        [lower_x0, y0, -half_thickness],
                        [lower_x1, y0, -half_thickness],
                        [upper_x1, mesh_y1, -half_thickness],
                        [upper_x0, mesh_y1, -half_thickness],
                        [lower_x0, y0, half_thickness],
                        [lower_x1, y0, half_thickness],
                        [upper_x1, mesh_y1, half_thickness],
                        [upper_x0, mesh_y1, half_thickness],
                    ]
                )
                local_faces = (
                    (4, 5, 6),
                    (4, 6, 7),
                    (0, 2, 1),
                    (0, 3, 2),
                    (0, 1, 5),
                    (0, 5, 4),
                    (1, 2, 6),
                    (1, 6, 5),
                    (2, 3, 7),
                    (2, 7, 6),
                    (3, 0, 4),
                    (3, 4, 7),
                )
                faces.extend(
                    [[base + index for index in face] for face in local_faces]
                )
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, dtype=int)
        mesh_name = self.naming_prefix + "triangle_visual_mesh"
        geom_name = self.naming_prefix + "triangle_visual"
        self.asset.append(
            ET.Element(
                "mesh",
                attrib={
                    "name": mesh_name,
                    "vertex": " ".join(str(value) for value in vertices.ravel()),
                    "face": " ".join(str(value) for value in faces.ravel()),
                },
            )
        )
        self.get_obj().append(
            ET.Element(
                "geom",
                attrib={
                    "name": geom_name,
                    "type": "mesh",
                    "mesh": mesh_name,
                    "rgba": " ".join(str(value) for value in TRIANGLE_RGBA),
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "1e-8",
                },
            )
        )
        self._visual_geoms.append("triangle_visual")


class RectanglePiece(_MultiHolePiece):
    """Four-hole rectangular piece built from horizontal primitive bands."""

    shape = "rectangle"

    def __init__(
        self,
        name="rectangle_piece",
        half_size=RECTANGLE_HALF_SIZE,
        hole_radius=HOLE_RADIUS,
        half_thickness=PIECE_HALF_THICKNESS,
        density=300.0,
    ):
        half_size = np.asarray(half_size, dtype=float)
        holes = SHAPE_SPECS[self.shape]["hole_positions"]
        if np.any(np.abs(holes) + hole_radius >= half_size):
            raise ValueError("Rectangle holes must fit inside the piece boundary")

        args = {}
        y_breaks = sorted(
            {
                -half_size[1],
                half_size[1],
                *(float(y - hole_radius) for _, y in holes),
                *(float(y + hole_radius) for _, y in holes),
            }
        )
        geom_index = 0
        for y0, y1 in zip(y_breaks[:-1], y_breaks[1:]):
            intervals = [(-half_size[0], half_size[0])]
            for hole_x, hole_y in holes:
                if y0 < hole_y + hole_radius and y1 > hole_y - hole_radius:
                    blocked = (hole_x - hole_radius, hole_x + hole_radius)
                    split = []
                    for x0, x1 in intervals:
                        if blocked[1] <= x0 or blocked[0] >= x1:
                            split.append((x0, x1))
                        else:
                            if x0 < blocked[0]:
                                split.append((x0, blocked[0]))
                            if blocked[1] < x1:
                                split.append((blocked[1], x1))
                    intervals = split
            for interval in intervals:
                _add_piece_box(
                    args,
                    f"rectangle_band_{geom_index}",
                    interval,
                    (y0, y1),
                    half_thickness,
                    RECTANGLE_RGBA,
                )
                geom_index += 1

        super().__init__(
            name=name,
            total_size=(*half_size, half_thickness),
            geom_types=args["geom_types"],
            geom_sizes=args["geom_sizes"],
            geom_locations=args["geom_locations"],
            geom_names=args["geom_names"],
            geom_rgbas=args["geom_rgbas"],
            geom_frictions=args["geom_frictions"],
            geom_condims=args["geom_condims"],
            density=density,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            locations_relative_to_center=True,
            joints=[dict(type="free", damping="0.0005")],
            sites=_piece_sites(
                half_thickness,
                half_size[0],
                holes,
                hole_radius,
            ),
            obj_types="all",
        )
