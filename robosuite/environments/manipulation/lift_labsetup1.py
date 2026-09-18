import numpy as np

from robosuite.environments.manipulation.lift import Lift

# Table dimensions (L=x/vertical-depth, W=y/horizontal-width, H=thickness), in meters.
TABLE_FULL_SIZE = (0.76, 1.34, 0.05)
TABLE_HEIGHT = 0.8  # table top height, matches Lift's default

# Nero7's base sits 8cm in from the table's near/bottom edge and 61cm in from its
# left edge (see robosuite/models/assets/robots/nero/README.md). Here the table is
# shifted instead of the robot, so the robot's base lands exactly on the world
# origin (x=0, y=0) while keeping that same physical mount point on the table.
_table_half_x, _table_half_y, _ = np.array(TABLE_FULL_SIZE) / 2
TABLE_OFFSET = np.array(
    [
        _table_half_x - 0.08,
        _table_half_y - 0.73,
        TABLE_HEIGHT,
    ]
)
NERO7_BASE_POS = np.array([0.0, 0.0, TABLE_HEIGHT])

# Dark grey, 4cm x 4cm x 4cm cube (BoxObject's "size" is half-extents).
CUBE_SIZE = (0.02, 0.02, 0.02)
CUBE_RGBA = (0.2, 0.2, 0.2, 1)


class LiftLabSetup1(Lift):
    """
    Lift task reproducing lab table setup #1: a 76cm x 134cm table with the Nero7
    robot's base centered at the world origin, table-mounted 8cm from the near edge
    and 61cm from the left edge.
    """

    def __init__(
        self,
        robots="Nero7",
        table_full_size=TABLE_FULL_SIZE,
        cube_size=CUBE_SIZE,
        cube_rgba=CUBE_RGBA,
        **kwargs,
    ):
        super().__init__(robots=robots, table_full_size=table_full_size, cube_size=cube_size, cube_rgba=cube_rgba, **kwargs)

    def _load_model(self):
        self.table_offset = TABLE_OFFSET
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(NERO7_BASE_POS)
