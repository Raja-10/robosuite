import numpy as np

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import LabWoodTableArena
from robosuite.models.arenas.lab_wood_table_arena import add_d435_camera

# Table dimensions (L=x/vertical-depth, W=y/horizontal-width, H=thickness), in meters --
# matches LiftLabSetup1 exactly, since it's the same physical lab table.
TABLE_FULL_SIZE = (0.76, 1.34, 0.05)
TABLE_HEIGHT = 0.8

# Nero7's base sits 8cm in from the table's near/bottom edge and 73cm in from its left
# edge -- see LiftLabSetup1 for the full derivation. The table is shifted (not the robot)
# so the base lands exactly at the world origin.
_table_half_x, _table_half_y, _ = np.array(TABLE_FULL_SIZE) / 2
TABLE_OFFSET = np.array(
    [
        _table_half_x - 0.08,
        _table_half_y - 0.73,
        TABLE_HEIGHT,
    ]
)
NERO7_BASE_POS = np.array([0.0, 0.0, TABLE_HEIGHT])

# Both cubes the same 4cm size and dark grey (matching LiftLabSetup1's cube), per request
# -- unlike Stack's default differently-sized red/green cubeA/cubeB.
CUBE_SIZE = (0.02, 0.02, 0.02)
CUBE_RGBA = (0.2, 0.2, 0.2, 1)

# Pick (spawn) region for cubeA/cubeB, offset from table_offset -- reuses the exact same
# verified-reachable zone found for LiftLabSetup1's single cube (gridded closed-loop OSC
# reach scan holding the gripper vertical). Narrowed in y (0.09-0.19 -> 0.12-0.19) to leave
# room below it for the stack goal, so cubes never spawn on top of the stacking target.
CUBE_X_RANGE = [-0.02, 0.12]
CUBE_Y_RANGE = [0.12, 0.19]
# Fixed point (offset from table_offset) where the scripted stack policy builds the stack --
# just below the pick region, still inside the broader verified-reachable area.
GOAL_OFFSET = [0.05, 0.02]


class StackLabSetup1(Stack):
    """
    Stack task reproducing lab table setup #1 (see LiftLabSetup1): the same 76cm x 134cm
    table with Nero7's base at the world origin, but with two matching 4cm dark grey
    cubes instead of Stack's default differently-sized red/green cubeA/cubeB.
    """

    arena_type = LabWoodTableArena

    def __init__(
        self,
        robots="Nero7",
        table_full_size=TABLE_FULL_SIZE,
        cube_size=CUBE_SIZE,
        cube_rgba=CUBE_RGBA,
        cube_x_range=CUBE_X_RANGE,
        cube_y_range=CUBE_Y_RANGE,
        goal_offset=GOAL_OFFSET,
        **kwargs,
    ):
        super().__init__(
            robots=robots,
            table_full_size=table_full_size,
            cube_size=cube_size,
            cube_rgba=cube_rgba,
            cube_x_range=cube_x_range,
            cube_y_range=cube_y_range,
            goal_offset=goal_offset,
            **kwargs,
        )

    def _load_model(self):
        self.table_offset = TABLE_OFFSET
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(NERO7_BASE_POS)
        add_d435_camera(self.robots[0].robot_model)
