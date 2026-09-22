from xml.etree import ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas.table_arena import TableArena
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements

# Pose of the real RealSense D435 (color optical frame) in the real robot's base frame,
# from hand-eye calibration (17 samples; mean spread 17.5mm / 2.2deg), with the x axis
# sign-flipped (x-translation, and the rotation entries coupling x with y/z) to match the
# real camera placement -- the raw calibration put the camera behind the robot.
# Uses the OpenCV optical convention: x right, y down, z forward (out of the lens).
# The real base frame is aligned with the sim world axes (the table spans y = -0.73..+0.61
# in it) and sits at the robot base position -- NOT the sim base_link frame, which is
# yawed ~180deg (euler="0 0 -3.14" in robot.xml).
T_BASE_D435 = np.array(
    [
        [0.893513161556756, -0.25411778899830456, -0.3702139644036584, 0.6259851419035427],
        [-0.44799019847340227, -0.4482125607811307, -0.7735698303513288, 0.612802136415748],
        [0.030643305902372964, 0.8570470521929473, -0.5143261009619691, 0.7173744049332957],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# fovy=42.5 is the D435's published RGB sensor vertical FOV (horizontal 69.4deg, diagonal
# 77deg, all +/-3deg; the separate depth module is wider: 87 x 58 x 95deg). Mujoco's
# <camera> only exposes a single vertical fovy, so this models the RGB sensor's FOV.
D435_FOVY = 42.5


class LabWoodTableArena(TableArena):
    """
    TableArena whose tabletop uses a real wood-laminate photo (see
    robosuite/models/assets/textures/table_texture.jpeg) instead of the default
    ceramic texture, to match a physical lab table. The externally mounted D435
    camera is attached to the robot's base body instead -- see add_d435_camera().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        wood = CustomMaterial(
            texture="textures/table_texture.png",
            tex_name="lab_wood_tex",
            mat_name="lab_wood_mat",
            tex_attrib={"type": "2d"},
            mat_attrib={"reflectance": "0.0", "shininess": "0.05", "specular": "0.1", "texrepeat": "1 1", "texuniform": "true"},
        )
        self.asset.append(ET.Element("texture", attrib=wood.tex_attrib))
        self.asset.append(ET.Element("material", attrib=wood.mat_attrib))
        self.table_visual.set("material", wood.name)


def add_d435_camera(robot_model, name="d435"):
    """
    Adds the fixed "d435" camera as a child of @robot_model's root body at the calibrated
    T_BASE_D435 pose, so it stays put relative to the robot base regardless of where the
    base is placed. The root body's own rotation (Nero7's base_link is yawed ~180deg) is
    undone, so T_BASE_D435 is applied in a world-aligned frame at the base position,
    matching the real robot's base frame.

    Args:
        robot_model (RobotModel): robot model whose root body gets the camera
        name (str): camera name
    """
    base = find_elements(
        root=robot_model.worldbody,
        tags="body",
        attribs={"name": robot_model.root_body},
        return_first=True,
    )
    # Rotation of the root body relative to its (world-aligned) parent
    if base.get("quat") is not None:
        base_rot = T.quat2mat(np.roll(np.array(base.get("quat").split(), dtype=float), -1))  # wxyz -> xyzw
    elif base.get("euler") is not None:
        base_rot = T.euler2mat(np.array(base.get("euler").split(), dtype=float))
    else:
        base_rot = np.eye(3)
    # OpenCV optical frame (z forward, y down) -> Mujoco camera frame (looks along -z, y up)
    rot = base_rot.T @ T_BASE_D435[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    pos = base_rot.T @ T_BASE_D435[:3, 3]
    quat = T.mat2quat(rot)  # xyzw
    base.append(
        ET.Element(
            "camera",
            attrib={
                "mode": "fixed",
                "name": name,
                "pos": array_to_string(pos),
                "quat": array_to_string(np.roll(quat, 1)),  # wxyz
                "fovy": str(D435_FOVY),
            },
        )
    )
