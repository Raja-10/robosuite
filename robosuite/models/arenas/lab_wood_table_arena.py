from xml.etree import ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas.table_arena import TableArena
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string


class LabWoodTableArena(TableArena):
    """
    TableArena whose tabletop uses a real wood-laminate photo (see
    robosuite/models/assets/textures/table_texture.jpeg) instead of the default
    ceramic texture, to match a physical lab table. Also adds a fixed, externally
    mounted camera (see D435_POS below) modeled after an Intel RealSense D435.
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

        # Mounted camera, positioned 30cm in from the table's near/robot-base edge
        # and 15cm in from its left edge, 55cm above the table surface -- pointed
        # at the table center. fovy=42.5 is the RealSense D435's published RGB
        # sensor vertical FOV (horizontal 69.4deg, diagonal 77deg, all +/-3deg; the
        # separate depth module is wider: 87 x 58 x 95deg). Mujoco's <camera> only
        # exposes a single vertical fovy, so this models the RGB sensor's FOV, not
        # the depth module's.
        near_edge_x = self.table_offset[0] - self.table_half_size[0]
        left_edge_y = self.table_offset[1] - self.table_half_size[1]
        cam_pos = np.array(
            [
                near_edge_x + 0.30,
                left_edge_y + 0.15,
                self.table_offset[2] + 0.55,
            ]
        )
        target = np.array(self.table_offset)
        z_cam = cam_pos - target
        z_cam /= np.linalg.norm(z_cam)
        x_cam = np.cross([0, 0, 1], z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        cam_quat = T.mat2quat(np.stack([x_cam, y_cam, z_cam], axis=1))  # xyzw
        cam_quat = np.array([cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2]])  # wxyz

        self.worldbody.append(
            ET.Element(
                "camera",
                attrib={
                    "mode": "fixed",
                    "name": "d435",
                    "pos": array_to_string(cam_pos),
                    "quat": array_to_string(cam_quat),
                    "fovy": "42.5",
                },
            )
        )
