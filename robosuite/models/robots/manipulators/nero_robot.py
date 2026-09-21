import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Nero7(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/nero/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.4, 0.4, 0.2, 0.2 , 0.1 , 0.1 , 0.1))/10)

    def add_gripper(self, gripper, arm_name=None):
        super().add_gripper(gripper, arm_name)
        # link5's and the gripper flange's collision meshes overlap at rest (mounting
        # artifact, not real interference). Excluded here rather than in a static XML
        # file because the gripper's body names only get their final prefix (e.g.
        # "gripper0_right_") once merged in below -- see the matching same-file
        # excludes for the arm's own overlapping links, base_link/link1 and
        # link5/link7, in robot.xml's own <contact> block.
        flange_name = next(
            name for name in self.get_element_names(self.worldbody, "body") if name.endswith("gripper_flange")
        )
        self.contact.append(
            ET.Element("exclude", {"name": "link5_flange", "body1": self.naming_prefix + "link5", "body2": flange_name})
        )

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "PiperGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_nero7"}

    @property
    def init_qpos(self):
        # All-zero qpos extends the arm straight up (eef ends up ~0.9m above the table),
        # far outside the workspace the Stack task expects. This pose instead points the
        # gripper straight down (local z-axis aligned with world -z, matching the
        # convention scripted policies assume) with the eef centered over the table at
        # the task's hover height (table height + 0.15m). This particular IK solution was
        # chosen (out of the arm's 1-DOF-redundant solution family for that pose) because
        # it keeps every joint well clear of its limits, leaving headroom to move in any
        # direction -- other solutions reach the same pose but pin 2+ joints near their
        # limits, which stalls the OSC controller as soon as it tries to move off-center.
        # return np.array([0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array([0.3312, 0.4951, -0.4556, 1.7133, 0.0425, -0.3086, 0.9589])

    @property
    def base_xpos_offset(self):
        # Nero7 uses NullMount (zero-height stand-in), unlike Panda/Sawyer whose x/y-only
        # offsets rely on a physical mount (e.g. RethinkMount) to lift the arm up to table
        # height. Without a mount, set_base_xpos() places the base at exactly these
        # coordinates, so the z-component here must equal the table height (0.8, matching
        # ManipulationEnv's default table_offset) for the base to sit on the table surface
        # rather than the floor.
        return {
            "bins": (-0.5, -0.1, 0.8),
            "empty": (-0.6, 0, 0.8),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0.8),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
