"""Tabletop mounting flange for the AgileX Nero7 arm."""

import numpy as np

from robosuite.models.bases.mount_model import MountModel
from robosuite.utils.mjcf_utils import xml_path_completion


class NeroTableMount(MountModel):
    """Fixed tabletop flange that preserves Nero7's validated base pose."""

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion("bases/nero_table_mount.xml"),
            idn=idn,
        )

    @property
    def top_offset(self):
        return np.array((0.2, 0.0, 0.85))

    @property
    def horizontal_radius(self):
        return 0.12

