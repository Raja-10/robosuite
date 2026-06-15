from .device import Device
from .keyboard import Keyboard
from .joystick import Joystick
from .nero_hardware_teleop import NeroHardwareTeleop

try:
    from .spacemouse import SpaceMouse
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
