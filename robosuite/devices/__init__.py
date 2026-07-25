from .device import Device
from .keyboard import Keyboard
from .joystick import Joystick

try:
    from .spacemouse import SpaceMouse
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )


def __getattr__(name):
    """Lazily load devices that depend on optional hardware SDKs."""
    if name == "NeroHardwareTeleop":
        try:
            from .nero_hardware_teleop import NeroHardwareTeleop
        except ModuleNotFoundError as exc:
            if exc.name == "pyAgxArm":
                raise ModuleNotFoundError(
                    "NeroHardwareTeleop requires the optional pyAgxArm SDK. "
                    "Install the AgileX Nero hardware SDK only when using the "
                    "physical leader arm; keyboard and SpaceMouse teleoperation "
                    "do not require it."
                ) from exc
            raise
        return NeroHardwareTeleop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
