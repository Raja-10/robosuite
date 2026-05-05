import pygame
import numpy as np
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected")

js = pygame.joystick.Joystick(0)
js.init()

print("Testing:", js.get_name())


def deadzone(x, dz=0.08):
    return 0.0 if abs(x) < dz else x


while True:
    pygame.event.pump()

    # Assumed Mapping
    left_analog_x = deadzone(-js.get_axis(1))
    left_analog_y = deadzone(-js.get_axis(0))

    right_analog_x = deadzone(-js.get_axis(3))
    right_analog_y = deadzone(-js.get_axis(2))
    left_trigger = js.get_axis(5)
    right_trigger   = js.get_axis(4)

    A = js.get_button(0)
    B = js.get_button(1)
    X = js.get_button(3)
    Y = js.get_button(4)   


    dpos = np.array([
        -left_analog_y,
         left_analog_x,
         (right_trigger - left_trigger) / 2.0,
    ])

    drot = np.array([
        right_analog_y,
        0,
        right_analog_x,
    ])

    print("=" * 60)
    print(f"Left Stick   : ({left_analog_x:.3f}, {left_analog_y:.3f})")
    print(f"Right Stick  : ({right_analog_x:.3f}, {right_analog_y:.3f})")

    print(f"Triggers     : LT={left_trigger:.3f}, RT={right_trigger:.3f}")

    print(f"Buttons      : A={A}, B={B}, X={X}, Y={Y}") 

    time.sleep(0.1)
