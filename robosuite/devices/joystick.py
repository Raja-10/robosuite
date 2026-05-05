import numpy as np
import pygame

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class Joystick(Device):

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")

        self.js = pygame.joystick.Joystick(0)
        self.js.init()

        print("Joystick:", self.js.get_name())

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = False

    def deadzone(self, x, dz=0.08):
        return 0.0 if abs(x) < dz else x

    def _reset_internal_state(self):
        super()._reset_internal_state()

        self.pos = np.zeros(3)
        self.last_pos = np.zeros(3)

        self.raw_drotation = np.zeros(3)
        self.last_drotation = np.zeros(3)

        self.prev_buttons = [0] * 15

        # same initial orientation as keyboard
        self.rotation = np.array([
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0, 0.0,-1.0]
        ])

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):

        pygame.event.pump()

        # ---------------- AXES ----------------
        right_x = self.deadzone(-self.js.get_axis(3))
        right_y = self.deadzone(-self.js.get_axis(2))

        left_x = self.deadzone(-self.js.get_axis(1))   # Z control

        lt_raw = self.js.get_axis(5)
        rt_raw = self.js.get_axis(4)

        # normalize triggers [-1,1] → [0,1]
        lt = (lt_raw + 1) / 2.0
        rt = (rt_raw + 1) / 2.0

        # ---------------- BUTTONS ----------------
        curr_buttons = [self.js.get_button(i) for i in range(self.js.get_numbuttons())]

        A = curr_buttons[0]
        B = curr_buttons[1]
        X = curr_buttons[3]
        Y = curr_buttons[4]

        # ---------------- POSITION ----------------
        self.pos += np.array([
            right_x,     # X
            right_y,     # Y
            left_x,      # Z
        ]) * 0.02 * self.pos_sensitivity

        # ---------------- ROTATION (Yaw via triggers) ----------------
        yaw = (rt - lt)

        if abs(yaw) < 0.05:
            yaw = 0.0

        yaw *= 0.1 * self.rot_sensitivity

        # rotation matrix (for consistency)
        drot = rotation_matrix(
            angle=yaw,
            direction=[0.0, 0.0, 1.0]
        )[:3, :3]

        self.rotation = self.rotation.dot(drot)

        # IMPORTANT: accumulate (not overwrite)
        self.raw_drotation[2] += yaw

        # ---------------- GRIPPER (toggle) ----------------
        if A and not self.prev_buttons[0]:
            self.grasp_states[self.active_robot][self.active_arm_index] = \
                not self.grasp_states[self.active_robot][self.active_arm_index]

        # ---------------- RESET ----------------
        if Y and not self.prev_buttons[4]:
            self._reset_state = 1
            self._enabled = False

        # ---------------- ZERO PITCH ----------------
        if X and not self.prev_buttons[3]:
            self.raw_drotation[1] = 0.0
            self.rotation[:, 1] = np.array([0, 1, 0])

        # ---------------- ZERO ROLL ----------------
        if B and not self.prev_buttons[1]:
            self.raw_drotation[0] = 0.0
            self.rotation[:, 0] = np.array([1, 0, 0])

        self.prev_buttons = curr_buttons

        # ---------------- DELTAS ----------------
        dpos = self.pos - self.last_pos
        self.last_pos = self.pos.copy()

        raw_drotation = self.raw_drotation - self.last_drotation
        self.last_drotation = self.raw_drotation.copy()

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 5
        dpos = dpos * 75

        return np.clip(dpos, -1, 1), np.clip(drotation, -1, 1)