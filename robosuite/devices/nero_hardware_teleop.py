import time
import numpy as np

from robosuite.devices import Device

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    NeroFW,
)


class NeroHardwareTeleop(Device):

    def __init__(
        self,
        env,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        super().__init__(env)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False

        # ----------------------------------
        # Connect leader arm
        # ----------------------------------

        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.DEFAULT,
            channel="can0",
        )

        self.robot = AgxArmFactory.create_arm(cfg)

        self.robot.connect()

        while not self.robot.enable():
            self.robot.set_normal_mode()
            time.sleep(0.01)

        self.robot.set_leader_mode()

        print("Leader arm connected")

    def _reset_internal_state(self):

        super()._reset_internal_state()

        self.rotation = np.eye(3)

        self.last_pos = None
        self.last_rpy = None

        self.raw_drotation = np.zeros(3)

    def start_control(self):

        self._reset_internal_state()

        self._reset_state = 0

        self._enabled = True

        # ----------------------------------
        # Read leader joints
        # ----------------------------------

        mja = self.robot.get_leader_joint_angles()

        if mja is None:
            print("Failed to read leader joints")
            return

        q = np.array(mja.msg)

        # ----------------------------------
        # Sync sim robot joints
        # ----------------------------------

        sim_robot = self.env.robots[0]

        self.env.sim.data.qpos[
            sim_robot._ref_joint_pos_indexes
        ] = q

        self.env.sim.forward()

        # ----------------------------------
        # Initialize FK reference
        # ----------------------------------

        pose = np.asarray(
            self.robot.fk(q),
            dtype=np.float64,
        )

        self.last_pos = pose[:3]
        self.last_rpy = pose[3:6]

        print("Leader/sim synchronized")

        print(
            "Initial pose:",
            np.round(pose, 4)
        )

    def get_controller_state(self):

        mja = self.robot.get_leader_joint_angles()

        if mja is None:

            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=int(self.grasp),
                reset=self._reset_state,
                base_mode=int(self.base_mode),
            )

        q = np.array(mja.msg)

        # FK returns:
        # [x, y, z, roll, pitch, yaw]

        pose = np.asarray(
            self.robot.fk(q),
            dtype=np.float64,
        )

        pos = pose[:3]
        rpy = pose[3:6]

        # ----------------------------------
        # Safety
        # ----------------------------------

        if self.last_pos is None:

            self.last_pos = pos.copy()
            self.last_rpy = rpy.copy()

            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=int(self.grasp),
                reset=self._reset_state,
                base_mode=int(self.base_mode),
            )

        # ----------------------------------
        # Delta pose
        # ----------------------------------

        dpos = pos - self.last_pos

        raw_drotation = rpy - self.last_rpy

        # ----------------------------------
        # Save
        # ----------------------------------

        self.last_pos = pos.copy()
        self.last_rpy = rpy.copy()

        self.raw_drotation = raw_drotation.copy()

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=self.raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _postprocess_device_outputs(
        self,
        dpos,
        drotation,
    ):

        dpos = (
            dpos
            * 10.0
            * self.pos_sensitivity
        )

        drotation = (
            drotation
            * 1.5
            * self.rot_sensitivity
        )

        dpos = np.clip(
            dpos,
            -1.0,
            1.0,
        )

        drotation = np.clip(
            drotation,
            -1.0,
            1.0,
        )

        return dpos, drotation