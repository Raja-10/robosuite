import time
import numpy as np
import robosuite as suite

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    NeroFW,
)

class RobotSim:

    def __init__(self, control_freq=100):

        # -----------------------------
        # REAL ROBOT
        # -----------------------------

        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.DEFAULT,
            channel="can0",
        )

        self.robot = AgxArmFactory.create_arm(cfg)

        # -----------------------------
        # ROBOSUITE
        # -----------------------------

        self.env = suite.make(
            env_name="Wipe",
            robots="Nero7",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            ignore_done=True,
            control_freq=control_freq,
        )

    def connect(self):

        self.robot.connect()

        while not self.robot.enable():
            self.robot.set_normal_mode()
            time.sleep(0.01)

        self.robot.set_leader_mode()

        self.env.reset()

        print("Robot + Simulation Ready")

    def get_real_joints(self):

        mja = self.robot.get_leader_joint_angles()

        if mja is None:
            return None

        return np.array(mja.msg)

    def step(self, action):

        return self.env.step(action)

    def get_sim_joints(self):

        return self.env.robots[0]._joint_positions

    def render_camera(self, camera_name, width, height):

        img = self.env.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

        return np.flipud(img)