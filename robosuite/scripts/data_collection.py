#!/usr/bin/env python3

import os
import cv2
import h5py
import time
import numpy as np
import robosuite as suite

from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    NeroFW,
)


# =====================================================
# CUSTOM WRAPPER
# =====================================================

class NeroDataCollectionWrapper(DataCollectionWrapper):

    def __init__(
        self,
        env,
        directory,
        collect_freq=1,
        flush_freq=1000000,
    ):
        super().__init__(
            env,
            directory,
            collect_freq,
            flush_freq,
        )

        self.q_real_log = []
        self.q_sim_log = []

        self.front_imgs = []
        self.wrist_imgs = []

        self.rewards = []
        self.dones = []

    def add_extra_observation(
        self,
        q_real,
        q_sim,
        front,
        wrist,
        reward,
        done,
    ):

        self.q_real_log.append(q_real.copy())
        self.q_sim_log.append(q_sim.copy())

        self.front_imgs.append(front.copy())
        self.wrist_imgs.append(wrist.copy())

        self.rewards.append(reward)
        self.dones.append(done)

    def _flush(self):

        if self.ep_directory is None:
            return

        filename = os.path.join(
            self.ep_directory,
            "demo.hdf5"
        )

        actions = []

        for info in self.action_infos:
            actions.append(info["actions"])

        with h5py.File(filename, "w") as f:

            obs_grp = f.create_group("observations")

            obs_grp.create_dataset(
                "q_real",
                data=np.asarray(self.q_real_log)
            )

            obs_grp.create_dataset(
                "q_sim",
                data=np.asarray(self.q_sim_log)
            )

            obs_grp.create_dataset(
                "front_rgb",
                data=np.asarray(self.front_imgs),
                compression="gzip"
            )

            obs_grp.create_dataset(
                "wrist_rgb",
                data=np.asarray(self.wrist_imgs),
                compression="gzip"
            )

            f.create_dataset(
                "states",
                data=np.asarray(self.states)
            )

            f.create_dataset(
                "actions",
                data=np.asarray(actions)
            )

            f.create_dataset(
                "rewards",
                data=np.asarray(self.rewards)
            )

            f.create_dataset(
                "dones",
                data=np.asarray(self.dones)
            )

            f.attrs["successful"] = self.successful

        print(f"Saved {filename}")

        self.states = []
        self.action_infos = []

        self.q_real_log = []
        self.q_sim_log = []

        self.front_imgs = []
        self.wrist_imgs = []

        self.rewards = []
        self.dones = []

        self.successful = False

    def close(self):

        if self.has_interaction:
            self._flush()

        self.env.close()


# =====================================================
# PARAMETERS
# =====================================================

SAVE_DIR = "hardware_demos"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

ENV_NAME = "Wipe"

# =====================================================
# CONNECT LEADER ARM
# =====================================================

cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.V111,
    channel="can0",
)

robot = AgxArmFactory.create_arm(cfg)

robot.connect()

while not robot.enable():
    robot.set_normal_mode()
    time.sleep(0.01)

robot.set_leader_mode()

print("Leader arm connected")

# =====================================================
# CREATE ENVIRONMENT
# =====================================================

env = suite.make(
    env_name=ENV_NAME,
    robots="Nero7",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    ignore_done=True,
    reward_shaping=True,
    control_freq=100,
)

env = NeroDataCollectionWrapper(
    env,
    directory=SAVE_DIR,
)

env.reset()

# =====================================================
# SYNC INITIAL STATE
# =====================================================

mja = robot.get_leader_joint_angles()

q_real = np.array(mja.msg)

env.sim.data.qpos[
    env.robots[0]._ref_joint_pos_indexes
] = q_real

env.sim.forward()

print()
print("s -> save episode")
print("r -> reset episode")
print("q -> quit")
print()

episode_id = 0

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    mja = robot.get_leader_joint_angles()

    if mja is None:
        continue

    q_real = np.asarray(mja.msg)

    action = np.zeros(env.action_dim)

    action[:7] = q_real[:7]

    obs, reward, done, info = env.step(action)

    q_sim = env.sim.data.qpos[
        env.robots[0]._ref_joint_pos_indexes
    ].copy()

    front = env.sim.render(
        width=640,
        height=480,
        camera_name="topview"
    )

    wrist = env.sim.render(
        width=640,
        height=480,
        camera_name="robot0_eye_in_hand"
    )

    front = np.flipud(front)
    wrist = np.flipud(wrist)

    env.add_extra_observation(
        q_real=q_real,
        q_sim=q_sim,
        front=front,
        wrist=wrist,
        reward=reward,
        done=done,
    )

    disp = cv2.cvtColor(
        front,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow(
        "Teleop",
        disp
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        env._flush()

        print("Episode Saved")

        env.reset()

    elif key == ord("r"):

        print("Episode Reset")

        env.reset()

    elif key == ord("q"):

        env.close()
        break

cv2.destroyAllWindows()