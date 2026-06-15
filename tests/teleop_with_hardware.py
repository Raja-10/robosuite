import time
import cv2
import numpy as np
import robosuite as suite

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    NeroFW,
)

# =====================================================
# CONNECT REAL ROBOT
# =====================================================

cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    channel="can0",
)

robot = AgxArmFactory.create_arm(cfg)

robot.connect()

while not robot.enable():
    robot.set_normal_mode()
    time.sleep(0.01)

robot.set_leader_mode()

print("Robot connected")

# =====================================================
# CREATE ENVIRONMENT
# =====================================================

env = suite.make(
    env_name="Wipe",
    robots="Nero7",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    ignore_done=True,
    control_freq=100,
)

env.reset()

# =====================================================
# IMPORTANT DEBUG (ONCE)
# =====================================================

print("\n================ DEBUG ================\n")

print("ACTION SPEC:")
print(env.action_spec)

print("\nJOINT RANGE J1:")
print(env.sim.model.jnt_range[0])

print("\nACT CTRL RANGE J1:")
print(env.sim.model.actuator_ctrlrange[0])

print("\n=======================================\n")

# =====================================================
# CAMERA SETUP
# =====================================================

camera_names = [
    "robot0_eye_in_hand",
    "sideview",
    "topview",
    "teleopview",
]

window_name = "MultiCam"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.moveWindow(window_name, 6144, 239)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)

# =====================================================
# INITIALIZE SIM WITH REAL ROBOT STATE
# =====================================================

mja = robot.get_leader_joint_angles()

if mja is None:
    print("Failed to read robot joints")
    exit()

q_real = np.array(mja.msg)

env.sim.data.qpos[
    env.robots[0]._ref_joint_pos_indexes
] = q_real

env.sim.forward()

print("Environment ready")

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    # -------------------------------------------------
    # READ REAL ROBOT
    # -------------------------------------------------

    mja = robot.get_leader_joint_angles()

    if mja is None:
        continue

    q_real = np.array(mja.msg)

    # -------------------------------------------------
    # ACTION
    # -------------------------------------------------

    action = np.zeros(env.action_dim)

    action[:7] = q_real[:7]

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------

    for _ in range(5):
        obs, reward, done, info = env.step(action)

    # -------------------------------------------------
    # RAW MUJOCO QPOS
    # -------------------------------------------------

    q_sim = env.sim.data.qpos[
        env.robots[0]._ref_joint_pos_indexes
    ].copy()

    # -------------------------------------------------
    # DEBUG
    # -------------------------------------------------

    qvel = env.sim.data.qvel[
        env.robots[0]._ref_joint_vel_indexes
    ]
    j1_lim = env.sim.model.jnt_range[5]

    print(
        f"J1 REAL : {q_real[5]: .3f} | "
        f"SIM : {q_sim[5]: .3f} | "
        f"LIMITS : [{j1_lim[0]: .3f}, {j1_lim[1]: .3f}]"
    )

    # print("\n----------------------------")

    print("REAL J1 :", round(q_real[0], 3))

    print("SIM  J1 :", round(q_sim[0], 3))

    # print("VEL  J1 :", round(qvel[0], 3))

    # print("CTRL J1 :", round(env.sim.data.ctrl[0], 3))

    # print(
    #     "QFRC ACT:",
    #     round(env.sim.data.qfrc_actuator[0], 3)
    # )

    # print(
    #     "QFRC CON:",
    #     round(env.sim.data.qfrc_constraint[0], 3)
    # )

    # print(
    #     "NCON :",
    #     env.sim.data.ncon
    # )

    # print(
    #     "QERR J1 :",
    #     round(q_real[0] - q_sim[0], 3)
    # )

    # print("----------------------------")

    for i in range(env.sim.data.ncon):

        c = env.sim.data.contact[i]

        g1 = env.sim.model.geom_id2name(c.geom1)
        g2 = env.sim.model.geom_id2name(c.geom2)

        print("CONTACT:", g1, "<->", g2)

    # -------------------------------------------------
    # RENDER CAMERAS
    # -------------------------------------------------

    imgs = []

    for cam_name in camera_names:

        img = env.sim.render(
            width=960,
            height=540,
            camera_name=cam_name,
        )

        img = np.flipud(img)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            img,
            cam_name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        imgs.append(img)

    # -------------------------------------------------
    # GRID
    # -------------------------------------------------

    top_row = np.hstack((imgs[0], imgs[1]))

    bottom_row = np.hstack((imgs[2], imgs[3]))

    final_img = np.vstack((top_row, bottom_row))

    # -------------------------------------------------
    # SHOW
    # -------------------------------------------------

    cv2.imshow(window_name, final_img)

    if cv2.waitKey(1) == 27:
        break

    # time.sleep(0.005)

# =====================================================
# CLEANUP
# =====================================================

cv2.destroyAllWindows()