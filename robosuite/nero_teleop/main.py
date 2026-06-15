import time
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_X11_NO_MITSHM"] = "1"
import cv2
import numpy as np

from config import *

from robot_sim import RobotSim
from visualization import Viewer, JointPlotter

# =====================================================
# INIT
# =====================================================

system = RobotSim(control_freq=CONTROL_FREQ)

system.connect()

viewer = Viewer(
    window_name="MultiCam",
    x=WINDOW_X,
    y=WINDOW_Y,
)

plotter = JointPlotter(history=HISTORY)

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    # -----------------------------------------
    # REAL ROBOT
    # -----------------------------------------

    q_real = system.get_real_joints()

    if q_real is None:
        continue

    # -----------------------------------------
    # STEP SIM
    # -----------------------------------------

    obs, reward, done, info = system.step(q_real)

    q_sim = system.get_sim_joints()

    # -----------------------------------------
    # RENDER CAMERAS
    # -----------------------------------------

    imgs = []

    for cam_name in CAMERA_NAMES:

        img = system.render_camera(
            camera_name=cam_name,
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR,
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

    final_img = viewer.make_grid(imgs)

    viewer.show(final_img)

    # -----------------------------------------
    # PLOTS
    # -----------------------------------------

    plotter.update(q_real, q_sim)

    # -----------------------------------------
    # DEBUG
    # -----------------------------------------

    print("REAL :", np.round(q_real, 3))
    print("SIM  :", np.round(q_sim, 3))

    # -----------------------------------------
    # EXIT
    # -----------------------------------------

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.005)

cv2.destroyAllWindows()