import cv2
import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtWidgets

class Viewer:

    def __init__(self, window_name, x, y):

        self.window_name = window_name

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        cv2.moveWindow(window_name, x, y)

        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    def make_grid(self, imgs):

        top = np.hstack((imgs[0], imgs[1]))
        bottom = np.hstack((imgs[2], imgs[3]))

        return np.vstack((top, bottom))

    def show(self, img):

        cv2.imshow(self.window_name, img)

class JointPlotter:

    def __init__(self, n_joints=7, history=500):

        self.app = QtWidgets.QApplication([])

        self.win = pg.GraphicsLayoutWidget(
            show=True,
            title="Joint Tracking",
        )

        self.history = history

        self.cmd_hist = [[] for _ in range(n_joints)]
        self.sim_hist = [[] for _ in range(n_joints)]

        self.curves_cmd = []
        self.curves_sim = []

        for j in range(n_joints):

            p = self.win.addPlot(row=j, col=0)

            p.setYRange(-3.5, 3.5)

            c1 = p.plot(pen='r')
            c2 = p.plot(pen='g')

            self.curves_cmd.append(c1)
            self.curves_sim.append(c2)

    def update(self, q_cmd, q_sim):

        for j in range(len(q_cmd)):

            self.cmd_hist[j].append(q_cmd[j])
            self.sim_hist[j].append(q_sim[j])

            self.cmd_hist[j] = self.cmd_hist[j][-self.history:]
            self.sim_hist[j] = self.sim_hist[j][-self.history:]

            self.curves_cmd[j].setData(self.cmd_hist[j])
            self.curves_sim[j].setData(self.sim_hist[j])

        self.app.processEvents()