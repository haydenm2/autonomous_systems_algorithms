#!/usr/bin/env python3
from ogm import OGM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.io import loadmat

# ------------------------------------------------------------------
# Summary:
# Example of implementation of the occupancy grid mapping class for a robot creating a 2D map



def InitPlot(ogm, body_radius, map):
    fig, ax = plt.subplots()
    lines, = ax.plot([], [], 'g--', zorder=1)
    robot_body = Circle((0, 0), body_radius, color='b', zorder=3)
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=4)
    ax.set_xlim([0, ogm.grid_size[0]])
    ax.set_ylim([0, ogm.grid_size[1]])
    ax.grid()
    img = ax.imshow(map, "Greys")
    return fig, lines, robot_body, robot_head, img


def UpdatePlot(fig, lines, robot_body, body_radius, robot_head, img, xt, map):
    lines.set_xdata(xt[0, :])
    lines.set_ydata(xt[1, :])

    robot_body.center = (xt[0, -1], xt[1, -1])
    headx = np.array([xt[0, -1], xt[0, -1] + body_radius * np.cos(xt[2, -1])])
    heady = np.array([xt[1, -1], xt[1, -1] + body_radius * np.sin(xt[2, -1])])
    robot_head.set_xdata(headx)
    robot_head.set_ydata(heady)
    img.set_data(map)
    img.autoscale()
    plt.pause(0.01)


if __name__ == "__main__":

    # Occupancy Graph Mapping object and incoming data initialization
    ogm = OGM(100, 100, 1)       # OGM algorithm object
    data = loadmat("state_meas_data.mat")
    X = data["X"]
    thk = data["thk"]
    z = data["z"]
    iter = 0

    body_radius = 1.5
    fig, lines, robot_body, robot_head, img = InitPlot(ogm, body_radius, ogm.map[2, :, :])
    xt = X[:, 0].reshape(3, 1)
    UpdatePlot(fig, lines, robot_body, body_radius, robot_head, img, xt, ogm.map[2, :, :])
    for i in range(len(X[0])):
        # truth model updates
        ogm.Update(X[:, iter].reshape(3, 1), z[:, :, iter])
        iter += 1
        UpdatePlot(fig, lines, robot_body, body_radius, robot_head, img, X[:, 0:iter], ogm.map[2, :, :])
        plt.pause(0.01)

    plt.show()