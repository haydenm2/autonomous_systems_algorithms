#!/usr/bin/env python3
from mdp import MDP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.io import loadmat

# ------------------------------------------------------------------
# Summary:
# Example of implementation of the markov decision process class for a robot planning through a 2D map


def InitPlot(mdp, body_radius, map):
    fig, ax = plt.subplots()
    lines, = ax.plot([], [], 'g--', zorder=1)
    robot_body = Circle((0, 0), body_radius, color='b', zorder=3)
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=4)
    ax.set_xlim([0, mdp.grid_size[0]])
    ax.set_ylim([0, mdp.grid_size[1]])
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

    # Markov Decision Process object and incoming data initialization
    data = loadmat("MDP_map.mat")
    N = data["N"].item(0)            # dimension of map without walls
    Np = data["Np"].item(0)        # dimension of map with walls
    map = data["map"]       # MDP map
    goal = data["goal"]       # MDP map
    obs1 = data["obs1"]     # obstacle 1 locations
    obs2 = data["obs2"]     # obstacle 2 locations
    obs3 = data["obs3"]     # obstacle 3 locations
    walls = data["walls"]   # wall locations

    obs = obs1 + obs2 + obs3 # total obstacle locations
    x0 = np.array((28,20))

    mdp = MDP(Np, Np, goal, obs, walls, x0)       # MDP algorithm object
    mdp.Solve()

    body_radius = 1.5
    fig, lines, robot_body, robot_head, img = InitPlot(mdp, body_radius, mdp.map[2, :, :])
    xt = mdp.X[:, 0].reshape(3, 1)
    UpdatePlot(fig, lines, robot_body, body_radius, robot_head, img, xt, mdp.map[2, :, :])

    plt.show()