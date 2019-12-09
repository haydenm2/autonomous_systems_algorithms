#!/usr/bin/env python3
from mdp import MDP
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------------------------------------------------------
# Summary:
# Example of implementation of the markov decision process class for a robot planning through a 2D map

if __name__ == "__main__":

    # Markov Decision Process object and incoming data initialization
    data = loadmat("MDP_map.mat")
    # data = loadmat("MDP_map2.mat")
    # data = loadmat("MDP_map3.mat")

    N = data["N"].item(0)            # dimension of map without walls
    Np = data["Np"].item(0)        # dimension of map with walls
    map = data["map"]       # MDP map
    goal = data["goal"]       # MDP map
    obs1 = data["obs1"]     # obstacle 1 locations
    obs2 = data["obs2"]     # obstacle 2 locations
    obs3 = data["obs3"]     # obstacle 3 locations
    try:
        obs4 = data["obs4"]     # obstacle 3 locations
    except:
        pass
    walls = data["walls"]   # wall locations
    try:
        obs = obs1 + obs2 + obs3 + obs4 # total obstacle locations
    except:
        obs = obs1 + obs2 + obs3 # total obstacle locations
    x0 = np.array((28,20))

    mdp = MDP(Np, Np, goal, obs, walls, x0)       # MDP algorithm object
    mdp.Solve()
    mdp.VisualizeMap()
    mdp.VisualizePolicyMap()

    plt.show()