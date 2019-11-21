#!/usr/bin/env python3

import numpy as np

# Generic Markov Decision Process Model (From Probablistic Robotics)

# Implement the discrete value iteration algorithm for MDP's outlined in Table 14.1.
#
# Demonstrate your algorithm by finding the optimal path through the map specified by this Matlab script: MDP_hw_map.m  Download
#
# Use a discount factor of 1 initially. Assign each cell of the map with a nominal cost of -2. Assign goal cells a reward of +100000, obstacles a cost of -5000, and walls a cost of -100.
#
# Assume that control actions result in your robot moving north, east, south, or west, and that your robot moves in the commanded direction with probability 0.8, 90 deg left of the commanded direction with probability 0.1, and 90 deg right of the commanded direction with probability 0.1.
#
# (1) Create a plot of the optimal policy for the specified map. Here's a function to draw arrows if you care to use it:draw_arrow.m  Download
#
# (2) Create a plot of the value function for specified map and robot characteristics.
#
# (3) With the robot starting in the initial state (28,20), plot the path to the goal region based on the optimal policy.
#
# (4) Exercise you algorithm by changing the map, the initial condition on the robot location, the costs/rewards in the map, the discount factor, and the uncertainty model associated with the robot motion p(xj | u, xi). Does the algorithm give the results you'd expect?


class MDP:
    def __init__(self, grid_l, grid_w, goal, obs, walls, x0):
        self.grid_size = np.array([grid_l, grid_w])  # size of grid space
        self.l_cells = self.grid_size[0]
        self.w_cells = self.grid_size[1]
        self.map = np.zeros((self.l_cells, self.w_cells))
        self.goal = goal * 1      # goal map type number
        self.obstacles = obs * 2       # obstacle map type number
        self.walls = walls * 3     # wall map type number
        self.map = self.map + self.goal + self.obstacles + self.walls
        self.num_cells = self.l_cells * self.w_cells

        # inverse range sensor model parameters
        self.discount_factor = 1

        # motion probability
        self.forward_probability = 0.8
        self.right_probability = 0.1
        self.left_probability = 0.1

        # costs
        self.nominal_cost = -2
        self.goal_reward = 100000
        self.obstacle_cost = -5000
        self.wall_cost = -100

        # robot data
        self.x0 = x0
        self.x = np.array((2, 0))

    def AssignValues(self):
        pass

    def AssignPolicies(self):
        pass

    def Solve(self):
        pass
