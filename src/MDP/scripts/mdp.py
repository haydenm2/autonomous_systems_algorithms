#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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

        # costs
        self.nominal_cost = -2
        self.goal_reward = 100000
        self.obstacle_cost = -5000
        self.wall_cost = -100

        # map values
        self.grid_size = np.array([grid_l, grid_w])  # size of grid space
        self.l_cells = self.grid_size[0]
        self.w_cells = self.grid_size[1]

        self.mask = np.logical_and(np.logical_and(np.logical_not(walls), np.logical_not(goal)), np.logical_not(obs))

        self.type_map = np.zeros((self.l_cells, self.w_cells))
        self.type_goal = goal * 1      # goal map type number
        self.type_obstacles = obs * -1       # obstacle map type number
        self.type_walls = walls * -2     # wall map type number
        self.type_map = self.type_map + self.type_goal + self.type_obstacles + self.type_walls

        self.value_map = np.zeros((self.l_cells, self.w_cells))
        self.value_goal = goal * self.goal_reward  # goal map values
        self.value_obstacles = obs * self.obstacle_cost  # obstacle map type values
        self.value_walls = walls * self.wall_cost  # wall map type values
        self.value_map = self.value_map + self.value_goal + self.value_obstacles + self.value_walls
        for i in range(self.l_cells-2):
            for j in range(self.w_cells-2):
                if self.type_map[i][j] == 0:
                    self.value_map[i][j] = self.nominal_cost

        self.policy_map = np.zeros((self.l_cells, self.w_cells))
        self.policy_goal = goal * -1  # goal map policies
        self.policy_obstacles = obs * -1  # obstacle map type policies
        self.policy_walls = walls * -1  # wall map type policies
        self.policy_map = self.policy_map + self.policy_goal + self.policy_obstacles + self.policy_walls

        self.num_cells = self.l_cells * self.w_cells

        # inverse range sensor model parameters
        self.discount_factor = 1.0

        # motion probability
        self.forward_probability = 0.8
        self.right_probability = 0.1
        self.left_probability = 0.1

        # robot data
        self.x0 = x0.reshape((-1, 1))
        self.optimal_policy = self.x0

        # flags
        self.convergence = False
        self.error_threshold = 0.1

        # map plotting
        fig, ax = plt.subplots()
        img = ax.imshow(self.value_map)
        fig.colorbar(img, ax=ax)
        lines, = ax.plot([], [], 'r--')
        self.map_handles = [ax, img, lines]

    def Solve(self):
        self.ValueIteration()
        self.PolicyAssignment()
        # self.Solve()

    def ValueIteration(self):
        while not self.convergence:
            map_prev = np.copy(self.value_map)
            map = np.zeros((4, self.l_cells-2, self.w_cells-2))
            nmap = self.value_map[2:self.l_cells, 1:self.w_cells-1]
            emap = self.value_map[1:self.l_cells-1, 0:self.w_cells-2]
            smap = self.value_map[0:self.l_cells-2, 1:self.w_cells-1]
            wmap = self.value_map[1:self.l_cells-1, 2:self.w_cells]
            map[0] = self.left_probability * wmap + self.forward_probability * nmap + self.right_probability * emap  # north
            map[1] = self.left_probability * nmap + self.forward_probability * emap + self.right_probability * smap  # east
            map[2] = self.left_probability * emap + self.forward_probability * smap + self.right_probability * wmap  # south
            map[3] = self.left_probability * smap + self.forward_probability * wmap + self.right_probability * nmap  # west
            self.value_map[1:self.l_cells-1, 1:self.w_cells-1] = self.discount_factor * np.multiply(np.max(map, axis=0) + self.nominal_cost, self.mask[1:self.l_cells-1, 1:self.w_cells-1])
            self.value_map += (self.value_goal + self.value_obstacles + self.value_walls)
            self.policy_map[1:self.l_cells-1, 1:self.w_cells-1] = np.multiply(np.argmax(map, axis=0), self.mask[1:self.l_cells-1, 1:self.w_cells-1])
            self.policy_map += (self.policy_goal + self.policy_obstacles + self.policy_walls)
            # self.VisualizeMap()

            # Check Convergence
            error = np.sum(np.sum(self.value_map - map_prev))
            if error < self.error_threshold:
                self.convergence = True

    def PolicyAssignment(self):
        while(True):
            i = self.optimal_policy[0, -1]
            j = self.optimal_policy[1, -1]
            current_policy = self.policy_map[self.optimal_policy[0, -1], self.optimal_policy[1, -1]]
            if current_policy == 0:  # north
                next_policy = np.array([[i + 1], [j]])
            elif current_policy == 1:  # east
                next_policy = np.array([[i], [j - 1]])
            elif current_policy == 2:  # south
                next_policy = np.array([[i - 1], [j]])
            elif current_policy == 3:  # west
                next_policy = np.array([[i], [j + 1]])
            self.optimal_policy = np.hstack((self.optimal_policy, next_policy))
            self.VisualizeMap()

            if self.policy_map[self.optimal_policy[0, -1], self.optimal_policy[1, -1]] == -1:
                break

    def VisualizeMap(self):
        (self.map_handles[1]).set_data(self.value_map)
        (self.map_handles[2]).set_xdata(self.optimal_policy[0, :])
        (self.map_handles[2]).set_ydata(self.optimal_policy[1, :])
        # (self.map_handles[1]).autoscale()
        plt.pause(0.1)



