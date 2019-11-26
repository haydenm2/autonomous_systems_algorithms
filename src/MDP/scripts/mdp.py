#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Generic Markov Decision Process Model (From Probablistic Robotics)

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
        img = ax.imshow(self.value_map, origin='lower')
        fig.colorbar(img, ax=ax)
        lines, = ax.plot([], [], 'r--')
        self.map_handles = [ax, img, lines]

    def Solve(self):
        print("Solving Value Map...")
        self.ValueIteration()
        print("Solving Optimal Policy...")
        self.PolicyAssignment()
        print("Solution Found!")

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
            # self.VisualizeMap()

            if self.policy_map[self.optimal_policy[0, -1], self.optimal_policy[1, -1]] == -1:
                break

    def VisualizeMap(self):
        (self.map_handles[1]).set_data(self.value_map.transpose())
        (self.map_handles[2]).set_xdata(self.optimal_policy[0, :])
        (self.map_handles[2]).set_ydata(self.optimal_policy[1, :])
        plt.pause(0.1)

    def VisualizePolicyMap(self):
        for i in range(self.l_cells):
            for j in range(self.w_cells):
                if self.policy_map[i, j] == 0:  # north
                    plt.arrow(i, j+0.5, 0.7, 0, length_includes_head=True, head_width=0.4, head_length=0.2)
                elif self.policy_map[i, j] == 1:  # east
                    plt.arrow(i+0.5, j+1, 0, -0.7, length_includes_head=True, head_width=0.4, head_length=0.2)
                elif self.policy_map[i, j] == 2:  # south
                    plt.arrow(i+1, j+0.5, -0.7, 0, length_includes_head=True, head_width=0.4, head_length=0.2)
                elif self.policy_map[i, j] == 3:  # west
                    plt.arrow(i + 0.5, j, 0, 0.7, length_includes_head=True, head_width=0.4, head_length=0.2)
                else:
                    continue
        plt.pause(0.1)
