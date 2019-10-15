#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

# Generic Monte Carlo Localization Filter Approach (From Probablistic Robotics)

class OGM:
    def __init__(self, grid_l, grid_w, cell_size):
        self.p = 0  # map certainty
        self.grid_size = np.array([grid_l, grid_w])  # size of grid space
        self.cell_size = cell_size
        self.init_val = 0.0
        self.map = np.ones((int(grid_l/self.cell_size), int(grid_w/self.cell_size)))*self.init_val
        self.num_cells = int(grid_l/self.cell_size) * int(grid_w/self.cell_size)

        # inverse range sensor model parameters
        self.alpha = 1
        self.beta = 5*np.pi/180
        self.z_max = 150

        # probabilities assigned for occupied and free cell detections
        self.p_occ = 0.7
        self.p_free = 0.3
        self.l_occ = np.log(self.p_occ/(1-self.p_occ))
        self.l_free = np.log(self.p_free/(1-self.p_free))

        pass

    def Update(self):
        # for i in range(self.num_cells):
        #     pass


        pass

    def InverseRangeSensorModel(self, m, xt, zt):
        xi = m[0]
        yi = m[1]
        l_0 = m[2]
        x = xt[0]
        y = xt[1]
        theta = xt[2]
        r = np.sqrt((xi-x)**2+(yi-y)**2)
        phi = np.arctan2(yi-y, xi-x)-theta
        k = np.argmin(np.abs(phi-zt[1, :]))
        if (r > min(self.z_max, zt[0, k] + self.alpha/2)) or (np.abs(phi-zt[1, k]) > self.beta/2):
            return l_0
        if (zt[0, k] < self.z_max) and (np.abs(r-zt[0, k]) < self.alpha/2):
            return self.l_occ
        if r <= zt[0, k]:
            return self.l_free
        pass

    def Wrap(self, th):
        th_wrap = np.fmod(th + np.pi, 2*np.pi)
        for i in range(len(th_wrap)):
            if th_wrap[i] < 0:
                th_wrap[i] += 2*np.pi
        return th_wrap - np.pi
