#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

# Generic Monte Carlo Localization Filter Approach (From Probablistic Robotics)

class OGM:
    def __init__(self, grid_l, grid_w, cell_size):
        self.p = 0  # map certainty
        self.grid_size = np.array([grid_l, grid_w])  # size of grid space
        self.cell_size = cell_size
        self.l_cells = int(grid_l/self.cell_size)
        self.w_cells = int(grid_w/self.cell_size)
        self.p_init = 0.5
        self.l_init = self.p2l(self.p_init)
        self.map = np.zeros((3, self.l_cells, self.w_cells))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.map[0, i, j] = self.cell_size * (0.5 + j)
                self.map[1, i, j] = self.cell_size * (0.5 + i)
                self.map[2, i, j] = self.p_init
        self.num_cells = self.l_cells * self.w_cells

        # inverse range sensor model parameters
        self.alpha = 1
        self.beta = 5*np.pi/180
        self.z_max = 150

        # probabilities assigned for occupied and free cell detections
        self.p_occ = 0.7
        self.p_free = 0.3
        self.l_occ = self.p2l(self.p_occ)
        self.l_free = self.p2l(self.p_free)

    def Update(self, x, z):
        # # Iterative Update Approach
        # for i in range(self.l_cells):
        #     for j in range(self.w_cells):
        #         self.InverseRangeSensorModel(self.map[:, i, j], x, z)

        # Vectorized Update Approach
        self.InverseRangeSensorModelVectorized(self.map, x, z)

        # Convert Map from log-odds to standard probability map
        self.map[2, :, :] = self.l2p(self.map[2, :, :])

    def InverseRangeSensorModelVectorized(self, m, xt, zt):
        xi = m[0, :, :]
        yi = m[1, :, :]
        l_prev = self.p2l(m[2, :, :])
        x = xt[0]
        y = xt[1]
        theta = xt[2]
        r = np.sqrt((xi-x)**2+(yi-y)**2)
        phi = np.arctan2(yi-y, xi-x)-theta
        zt = np.nan_to_num(zt, nan=0)
        zphidiff = np.zeros((zt.shape[1], phi.shape[0], phi.shape[1]))
        for i in range(zt.shape[1]):
            zphidiff[i, :, :] = np.abs(self.Wrap(phi - zt[1, i]))
        k = np.argmin(zphidiff, axis=0)

        # inverse sensor model logical grid
        table1 = np.logical_or((r > np.minimum(self.z_max, zt[0, k] + self.alpha/2)), (np.abs(self.Wrap(phi-zt[1, k])) > self.beta/2))
        table2 = np.logical_and(np.logical_and((zt[0, k] < self.z_max), (np.abs(r-zt[0, k]) < self.alpha/2)), np.logical_not(table1))
        table3 = np.logical_and(np.logical_and((r <= zt[0, k]), np.logical_not(table1)), np.logical_not(table2))

        # convert logic tables from bool to double
        table1 = (table1).astype(np.double)
        table2 = (table2).astype(np.double)
        table3 = (table3).astype(np.double)

        # assign conditional values for each table
        table1 *= l_prev + self.l_init - self.l_init
        table2 *= l_prev + self.l_occ - self.l_init
        table3 *= l_prev + self.l_free - self.l_init

        # overlay table probabilities
        m[2, :, :] = table1 + table2 + table3

    def InverseRangeSensorModel(self, m, xt, zt):
        xi = m[0]
        yi = m[1]
        l_prev = self.p2l(m[2])
        x = xt[0]
        y = xt[1]
        theta = xt[2]
        r = np.sqrt((xi-x)**2+(yi-y)**2)
        phi = np.arctan2(yi-y, xi-x)-theta
        zt = np.nan_to_num(zt, nan=0)
        k = np.argmin(np.abs(self.Wrap(phi-zt[1, :])))
        if (r > min(self.z_max, zt[0, k] + self.alpha/2)) or (np.abs(self.Wrap(phi-zt[1, k])) > self.beta/2):
            m[2] = l_prev + self.l_init - self.l_init
        elif (zt[0, k] < self.z_max) and (np.abs(r-zt[0, k]) < self.alpha/2):
            m[2] = l_prev + self.l_occ - self.l_init
        elif r <= zt[0, k]:
            m[2] = l_prev + self.l_free - self.l_init

    def Wrap(self, th):
        th_wrap = (th + np.pi) % (2 * np.pi) - np.pi
        return th_wrap

    def p2l(self, p):
        l = np.log(p / (1 - p))
        return l

    def l2p(self, l):
        p = 1 - 1/(1 + np.exp(l))
        return p