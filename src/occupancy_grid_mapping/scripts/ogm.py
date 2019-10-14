#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

# Generic Monte Carlo Localization Filter Approach (From Probablistic Robotics)

class OGM:
    def __init__(self):
        pass

    def Propogate(self, u, z):
        pass

    def PropogateMotionModel(self, u, xprev):
        pass

    def MeasurementModel(self, z, x):
        pass

    def Wrap(self, th):
        th_wrap = np.fmod(th + np.pi, 2*np.pi)
        for i in range(len(th_wrap)):
            if th_wrap[i] < 0:
                th_wrap[i] += 2*np.pi
        return th_wrap - np.pi
