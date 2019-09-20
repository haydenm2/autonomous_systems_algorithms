#!/usr/bin/env python3

import numpy as np
import control as ct

# Generic model of Two-Wheeled Robot operating on a 20mx20m field. Three landmarks
# are continuously visible to the robot. Robot can measure range and bearing to each landmark.
#
# This model takes in an initial position and propagates states and noisy measurements forward
#
# Two-Wheeled Robot (TWR) Dynamics defined by Probablistic Robotics Ch5 motion model:
#
# Initial conditions: x0 = -5m, y0 = -3m, theta0 = 90deg
# Velocity commands given by:
#
# v_c = 1 + 0.5*cos*2*pi*(0.2)*t)
# w_c = -0.2 + 2*cos(2*pi*(o.6)*t)
#
# Velocities experienced by robot are noisy versions of commanded velocities with noise characteristics:
# a_1 = a_4 = 0.1 and a_2 = a_3 = 0.01
# Landmark locations are (6,4),(-7,8), and (6,-4)
# Standard deviation of range and bearing sensor noise for each landmark is given by:
# sigma_r = 0.1m and sigma_phi = 0.05 rad
# Sample period of 0.1s and duration of 20s


class TWR:
    def __init__(self, x0, duration, dt):
        # Time parameters
        self.duration = duration  # duration of run
        self.dt = dt              # time step
        self.i = 0                # iterator variable

        # Noise characteristics of motion
        self.a_1 = 0.1
        self.a_2 = 0.01
        self.a_3 = self.a_2
        self.a_4 = self.a_1

        # plot data containers
        self.x = np.zeros([int(self.duration / self.dt) + 1, 3])  # state truth vector
        self.x[self.i] = x0                                       # initial states
        self.z = np.zeros([int(self.duration / self.dt) + 1, 2])  # measurement vector
        self.u = np.zeros([int(self.duration / self.dt) + 1, 2])  # input command vector
        self.t = np.zeros([int(self.duration / self.dt) + 1, 1])  # time vector


    def Propagate(self, u):
        # random noise variables
        epsilon = np.array([np.sqrt(self.R.item((0, 0))) * np.random.randn(), np.sqrt(self.R.item((1, 1))) * np.random.randn()])
        delta = np.array(np.sqrt(self.Q) * np.random.randn())

        # update truth/measurement data vectors
        self.t[self.i + 1] = self.i * self.dt
        self.x[self.i + 1] = self.A @ self.x[self.i] + self.B @ (np.array([u]).transpose()) + epsilon
        self.z[self.i + 1] = self.C @ self.x[self.i] + delta
        self.i += 1

    def Getx(self):
        return self.x[self.i]

    def Getz(self):
        return self.z[self.i]


