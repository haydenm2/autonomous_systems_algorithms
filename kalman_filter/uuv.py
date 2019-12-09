#!/usr/bin/env python3

import numpy as np
import control as ct

# Generic Model of Unmanned Underwater Vehicle with basic position sensor
# This model takes in an initial position and propagates states and noisy measurements forward
#
# Unmanned Underwater Vehicle (UUV) system defined by:
#
# m*vdot+b*v=F(t)
# xdot = v
#
# where v is velocity, x is position, F is prop thrust.
# mass(m) = 100 kg, linear drag coeff(b) = 20 N-s/m.


class UUV:
    def __init__(self, x0, duration, dt, R, Q):
        # Time parameters
        self.duration = duration  # duration of run
        self.dt = dt              # time step
        self.i = 0                # iterator variable

        # Model parameters
        self.m = 100   # mass
        self.b = 20    # drag coefficient
        self.R = R     # process covariance
        self.Q = Q     # measurement covariance

        # plot data containers
        self.x = np.zeros([int(self.duration / self.dt) + 1, 2])  # state truth vector
        self.x[self.i] = x0                                       # initial states
        self.z = np.zeros([int(self.duration / self.dt) + 1, 1])  # measurement vector
        self.u = np.zeros([int(self.duration / self.dt) + 1, 1])  # input command vector
        self.t = np.zeros([int(self.duration / self.dt) + 1, 1])  # time vector

        # state space model
        A = np.array([[-self.b / self.m, 0], [1, 0]])       # continuous state space A-matrix
        B = np.array([[1 / self.m], [0]])                   # continuous state space B-matrix
        C = np.array([0, 1])                                # continuous state space C-matrix
        D = np.array(0)                                     # continuous state space D-matrix
        sys = ct.ss(A, B, C, D)                             # continuous LTI system object
        sysd = ct.c2d(sys, self.dt)                         # discrete LTI system object

        self.A = sysd.A                                  # discrete state A-matrix
        self.B = sysd.B                                  # discrete state B-matrix
        self.C = sysd.C                                  # discrete measurement C-matrix

    def Propagate(self, u):
        # random noise variables
        epsilon = np.array([np.sqrt(self.R.item((0, 0))) * np.random.randn(), np.sqrt(self.R.item((1, 1))) * np.random.randn()])
        delta = np.array(np.sqrt(self.Q) * np.random.randn())

        # update truth/measurement data vectors
        self.t[self.i + 1] = self.i * self.dt
        self.x[self.i + 1] = self.A @ self.x[self.i] + self.B @ (np.array([u]).transpose()) + epsilon
        self.z[self.i + 1] = self.C @ self.x[self.i + 1] + delta
        self.i += 1

    def Getx(self):
        return self.x[self.i]

    def Getz(self):
        return self.z[self.i]


