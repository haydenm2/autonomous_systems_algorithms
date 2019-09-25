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
    def __init__(self, t_end=20, dt=0.1):

        # Time parameters
        self.t_end = t_end        # completion time
        self.dt = dt              # time step

        # Noise characteristics of motion
        self.a_1 = 0.1
        self.a_2 = 0.01
        self.a_3 = 0.01
        self.a_4 = 0.1
        self.sig_r = 0.1
        self.sig_phi = 0.05

        # Plot data containers
        self.x = np.zeros([3, 1])  # state truth vector
        self.x = np.array([[-5], [-3], [90*(np.pi/180)]])   # initial states
        self.z1 = np.zeros([2, 1])  # measurement vector
        self.z2 = np.zeros([2, 1])  # measurement vector
        self.z3 = np.zeros([2, 1])  # measurement vector
        self.u = np.zeros([2, 1])  # input command vector
        self.t = np.zeros(1)       # time vector

        # Landmark Locations
        self.l1 = np.array([6, 4])
        self.l2 = np.array([-7, 8])
        self.l3 = np.array([6, -4])
        self.c = np.vstack((self.l1, self.l2, self.l3))

        # Truth Propogation Containers
        self.x_new = np.zeros([3, 1])
        self.u_new = np.array([[1.5], [1.8]])
        self.t_new = np.zeros(1)
        self.z1_new = np.zeros([2, 1])
        self.z2_new = np.zeros([2, 1])
        self.z3_new = np.zeros([2, 1])

    def Propagate(self):

        # motion model calculations
        self.t_new[0] = self.t[np.size(self.t)-1] + self.dt
        self.u_new[0] = 1 + 0.5 * np.cos(2 * np.pi * (0.2) * self.t_new)
        self.u_new[1] = -0.2 + 2 * np.cos(2 * np.pi * (0.6) * self.t_new)

        v = self.u_new[0] #+ np.sqrt(self.a_1 * np.power(self.u_new[0], 2) + self.a_2 * np.power(self.u_new[1], 2)) * np.random.randn()
        w = self.u_new[1] #+ np.sqrt(self.a_3 * np.power(self.u_new[0], 2) + self.a_4 * np.power(self.u_new[1], 2)) * np.random.randn()
        gamma = 0
        self.x_new[0] = self.x[0, len(self.x[0])-1] - (v/w) * np.sin(self.x[2, len(self.x[0])-1]) + (v/w) * np.sin(self.x[2, len(self.x[0])-1] + w*self.dt)
        self.x_new[1] = self.x[1, len(self.x[0])-1] + (v/w) * np.cos(self.x[2, len(self.x[0])-1]) - (v/w) * np.cos(self.x[2, len(self.x[0])-1] + w*self.dt)
        self.x_new[2] = self.x[2, len(self.x[0])-1] + w * self.dt + gamma * self.dt

        self.z1_new[0] = np.sqrt(np.power(self.l1[0] - self.x_new[0], 2) + np.power(self.l1[1] - self.x_new[1], 2))
        self.z1_new[1] = np.arctan2(self.l1[1] - self.x_new[1], self.l1[0] - self.x_new[0]) - self.x_new[2]

        self.z2_new[0] = np.sqrt(np.power(self.l2[0] - self.x_new[0], 2) + np.power(self.l2[1] - self.x_new[1], 2))
        self.z2_new[1] = np.arctan2(self.l2[1] - self.x_new[1], self.l2[0] - self.x_new[0]) - self.x_new[2]

        self.z3_new[0] = np.sqrt(np.power(self.l3[0] - self.x_new[0], 2) + np.power(self.l3[1] - self.x_new[1], 2))
        self.z3_new[1] = np.arctan2(self.l3[1] - self.x_new[1], self.l3[0] - self.x_new[0]) - self.x_new[2]

        # update truth/measurement data vectors
        self.t = np.hstack((self.t, self.t_new))
        self.x = np.hstack((self.x, self.x_new))
        self.u = np.hstack((self.u, self.u_new))
        self.z1 = np.hstack((self.z1, self.z1_new))
        self.z2 = np.hstack((self.z2, self.z2_new))
        self.z3 = np.hstack((self.z3, self.z3_new))

    def Getx(self):
        return self.x_new

    def Getu(self):
        return self.u_new

    def Getz1(self):
        return self.z1_new

    def Getz2(self):
        return self.z2_new

    def Getz3(self):
        return self.z3_new

    def Getz(self):
        return (np.vstack((self.z1_new, self.z2_new, self.z3_new)))


