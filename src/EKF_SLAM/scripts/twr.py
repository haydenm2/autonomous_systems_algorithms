#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

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
    def __init__(self, t_end=60, dt=0.1, n=8):

        # Time parameters
        self.t_end = t_end        # completion time
        self.dt = dt              # time step
        self.init = True

        # Noise characteristics of motion
        self.a_1 = 0.1 /2
        self.a_2 = 0.01 /2
        self.a_3 = 0.01 /2
        self.a_4 = 0.1 /2

        # Sensor Parameters
        self.fov = 45 * np.pi/180
        self.sig_r = 0.1
        self.sig_phi = 0.05

        # Landmark Locations
        self.nl = n
        self.c = np.zeros([self.nl, 2])

        # Random Landmark Generator
        for k in range(self.nl):
            self.c[k] = np.array([(np.random.rand()*2-1)*10, (np.random.rand()*2-1)*10])

        # Plot data containers
        self.x0 = np.array([[-5], [-3], [90*(np.pi/180)]])
        self.x = self.x0
        self.z = np.zeros([2, self.nl])
        for i in range(self.nl):
            self.z[0, i] = np.sqrt(np.power(self.c[i, 0] - self.x[0], 2) + np.power(self.c[i, 1] - self.x[1], 2))
            self.z[1, i] = self.Wrap(np.arctan2(self.c[i, 1] - self.x[1], self.c[i, 0] - self.x[0]) - self.x[2])
            if np.abs(self.z[1, i]) > self.fov/2:
                self.z[:, i] *= np.nan
            else:
                self.z[0, i] += + self.sig_r * np.random.randn()
                self.z[1, i] = self.Wrap(self.z[1, i] + self.sig_phi * np.random.randn())
        self.u = np.zeros([2, 1])  # input command vector
        self.t = np.zeros(1)       # time vector

        # Truth Propogation Containers
        self.x_new = np.copy(self.x)
        self.u_new = np.array([[1.5], [1.8]])
        self.t_new = np.zeros(1)
        self.z_new = np.zeros([2, self.nl])


    def Propagate(self):

        # motion model calculations
        self.t_new[0] = self.t[np.size(self.t)-1] + self.dt

        if self.init:
            self.init = False
        else:
            self.u_new[0] = 1 + 0.5 * np.cos(2 * np.pi * (0.2) * self.t_new)
            self.u_new[1] = -0.2 + 2 * np.cos(2 * np.pi * (0.6) * self.t_new)

        v = self.u_new[0] + np.sqrt(self.a_1 * np.power(self.u_new[0], 2) + self.a_2 * np.power(self.u_new[1], 2)) * np.random.randn()
        w = self.u_new[1] + np.sqrt(self.a_3 * np.power(self.u_new[0], 2) + self.a_4 * np.power(self.u_new[1], 2)) * np.random.randn()
        gamma = 0
        self.x_new[0] = self.x[0, len(self.x[0])-1] - (v/w) * np.sin(self.x[2, len(self.x[0])-1]) + (v/w) * np.sin(self.x[2, len(self.x[0])-1] + w*self.dt)
        self.x_new[1] = self.x[1, len(self.x[0])-1] + (v/w) * np.cos(self.x[2, len(self.x[0])-1]) - (v/w) * np.cos(self.x[2, len(self.x[0])-1] + w*self.dt)
        self.x_new[2] = self.x[2, len(self.x[0])-1] + w * self.dt + gamma * self.dt

        for i in range(self.nl):
            self.z_new[0, i] = np.sqrt(np.power(self.c[i, 0] - self.x_new[0], 2) + np.power(self.c[i, 1] - self.x_new[1], 2)) + self.sig_r*np.random.randn()
            self.z_new[1, i] = self.Wrap(np.arctan2(self.c[i, 1] - self.x_new[1], self.c[i, 0] - self.x_new[0]) - self.x_new[2] + self.sig_phi*np.random.randn())
            if np.abs(self.z_new[1, i]) > self.fov/2:
                self.z_new[:, i] *= np.nan
        # update truth/measurement data vectors
        self.t = np.hstack((self.t, self.t_new))
        self.x = np.hstack((self.x, self.x_new))
        self.u = np.hstack((self.u, self.u_new))
        self.z = np.hstack((self.z, self.z_new))

    def Getx(self):
        return self.x_new

    def Getu(self):
        return self.u_new

    def Getz(self):
        z = (self.z[:, len(self.z[0])-self.nl:len(self.z[0])]).transpose()
        return z.reshape((2*self.nl, 1))

    # Get position estimations in x,y coordinates of sensor values
    def Getzpos(self):
        z = self.z[:, len(self.z[0])-self.nl:len(self.z[0])]
        xz = z
        x = self.x_new[0, 0]
        y = self.x_new[1, 0]
        theta = self.x_new[2, 0]
        for i in range(self.nl):
            xz[:, i] = (np.array([np.cos(theta+z[1, i])*z[0, i]+x, np.sin(theta+z[1, i])*z[0, i]+y])).transpose()
        return xz

    def g(self, u, mu):
        x = mu[0] + (-(u[0] / u[1]) * np.sin(mu[2]) + (u[0] / u[1]) * np.sin(mu[2] + u[1] * self.dt))
        y = mu[1] + (u[0] / u[1]) * np.cos(mu[2]) - (u[0] / u[1]) * np.cos(mu[2] + u[1] * self.dt)
        theta = mu[2] + u[1] * self.dt
        return np.array((x, y, theta)).reshape(len(mu), 1)

    def h(self, mu, c):
        dx = c[0]-mu[0]
        dy = c[1]-mu[1]
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - mu[2]
        return np.array((r, self.Wrap(phi))).reshape(2, 1)

    def Wrap(self, th):
        if type(th) is np.ndarray:
            th_wrap = np.fmod(th + np.pi, 2*np.pi)
            for i in range(len(th_wrap)):
                if th_wrap[i] < 0:
                    th_wrap[i] += 2*np.pi
        else:
            th_wrap = np.fmod(th + np.pi, 2 * np.pi)
            if th_wrap < 0:
                th_wrap += 2 * np.pi
        return th_wrap - np.pi
