#!/usr/bin/env python3

import numpy as np
import control as ct

# Generic Unscented Kalman Filter Approach (From Probablistic Robotics)


class UKF:
    def __init__(self, c, nl, dt=0.1, x0=np.array([[-5], [-3], [90*np.pi/180.0]])):
        self.u = np.zeros([2, 1])             # input command history
        self.z = np.zeros([3, 1])             # measurement history
        self.mu = np.copy(x0)                        # state mean vector
        self.mu_a = np.hstack((self.mu.transpose(), np.zeros((1, 4)))).transpose()
        self.mu_bar = np.copy(x0)                 # state mean prediction vector
        self.cov = np.eye(3) * 0.1                 # state covariance
        self.cov_bar = np.eye(3)               # state covariance prediction
        self.cov_a = np.zeros([7, 7])
        # self.G = np.eye(3)
        # self.V = np.zeros([3, 2])
        self.K = np.zeros([6, 1])
        self.M = np.zeros([2, 2])
        self.Q = np.zeros([2, 2])
        self.c = np.zeros([3, 2])
        self.pz = 0
        self.dt = dt

        # Noise Parameters
        self.a_1 = 0.1
        self.a_2 = 0.01
        self.a_3 = 0.01
        self.a_4 = 0.1
        self.sig_r = 0.1
        self.sig_phi = 0.05

        # Sigma Point Variables
        self.n = len(self.mu)
        self.kappa = 4
        self.alpha = 0.4
        self.beta = 2
        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lmbda)
        self.sigma_pts_x = np.zeros([self.n, 15])
        self.sigma_pts_u = np.zeros([2, 15])
        self.sigma_pts_z = np.zeros([2, 15])

        # Landmark Locations
        self.nl = nl
        self.c = c

    def Propogate(self, u, z):
        self.PredictState(u)
        self.AddMeasurement(z)

    def PredictState(self, u):

        # self.G[0, 2] = -(vt/wt)*np.cos(theta)+(vt/wt)*np.cos(theta+wt*self.dt)
        # self.G[1, 2] = -(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt)

        # self.V[0, 0] = (-np.sin(theta) + np.sin(theta + wt * self.dt)) / wt
        # self.V[0, 1] = vt * (np.sin(theta) - np.sin(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.cos(theta + wt * self.dt) * self.dt) / wt
        # self.V[1, 0] = (np.cos(theta) - np.cos(theta + wt * self.dt)) / wt
        # self.V[1, 1] = -vt * (np.cos(theta) - np.cos(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.sin(theta + wt * self.dt) * self.dt) / wt
        # self.V[2, 1] = self.dt

        # Generate Augmented Mean and Covariance Matrices
        vt = u[0, 0]
        wt = u[1, 0]

        self.M[0, 0] = self.a_1*np.power(vt, 2) + self.a_2*np.power(wt, 2)
        self.M[1, 1] = self.a_3*np.power(vt, 2) + self.a_4*np.power(wt, 2)

        self.Q[0, 0] = np.power(self.sig_r, 2)
        self.Q[1, 1] = np.power(self.sig_phi, 2)

        self.mu_a = np.hstack([self.mu.transpose(), np.zeros([1, 4])]).transpose()

        self.cov_a = np.block([[self.cov, np.zeros([3, 4])], [np.zeros([2, 3]), self.M, np.zeros([2, 2])], [np.zeros([2, 5]), self.Q]])

        # Generate Sigma Points
        self.GenerateSigmaPoints()
        self.PropogateSigmaPoints(u + self.sigma_pts_u, self.sigma_pts_x)


    def AddMeasurement(self, z):
        for j in range(self.nl):
            dx = (self.c[j, 0] - self.mu_bar[0])[0]
            dy = (self.c[j, 1] - self.mu_bar[1])[0]
            q = np.power(dx, 2) + np.power(dy, 2)
            z_hat = np.array([[np.sqrt(q)], [np.arctan2(dy, dx) - self.mu_bar[2, 0]]])
            H = np.array([[-dx/np.sqrt(q), -dy/np.sqrt(q), 0], [dy/q, -dx/q, -1]])
            S = H @ (self.cov_bar @ H.transpose()) + self.Q
            K = self.cov_bar @ (H.transpose() @ np.linalg.inv(S))
            self.mu_bar = self.mu_bar + K @ (z[(0+j*2):(2+j*2)] - z_hat)
            self.cov_bar = (np.eye(3) - (K @ H)) @ self.cov_bar
        self.mu = self.mu_bar
        self.cov = self.cov_bar
        self.K = K.reshape((6, 1))

    def GenerateSigmaPoints(self):
        sigma_pts_a = np.hstack([self.mu_a, self.mu_a + self.gamma*np.linalg.cholesky(self.cov_a), self.mu_a - self.gamma*np.linalg.cholesky(self.cov_a)])
        L = 2*self.n + 1
        i = 2*L + 1
        self.sigma_pts_x = sigma_pts_a[0:self.n, 0:i]
        self.sigma_pts_u = sigma_pts_a[self.n:self.n + 2, 0:i]
        self.sigma_pts_z = sigma_pts_a[self.n+2:self.n + 4, 0:i]

    def PropogateSigmaPoints(self, ctl_pts, state_pts):
        sigma_pts_x_bar = np.zeros(np.shape(self.sigma_pts_x))

        for i in range(len(sigma_pts_x_bar[0])):
            vt = ctl_pts[0, i]
            wt = ctl_pts[1, i]
            x = state_pts[0, i]
            y = state_pts[1, i]
            theta = state_pts[2, i]
            sigma_pts_x_bar[0, i] = x + (-(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt))
            sigma_pts_x_bar[1, i] = y + (vt/wt)*np.cos(theta)-(vt/wt)*np.cos(theta+wt*self.dt)
            sigma_pts_x_bar[2, i] = theta + wt*self.dt

        # Define weights
        wm = np.zeros([len(self.sigma_pts_x[0]), 1])
        wc = np.zeros([len(self.sigma_pts_x[0]), 1])
        wm[0] = self.lmbda / (self.n + self.lmbda)
        wc[0] = wm[0] + (1 - self.alpha ** 2 + self.beta)
        for j in range(1, len(sigma_pts_x_bar[0])):
            wm[j] = 1/(2*(self.n + self.lmbda))
            wc[j] = 1/(2*(self.n + self.lmbda))

        self.mu_bar = sigma_pts_x_bar @ wm
        cov_temp = np.zeros(np.shape(self.cov))
        for k in range(len(sigma_pts_x_bar[0])):
            cov_temp = cov_temp + (wc[k] * ((sigma_pts_x_bar[:, k] - self.mu_bar) @ (sigma_pts_x_bar[:, k] - self.mu_bar).transpose()))
        self.cov_bar = cov_temp


