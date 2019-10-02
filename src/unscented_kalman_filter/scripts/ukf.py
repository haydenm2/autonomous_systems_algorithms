#!/usr/bin/env python3

import numpy as np
import control as ct

# Generic Unscented Kalman Filter Approach (From Probablistic Robotics)


class UKF:
    def __init__(self, c, nl, dt=0.1, x0=np.array([[-5], [-3], [90*np.pi/180.0]])):
        self.mu = np.copy(x0)                        # state mean vector
        self.mu_a = np.hstack((self.mu.transpose(), np.zeros((1, 4)))).transpose()
        self.mu_bar = np.copy(x0)                 # state mean prediction vector
        self.cov = np.eye(3) * 0.1                 # state covariance
        self.cov_bar = np.eye(3) * 0.1             # state covariance prediction
        self.cov_a = np.zeros([7, 7])
        self.K = np.zeros([6, 1])
        self.M = np.zeros([2, 2])
        self.Q = np.zeros([2, 2])
        self.c = np.zeros([3, 2])
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
        self.L = 2 * self.n + 1
        self.kappa = 4
        self.alpha = 0.4
        self.beta = 2
        self.lmbda = self.alpha**2 * (self.L + self.kappa) - self.n
        self.gamma = np.sqrt(self.L + self.lmbda)
        self.sigma_pts_x = np.zeros([self.n, 15])
        self.sigma_pts_x_bar = np.zeros([self.n, 15])
        self.sigma_pts_u = np.zeros([2, 15])
        self.sigma_pts_z = np.zeros([2, 15])
        self.wm = np.zeros([len(self.sigma_pts_x[0]), 1])
        self.wc = np.zeros([len(self.sigma_pts_x[0]), 1])

        # Landmark Locations
        self.nl = nl
        self.c = c

    def Propogate(self, u, z):
        self.PredictState(u)
        for i in range(int(len(z)/2)):
            self.AddMeasurement(z[2*i:2*(i+1)], i)

    def PredictState(self, u):

        # Generate Augmented Mean and Covariance Matrices
        vt = u[0, 0]
        wt = u[1, 0]

        self.M[0, 0] = self.a_1*np.power(vt, 2) + self.a_2*np.power(wt, 2)
        self.M[1, 1] = self.a_3*np.power(vt, 2) + self.a_4*np.power(wt, 2)

        self.Q[0, 0] = np.power(self.sig_r, 2)
        self.Q[1, 1] = np.power(self.sig_phi, 2)

        # Generate Sigma Points
        self.GenerateSigmaPoints(self.mu, self.cov)
        self.PropogateSigmaPoints(u + self.sigma_pts_u, self.sigma_pts_x)

    def GenerateSigmaPoints(self, mu, cov):
        mu_a = np.hstack([mu.transpose(), np.zeros([1, 4])]).transpose()
        cov_a = np.block([[cov, np.zeros([3, 4])], [np.zeros([2, 3]), self.M, np.zeros([2, 2])], [np.zeros([2, 5]), self.Q]])
        sigma_pts_a = np.hstack([mu_a, mu_a + self.gamma*np.linalg.cholesky(cov_a), mu_a - self.gamma*np.linalg.cholesky(cov_a)])
        i = 2*self.L + 1
        self.sigma_pts_x = sigma_pts_a[0:self.n, 0:i]
        self.sigma_pts_u = sigma_pts_a[self.n:self.n + 2, 0:i]
        self.sigma_pts_z = sigma_pts_a[self.n + 2:self.n + 4, 0:i]

    def PropogateSigmaPoints(self, ctl_pts, state_pts):
        self.sigma_pts_x_bar = np.zeros(np.shape(self.sigma_pts_x))

        for i in range(len(self.sigma_pts_x_bar[0])):
            vt = ctl_pts[0, i]
            wt = ctl_pts[1, i]
            x = state_pts[0, i]
            y = state_pts[1, i]
            theta = state_pts[2, i]
            self.sigma_pts_x_bar[0, i] = x + (-(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt))
            self.sigma_pts_x_bar[1, i] = y + (vt/wt)*np.cos(theta)-(vt/wt)*np.cos(theta+wt*self.dt)
            self.sigma_pts_x_bar[2, i] = theta + wt*self.dt

        # Define weights
        self.wm[0] = self.lmbda / (self.L + self.lmbda)
        self.wc[0] = self.wm[0] + (1 - self.alpha ** 2 + self.beta)
        for j in range(1, len(self.sigma_pts_x_bar[0])):
            self.wm[j] = 1/(2*(self.L + self.lmbda))
            self.wc[j] = 1/(2*(self.L + self.lmbda))

        self.mu_bar = self.sigma_pts_x_bar @ self.wm
        cov_temp = np.zeros(np.shape(self.cov))
        for k in range(len(self.sigma_pts_x_bar[0])):
            cov_temp += (self.wc[k] * ((self.sigma_pts_x_bar[:, k].reshape(np.shape(self.mu_bar)) - self.mu_bar) @ (self.sigma_pts_x_bar[:, k].reshape(np.shape(self.mu_bar)) - self.mu_bar).transpose()))
        self.cov_bar = cov_temp

    def AddMeasurement(self, z, i):
        self.GenerateSigmaPoints(self.mu_bar, self.cov_bar)
        Z_bar = np.zeros(np.shape(self.sigma_pts_z))
        for j in range(len(self.sigma_pts_x[0])):
            x = self.sigma_pts_x[0, j]
            y = self.sigma_pts_x[1, j]
            theta = self.sigma_pts_x[2, j]
            Z_bar[0, j] = np.sqrt(np.power(self.c[i, 0] - x, 2) + np.power(self.c[i, 1] - y, 2))
            Z_bar[1, j] = np.arctan2(self.c[i, 1] - y, self.c[i, 0] - x) - theta
        Z_bar = Z_bar + self.sigma_pts_z
        z_hat = Z_bar @ self.wm
        S = np.zeros([len(z_hat), len(z_hat)])
        for k in range(len(Z_bar[0])):
            S += (self.wc[k] * ((Z_bar[:, k].reshape(np.shape(z_hat)) - z_hat) @ (Z_bar[:, k].reshape(np.shape(z_hat)) - z_hat).transpose()))
        cov_xz = np.zeros([len(self.sigma_pts_x), len(z_hat)])
        for l in range(len(Z_bar[0])):
            cov_xz += (self.wc[l] * ((self.sigma_pts_x[:, l].reshape(np.shape(self.mu_bar)) - self.mu_bar) @ (Z_bar[:, l].reshape(np.shape(z_hat)) - z_hat).transpose()))
        K = cov_xz @ np.linalg.inv(S)
        self.mu = self.mu_bar + K @ (z-z_hat)
        self.cov = self.cov_bar - K @ (S @ K.transpose())
        self.mu_bar = self.mu
        self.cov_bar = self.cov
        self.K = K.reshape((6, 1))
