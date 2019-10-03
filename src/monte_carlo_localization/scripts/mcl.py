#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

# Generic Monte Carlo Localization Filter Approach (From Probablistic Robotics)

class MCL:
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

        # Particle parameters
        self.pts = 1000
        self.X_prev = np.zeros([self.n + 1, self.pts])
        for i in range(self.pts):
            self.X_prev[0, i] = (2*np.random.rand()-1)*10
            self.X_prev[1, i] = (2*np.random.rand()-1)*10
        self.X_bar = np.empty([self.n + 1, 0])
        # self.X = np.empty([self.n + 1, 0])
        self.X = np.copy(self.X_prev)

    def Propogate(self, u, z):
        self.PredictState(u)
        for i in range(int(len(z)/2)):
            self.AddMeasurement(z[2*i:2*(i+1)], i)

        for m in range(self.pts):
            xt = self.SampleMotionModel(u, self.X_prev[:, m].reshape(self.n+1, 1))
            wt = self.MeasurementModel(z, xt)
            x = np.vstack((xt, wt))
            self.X_bar = np.hstack((self.X_bar, x))
        self.X = self.X_bar
        self.X_bar = np.empty([self.n + 1, 0])
        self.X_prev = self.X

    def SampleMotionModel(self, u, xprev):
        v_hat = u[0] + np.sqrt(self.a_1 * np.power(u[0], 2) + self.a_2 * np.power(u[1], 2)) * np.random.randn()
        w_hat = u[1] + np.sqrt(self.a_3 * np.power(u[0], 2) + self.a_4 * np.power(u[1], 2)) * np.random.randn()
        x = np.zeros([3, 1])
        x[0] = xprev[0] - (v_hat / w_hat) * np.sin(xprev[2]) + (v_hat / w_hat) * np.sin(xprev[2] + w_hat * self.dt)
        x[1] = xprev[1] + (v_hat / w_hat) * np.cos(xprev[2]) - (v_hat / w_hat) * np.cos(xprev[2] + w_hat * self.dt)
        x[2] = xprev[2] + w_hat * self.dt
        return x

    def MeasurementModel(self, z, x):
        q = np.zeros([1, int(len(z)/2)])
        for i in range(int(len(z)/2)):
            r = np.sqrt((self.c[i, 0] - x[0])**2 + (self.c[i, 1] - x[1])**2)
            phi = np.arctan2(self.c[i, 1] - x[1], self.c[i, 0] - x[0])
            q[0, i] = norm(0, self.sig_r).pdf(z[2*i]-r[0]) * norm(0, self.sig_phi).pdf(np.mod(z[2*i+1]-phi[0], np.pi))
        w = np.array([np.prod(q)])
        return w

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
