#!/usr/bin/env python3

import numpy as np

# Generic FAST SLAM Filter Approach (From Probablistic Robotics)

class FAST_SLAM:
    def __init__(self, twr):
        # Landmark Locations
        self.nl = twr.nl
        self.n = twr.n

        self.mu = np.vstack((twr.x0, np.zeros([2*self.nl, 1])))       # state mean vector
        self.mu_bar = np.copy(self.mu)                            # state mean prediction vector
        self.cov = np.eye(3 + 2*self.nl)*10000                         # state covariance
        self.cov[0, 0] = 0
        self.cov[1, 1] = 0
        self.cov[2, 2] = 0
        self.cov_bar = np.eye(3 + 2*self.nl)*10000                          # state covariance prediction
        self.cov_bar[0, 0] = 0
        self.cov_bar[1, 1] = 0
        self.cov_bar[2, 2] = 0  
        self.G = np.eye(3 + 2*self.nl)
        self.V = np.zeros([3, 2])
        self.K = np.zeros([2*(3 + 2*self.nl), 1])
        self.M = np.zeros([2, 2])
        self.R = np.zeros([3, 3])
        self.Q = np.zeros([2, 2])

        self.a_1 = twr.a_1
        self.a_2 = twr.a_2
        self.a_3 = twr.a_3
        self.a_4 = twr.a_4
        self.sig_r = twr.sig_r
        self.sig_phi = twr.sig_phi
        self.dt = twr.dt
        self.landmark_seen = np.zeros((1, self.nl))

        self.Fx = np.hstack((np.eye(3), np.zeros([3, 2*self.nl])))

        self.Q[0, 0] = np.power(self.sig_r, 2)
        self.Q[1, 1] = np.power(self.sig_phi, 2)

        self.g = twr.g
        self.h = twr.h
        self.h_inv = twr.h_inv

        # Particle parameters
        self.M = 1000
        self.X = np.zeros([self.n + 2*self.nl, self.M])
        for i in range(self.M):
            self.X[0, i] = self.mu[0]  # (2 * np.random.rand() - 1) * 10
            self.X[1, i] = self.mu[1]  # (2 * np.random.rand() - 1) * 10
            self.X[2, i] = self.mu[2]  # (2 * np.random.rand() - 1) * np.pi
        self.X_bar = np.empty([self.n + 2*self.nl, 0])
        self.W_bar = np.empty([1, 0])
        self.W = np.copy(self.W_bar)
        self.EstimateMeanCov()


    def Propogate(self, u, z):
        self.X_bar = np.empty([self.n, 0])
        self.W_bar = np.empty([1, 0])

        self.X_bar = self.PropogatePoints(u, self.X)

        if not(np.count_nonzero(np.isnan(z)) == len(z)):
            self.W_bar = self.AddMeasurement(z, self.X_bar)
            self.W_bar = self.W_bar / np.sum(self.W_bar)
            self.W = self.W_bar
            self.Resample()
            self.EstimateMeanCov()
        else:
            self.EstimateMeanCov()


    def PredictState(self, u):
        theta = (self.mu[2])[0]
        vt = u[0, 0]
        wt = u[1, 0]

        mu_tilde = np.zeros([3, 1])
        mu_tilde[0] = (-(vt / wt) * np.sin(theta) + (vt / wt) * np.sin(theta + wt * self.dt))
        mu_tilde[1] = (vt / wt) * np.cos(theta) - (vt / wt) * np.cos(theta + wt * self.dt)
        mu_tilde[2] = wt * self.dt
        self.mu_bar = self.mu + self.Fx.transpose() @ mu_tilde

        G_tilde = np.zeros([3, 3])
        G_tilde[0, 2] = -(vt/wt)*np.cos(theta)+(vt/wt)*np.cos(theta+wt*self.dt)
        G_tilde[1, 2] = -(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt)
        self.G = np.eye(3 + 2*self.nl) + self.Fx.transpose() @ G_tilde @ self.Fx

        self.V[0, 0] = (-np.sin(theta) + np.sin(theta + wt * self.dt)) / wt
        self.V[0, 1] = vt * (np.sin(theta) - np.sin(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.cos(theta + wt * self.dt) * self.dt) / wt
        self.V[1, 0] = (np.cos(theta) - np.cos(theta + wt * self.dt)) / wt
        self.V[1, 1] = -vt * (np.cos(theta) - np.cos(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.sin(theta + wt * self.dt) * self.dt) / wt
        self.V[2, 1] = self.dt
        self.M[0, 0] = self.a_1*np.power(vt, 2) + self.a_2*np.power(wt, 2)
        self.M[1, 1] = self.a_3*np.power(vt, 2) + self.a_4*np.power(wt, 2)
        self.R = self.V @ self.M @ self.V.transpose()
        self.cov_bar = self.G @ self.cov @ self.G.transpose() + self.Fx.transpose() @ self.R @ self.Fx

    def AddMeasurement(self, z):
        for j in range(self.nl):
            zj = z[(2*j):(2*j + 2)]
            zr = zj[0]
            zphi = zj[1]
            if np.isnan(zr) or np.isnan(zphi):
                continue
            if not(self.landmark_seen[0, j]):
                self.mu_bar[3 + 2*j] = self.mu_bar[0] + zr*np.cos(zphi + self.mu_bar[2])
                self.mu_bar[3 + 2*j + 1] = self.mu_bar[1] + zr*np.sin(zphi + self.mu_bar[2])
                self.landmark_seen[0, j] = 1
            dx = (self.mu_bar[3 + 2 * j] - self.mu_bar[0])[0]
            dy = (self.mu_bar[3 + 2 * j + 1] - self.mu_bar[1])[0]
            delta = np.array([[dx], [dy]])
            q = (delta.transpose() @ delta)[0, 0]
            zhat = np.array([[np.sqrt(q)], [self.Wrap(np.arctan2(dy, dx) - self.mu_bar[2, 0])]])
            H_base = np.array([[-np.sqrt(q)*dx, -np.sqrt(q)*dy, 0, np.sqrt(q)*dx, np.sqrt(q)*dy], [dy, -dx, -q, -dy, dx]])
            H = 1/q * np.hstack((H_base[:, 0:3], np.zeros((2, 2*j)), H_base[:, 3:5], np.zeros((2, 2*self.nl - 2*j - 2))))
            K = self.cov_bar @ H.transpose() @ np.linalg.inv(H @ self.cov_bar @ H.transpose() + self.Q)
            zdiff = zj - zhat
            zdiff[1] = self.Wrap(zdiff[1])
            self.mu_bar = self.mu_bar + K @ zdiff
            self.cov_bar = (np.eye(3 + 2*self.nl) - K @ H) @ self.cov_bar
        self.mu = self.mu_bar
        self.cov = self.cov_bar
        self.K = K.reshape((2*(3 + 2*self.nl), 1))

    def Resample(self):
        X_bar = np.empty([self.n, 0])
        self.W_bar = np.empty([1, 0])
        Minv = 1/self.M
        r = np.random.uniform(low=0, high=Minv)
        c = self.W[0, 0]
        i = 0
        for m in range(self.M):
            U = r + m*Minv
            while U > c:
                i = i+1
                c = c + self.W[0, i]
            X_bar = np.hstack((X_bar, self.X_bar[:, i].reshape((3, 1))))
        uniq = len(np.unique(X_bar))

        # introduce synthetic noise if convergence is too fast
        if uniq/self.M < 0.5:
            Q = self.cov/((self.M*uniq)**(1/self.n))
            X_bar = X_bar + Q @ np.random.randn(np.shape(X_bar)[0], np.shape(X_bar)[1])
        self.X = X_bar


    def EstimateMeanCov(self):
        self.mu = np.mean(self.X, axis=1).reshape((self.n + 2*self.nl, 1))
        E = (self.X - self.mu)
        E[2, :] = self.Wrap(E[2, :])
        self.cov = 1/(len(self.X[0]) - 1) * (E @ E.transpose())
        pass

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

