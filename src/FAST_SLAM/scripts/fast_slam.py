#!/usr/bin/env python3

import numpy as np

# Generic FAST SLAM Filter Approach (From Probablistic Robotics)

class FAST_SLAM:
    def __init__(self, twr):
        # Landmark Locations
        self.nl = twr.nl
        self.n = twr.n

        # Particle parameters
        self.M = 1000
        self.X = np.zeros([self.n + 2 * self.nl, self.M])
        for i in range(self.M):
            self.X[0, i] = twr.x0[0]  # (2 * np.random.rand() - 1) * 10
            self.X[1, i] = twr.x0[1]  # (2 * np.random.rand() - 1) * 10
            self.X[2, i] = twr.x0[2]  # (2 * np.random.rand() - 1) * np.pi
        self.X_bar = np.empty([self.n + 2 * self.nl, 0])
        self.W_bar = np.empty([1, 0])
        self.W = np.copy(self.W_bar)
        self.p0 = 1/self.M
        self.EstimateMeanCov()

        self.mu = np.vstack((twr.x0, np.zeros([2*self.nl, 1])))
        self.mu_bar = np.copy(self.mu)
        self.cov = np.tile(np.eye(3 + 2*self.nl), self.M)*10000
        self.cov[0, 0] = 0
        self.cov[1, 1] = 0
        self.cov[2, 2] = 0
        self.cov_bar = np.eye(3 + 2 * self.nl)
        self.cov_bar[0, 0] = 0
        self.cov_bar[1, 1] = 0
        self.cov_bar[2, 2] = 0

        self.G = np.eye(3 + 2*self.nl)
        self.V = np.zeros([3, 2])
        self.R = np.zeros([3, 3])
        self.Q = np.zeros([2, 2])

        self.a_1 = twr.a_1
        self.a_2 = twr.a_2
        self.a_3 = twr.a_3
        self.a_4 = twr.a_4
        self.sig_r = twr.sig_r
        self.sig_phi = twr.sig_phi
        self.dt = twr.dt
        self.landmark_seen = np.zeros((self.M, self.nl))

        self.Fx = np.hstack((np.eye(3), np.zeros([3, 2*self.nl])))

        self.Q[0, 0] = np.power(self.sig_r, 2)
        self.Q[1, 1] = np.power(self.sig_phi, 2)

        self.g = twr.g
        self.h = twr.h
        self.h_inv = twr.h_inv
        self.H = twr.H


    def Propogate(self, u, z):
        self.X_bar = np.empty([self.n, 0])
        self.W_bar = np.ones([1, self.M])*1/self.M

        self.X_bar = self.PropogatePoints(u, self.X)

        if not(np.count_nonzero(np.isnan(z)) == len(z)):
            self.MeasurementModel(z)
            self.W_bar = self.W_bar / np.sum(self.W_bar)
            self.W = self.W_bar
            self.Resample()
            self.EstimateMeanCov()
        else:
            self.EstimateMeanCov()

    def PropogatePoints(self, u, xprev):
        v_hat = u[0] + np.sqrt(self.a_1 * u[0] ** 2 + self.a_2 * u[1] ** 2) * np.random.randn(1, len(xprev[0]))
        w_hat = u[1] + np.sqrt(self.a_3 * u[0] ** 2 + self.a_4 * u[1] ** 2) * np.random.randn(1, len(xprev[0]))
        # x = np.zeros(np.shape(xprev))
        xprev[0, :] = xprev[0, :] - (v_hat / w_hat) * np.sin(xprev[2, :]) + (v_hat / w_hat) * np.sin(
            xprev[2, :] + w_hat * self.dt)
        xprev[1, :] = xprev[1, :] + (v_hat / w_hat) * np.cos(xprev[2, :]) - (v_hat / w_hat) * np.cos(
            xprev[2, :] + w_hat * self.dt)
        xprev[2, :] = xprev[2, :] + w_hat * self.dt
        return xprev

    def MeasurementModel(self, z):
        for i in range(self.M):
            for j in range(self.nl):
                zj = z[(2*j):(2*j + 2)]
                zr = zj[0]
                zphi = zj[1]
                indx = ((3 + 2 * self.nl) * i + (3 + 2 * j))
                indy = (3 + 2 * j)
                if np.isnan(zr) or np.isnan(zphi):
                    continue
                if not(self.landmark_seen[i, j]):
                    self.X_bar[(3 + 2 * j):(3 + 2 * j + 2), i] = self.h_inv(self.X_bar[0:3, i], zj).flatten()
                    H_inv = np.linalg.inv(self.H(self.X_bar[(3 + 2*j):(3 + 2*j + 2), i], self.X_bar[0:3, i]))
                    self.cov[indy:(indy+2), indx:(indx+2)] = H_inv @ self.Q @ H_inv.transpose()
                    self.W_bar[0, i] = self.p0
                    self.landmark_seen[i, j] = 1
                else:
                    zhat = self.h(self.X_bar[0:3, i], self.X_bar[(3 + 2*j):(3 + 2*j+2), i])
                    H = self.H(self.X_bar[(3 + 2*j):(3 + 2*j + 2), i], self.X_bar[0:3, i])
                    Q = H @ self.cov[indy:(indy+2), indx:(indx+2)] @ H.transpose() + self.Q
                    K = self.cov[indy:(indy+2), indx:(indx+2)] @ H.transpose() @ np.linalg.inv(Q)
                    self.X_bar[(3 + 2 * j):(3 + 2 * j + 2), i] = self.X_bar[(3 + 2 * j):(3 + 2 * j + 2), i] + (K @ (zj - zhat)).flatten()
                    self.cov[indy:(indy + 2), indx:(indx + 2)] = (np.eye(2) - K @ H) @ self.cov[indy:(indy+2), indx:(indx+2)]
                    zdiff = zj - zhat
                    self.W_bar[0, i] = np.linalg.det(2*np.pi*Q)**(-1/2) * np.exp(-1/2 * zdiff.transpose() @ np.linalg.inv(Q) @ zdiff)

    def Resample(self):
        X_bar = np.empty([self.n + 2*self.nl, 0])
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
            X_bar = np.hstack((X_bar, self.X_bar[:, i].reshape((self.n + 2*self.nl, 1))))
        uniq = len(np.unique(X_bar))

        # introduce synthetic noise if convergence is too fast
        if uniq/self.M < 0.5:
            Q = self.cov_bar/((self.M*uniq)**(1/self.n))
            X_bar = X_bar + Q @ np.random.randn(np.shape(X_bar)[0], np.shape(X_bar)[1])
        self.X = X_bar


    def EstimateMeanCov(self):
        self.mu_bar = np.mean(self.X, axis=1).reshape((self.n + 2*self.nl, 1))
        E = (self.X - self.mu_bar)
        E[2, :] = self.Wrap(E[2, :])
        self.cov_bar = 1/(len(self.X[0]) - 1) * (E @ E.transpose())
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
