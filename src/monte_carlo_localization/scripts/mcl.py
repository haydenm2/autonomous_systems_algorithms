#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

# Generic Monte Carlo Localization Filter Approach (From Probablistic Robotics)

class MCL:
    def __init__(self, c, nl, dt=0.1, x0=np.array([[-5], [-3], [90*np.pi/180.0]])):
        self.mu = np.copy(x0)                        # state mean vector            # state mean prediction vector
        self.cov = np.eye(3) * 0.1                 # state covariance
        self.dt = dt

        # Noise Parameters
        self.a_1 = 0.1
        self.a_2 = 0.01
        self.a_3 = 0.01
        self.a_4 = 0.1
        self.a_5 = 0.01
        self.a_6 = 0.01
        self.sig_r = 0.1
        self.sig_phi = 0.05

        # Sigma Point Variables
        self.n = len(self.mu)

        # Landmark Locations
        self.nl = nl
        self.c = c

        # Particle parameters
        self.M = 1000
        self.X = np.zeros([self.n, self.M])
        for i in range(self.M):
            self.X[0, i] = (2*np.random.rand()-1)*10
            self.X[1, i] = (2*np.random.rand()-1)*10
            self.X[2, i] = (2*np.random.rand()-1)*np.pi
        self.X_bar = np.empty([self.n, 0])
        self.W_bar = np.empty([1, 0])
        self.W = np.copy(self.W_bar)
        self.EstimateMeanCov()

    def Propogate(self, u, z):
        self.X_bar = np.empty([self.n, 0])
        self.W_bar = np.empty([1, 0])

        # Vectorized Approach
        self.X_bar = self.PropogateMotionModel(u, self.X)
        self.W_bar = self.MeasurementModel(z, self.X_bar)

        # # iterative approach
        # for m in range(self.M):
        #     x = self.PropogateMotionModel(u, self.X[:, m].reshape(self.n, 1))
        #     w = self.MeasurementModel(z, x)
        #     self.X_bar = np.hstack((self.X_bar, x))
        #     self.W_bar = np.hstack((self.W_bar, w.reshape((1, 1))))

        self.W_bar = self.W_bar / np.sum(self.W_bar)
        self.W = self.W_bar
        self.Resample()
        self.EstimateMeanCov()

    # for vectorized approach
    def PropogateMotionModel(self, u, xprev):
        v_hat = u[0] + np.sqrt(self.a_1 * u[0]**2 + self.a_2 * u[1]**2) * np.random.randn(1, len(xprev[0]))
        w_hat = u[1] + np.sqrt(self.a_3 * u[0]**2 + self.a_4 * u[1]**2) * np.random.randn(1, len(xprev[0]))
        x = np.zeros(np.shape(xprev))
        x[0, :] = xprev[0, :] - (v_hat / w_hat) * np.sin(xprev[2, :]) + (v_hat / w_hat) * np.sin(xprev[2, :] + w_hat * self.dt)
        x[1, :] = xprev[1, :] + (v_hat / w_hat) * np.cos(xprev[2, :]) - (v_hat / w_hat) * np.cos(xprev[2, :] + w_hat * self.dt)
        x[2, :] = xprev[2, :] + w_hat * self.dt
        return x

    def MeasurementModel(self, z, x):
        q = np.ones([1, len(x[0])])
        for i in range(int(len(z)/2)):
            r = np.sqrt((self.c[i, 0] - x[0, :])**2 + (self.c[i, 1] - x[1, :])**2).reshape((1, len(x[0])))
            phi = (np.arctan2(self.c[i, 1] - x[1, :], self.c[i, 0] - x[0, :]) - x[2, :]).reshape((1, len(x[0])))
            q = np.multiply(q, norm(0, self.sig_r).pdf(z[2*i]-r[0, :]) * norm(0, self.sig_phi).pdf(self.Wrap(z[2*i+1]-phi[0, :])))
        return q

    # # for iterative approach
    # def PropogateMotionModel(self, u, xprev):
    #     v_hat = u[0] + np.sqrt(self.a_1 * u[0]**2 + self.a_2 * u[1]**2) * np.random.randn()
    #     w_hat = u[1] + np.sqrt(self.a_3 * u[0]**2 + self.a_4 * u[1]**2) * np.random.randn()
    #     x = np.zeros([3, 1])
    #     x[0] = xprev[0] - (v_hat / w_hat) * np.sin(xprev[2]) + (v_hat / w_hat) * np.sin(xprev[2] + w_hat * self.dt)
    #     x[1] = xprev[1] + (v_hat / w_hat) * np.cos(xprev[2]) - (v_hat / w_hat) * np.cos(xprev[2] + w_hat * self.dt)
    #     x[2] = xprev[2] + w_hat * self.dt
    #     return x
    #
    # def MeasurementModel(self, z, x):
    #     q = np.zeros([1, int(len(z)/2)])
    #     for i in range(int(len(z)/2)):
    #         r = np.sqrt((self.c[i, 0] - x[0])**2 + (self.c[i, 1] - x[1])**2)
    #         phi = np.arctan2(self.c[i, 1] - x[1], self.c[i, 0] - x[0]) - x[2]
    #         q[0, i] = norm(0, self.sig_r).pdf(z[2*i]-r[0]) * norm(0, self.sig_phi).pdf(self.Wrap(z[2*i+1]-phi[0]))
    #     w = np.array([np.sum(q)])
    #     return w

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

    def Wrap(self, th):
        th_wrap = np.fmod(th + np.pi, 2*np.pi)
        for i in range(len(th_wrap)):
            if th_wrap[i] < 0:
                th_wrap[i] += 2*np.pi
        return th_wrap - np.pi

    def EstimateMeanCov(self):
        self.mu = np.mean(self.X, axis=1).reshape((self.n, 1))
        E = (self.X - self.mu)
        E[2, :] = self.Wrap(E[2, :])
        self.cov = 1/(len(self.X[0]) - 1) * (E @ E.transpose())
        pass
