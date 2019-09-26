#!/usr/bin/env python3

import numpy as np
import control as ct

# Generic Kalman Filter Approach (From Probablistic Robotics)
#
# ALGORITHM_KALMAN_FILTER(mu_(t-1),cov(t-1),u_t,z_t):
#   mubar_t = A_t*mu_(t-1) + B_t*u_t
#   covbar_t = A_t*cov_(t-1)*A_t^T + R_t
#   K_t = covbar_t*C_t^T*(C_t*covbar_t*C_t^T + Q_t)^(-1)
#   mu_t = mubar_t + K_t*(z_t - C_t*mubar_t)
#   cov_t = (I - K_t*C_t)*covbar_t
#   return (mu_t, cov_t)
#


class EKF:
    def __init__(self, dt=0.1, x0=np.array([[-5], [-3], [90*np.pi/180.0]])):
        self.u = np.zeros([2, 1])             # input command history
        self.z = np.zeros([3, 1])             # measurement history
        self.mu = np.copy(x0)                        # state mean vector
        self.mu_bar = np.copy(x0)                 # state mean prediction vector
        self.cov = np.eye(3) * 0.1                 # state covariance
        self.cov_bar = np.eye(3)               # state covariance prediction
        self.G = np.eye(3)
        self.V = np.zeros([3, 2])
        self.K = np.zeros([6, 1])
        self.M = np.zeros([2, 2])
        self.Q = np.zeros([2, 2])
        self.c = np.zeros([3, 2])
        self.pz = 0
        self.a_1 = 0.1
        self.a_2 = 0.01
        self.a_3 = 0.01
        self.a_4 = 0.1
        self.sig_r = 0.1
        self.sig_phi = 0.05
        self.dt = dt

        # Landmark Locations
        self.l1 = np.array([6, 4])
        self.l2 = np.array([-7, 8])
        self.l3 = np.array([6, -4])
        self.c = np.vstack((self.l1, self.l2, self.l3))


    def Propogate(self, u, z):
        self.PredictState(u)
        self.AddMeasurement(z)

    def PredictState(self, u):
        theta = (self.mu[2])[0]
        vt = u[0, 0]
        wt = u[1, 0]
        self.G[0, 2] = -(vt/wt)*np.cos(theta)+(vt/wt)*np.cos(theta+wt*self.dt)
        self.G[1, 2] = -(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt)

        self.V[0, 0] = (-np.sin(theta) + np.sin(theta + wt * self.dt)) / wt
        self.V[0, 1] = vt * (np.sin(theta) - np.sin(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.cos(theta + wt * self.dt) * self.dt) / wt
        self.V[1, 0] = (np.cos(theta) - np.cos(theta + wt * self.dt)) / wt
        self.V[1, 1] = -vt * (np.cos(theta) - np.cos(theta + wt * self.dt)) / np.power(wt, 2) + (vt * np.sin(theta + wt * self.dt) * self.dt) / wt
        self.V[2, 1] = self.dt

        self.M[0, 0] = self.a_1*np.power(vt, 2) + self.a_2*np.power(wt, 2)
        self.M[1, 1] = self.a_3*np.power(vt, 2) + self.a_4*np.power(wt, 2)

        self.mu_bar[0] = self.mu[0] + (-(vt/wt)*np.sin(theta)+(vt/wt)*np.sin(theta+wt*self.dt))
        self.mu_bar[1] = self.mu[1] + (vt/wt)*np.cos(theta)-(vt/wt)*np.cos(theta+wt*self.dt)
        self.mu_bar[2] = self.mu[2] + wt*self.dt

        self.cov_bar = self.G @ (self.cov @ self.G.transpose()) + self.V @ (self.M @ self.V.transpose())

        self.Q[0, 0] = np.power(self.sig_r, 2)
        self.Q[1, 1] = np.power(self.sig_phi, 2)

    def AddMeasurement(self, z):
        for j in range(3):
            dx = (self.c[j, 0] - self.mu_bar[0])[0]
            dy = (self.c[j, 1] - self.mu_bar[1])[0]
            q = np.power(dx, 2) + np.power(dy, 2)
            zhat = np.array([[np.sqrt(q)], [np.arctan2(dy, dx) - self.mu_bar[2, 0]]])
            H = np.array([[-dx/np.sqrt(q), -dy/np.sqrt(q), 0], [dy/q, -dx/q, -1]])
            S = H @ (self.cov_bar @ H.transpose()) + self.Q
            K = self.cov_bar @ (H.transpose() @ np.linalg.inv(S))
            self.mu_bar = self.mu_bar + K @ (z[(0+j*2):(2+j*2)] - zhat)
            self.cov_bar = (np.eye(3) - (K @ H)) @ self.cov_bar
        self.mu = self.mu_bar
        self.cov = self.cov_bar
        self.K = K.reshape((6, 1))
