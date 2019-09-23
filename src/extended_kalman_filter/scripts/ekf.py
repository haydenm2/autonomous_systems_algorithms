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
    def __init__(self):
        self.u = np.zeros([2, 1])             # input command history
        self.z = np.zeros([3, 1])             # measurement history
        self.mu = np.zeros([3, 1])            # state mean vector
        self.mu_bar = self.mu                       # state mean prediction vector
        self.cov = np.eye(3)                  # state covariance
        self.cov_bar = self.cov                     # state covariance prediction
        self.G = np.zeros([3, 3])
        self.V = np.zeros([3, 2])
        self.M = np.zeros([2, 2])
        self.Q = np.zeros([2, 2])
        self.c = np.zeros([3, 2])
        self.H = np.zeros([2, 3])
        self.z_hat = np.zeros([2, 1])
        self.S = np.zeros([2, 2])
        self.K = np.zeros([3, 2])
        self.pz = 0


    def Propogate(self, u, z):
        self.PredictState(u)
        self.AddMeasurement(z)

    def PredictState(self, u):
        self.mu_bar = self.A @ self.mu + (self.B @ u).transpose()

    def AddMeasurement(self, z):
        temp = z - self.C @ self.mu_bar
        self.mu = self.mu_bar + self.K @ temp
