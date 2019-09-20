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


class Kalman:
    def __init__(self, A, B, C, R, Q):
        self.nx = np.size(A, axis=0)                # number of state variables
        self.nu = np.size(B, axis=1)                # number of input types
        self.nz = np.size(C, axis=0)                # number of measurement types
        self.u = np.zeros([self.nu, 1])             # input command history
        self.z = np.zeros([self.nz, 1])             # measurement history
        self.mu = np.zeros([self.nx, 1])            # state mean vector
        self.mu_bar = self.mu                       # state mean prediction vector
        self.R = R                                  # process covariance
        self.Q = Q                                  # measurement covariance
        self.cov = np.eye(self.nx)                  # state covariance
        self.cov_bar = self.cov                     # state covariance prediction
        self.K = np.zeros([self.nx, self.nz])       # kalman gains
        self.A = A                                  # discrete state A-matrix
        self.B = B                                  # discrete state B-matrix
        self.C = C                                  # discrete measurement C-matrix

    def Propogate(self, u, z):
        self.PredictState(u)
        self.PredictCovariance()
        self.UpdateGains()
        self.UpdateState(z)
        self.UpdateCovariance()

    def PredictState(self, u):
        self.mu_bar = self.A @ self.mu + (self.B @ u).transpose()

    def PredictCovariance(self):
        self.cov_bar = self.A @ (self.cov @ self.A.transpose()) + self.R

    def UpdateGains(self):
        temp = np.linalg.inv(self.C @ (self.cov_bar @ self.C.transpose()) + self.Q)
        self.K = self.cov_bar @ (self.C.transpose() @ temp)

    def UpdateState(self, z):
        temp = z - self.C @ self.mu_bar
        self.mu = self.mu_bar + self.K @ temp

    def UpdateCovariance(self):
        temp = np.eye(self.nx) - self.K @ self.C
        self.cov = temp @ self.cov_bar
