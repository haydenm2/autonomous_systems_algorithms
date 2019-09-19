#!/usr/bin/env python3
from kalman import Kalman
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Summary:
# Example of implementation of kalman class on a simple Unmanned Underwater Vehicle (UUV) system defined by:
#
# m*vdot+b*v=F(t)
# xdot = v
#
# where v is velocity, x is position, F is prop thrust.
# mass(m) = 100 kg, linear drag coeff(b) = 20 N-s/m.
# This test will simulate UUV position for 50 s with the following thrust
#
# F(t) = {50 if 0<=t<5}
#        {-50 if 25<=t<30}
#        {0 otherwise}
# 
# We will assume:
#
# - Position measurement noise covariance of 0.001 m^2
# - Velocity process noise covariance of 0.01 m^2/s^2
# - Position process noise covariance of 0.0001 m^2
# - Sample period of 0.05 s
# 
# The output plots will be:
#
# - Position and velocity states and estimates vs time
# - Estimation error and error covariance vs time
# - Kalman gains vs time

if __name__ == "__main__":

    # Model parameters
    m = 100                                 # mass
    b = 20                                  # drag coefficient
    dt = 0.05                               # time step
    R = np.array([[0.01, 0], [0, 0.0001]])  # process covariance
    Q = np.array([0.001])                   # measurement covariance

    # plot data containers
    x = np.zeros([int(50 / dt) + 1, 2])             # state truth vector
    z = np.zeros([int(50 / dt) + 1, 1])             # measurement vector
    t = np.zeros([int(50 / dt) + 1, 1])             # time vector
    two_sig_v = np.zeros([int(50 / dt) + 1, 2])     # two-sigma velocity boundary
    two_sig_x = np.zeros([int(50 / dt) + 1, 2])     # two-sigma position boundary
    mu = np.zeros([int(50 / dt) + 1, 2])            # state estimation vector
    K = np.zeros([int(50 / dt) + 1, 2])             # Kalman gain vector

    # state space model
    A = np.array([[-b / m, 0], [1, 0]])             # continuous state space A-matrix
    B = np.array([[1 / m], [0]])                    # continuous state space B-matrix
    C = np.array([0, 1])                            # continuous state space C-matrix
    D = np.array(0)                                 # continuous state space D-matrix
    sys = ct.ss(A, B, C, D)                         # continuous LTI system object
    sysd = ct.c2d(sys, dt)                          # discrete LTI system object

    # Kalman Filter Init
    UUV = Kalman(sysd.A, sysd.B, sysd.C, R, Q)

    # Input Command Simulation
    F = np.zeros([int(50 / dt)])
    for i in range(int(50 / dt)):
        if i < int(5 / dt):
            F[i] = 50
        elif i < int(25 / dt):
            F[i] = 0
        elif i < int(30 / dt):
            F[i] = -50
        else:
            F[i] = 0

        # random noise variables
        epsilon = np.array([np.sqrt(R.item((0, 0))) * np.random.randn(), np.sqrt(R.item((1, 1))) * np.random.randn()])
        delta = np.array(np.sqrt(Q) * np.random.randn())

        # update truth/measurement data vectors
        t[i + 1] = i * dt
        x[i + 1] = sysd.A @ x[i] + sysd.B @ (np.array([F[i]]).transpose()) + epsilon
        z[i + 1] = sysd.C @ x[i] + delta

        # kalman updates
        UUV.Execute(np.array([F[i]]), z[i + 1])
        mu[i + 1] = UUV.mu.transpose()
        K[i + 1] = UUV.K.transpose()
        two_sig_v[i + 1] = np.array([2 * np.sqrt(UUV.cov.item((0, 0))), -2 * np.sqrt(UUV.cov.item((0, 0)))])
        two_sig_x[i + 1] = np.array([2 * np.sqrt(UUV.cov.item((1, 1))), -2 * np.sqrt(UUV.cov.item((1, 1)))])

    # Plotting Functions
    ve = (mu.transpose())[0, :]  # velocity estimation
    xe = (mu.transpose())[1, :]  # position estimation
    vt = (x.transpose())[0, :]  # velocity truth
    xt = (x.transpose())[1, :]  # position truth
    verr = ve-vt  # velocity error
    xerr = xe-xt  # position error

    vc_upper = (two_sig_v.transpose())[0, :]  # velocity two sigma covariance upper bound
    vc_lower = (two_sig_v.transpose())[1, :]  # velocity two sigma covariance lower bound
    xc_upper = (two_sig_x.transpose())[0, :]  # position two sigma covariance upper bound
    xc_lower = (two_sig_x.transpose())[1, :]  # position two sigma covariance lower bound

    Kv = (K.transpose())[0, :]  # velocity kalman gains
    Kx = (K.transpose())[1, :]  # position kalman gains

    # plot state truth and state estimates
    plt.figure(1)
    plt.plot(t, xe, 'c-', label='Position Estimate')
    plt.plot(t, ve, 'm-', label='Velocity Estimate')
    plt.plot(t, xt, 'b-', label='Position Truth')
    plt.plot(t, vt, 'r-', label='Velocity Truth')
    plt.plot(t, z, 'g--', label='Position Measurement')
    plt.ylabel('mu')
    plt.xlabel('t (s)')
    plt.title('States and State Estimates')
    plt.legend()
    plt.grid(True)

    # plot error and covariance of states
    plt.figure(2)
    plt.plot(t, verr, 'm-', label='Velocity Error')
    plt.plot(t, xerr, 'c-', label='Position Error')
    plt.plot(t, vc_upper, 'r--', label='Velocity Covariance Bounds')
    plt.plot(t, vc_lower, 'r--')
    plt.plot(t, xc_upper, 'b--', label='Position Covariance Bounds')
    plt.plot(t, xc_lower, 'b--')
    plt.ylabel('cov')
    plt.xlabel('t (s)')
    plt.title('Estimation Error and Covariance Behavior')
    plt.legend()
    plt.grid(True)

    # plot kalman gain propogation
    plt.figure(3)
    plt.plot(t, Kv, 'b-', label='Velocity Kalman Gain')
    plt.plot(t, Kx, 'g-', label='Position Kalman Gain')
    plt.ylabel('K')
    plt.xlabel('t (s)')
    plt.title('Kalman Gain Behavior')
    plt.legend()
    plt.grid(True)
    plt.show()
