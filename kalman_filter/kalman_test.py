#!/usr/bin/env python3
from kalman import Kalman
from uuv import UUV
import numpy as np
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

    # UUV parameters
    x0 = np.zeros([1, 2])                   # initial states
    duration = 50                           # model test duration
    dt = 0.05                               # time step
    R = np.array([[0.01, 0], [0, 0.0001]])  # process covariance
    Q = np.array([0.001])                   # measurement covariance
    uuv = UUV(x0, duration, dt, R, Q)       # UUV model object

    # Kalman Filter Init
    kalman = Kalman(uuv.A, uuv.B, uuv.C, uuv.R, uuv.Q)

    # plot data containers
    two_sig_v = np.zeros([int(duration / dt) + 1, 2])     # two-sigma velocity boundary
    two_sig_x = np.zeros([int(duration / dt) + 1, 2])     # two-sigma position boundary
    mu = np.zeros([int(duration / dt) + 1, 2])            # state estimation vector
    K = np.zeros([int(duration / dt) + 1, 2])             # Kalman gain vector

    # Input Command Simulation
    F = np.zeros([int(duration / dt)])
    for i in range(int(duration / dt)):
        if i < int(5 / dt):
            F[i] = 500
        elif i < int(25 / dt):
            F[i] = 0
        elif i < int(30 / dt):
            F[i] = -500
        else:
            F[i] = 0

        # truth model updates
        uuv.Propagate(F[i])

        # kalman updates
        kalman.Propogate(np.array([F[i]]), uuv.Getz())
        mu[i + 1] = kalman.mu.transpose()
        K[i + 1] = kalman.K.transpose()
        two_sig_v[i + 1] = np.array([2 * np.sqrt(kalman.cov.item((0, 0))), -2 * np.sqrt(kalman.cov.item((0, 0)))])
        two_sig_x[i + 1] = np.array([2 * np.sqrt(kalman.cov.item((1, 1))), -2 * np.sqrt(kalman.cov.item((1, 1)))])

    # Plotting Vectors
    ve = (mu.transpose())[0, :]  # velocity estimation
    xe = (mu.transpose())[1, :]  # position estimation
    vt = (uuv.x.transpose())[0, :]  # velocity truth
    xt = (uuv.x.transpose())[1, :]  # position truth
    verr = ve-vt  # velocity error
    xerr = xe-xt  # position error

    vc_upper = (two_sig_v.transpose())[0, :]  # velocity two sigma covariance upper bound
    vc_lower = (two_sig_v.transpose())[1, :]  # velocity two sigma covariance lower bound
    xc_upper = (two_sig_x.transpose())[0, :]  # position two sigma covariance upper bound
    xc_lower = (two_sig_x.transpose())[1, :]  # position two sigma covariance lower bound

    Kv = (K.transpose())[0, :]  # velocity kalman gains
    Kx = (K.transpose())[1, :]  # position kalman gains

    # Plot state truth and state estimates
    plt.figure(1)
    plt.plot(uuv.t, xe, 'c-', label='Position Estimate')
    plt.plot(uuv.t, ve, 'm-', label='Velocity Estimate')
    plt.plot(uuv.t, xt, 'b-', label='Position Truth')
    plt.plot(uuv.t, vt, 'r-', label='Velocity Truth')
    plt.plot(uuv.t, uuv.z, 'g--', label='Position Measurement')
    plt.ylabel('mu')
    plt.xlabel('t (s)')
    plt.title('States and State Estimates')
    plt.legend()
    plt.grid(True)

    # Plot error and covariance of states
    plt.figure(2)
    plt.plot(uuv.t, verr, 'm-', label='Velocity Error')
    plt.plot(uuv.t, xerr, 'c-', label='Position Error')
    plt.plot(uuv.t, vc_upper, 'r--', label='Velocity Covariance Bounds')
    plt.plot(uuv.t, vc_lower, 'r--')
    plt.plot(uuv.t, xc_upper, 'b--', label='Position Covariance Bounds')
    plt.plot(uuv.t, xc_lower, 'b--')
    plt.ylabel('cov')
    plt.xlabel('t (s)')
    plt.title('Estimation Error and Covariance Behavior')
    plt.legend()
    plt.grid(True)

    # Plot kalman gain propogation
    plt.figure(3)
    plt.plot(uuv.t, Kv, 'b-', label='Velocity Kalman Gain')
    plt.plot(uuv.t, Kx, 'g-', label='Position Kalman Gain')
    plt.ylabel('K')
    plt.xlabel('t (s)')
    plt.title('Kalman Gain Behavior')
    plt.legend()
    plt.grid(True)
    plt.show()
