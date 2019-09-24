#!/usr/bin/env python3
from ekf import EKF
from twr import TWR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ------------------------------------------------------------------
# Summary:
# Example of implementation of ekf class on a simple Two-Wheeled Robot system defined by
# the motion model described in Chapter 5 of Probablistic Robotics
#
# Commanded as follows:
# v_c = 1 + 0.5*cos(2*pi*(0.2)*t)
# w_c = -0.2 + 2*cos(2*pi*(0.6)*t)
#
# We will assume:
#
# - Range measurement covariance of 0.1 m
# - Bearing measurement covariance of 0.05 rad
# - Noise characteristics: a1 = a4 = 0.1 and a2 = a3 = 0.01
# - Sample period of 0.1 s
#

if __name__ == "__main__":

    # Two-Wheeled Robot Init
    twr = TWR()       # TWR model object

    # Kalman Filter Init
    ekf = EKF()

    # # plot data containers
    # two_sig_v = np.zeros([1, 2])     # two-sigma velocity boundary
    # two_sig_x = np.zeros([1, 2])     # two-sigma position boundary
    # mu = np.zeros([1, 2])            # state estimation vector
    # K = np.zeros([1, 2])             # Kalman gain vector

    fig, ax = plt.subplots()
    lines, = ax.plot([], [], 'r-', zorder=1)
    body_radius = 0.3
    robot_body = Circle((0, 0), body_radius, color='b', zorder=2)
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=3)
    ax.plot(twr.l1[0], twr.l1[1], 'o', zorder=4)
    ax.plot(twr.l2[0], twr.l2[1], 'o', zorder=4)
    ax.plot(twr.l3[0], twr.l3[1], 'o', zorder=4)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.grid()

    for i in range(int(twr.t_end/twr.dt)):
        # truth model updates
        twr.Propagate()
        xt = twr.x[0:2, :]  # position truth
        lines.set_xdata(xt[0, :])
        lines.set_ydata(xt[1, :])
        robot_body.center = ((twr.Getx())[0], (twr.Getx())[1])
        fig.canvas.draw()
        plt.pause(0.001)

        # # kalman updates
        # kalman.Propogate(np.array([F[i]]), uuv.Getz())
        # mu[i + 1] = kalman.mu.transpose()
        # K[i + 1] = kalman.K.transpose()
        # two_sig_v[i + 1] = np.array([2 * np.sqrt(kalman.cov.item((0, 0))), -2 * np.sqrt(kalman.cov.item((0, 0)))])
        # two_sig_x[i + 1] = np.array([2 * np.sqrt(kalman.cov.item((1, 1))), -2 * np.sqrt(kalman.cov.item((1, 1)))])

    # Plotting Vectors
    # ve = (mu.transpose())[0, :]  # velocity estimation
    # xe = (mu.transpose())[1, :]  # position estimation
    # vt = (uuv.x.transpose())[0, :]  # velocity truth
    # xt = twr.x[0:1, :]  # position truth
    # verr = ve-vt  # velocity error
    # xerr = xe-xt  # position error

    # vc_upper = (two_sig_v.transpose())[0, :]  # velocity two sigma covariance upper bound
    # vc_lower = (two_sig_v.transpose())[1, :]  # velocity two sigma covariance lower bound
    # xc_upper = (two_sig_x.transpose())[0, :]  # position two sigma covariance upper bound
    # xc_lower = (two_sig_x.transpose())[1, :]  # position two sigma covariance lower bound
    #
    # Kv = (K.transpose())[0, :]  # velocity kalman gains
    # Kx = (K.transpose())[1, :]  # position kalman gains

    # Plot state truth and state estimates
    fig1 = plt.figure(1)
    plt.plot(twr.t, xe, 'c-', label='Position Estimate')
    plt.plot(twr.t, ve, 'm-', label='Velocity Estimate')
    plt.plot(twr.t, xt, 'b-', label='Position Truth')
    plt.plot(twr.t, vt, 'r-', label='Velocity Truth')
    plt.plot(twr.t, twr.z, 'g--', label='Position Measurement')
    plt.ylabel('mu')
    plt.xlabel('t (s)')
    plt.title('States and State Estimates')
    plt.legend()
    plt.grid(True)

    # Plot error and covariance of states
    plt.figure(2)
    plt.plot(twr.t, verr, 'm-', label='Velocity Error')
    plt.plot(twr.t, xerr, 'c-', label='Position Error')
    plt.plot(twr.t, vc_upper, 'r--', label='Velocity Covariance Bounds')
    plt.plot(twr.t, vc_lower, 'r--')
    plt.plot(twr.t, xc_upper, 'b--', label='Position Covariance Bounds')
    plt.plot(twr.t, xc_lower, 'b--')
    plt.ylabel('cov')
    plt.xlabel('t (s)')
    plt.title('Estimation Error and Covariance Behavior')
    plt.legend()
    plt.grid(True)

    # Plot kalman gain propogation
    plt.figure(3)
    plt.plot(twr.t, Kv, 'b-', label='Velocity Kalman Gain')
    plt.plot(twr.t, Kx, 'g-', label='Position Kalman Gain')
    plt.ylabel('K')
    plt.xlabel('t (s)')
    plt.title('Kalman Gain Behavior')
    plt.legend()
    plt.grid(True)
    plt.show()
