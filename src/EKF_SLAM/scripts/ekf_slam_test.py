#!/usr/bin/env python3
from ekf_slam import EKF_SLAM
from twr import TWR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ------------------------------------------------------------------
# Summary:
# Example of implementation of ekf_slam class on a simple Two-Wheeled Robot system defined by
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


def InitPlot(twr, body_radius):
    fig, ax = plt.subplots()
    lines, = ax.plot([], [], 'g-', zorder=1)
    lines_est, = ax.plot([], [], 'r--', zorder=2)
    robot_body = Circle((0, 0), body_radius, color='b', zorder=3)
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=4)
    msensor, = ax.plot([], [], 'ro', zorder=5)
    for i in range(twr.nl):
        ax.plot(twr.c[i, 0], twr.c[i, 1], 'o', zorder=6)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.grid()
    return fig, lines, lines_est, msensor, robot_body, robot_head


def UpdatePlot(fig, lines, lines_est, msensor, robot_body, body_radius, robot_head, twr, mu, zpos):
    xt = twr.x[0:2, :]  # position truth
    lines.set_xdata(xt[0, :])
    lines.set_ydata(xt[1, :])
    lines_est.set_xdata(mu[0, :])
    lines_est.set_ydata(mu[1, :])
    msensor.set_xdata(zpos[0, :])
    msensor.set_ydata(zpos[1, :])

    robot_body.center = ((twr.Getx())[0], (twr.Getx())[1])
    headx = np.array([twr.Getx()[0], twr.Getx()[0] + body_radius * np.cos(twr.Getx()[2])])
    heady = np.array([twr.Getx()[1], twr.Getx()[1] + body_radius * np.sin(twr.Getx()[2])])
    robot_head.set_xdata(headx)
    robot_head.set_ydata(heady)

    fig.canvas.draw()
    plt.pause(0.01)


if __name__ == "__main__":

    # Two-Wheeled Robot Init
    twr = TWR()       # TWR model object

    # Kalman Filter Init
    ekf_slam = EKF_SLAM(twr.c, twr.nl)

    body_radius = 0.3
    fig, lines, lines_est, msensor, robot_body, robot_head = InitPlot(twr, body_radius)
    mu = ekf_slam.mu
    K = ekf_slam.K
    two_sig_x = np.array([[2 * np.sqrt(ekf_slam.cov.item((0, 0)))], [-2 * np.sqrt(ekf_slam.cov.item((0, 0)))]])
    two_sig_y = np.array([[2 * np.sqrt(ekf_slam.cov.item((1, 1)))], [-2 * np.sqrt(ekf_slam.cov.item((1, 1)))]])
    two_sig_theta = np.array([[2 * np.sqrt(ekf_slam.cov.item((2, 2)))], [-2 * np.sqrt(ekf_slam.cov.item((2, 2)))]])
    for i in range(int(twr.t_end/twr.dt)):
        # truth model updates
        twr.Propagate()
        ekf_slam.Propogate(twr.Getu(), twr.Getz())

        # plotter updates
        mu = np.hstack((mu, ekf_slam.mu))
        K = np.hstack((K, ekf_slam.K))
        # zpos = np.hstack((zpos, twr.Getzpos())) # Historical sensor plotting
        zpos = twr.Getzpos() # Immediate sensor plotting
        UpdatePlot(fig, lines, lines_est, msensor, robot_body, body_radius, robot_head, twr, mu, zpos)
        two_sig_x = np.hstack((two_sig_x, np.array([[2 * np.sqrt(ekf_slam.cov.item((0, 0)))], [-2 * np.sqrt(ekf_slam.cov.item((0, 0)))]])))
        two_sig_y = np.hstack((two_sig_y, np.array([[2 * np.sqrt(ekf_slam.cov.item((1, 1)))], [-2 * np.sqrt(ekf_slam.cov.item((1, 1)))]])))
        two_sig_theta = np.hstack((two_sig_theta, np.array([[2 * np.sqrt(ekf_slam.cov.item((2, 2)))], [-2 * np.sqrt(ekf_slam.cov.item((2, 2)))]])))

    # Plotting Vectors
    xe = mu[0, :]  # position x estimation
    ye = mu[1, :]  # position y estimation
    thetae = mu[2, :]  # position angle estimation
    xt = twr.x[0, :]  # position x truth
    yt = twr.x[1, :]  # position y truth
    thetat = twr.x[2, :]  # position angle truth
    xerr = xe-xt  # position x error
    yerr = ye-yt  # position y error
    thetaerr = thetae-thetat  # position y error

    xc_upper = two_sig_x[0, :]  # position x two sigma covariance upper bound
    xc_lower = two_sig_x[1, :]  # position x two sigma covariance lower bound
    yc_upper = two_sig_y[0, :]  # position y two sigma covariance upper bound
    yc_lower = two_sig_y[1, :]  # position y two sigma covariance lower bound
    thetac_upper = two_sig_theta[0, :]  # position theta two sigma covariance upper bound
    thetac_lower = two_sig_theta[1, :]  # position theta two sigma covariance lower bound

    # Plot position x truth and estimate
    plt.figure(2)
    plt.subplot(311)
    plt.plot(twr.t, xe, 'c-', label='Position X Estimate')
    plt.plot(twr.t, xt, 'b-', label='Position X Truth')
    plt.ylabel('x (m)')
    plt.title('Position Truth and Estimate')
    plt.legend()
    plt.grid(True)

    # Plot position y truth and estimate
    plt.subplot(312)
    plt.plot(twr.t, ye, 'c-', label='Position Y Estimate')
    plt.plot(twr.t, yt, 'b-', label='Position Y Truth')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)

    # Plot position theta truth and estimate
    plt.subplot(313)
    plt.plot(twr.t, thetae, 'c-', label='Position Theta Estimate')
    plt.plot(twr.t, thetat, 'b-', label='Position Theta Truth')
    plt.ylabel('theta (rad)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)

    # Plot position x error and covariance of states
    plt.figure(3)
    plt.subplot(311)
    plt.plot(twr.t, xerr, 'm-', label='Position X Error')
    plt.plot(twr.t, xc_upper, 'b--', label='Position X Covariance Bounds')
    plt.plot(twr.t, xc_lower, 'b--')
    plt.ylabel('x (m)')
    plt.title('Position Estimation Error and Covariance Behavior')
    plt.legend()
    plt.grid(True)

    # Plot position y error and covariance of states
    plt.subplot(312)
    plt.plot(twr.t, yerr, 'm-', label='Position Y Error')
    plt.plot(twr.t, yc_upper, 'b--', label='Position Y Covariance Bounds')
    plt.plot(twr.t, yc_lower, 'b--')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)

    # Plot position theta error and covariance of states
    plt.subplot(313)
    plt.plot(twr.t, thetaerr, 'm-', label='Position Theta Error')
    plt.plot(twr.t, thetac_upper, 'b--', label='Position Theta Covariance Bounds')
    plt.plot(twr.t, thetac_lower, 'b--')
    plt.ylabel('theta (rad)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)

    # Plot kalman gain propogation
    plt.figure(4)
    plt.plot(twr.t, K[0, :], label='Kalman Gain 1')
    plt.plot(twr.t, K[1, :], label='Kalman Gain 2')
    plt.plot(twr.t, K[2, :], label='Kalman Gain 3')
    plt.plot(twr.t, K[3, :], label='Kalman Gain 4')
    plt.plot(twr.t, K[4, :], label='Kalman Gain 5')
    plt.plot(twr.t, K[5, :], label='Kalman Gain 6')
    plt.ylabel('K')
    plt.xlabel('t (s)')
    plt.title('Kalman Gain Behavior')
    plt.legend()
    plt.grid(True)
    plt.show()