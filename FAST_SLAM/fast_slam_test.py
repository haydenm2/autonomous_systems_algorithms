#!/usr/bin/env python3
from fast_slam import FAST_SLAM
from twr import TWR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

# ------------------------------------------------------------------
# Summary:
# Example of implementation of fast_slam class on a simple Two-Wheeled Robot system defined by
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
    lines, = ax.plot([], [], 'g-', zorder=1, label='True Path')
    lines_est, = ax.plot([], [], 'r--', zorder=2, label='Estimated Path')
    robot_body = Circle((0, 0), body_radius, color='b', zorder=3, label='Robot Truth')
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=4)
    scan_angle = Wedge((twr.x_new[0, 0], twr.x_new[1, 0]), 30, (twr.x_new[2, 0]-twr.fov/2)*180/np.pi, (twr.x_new[2, 0]+twr.fov/2)*180/np.pi, color='c', zorder=7, alpha=0.1)
    ax.add_artist(scan_angle)
    clandmarks, = ax.plot([], [], 'r.', zorder=6)
    mlandmarks = []
    for i in range(twr.nl):
        han, = ax.plot([], [], 'r-', zorder=6)
        mlandmarks.append(han)
    mparticles, = ax.plot([], [], 'r.', zorder=7, markersize=3)
    ax.add_artist(robot_body)
    ax.plot(twr.c[:, 0], twr.c[:, 1], 'o', zorder=5, label='True Landmark Positions')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    plt.title('Map with Robot and Landmark Truth and Estimate')
    ax.legend()
    ax.grid()
    return fig, lines, lines_est, mparticles, mlandmarks, clandmarks, robot_body, robot_head, scan_angle


def UpdatePlot(fig, lines, lines_est, mparticles, clandmarks, robot_body, body_radius, robot_head, scan_angle, twr, mu, mu_landmarks, w_landmarks, h_landmarks, ang_landmarks, X):
    xt = twr.x[0:2, :]  # position truth
    lines.set_xdata(xt[0, :])
    lines.set_ydata(xt[1, :])
    lines_est.set_xdata(mu[0, :])
    lines_est.set_ydata(mu[1, :])
    mparticles.set_xdata(X[0, :])
    mparticles.set_ydata(X[1, :])
    mu_landmarks_x = []
    mu_landmarks_y = []
    for i in range(int(len(mu_landmarks)/2)):
        mu_landmarks_x = np.hstack((mu_landmarks_x, mu_landmarks[2*i]))
        mu_landmarks_y = np.hstack((mu_landmarks_y, mu_landmarks[2*i+1]))

    robot_body.center = ((twr.Getx())[0], (twr.Getx())[1])
    headx = np.array([twr.Getx()[0], twr.Getx()[0] + body_radius * np.cos(twr.Getx()[2])])
    heady = np.array([twr.Getx()[1], twr.Getx()[1] + body_radius * np.sin(twr.Getx()[2])])
    robot_head.set_xdata(headx)
    robot_head.set_ydata(heady)

    scan_angle.set_center((twr.x_new[0, 0], twr.x_new[1, 0]))
    scan_angle.set_theta1((twr.x_new[2, 0] - twr.fov / 2)*180/np.pi)
    scan_angle.set_theta2((twr.x_new[2, 0] + twr.fov / 2)*180/np.pi)

    fig.canvas.draw()
    plt.pause(0.01)


if __name__ == "__main__":

    plot_live = True

    # Two-Wheeled Robot Init
    twr = TWR()       # TWR model object

    # FAST SLAM filter Init
    fast_slam = FAST_SLAM(twr)

    body_radius = 0.3
    fig, lines, lines_est, mparticles, mlandmarks, clandmarks, robot_body, robot_head, scan_angle = InitPlot(twr, body_radius)
    mu = fast_slam.mu_bar[0:3].reshape([3, 1])
    mu_landmarks = fast_slam.mu_bar[3:].reshape([len(fast_slam.mu_bar[3:]), 1])
    w_landmarks = np.zeros((twr.nl, 1))
    h_landmarks = np.zeros((twr.nl, 1))
    ang_landmarks = np.zeros((twr.nl, 1))
    two_sig_x = np.array([[2 * np.sqrt(fast_slam.cov_bar.item((0, 0)))], [-2 * np.sqrt(fast_slam.cov_bar.item((0, 0)))]])
    two_sig_y = np.array([[2 * np.sqrt(fast_slam.cov_bar.item((1, 1)))], [-2 * np.sqrt(fast_slam.cov_bar.item((1, 1)))]])
    two_sig_theta = np.array([[2 * np.sqrt(fast_slam.cov_bar.item((2, 2)))], [-2 * np.sqrt(fast_slam.cov_bar.item((2, 2)))]])
    for i in range(int(twr.t_end/twr.dt)):
        # truth model updates
        twr.Propagate()
        fast_slam.Propogate(twr.Getu(), twr.Getz())

        # plotter updates
        mu = np.hstack((mu, fast_slam.mu_bar[0:3].reshape([3, 1])))
        mu_landmarks = np.hstack((mu_landmarks, fast_slam.mu_bar[3:].reshape([len(fast_slam.mu_bar[3:]), 1])))
        if plot_live:
            X = fast_slam.X
            clmx = []
            clmy = []
            for i in range(twr.nl):
                if ~(fast_slam.mu_bar[3 + 2*i] == 0 and fast_slam.mu_bar[3 + 2*i + 1] == 0):
                    A = (fast_slam.cov_bar[3:, 3:])[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2]
                    [U, S, _] = np.linalg.svd(A)
                    C = U * 2*np.sqrt(S)
                    theta = np.linspace(0, 2*np.pi, 100)
                    circle = np.array([np.cos(theta), np.sin(theta)])
                    ellipse = C @ circle
                    ellipse[0, :] += fast_slam.mu_bar[3 + 2*i]
                    ellipse[1, :] += fast_slam.mu_bar[3 + 2*i + 1]

                    # update landmark covariance ellipses
                    mlandmarks[i].set_xdata(ellipse[0, :])
                    mlandmarks[i].set_ydata(ellipse[1, :])

                    clmx.append(fast_slam.mu_bar[3 + 2 * i])
                    clmy.append(fast_slam.mu_bar[4 + 2 * i])

            clandmarks.set_xdata(clmx)
            clandmarks.set_ydata(clmy)
            UpdatePlot(fig, lines, lines_est, mparticles, clandmarks, robot_body, body_radius, robot_head, scan_angle, twr, mu, mu_landmarks[:, -1], w_landmarks, h_landmarks, ang_landmarks, X)
        two_sig_x = np.hstack((two_sig_x, np.array([[2 * np.sqrt(fast_slam.cov_bar.item((0, 0)))], [-2 * np.sqrt(fast_slam.cov_bar.item((0, 0)))]])))
        two_sig_y = np.hstack((two_sig_y, np.array([[2 * np.sqrt(fast_slam.cov_bar.item((1, 1)))], [-2 * np.sqrt(fast_slam.cov_bar.item((1, 1)))]])))
        two_sig_theta = np.hstack((two_sig_theta, np.array([[2 * np.sqrt(fast_slam.cov_bar.item((2, 2)))], [-2 * np.sqrt(fast_slam.cov_bar.item((2, 2)))]])))
    if ~plot_live:
        X = fast_slam.X
        clmx = []
        clmy = []
        for i in range(twr.nl):
            if ~(fast_slam.mu_bar[3 + 2 * i] == 0 and fast_slam.mu_bar[3 + 2 * i + 1] == 0):
                A = (fast_slam.cov_bar[3:, 3:])[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2]
                [U, S, _] = np.linalg.svd(A)
                C = U * 2 * np.sqrt(S)
                theta = np.linspace(0, 2 * np.pi, 100)
                circle = np.array([np.cos(theta), np.sin(theta)])
                ellipse = C @ circle
                ellipse[0, :] += fast_slam.mu_bar[3 + 2 * i]
                ellipse[1, :] += fast_slam.mu_bar[3 + 2 * i + 1]

                # update landmark covariance ellipses
                mlandmarks[i].set_xdata(ellipse[0, :])
                mlandmarks[i].set_ydata(ellipse[1, :])

                clmx.append(fast_slam.mu_bar[3 + 2 * i])
                clmy.append(fast_slam.mu_bar[4 + 2 * i])

        clandmarks.set_xdata(clmx)
        clandmarks.set_ydata(clmy)

        UpdatePlot(fig, lines, lines_est, mparticles, clandmarks, robot_body, body_radius, robot_head, scan_angle, twr, mu, mu_landmarks[:, -1], w_landmarks, h_landmarks, ang_landmarks, X)

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

    plt.show()