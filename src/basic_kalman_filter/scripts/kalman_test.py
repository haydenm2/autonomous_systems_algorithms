#!/usr/bin/env python3
from kalman import Kalman
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
#
#

if(__name__ == "__main__"):
    AUV = Kalman()

