#!/usr/bin/env python3
from kalman import Kalman
import numpy as np
import control as ct
import matplotlib as plot
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

    #Model parameters
    m = 100
    b = 20
    dt = 0.05
    R = np.array([[0.01,0],[0,0.0001]])
    Q = np.array([0.001])

    #plot data containers
    x = np.zeros([(int)(50/dt)+1,2])
    z = np.zeros([(int)(50/dt)+1,1])
    mu = np.zeros([(int)(50/dt)+1,2])
    
    #random noise variables
    epsilon = np.random.multivariate_normal([0,0],R.tolist())
    delta = np.random.normal([0],Q.tolist())

    #state space model
    A = np.array([[-b/m,0],[1,0]])
    B = np.array([[1/m],[0]])
    C = np.array([0,1])
    D = np.array(0)
    sys = ct.ss(A,B,C,D)
    sysd = ct.c2d(sys,dt)

    #Kalman Filter Init
    UUV = Kalman(sysd.A,sysd.B,sysd.C,R,Q)

    # #Kalman Filter Test
    # UUV.Execute(np.array(50),np.array(0))

    #Input Command Simulation
    F = np.zeros([(int)(50/dt)])
    for t in range((int)(50/dt)):
        if(t<(int)(5/dt)):
            F[t] = 50
        elif(t<(int)(25/dt)):
            F[t] = 0
        elif(t<(int)(30/dt)):
            F[t] = -50
        else:
            F[t] = 0

        x[t+1] = sysd.A.dot(x[t])+sysd.B.dot(F[t]).transpose()
        print("t: \n",t*0.05)
        print("x: \n",x[t+1])
        z[t+1] = sysd.C.dot(x[t])+delta
        print("z: \n",z[t+1])

        UUV.Execute(np.array([F[t]]),z[t+1])
        mu[t+1] = UUV.mu
        print("mu: \n",mu[t+1])
        print('')
