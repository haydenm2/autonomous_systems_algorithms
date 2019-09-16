#!/usr/bin/env python3

import numpy as np

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
    def __init__(self,A,B,C,D,cov,Q):
        self.t = 0                   #time iterator
        self.nx = np.size(A,axis=0)   #number of state variables
        self.nu = np.size(D,axis=1)   #number of input types
        self.nz = np.size(C,axis=0)   #number of measurement types
        self.ind = 0                 #history iterator
        self.u = np.array([])        #input command history
        self.z = np.array([])        #measurement history
        self.mu = np.zeros(n)        #state vector
        self.mu_bar = np.zeros(n)    #state prediction vector
        self.cov = np.zeros(shape=(n,n)) #covariance 
        self.cov_bar = np.zeros(shape=(n,n)) #covariance prediction
        pass

    def Execute(self,u,z):
        self.
        self.PredictState()
        self.PredictCovariance()
        self.UpdateGains()
        self.UpdateState()
        self.UpdateCovariance()
        pass

    def Update(self):
        pass
    
    def PredictState(self):
        pass

    def PredictCovariance(self):
        pass
    
    def UpdateGains(self):
        pass

    def UpdateState(self):
        pass
    
    def UpdateCovariance(self):
        pass

    
