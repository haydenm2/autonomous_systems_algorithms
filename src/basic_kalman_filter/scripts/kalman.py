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
    def __init__(self,A,B,C,R,Q):
        self.nx = np.size(A,axis=0)   #number of state variables
        self.nu = np.size(B,axis=1)   #number of input types
        self.nz = np.size(C,axis=0)   #number of measurement types
        self.u = np.zeros(self.nu)        #input command history
        self.z = np.zeros(self.nz)        #measurement history
        self.mu = np.zeros(self.nx)        #state mean vector
        self.mu_bar = self.mu        #state mean prediction vector
        self.R = R                   #process covariance 
        self.Q = Q                   #measurement covariance 
        self.cov = np.eye(self.nx)         #state covariance
        self.cov_bar = self.cov      #state covariance prediction
        self.K = np.zeros([self.nx,self.nz])   #kalman gains
        self.A = A
        self.B = B
        self.C = C
        pass

    def Execute(self,u,z):
        self.PredictState(u)
        self.PredictCovariance()
        self.UpdateGains()
        self.UpdateState(z)
        self.UpdateCovariance()
        pass

    def Update(self):
        pass
    
    def PredictState(self,u):
        self.mu_bar = self.A.dot(self.mu.transpose())+self.B.dot(u)
        pass

    def PredictCovariance(self):
        self.cov_bar = np.dot(self.A,np.dot(self.cov,self.A.transpose()))+self.R
        pass
    
    def UpdateGains(self):
        temp = np.linalg.inv(np.dot(self.C,np.dot(self.cov_bar,self.C.transpose()))+self.Q)
        self.K = np.dot(self.cov_bar,np.dot(self.C.transpose(),temp))
        pass

    def UpdateState(self,z):
        print("self.mu: \n",self.mu)
        print("self.C: \n",self.C)
        print("self.mu_bar: \n",self.mu_bar)
        temp = z - np.dot(self.C,self.mu_bar.transpose())
        print("temp: \n",temp)
        self.mu = self.mu_bar+np.dot(self.K,temp).transpose()
        print("self.K: \n",self.K)
        print("np.dot(self.K,temp): \n",np.dot(self.K,temp))
        pass
    
    def UpdateCovariance(self):
        temp = np.eye(self.nx)-np.dot(self.K,self.C)
        self.cov = np.dot(temp,self.cov_bar)
        pass

    
