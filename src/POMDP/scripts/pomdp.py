#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Generic Partially Observable Markov Decision Process Model (From Probablistic Robotics)

# Implement algorithm 15.1 with the objective of representing the value function for the illustrative example of section 15.2
# Compute the value function for the example for a time horizon of 2 and show that you obtain the linear value function constraints of equation 15.31 (along with a few others that have not been pruned away).
# Develop a simple pruning strategy to prune away some of the obviously superfluous constraints (replicates of initial payoff constraints, constraints dominated by (lying below) the payoff constraint for u3).
# Modify the state transition probability for u3 as well as the measurement probabilities for states x1 and x2. Compute value functions for different probabilities. Do your results make sense? Change the payoffs associated with the control actions and compute the value function. Do the results make sense?
# Using the probability and payoff parameters of your choice, use the associated value function to choose control actions. Assume that your true initial state is x1 and that your belief is 0.6. What outcomes do you obtain for 10 trials? Do you outcomes align with your expectations? Did your value function produce good results?

class POMDP:
    def __init__(self, t=1):
        self.T = t  # time horizon
        self.gamma = 1.0  # discount factor
        self.Y = np.array([[0, 0, 0]])

        # dimension space of problem
        self.N = 2  # number of states (x1,x2) = (facing forward, facing backward)
        self.Nu = 3  # number of control inputs (u1,u2,u3) = (drive forward, drive backward, turn around)
        self.Nz = 2  # number of measurements (z1,z2) = (sense forward, sense backward)

        self.r = np.array([[-100, 100, -1], [100, -50, -1]])  # reward r(x_i, u_iu)
        self.pt = np.array([[0.2, 0.8], [0.8, 0.2]])  # transition probabilities pt(x_i ' | x_j, u_iu)
        self.pz = np.array([[0.7, 0.3], [0.3, 0.7]])  # measurement probabilites px(z_iz | x_j)

        self.K = 1  # number of linear constraint functions
        self.cost = 0  # cost accrued
        self.p1 = 0.5  # initial belief of state being x1

    def Run(self):
        for tau in range(self.T):
            Ypr = []
            #  calculate linear constraints for all u_pr
            v = np.zeros((self.K, self.Nu, self.Nz, self.N))
            for k in range(self.K):  # for every line k
                for iu in range(self.Nu):  # consider all possible control options
                    for iz in range(self.Nz):  # consider all possible measurements
                        for j in range(self.N):  # consider all possible states
                            # Calculate values v(k, iu, iz, j)
                            v[k, iu, iz, j] = self.Y[k, 1]*self.pz[iz, 0]*self.pt[0, j] + self.Y[k, 2]*self.pz[iz, 1]*self.pt[1, j]
            # calculate linear constraints of new value function
            for iu in range(self.Nu):  # consider all possible control options
                # For our problem, there are Nz = 2 nested loops from 1: K
                for k1 in range(self.K):
                    for k2 in range(self.K):
                        ypr_temp = np.zeros((1,self.N + 1))
                        ypr_temp[0] = iu
                        for i in range(self.N):
                            pass
                            # Calculate end points of new set of linear functions vpr(1) and vpr(2)
                            # ypr_temp(i+1) = self.gamma * (self.r[i,iu] + )
                        # Ypr = []  # Augment Ypr
            # Prune unnecessary linear functions
            self.Prune(Ypr)

            # Assign pruned linear functions to final set
            self.Y = Ypr

            # Plot current linear function set
            self.Visualize()

    def Prune(self, Ypr):
        # self.K = None
        pass

    def Visualize(self):
        for i in range(self.K):
            pass
            # plt.plot([0, self.Y[]],)

