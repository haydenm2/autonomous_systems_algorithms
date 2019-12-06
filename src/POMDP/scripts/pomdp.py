#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Generic Partially Observable Markov Decision Process Model (From Probablistic Robotics)

# Implement algorithm 15.1 with the objective of representing the value function for the illustrative example of section 15.2
# Compute the value function for the example for a time horizon of 2 and show that you obtain the linear value function constraints of equation 15.31 (along with a few others that have not been pruned away).
# Develop a simple pruning strategy to prune away some of the obviously superfluous constraints (replicates of initial payoff constraints, constraints dominated by (lying below) the payoff constraint for u3).
# Modify the state transition probability for u3 as well as the measurement probabilities for states x1 and x2. Compute value functions for different probabilities. Do your results make sense? Change the payoffs associated with the control actions and compute the value function. Do the results make sense?

class POMDP:
    def __init__(self, t=20):
        self.T = t  # time horizon
        self.gamma = 1.0  # discount factor

        # dimension space of problem
        self.N = 2  # number of states (x1,x2) = (facing forward, facing backward)
        self.Nu = 3  # number of control inputs (u1,u2,u3) = (drive forward, drive backward, turn around)
        self.Nz = 2  # number of measurements (z1,z2) = (sense forward, sense backward)

        # rewards and probabilities
        self.r = np.array([[-100, 100, -1], [100, -50, -1]])  # reward r(x_i, u_iu)
        self.pt = np.array([[0.2, 0.8], [0.8, 0.2]])  # transition probabilities pt(x_i ' | x_j, u_iu)
        self.pz = np.array([[0.7, 0.3], [0.3, 0.7]])  # measurement probabilites px(z_iz | x_j)

        self.K = 1  # number of linear constraint functions
        self.Y = np.zeros((self.K, self.N))
        self.Y0 = np.hstack((self.r[0, 0:self.N].reshape(-1, 1), self.r[1, 0:self.N].reshape(-1, 1)))
        self.pruning_res = 0.0001
        self.Y_final_w_commands = self.Y

        # action simulation params
        self.action_command = []
        self.x_true = [0]
        self.x_estimated = [1]
        self.z_received = []
        self.p1 = 0.5  # initial belief of state being x1
        self.cost = 0

        # plotter init
        plt.figure(1)
        plt.title('Value Functions')
        plt.ylabel('Reward (r)')
        plt.xlabel('Belief in State 1 (b(x1))')
        self.live_viz = False

    def CreateValueMap(self):
        for tau in range(self.T):
            if self.live_viz:
                self.VisualizeValues()
            self.Sense()
            self.Prune()
            self.Prediction()
            self.Prune()
        self.VisualizeValues()
        self.Y_final_w_commands = np.hstack((np.hstack((0, 1, np.ones(len(self.Y)-2)*2)).reshape(-1, 1), self.Y))
        print(self.Y)

    def Sense(self):
        Ypr1 = np.multiply(self.Y, self.pz[:, 0])
        Ypr2 = np.multiply(self.Y, self.pz[:, 1])
        rng = np.arange(0, len(self.Y))
        combos = np.vstack((np.tile(rng, self.K), np.repeat(rng, self.K)))
        self.Y = Ypr1[combos[0, :]] + Ypr2[combos[1, :]]

    def Prediction(self):
        self.Y = self.gamma*((self.Y @ self.pt) - 1)
        self.Y = np.vstack((self.Y0, self.Y))

    def Prune(self):
        probs = np.vstack([np.arange(0, 1+self.pruning_res, self.pruning_res), np.arange(0, 1+self.pruning_res, self.pruning_res)[::-1]])
        lines = self.Y @ probs
        index = np.unique(np.argmax(lines, axis=0))
        self.Y = self.Y[index]
        self.K = len(self.Y)

    def VisualizeValues(self):
        plt.clf()
        for i in range(self.K):
            plt.plot([self.Y[i, 1], self.Y[i, 0]], 'r-')
        plt.pause(0.1)

    def Play(self):
        while(True):
            # determine true and estimated sensor response
            r = np.random.rand()
            if r <= self.pz[self.x_true[-1], self.x_true[-1]]:
                self.z_received.append(self.x_true[-1])
                print('good')
            else:
                self.z_received.append(int(not(self.x_true[-1])))
                print('bad')

            # update belief from sensor
            self.p1 = self.pz[0, 0]*self.p1/(self.pz[0, 0]*self.p1 + self.pz[0, 1]*(1-self.p1))
            print(self.p1)

            # determine intended action and cost
            self.action_command.append(int(self.Y_final_w_commands[np.argmax(self.Y @ np.array([self.p1, 1 - self.p1])), 0]))  # propogate
            self.cost += self.r[self.x_true[-1], self.action_command[-1]]

            # determine true and estimated states
            if self.action_command[-1] == 2:
                self.x_estimated.append(int(not(self.x_estimated[-1])))
                r = np.random.rand()
                if r <= self.pt[self.x_true[-1], int(not(self.x_true[-1]))]:
                    self.x_true.append(int(not(self.x_true[-1])))
                    print('good')
                else:
                    self.x_true.append(self.x_true[-1])
                    print('bad')
            else:
                pass

            # update belief from action
            self.p1 = self.pt[0, 0]*self.p1 + self.pt[0, 1]*(1-self.p1)
            print(self.p1)
            print("")

            # check if terminal action
            if self.action_command[-1] != 2:
                print("Final Score: ", self.cost)
                if (self.r[self.x_true[-1], self.action_command[-1]] == self.r[0,0]) or (self.r[self.x_true[-1], self.action_command[-1]] == self.r[1,1]):
                    if self.action_command[-1] == 0:
                        print("Drove Forward into lava. You Lose...")
                    else:
                        print("Drove Backward into lava. You Lose...")
                else:
                    if self.action_command[-1] == 0:
                        print("Drove Foward through Door. You Win!")
                    else:
                        print("Drove Backward through Door. You Win!")
                break





