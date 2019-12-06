#!/usr/bin/env python3
from pomdp import POMDP
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Summary:
# Example of implementation of the partially observable markov decision process class for a simplified two-state robot model

if __name__ == "__main__":
    pomdp = POMDP()       # MDP algorithm object
    pomdp.CreateValueMap()
    plt.show()