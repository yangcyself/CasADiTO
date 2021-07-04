"""
Given an observation, try to find the config parameters of the model
"""

import sys
import matplotlib.pyplot as plt
sys.path.append(".")

from pendulum.pendulumModel import Pendulum
import pickle as pkl


m = Pendulum(symbolWeight = True)

with open("pendulum/data/nlpSol1625383068.pkl", "rb")  as f:
    data = pkl.load(f)
print(data.keys())