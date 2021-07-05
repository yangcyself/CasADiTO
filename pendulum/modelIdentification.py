"""
Given an observation, try to find the config parameters of the model
"""

import sys
import matplotlib.pyplot as plt
sys.path.append(".")

from pendulum.pendulumModel import Pendulum
import pickle as pkl
from optGen.trajOptimizer import optGen
import casadi as ca

m = Pendulum(symbolWeight = True)

with open("pendulum/data/nlpSol1625383068.pkl", "rb")  as f:
    data = pkl.load(f)
print(data['sol'].keys())
ddq_plot = data['sol']['ddq_plot'].T
x_plot = data['sol']['Xgen']['x_plot'].T
u_plot = data['sol']['Ugen']['u_plot'].T
print(ddq_plot.size())
print(x_plot.size())
print(u_plot.size())

dyn_func = m.dyn_func_sym # (sym, x,u) -> (ddq)

print(dyn_func([1,1,1,1], x_plot[1,:].T, u_plot[1,:].T))
print(ddq_plot[0,:].T)

opt = optGen()
var = ca.SX.sym("Masses", len(m._syms))
opt._w.append(var)
opt._lbw.append(ca.DM([0]*4))
opt._ubw.append(ca.DM([ca.inf]*4))
opt._w0.append(ca.DM([0.5]*4))
opt._state.update({"m":var})

for i in range(100):
    ind = 5*i
    x = x_plot[ind+1,:].T
    u = u_plot[ind+1,:].T
    ddq = ddq_plot[ind+0,:].T
    x = x+(ca.DM.rand(x.size())- ca.DM.rand(x.size())) * 0.01
    u = u+(ca.DM.rand(u.size())- ca.DM.rand(u.size())) * 0.01
    ddq = ddq+(ca.DM.rand(ddq.size())- ca.DM.rand(ddq.size())) * 0.01
    opt.addCost(lambda m: ca.sum1((dyn_func(m, x, u) - ddq)**2))

res = opt.solve()

print(res.keys())
print(res['_w'])
