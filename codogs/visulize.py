import sys
sys.path.append(".")

import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import sys
import pandas as pd
import numpy as np
from codogs.heavyRopeLoad import HeavyRopeLoad
import casadi as ca
if(len(sys.argv)<2):
    print("please input the solution file name")
    exit()

solFile = sys.argv[1]

with open(solFile, "rb") as f:
    solraw = pkl.load(f)
    sol = solraw["sol"]
    nc = solraw["nc"]
    # sol_ddq = sol["ddq_plot"].full().T
    sol_x= sol['Xgen']['x_plot'].full().T
    sol_u= sol['Ugen']['u_plot'].full().T

model = HeavyRopeLoad(nc)
model.setHyperParamValue({
    # "r": ca.DM([1, 1, 1]),
    "r": ca.DM([1]),
    "pc": ca.DM([1,0]),
    # "pc": ca.DM([-1,0, 
    #               1,0,
    #               0,1]), 
    # "pa": ca.DM([0,0, # not used
    #              0,0,
    #              0,0]), 
    # "Q": np.diag([1,1,1]), 
    # "xold": ca.DM([0,0,0])
})

pfuncs = model.pFuncs

fig, ax = plt.subplots()
def animate(i):
    ind = i%len(sol_x)
    xsol = sol_x[ind]
    usol = sol_u[ind]
    ax.clear()
    
    sth = np.sin(xsol[2])
    cth = np.cos(xsol[2])
    L,W = 2,2
    hl,hw = L/2, W/2

    box, = ax.plot(np.array([cth*hl-sth*hw, -cth*hl-sth*hw, -cth*hl+sth*hw, cth*hl+sth*hw, cth*hl-sth*hw])+xsol[0], 
                  np.array([sth*hl+cth*hw, -sth*hl+cth*hw, -sth*hl-cth*hw, sth*hl-cth*hw, sth*hl+cth*hw])+xsol[1])
    pcs = pfuncs(xsol)
    lines=[
        ax.plot([pcs[i,0], usol[2*i]],[pcs[i,1], usol[2*i+1]])
        for i in range(nc)
    ]

    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)

    
    return box,lines[0][0]
    # return linesol,til

ani = animation.FuncAnimation(
    fig, animate, interval=100, blit=True, save_count=50)

plt.show()