# from collocationMain4 import *
from directMain import rounge_Kutta, DynFuncs
import vis
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import pandas as pd
import numpy as np

if(len(sys.argv)<2):
    print("please input the solution file name")
    exit()

solFile = sys.argv[1]
dT = 0.01

with open(solFile, "rb") as f:
    solraw = pkl.load(f)
    sol = solraw["sol"]
    sol_x=solraw['sol_x'].T
    sol_u=solraw['sol_u'].T
    Scheme = solraw["Scheme"]
    x_init = solraw["x_init"]


print(sol.keys())
print(solraw.keys())
print(sol_u.shape)
print(sol_x.shape)

# opt.loadSol(sol)


#     # # Plot the solution
u_opt = sol_u
x_opt = sol_x


# exit()
# u_opt = solraw["sol_u"].T
# x_opt = solraw["sol_x"].T
# x_sim = x_opt


print("u_optShape", u_opt.shape)
print("x_optShape", x_opt.shape)

print("x_opt[0]",x_opt[0])
print("x_opt[1]",x_opt[1])
ddq = (6 * x_opt[1][:7] - 2*x_opt[1][7:]*dT 
            - 6 * x_opt[0][:7] - 4*x_opt[0][7:]*dT)/(dT**2)
print("ddq0",ddq)
print("U0F0",u_opt[0])

phase = ["init"]
x_sim = [x_opt[0]]
u_count = 0
for cons, N, name in Scheme:
    dynF = DynFuncs[cons]
    for i in range(N):
        phase.append(name)
        x_sim.append( np.array(rounge_Kutta(x_sim[-1], u_opt[u_count][:4], 
            lambda x,u : dynF(x=x,u=u)["dx"])).reshape(-1))
        u_count += 1
        


# Animate
fig, ax = plt.subplots()
# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])
print("len(phase)：",len(phase))
print("len(x_sim)",len(x_sim))
print("len(x_opt)",len(x_opt))

def animate(i):
    Total = len(x_sim)
    xsim = x_sim[i%Total]
    xsol = x_opt[i%Total]
    xini = x_init[i%Total]

    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    # robotLinesim = vis.visFunc(xsim[:7])
    robotLinesol = vis.visFunc(xsol[:7])
    robotLineini = vis.visFunc(xini[:7])
    # linesim, = ax.plot(robotLinesim[:,0], robotLinesim[:,1])
    linesol, = ax.plot(robotLinesol[:,0], robotLinesol[:,1])
    lineini, = ax.plot(robotLineini[:,0], robotLineini[:,1])
    til = ax.set_title(phase[i%Total])
    # til = ax.set_title(phase[i%Total])
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    return linesol,lineini,til
    # return linesim,linesol

ani = animation.FuncAnimation(
    fig, animate, frames= 150, interval=25, blit=True, save_count=50)

# To save the animation, use e.g.
#
ani.save("data/animation/collocation.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()

plt.figure()
plt.plot(u_opt)
plt.legend(["u1","u2","u3","u4"])
plt.figure()
plt.plot(x_opt[:,:7])
plt.legend(["x","y","th","q1","q2","q3","q4"])
plt.show()