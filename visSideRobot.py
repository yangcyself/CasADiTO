# from collocationMain4 import *
# from directMain import rounge_Kutta, DynFuncs
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vis import saveSolution

import sys
import pandas as pd
import numpy as np

from model.leggedRobotX import LeggedRobotX
model = LeggedRobotX.fromYaml("data/robotConfigs/JYminiLitev2.yaml")


if(len(sys.argv)<2):
    print("please input the solution file name")
    exit()

solFile = sys.argv[1]
dT = 0.01

with open(solFile, "rb") as f:
    solraw = pkl.load(f)
    sol = solraw["sol"]
    sol_x= sol['Xgen']['x_plot'].full().T
    sol_u= sol['Ugen']['u_plot'].full().T
    terrian = sol['Xgen']['terrain_plot'].full()
    Scheme = solraw["Scheme"]
    x_init = solraw["x_init"].full().T
    timeStamps = sol['dTgen']['t_plot'].full()

print(sol_x[0])
print(sol_x[1])


print(sol.keys())
print(solraw.keys())
print(sol_u.shape)
print(sol_x.shape)

# opt.loadSol(sol)


#     # # Plot the solution
u_opt = sol_u
x_opt = sol_x
print("u_optShape", u_opt.shape)
print("x_optShape", x_opt.shape)


phase = ["init"]
x_sim = [x_opt[0]]
u_count = 0
for cons, N, name in Scheme:
    # dynF = DynFuncs[cons]
    for i in range(N):
        phase.append(name)

# Animate
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(4,8))
# fig = plt.figure(figsize=(10,14))
# ax1 = fig.add_axes((0, 0, 10, 10))
# ax2 = fig.add_axes((0, 10, 4, 4))


# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])
print("len(phase)：",len(phase))
print("len(x_sim)",len(x_sim))
print("len(x_opt)",len(x_opt))
print("len(timestamp)",len(timeStamps))
print("TIME STEP LENGTH of EACH PHASE:")
for dt, (c,n,m) in zip(sol['dTgen']['_w'].full(), Scheme):
    print(m, "\tContact: ", c, "\tN: ",n, "\tdT:", dt )

def animate(i):
    t = (i*0.01) % timeStamps[-1]
    ind = 0
    while(ind<len(timeStamps)-2 and timeStamps[ind]<t-1e-9 ):
        ind+=1
    # xsim = x_sim[ind]
    xsol = x_opt[ind]
    xini = x_init[ind]
    ax1.clear()
    
    linesol = model.visulize(xsol,ax1)
    lineini = model.visulize(xini,ax1)

    terrianLine = ax1.plot(terrian[:,0],terrian[:,1])

    til = ax1.set_title(phase[ind])
    # til = ax1.set_title(phase[i%Total])
    ax1.set_xlim(-0.5,1.5)
    ax1.set_ylim(-0.5,1.5)


    ax2.clear()
    legl, legr = model.visulizeLocal(xsol,ax2)
    return linesol,lineini,til, legl, legr
    # return linesol,til

ani = animation.FuncAnimation(
    fig, animate, frames= int(timeStamps[-1]/0.01), interval=25, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("data/animation/visSideRobot.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

# saveSolution("out.csv", sol_x, sol_u, timeStamps.reshape(-1))

plt.show()

plt.figure()
plt.plot(u_opt)
plt.legend(["u1","u2","u3","u4"])
plt.figure()
plt.plot(x_opt[:,:7])
plt.legend(["x","y","th","q1","q2","q3","q4"])
plt.show()