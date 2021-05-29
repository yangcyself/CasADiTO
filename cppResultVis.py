# from collocationMain4 import *
# import vis
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import pandas as pd
import numpy as np

from model.leggedRobot2D import LeggedRobot2D
model = LeggedRobot2D.fromYaml("data/robotConfigs/JYminiLitev2.yaml")


if(len(sys.argv)<2):
    print("please input the solution file prefix")
    exit()

prefix = sys.argv[1]
dT = 0.01

x_opt = np.genfromtxt("%s/X_out.csv"%prefix, delimiter=',')
u_opt = np.genfromtxt("%s/U_out.csv"%prefix, delimiter=',')
timeStamps = np.genfromtxt("%s/T_out.csv"%prefix, delimiter=',')


# Animate
fig, ax = plt.subplots()
# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])
print("len(x_opt)",len(x_opt))
print("len(timestamp)",len(timeStamps))
print("TIME STEP LENGTH of EACH PHASE:")

def animate(i):
    t = (i*0.01) % timeStamps[-1]
    ind = 0
    while(ind<len(timeStamps)-2 and timeStamps[ind]<t-1e-9 ):
        ind+=1
    xsol = x_opt[ind]

    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    # robotLinesol = vis.visFunc(xsol[:7])
    # linesol, = ax.plot(robotLinesol[:,0], robotLinesol[:,1])
    linesol = model.visulize(xsol)
    # til = ax.set_title(phase[i%Total])
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    return linesol,

ani = animation.FuncAnimation(
    fig, animate, frames= int(timeStamps[-1]/0.01), interval=25, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("data/animation/collocation.mp4")
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