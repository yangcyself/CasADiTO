from collocationMain4 import *
from directMain import rounge_Kutta, DynFuncs
import vis
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import pandas as pd

if(len(sys.argv)<2):
    print("please input the solution file name")
    exit()

solFile = sys.argv[1]

with open(solFile, "rb") as f:
    solraw = pkl.load(f)
    sol = solraw["sol"]

print(sol.keys())
print(solraw.keys())

opt.loadSol(sol)


#     # # Plot the solution
u_opt = opt.getSolU().T
x_opt = opt.getSolX().T

df_x = pd.DataFrame(x_opt, 
    columns = ["x", "y", "r", "bh", "bt", "fh", "ft",
            "dx", "dy", "dr", "dbh", "dbt", "dfh", "dft"], 
    index = [dT * i for i in range(x_opt.shape[0])]
)

df_u = pd.DataFrame(u_opt, 
    columns = ["ubh", "ubt", "ufh", "uft", "Fbx", "Fby", "Ffx", "Ffy"], 
    index = [dT * i for i in range(u_opt.shape[0])]
)

df = pd.concat([df_x,df_u],axis = 1)
print(df.head())

# Frame shift

df["bh"] = df["bh"] + np.math.pi/2
df["fh"] = df["fh"] + np.math.pi/2
df.to_csv('TOoutput.csv', index_label = "t", 
        columns = ["x", "y", "r", "bh", "bt", "fh", "ft", 
        "dx", "dy", "dr", "dbh", "dbt", "dfh", "dft", 
        "ubh", "ubt", "ufh", "uft"])

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
print("EOMFfunction",EoMFuncs[(1,1)](x_opt[0], u_opt[0][:4],u_opt[0][4:],ddq))

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

def animate(i):
    Total = len(x_sim)
    xsim = x_sim[i%Total]

    Total = len(x_opt)
    xsol = x_opt[i%Total]

    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    # robotLinesim = vis.visFunc(xsim[:7])
    robotLinesol = vis.visFunc(xsol[:7])
    # linesim, = ax.plot(robotLinesim[:,0], robotLinesim[:,1])
    linesol, = ax.plot(robotLinesol[:,0], robotLinesol[:,1])
    til = ax.set_title(phase[i%Total])
    til = ax.set_title(phase[i%Total])
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    return linesol,til
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