from directMain import *
import vis
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("directSol_with_init_toyProblem.pkl", "rb") as f:
    sol = pkl.load(f)

print(sol.keys())

opt.loadSol(sol)


    # # Plot the solution
u_opt = opt.getSolU().T
x_opt = opt.getSolX().T

x_sim = [x_opt[0]]
print("u_optShape", u_opt.shape)
print("x_optShape", x_opt.shape)


for cons, N, name in Scheme:
    dynF = DynFuncs[cons]
    for i in range(N):
        x_sim.append(rounge_Kutta(x_sim[-1], u_opt[i], 
            lambda x,u : dynF(x=x,u=u)["dx"]))
        

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
    robotLinesim = vis.visFunc(xsim[:7])
    robotLinesol = vis.visFunc(xsol[:7])
    linesim, = ax.plot(robotLinesim[:,0], robotLinesim[:,1])
    linesol, = ax.plot(robotLinesol[:,0], robotLinesol[:,1])
    ax.set_xlim(-1.5,2.5)
    ax.set_ylim(-0.5,3.5)
    return linesim,linesol

ani = animation.FuncAnimation(
    fig, animate, interval=200, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
