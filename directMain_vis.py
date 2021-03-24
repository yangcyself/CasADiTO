from directMain import *
import vis
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("data/nlpSol/direct1616576464.pkl", "rb") as f:
    sol = pkl.load(f)["sol"]

print(sol.keys())

opt.loadSol(sol)


    # # Plot the solution
u_opt = opt.getSolU().T
x_opt = opt.getSolX().T

x_sim = [x_opt[0]]
print("u_optShape", u_opt.shape)
print("x_optShape", x_opt.shape)

u_count = 0
for cons, N, name in Scheme:
    dynF = DynFuncs[cons]
    for i in range(N):
        x_sim.append( np.array(rounge_Kutta(x_sim[-1], u_opt[u_count], 
            lambda x,u : dynF(x=x,u=u)["dx"])).reshape(-1))
        u_count += 1
        
        # x = x_sim[-1]
        # for n,pfunc in model.pFuncs.items():
        #     if(pfunc(x)[1]<0):
        #         print(n, pfunc(x)[1])

# for x in x_opt:
#     for n,pfunc in model.pFuncs.items():
#         if(pfunc(x)[1]<0):
#             print(n, pfunc(x)[1])

print("x_simShape", np.array(x_sim).shape)
print(x_sim[:2])

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
    ax.set_xlim(-1.5,3.5)
    ax.set_ylim(-0.5,4.5)
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

plt.figure()
plt.plot(u_opt)
plt.legend(["u1","u2","u3","u4"])
plt.figure()
plt.plot(x_opt[:,:7])
plt.legend(["x","y","th","q1","q2","q3","q4"])
plt.show()