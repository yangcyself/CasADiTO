from optimain import *
import vis
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("log/21_3_19_shooting_solution.pkl", "rb") as f:
    sol = pkl.load(f)

print(sol.keys())

w_opt = sol['x']

    # # Plot the solution
u_opt = w_opt

x_opt = [Xinit]

count = 0

robotLines = []
for cons, N, name in Scheme:
    DynF = DynFuncs[cons]
    F = Fs[cons]

    for k in range(N):
        # New NLP variable for the control
        Uk = u_opt[count: count+4]
        count+=4
        Xk = x_opt[-1]
        # robotLines.append(vis.visFunc([:7]))

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        x_opt += [Fk['xf'].full()]



# Animate
fig, ax = plt.subplots()
# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])

def animate(i):
    Total = len(x_opt)
    x = x_opt[i%Total]
    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    robotLine = vis.visFunc(x[:7])
    line, = ax.plot(robotLine[:,0], robotLine[:,1])
    ax.set_xlim(-1.5,2.5)
    ax.set_ylim(-0.5,3.5)
    return line,

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
