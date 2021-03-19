import model, vis
from casadi import np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_val = np.array([0,0,0,-0.3,-2.5, -0.3, -2.5,
                  0,0,0,0,    0,    0,    0])

# x_val = np.array([0,0,0,-np.math.pi/2, 0, -0.3, -2.5,
#                   0,0,0,0,    0,    0,    0])

robotLines = []

for i in range(3000):
    x_val += model.DynF(x_val) * 0.001
    
    if(not i %10):
        robotLines.append(vis.visFunc(x_val[:7]))

        # visState(x_val[:7])
    # break

# Animate

fig, ax = plt.subplots()

# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])

def animate(i):
    i = i%300
    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    line, = ax.plot(robotLines[i][:,0], robotLines[i][:,1])
    ax.set_xlim(-2,2)
    ax.set_ylim(-3.5,0.5)
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=60, blit=True, save_count=50)

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
