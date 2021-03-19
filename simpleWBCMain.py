import model, vis
from casadi import np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_val = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])
# x_val = np.array([0,0,0,-np.math.pi/2, 0, -0.3, -2.5,
#                   0,0,0,0,    0,    0,    0])


def Ctrl(x):
    
    CoM = model.pComFuncs["pMtor"](x)
    dCoM = model.dpComFuncs["dpMtor"](x)
    
    pbtoe = model.pComFuncs["phbLeg2"](x) - CoM
    pftoe = model.pComFuncs["phfLeg2"](x) - CoM

    F = 10*(np.array([0.5, 1.3]) - CoM) + ( - dCoM) \
        + np.array([0, model.params["G"]])*(model.params["torM"] + 4* model.params["legM"])
    Fth = 10 * (0 - x[2]) + (-x[9])

    MA = np.zeros((3,4))
    MA[:2,:2] =  np.eye(2)
    MA[:2,2:4] = np.eye(2)
    MA[2,:] = np.array([-pbtoe[1], pbtoe[0], -pftoe[1], pftoe[0]])

    FootF = np.linalg.pinv(MA) @ np.concatenate([F,Fth])
    
    Jac = np.concatenate([model.JacFuncs["Jbtoe"](x), model.JacFuncs["Jftoe"](x)])
    u = Jac.T @ FootF

    return u

robotLines = []

for i in range(3000):
    sol = model.DynF(x = x_val,u = np.zeros(4))
    x_val += sol["dx"] * 0.001
    
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
