import model, vis
from casadi import np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dT = 0.001

def colloF(Xk,Xk_puls_1):
    q0 = Xk[:7]
    dq0 = Xk[7:]
    q1 = Xk_puls_1[:7]
    dq1 = Xk_puls_1[7:]

    a0 = q0
    a1 = dq0
    a2 = -(3*(q0 - q1) + dT*(2*dq0 + dq1))/(dT**2)
    a3 = (2*(q0 - q1) + dT*(dq0 + dq1))/(dT**3)

    ddq0 = 2*a2 # (6 * q1 - 2*dq1*dT - 6 * q0 - 4*dq0*dT)/(dT**2) # 6q1 - 2dq1dt - 6q0 - 4dq0dt

    qc = a0 + a1*(dT/2) + a2 * (dT/2)**2 + a3 * (dT/2)**3
    dqc = a1 + 2 * a2 * (dT/2) + 3* a3 * (dT/2)**2
    ddqc = 2 * a2 + 6 * a3 * (dT/2)
    print("q0 ",q0)
    print("qc ",qc)
    print("q1 ",q1)
    print("dq0 ",dq0)
    print("dqc ",dqc)
    print("dq1 ",dq1)
    return (a0,a1,a2,a3,ddq0,qc,dqc,ddqc)


x_val = np.array([0,0,0,-0.3,-2.5, -0.3, -2.5,
                  0,0,0,0,    0,    0,    0])

# x_val = np.array([0,0,0,-np.math.pi/2, 0, -0.3, -2.5,
#                   0,0,0,0,    0,    0,    0])

robotLines = []

# EOMF = model.buildEOMF((0,0))
EOMF = model.buildEOMF((1,1))
# EOMF = model.buildEOMF((1,0))

# DynF = model.buildDynF([],"all_leg", ["btoe","ftoe"])
DynF = model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"])
# DynF = model.buildDynF([model.phbLeg2],"all_leg", ["btoe"])
N = 30000
for i in range(N ):
    u = np.random.random(4)
    sol = DynF(x = x_val,u = u)
    # print(EOMF(x = x_val,u = u,F=np.concatenate([sol["F0"],np.array([0,0]).reshape(sol["F0"].size())]),ddq = sol["ddq"]))
    eomf_check = EOMF(x = x_val,u = u,F=np.concatenate([sol["F0"],sol["F1"]]),ddq = sol["ddq"])['EOM'].full()
    print(eomf_check.reshape(-1))
    assert(all(abs(eomf_check)<1e-9))
    x_val_last = x_val

    x_val += sol["dx"] * dT

    if(not i%500):
        plt.figure()
        plotX = np.linspace(0,dT,100)
        a0,a1,a2,a3,ddq0,qc,dqc,ddqc = colloF(x_val_last,x_val)
        ind = int(np.random.random()*7)
        plt.plot(plotX, (np.array(a3[ind]).reshape(-1,1)*plotX**3).T
                        +(np.array(a2[ind]).reshape(-1,1)*plotX**2).T
                        +(np.array(a1[ind]).reshape(-1,1)*plotX).T
                        +(np.array(a0[ind]).reshape(-1,1)).T)
        plt.show()

    # print(sol['FA'])
    # print(sol['Fb'])
    if(not i %10):
        robotLines.append(vis.visFunc(x_val[:7]))

        # visState(x_val[:7])
    # break

# Animate

fig, ax = plt.subplots()

# line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])

def animate(i):
    i = i%len(robotLines)
    # line.set_xdata(robotLines[i][:,0])  # update the data.
    # line.set_ydata(robotLines[i][:,1])  # update the data.
    ax.clear()
    line, = ax.plot(robotLines[i][:,0], robotLines[i][:,1])
    ax.set_xlim(-2,2)
    ax.set_ylim(-3.5,0.5)
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=6, blit=True, save_count=50)

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
