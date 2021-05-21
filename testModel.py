from model.leggedRobotX import LeggedRobotX
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation

model = LeggedRobotX.fromYaml("data/robotConfigs/JYminiLitev2.yaml")

fl = model.pFuncs["pl2"]
fr = model.pFuncs["pr2"]
import numpy as np
# x_val = ca.DM([0,1,0, 1/2*ca.pi, -1/2*ca.pi, 0, 1/2*ca.pi, -1/2*ca.pi, 0 ]+[0]*9)
x_val = ca.DM([0,1,0, 0, -np.math.pi*5/6,np.math.pi*2/3, 0, -np.math.pi*5/6,np.math.pi*2/3 ]+[0]*9)
print(ca.symvar(model.l2._p_proj(model.l2.bdy.points["b"])))
print(fl(x_val))
print(fr(x_val))

### Kinematic Test

def x_f(i):
    add = ca.DM.zeros(18)
    add[8] = i/100
    return x_val + add

# print(f(x_f(0)))
# print(f(x_f(ca.pi/2*100)))

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    line = model.visulize(x_f(i))
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    # return linesol,lineini,til
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=25, blit=True, save_count=50)

plt.show()

# import numpy as np
initHeight = (model.params["legL2"] + model.params["legL1"])/2 # assume 30 angle of legs
X0 = np.array([0, initHeight,0, 0.5*np.math.pi, -np.math.pi*5/6,np.math.pi*2/3, 0.5*np.math.pi, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0, 0,0,0, 0,0,0])

# fleg_local_l = model.pLocalFuncs['pl2']
# fleg_local_r = model.pLocalFuncs['pr2']
# print(fleg_local_l(X0))
# print(fleg_local_r(X0))

consJ = ca.jacobian( model.r2._p_proj(model.r2.points["b"]), model.q)
consJf = ca.Function("f", [model.x], [consJ])
localConsJ = ca.jacobian( model.r2.points["b"], model.q)
localConsJf = ca.Function("f", [model.x], [localConsJ])
print(consJf(X0))
print()
print(localConsJf(X0))
