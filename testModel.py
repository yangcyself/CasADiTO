from model.leggedRobotX import LeggedRobotX
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation

model = LeggedRobotX.fromYaml("data/robotConfigs/JYminiLitev2.yaml")

f = model.pFuncs["pl2"]

x_val = ca.DM([0,1,0, 1/2*ca.pi, -1/2*ca.pi, 0, -ca.pi,0,0 ]+[0]*9)
print(ca.symvar(model.lhip._p_proj(model.l2.points["b"])))
print(f(x_val))

f_xy = model.pFuncs["pltoexy"]
print(f_xy(x_val) )

### Kinematic Test

# def x_f(i):
#     add = ca.DM.zeros(18)
#     add[5] = i/100
#     return x_val + add

# # print(f(x_f(0)))
# # print(f(x_f(ca.pi/2*100)))

# fig, ax = plt.subplots()
# def animate(i):
#     ax.clear()
#     line = model.visulize(x_f(i))
#     ax.set_xlim(-0.5,1.5)
#     ax.set_ylim(-0.5,1.5)
#     # return linesol,lineini,til
#     return line,

# ani = animation.FuncAnimation(
#     fig, animate, interval=25, blit=True, save_count=50)

# plt.show()