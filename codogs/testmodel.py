import sys
sys.path.append(".")

from heavyRopeLoad import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


m = HeavyRopeLoad(nc = 2)

x = ca.DM([-1,0,0])
m.setHyperParamValue({
    "r": ca.DM([1, 0.5]),
    "pc": ca.DM([-1,1,
                 1, 0]), 
    "pa": ca.DM([-2.5,1,
                0,0]), 
    "Q": np.diag([1,1,1]), 
    "xold": x
})

x_plot = []
pa1_plot = []
pa2_plot = []

for i in range(1000):
    try:
        r = i/300
        pa1 = ca.DM([-3*ca.cos(r) , 1+3*ca.sin(r)])
        pa2 = ca.DM([0.5-0.5*ca.cos(r), 0.5*ca.sin(r)])
        x = m.dynam(x, ca.vertcat(pa1, pa2))

        x_plot.append(x)
        pa1_plot.append(pa1)
        pa2_plot.append(pa2)
    except Exception as e:
        print(e)
        print("x:", x)
        print("pa1:", pa1)
        print("pa2:", pa2)
        break
pfuncs = m.pFuncs

fig, ax = plt.subplots()

def animate(i):
    ind = i%len(x_plot)
    x = x_plot[ind]
    pa1 = pa1_plot[ind]
    pa2 = pa2_plot[ind]
    ps = pfuncs(x)

    linesol = ax.plot( 
        [pa1[0], ps[0,0], x[0], ps[1,0], pa2[0]],
        [pa1[1], ps[0,1], x[1], ps[1,1], pa2[1]]
    )
    # lineini = model.visulize(xini)

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    # return linesol,lineini,til
    return linesol

ani = animation.FuncAnimation(
    fig, animate, frames= len(x_plot), interval=25, blit=True, save_count=50)
plt.show()