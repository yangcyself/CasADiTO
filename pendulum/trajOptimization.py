import sys
sys.path.append(".")
from optGen.trajOptimizer import *
import pendulumModel
import pickle as pkl

from ExperimentSecretary.Core import Session
import os
import time
import numpy as np


pend = pendulumModel.Pendulum()
dynf = pend.dyn_func
mDim = pend.dim

dT0 = 0.01

X0 = np.array([0.]* mDim *2)
XDes = X0.copy()
XDes[1] += np.math.pi/2


xlim = [[-np.inf,np.inf]] * mDim *2
ulim = [[-40.,40.]]



opt = TowrCollocationDefault(mDim*2, 1, 0, xlim, ulim, [[]], dT0)
opt.begin(x0=X0, u0=[0.], F0 = [])


for i in range(500):
    opt.step(lambda dx,x,u : dynf(x,u) - dx[mDim:],
            x0 = X0 + np.random.random(mDim*2) - np.random.random(mDim*2),# + np.array([0, i/200 * np.math.pi]+[0]*8),
            u0 = np.random.random(1) - np.random.random(1), F0 = [])

    # opt.addCost(lambda x,u: u*u)
    opt.addCost(lambda x,u: x[0]**2)
    opt.addCost(lambda x,u: -pend.CoMposValue(x)[-1, 1]**3)
 
# opt.addConstraint(lambda x: x-XDes, [0.]*mDim*2, [0.]*mDim*2)
# opt.addConstraint(lambda x: (x-XDes)[0], [0.], [0.])

opt.step(lambda dx,x,u : dynf(x,u) - dx[mDim:], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        x0 = XDes, u0 = [0.], F0=[])


if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    with Session(__file__,terminalLog = True) as ss:
    # if(True):
        res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            # "expand" : True,
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 10000, # unkown option
                }
            })

        x_plot = res["Xgen"]["x_plot"].full().T
        


        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        def animate(i):
            i = (i*1)%len(x_plot)
            # line.set_xdata(robotLines[i][:,0])  # update the data.
            # line.set_ydata(robotLines[i][:,1])  # update the data.
            ax.clear()
            line = pend.visulize(x_plot[i])
            # ax.set_xlim(-5+Xs[i][0],5+Xs[i][0])
            ax.set_xlim(-8,8)
            ax.set_ylim(-8,8)
            return line,

        ani = animation.FuncAnimation(
            fig, animate, interval=100, blit=True, save_count=5000)
        plt.show()
        ani.save("penduMovie2.mp4")

        fig, ax = plt.subplots()
        plt.plot(x_plot[:-1,:2])
        fig, ax = plt.subplots()
        plt.plot([pend.CoMposValue(x)[1,1] for x in x_plot] )
        plt.show()


        dumpname = os.path.abspath(os.path.join("pendulum/data", "nlpSol%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":res,
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("Note","TO of the pendulum")