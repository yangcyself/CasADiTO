from trajOptimizer import *
import model, vis
import pickle as pkl
from trajOptimizerHelper import *

from ExperimentSecretary.Core import Session
import os
import time

"""
use direct scription method, use ronge kuta
"""

dT = 0.01

DynFuncs = {
    (1,1): model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"]),
    (1,0): model.buildDynF([model.phbLeg2],"back_leg", ["btoe"]),
    (0,0): model.buildDynF([],"fly")
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 50, "start"),
    ((1,0), 50, "lift"),
    ((0,0), 50, "fly"),
    # # ([model.phbLeg2], 3, "land"),
    ((1,1), 50, "finish")
]

X0 = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([1.5,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

Xlift0 = X0.copy()
Xlift0[2] = np.math.pi/6

References = [
    lambda i:( # start
        X0,
        [1,125,1,150]
    ),
    lambda i:( # lift
        Xlift0,
        [100,200,0,0]
    ),
    lambda i:( # lift
        X0 + np.concatenate([np.array([1.5/50*i, 2*1.5/50*i*(1.5-1.5/50*i)]), np.zeros(12)]),
        [0,0,0,0]
    ),
    lambda i:( # lift
        XDes,
        [1,125,1,150]
    ),

]

# input dims: [ux4,Fbx2,Ffx2]
opt = DirectOptimizer(14, 4, [np.inf, np.inf, np.math.pi, np.math.pi, np.math.pi, np.math.pi, np.math.pi],
                        [-200, 200], dT)


def rounge_Kutta(x,u,dynF):
    DT = dT
    k1 = dynF(x, u)
    k2 = dynF(x + DT/2 * k1, u)
    k3 = dynF(x + DT/2 * k2, u)
    k4 = dynF(x + DT * k3, u)
    x=x+DT/6*(k1 +2*k2 +2*k3 +k4)
    return x


opt.init(X0)

for (cons, N, name),R in zip(Scheme,References):
    dynF = DynFuncs[cons]
    for i in range(N):
        x_0, u_0 = R(i)
        opt.step(lambda x,u : dynF(x=x,u=u)["dx"],
                rounge_Kutta,
                np.array(u_0),x_0)
        opt.addCost(lambda x,u: 0.000001*ca.dot(u,u))

        addAboveGoundConstraint(opt)

        if(sum(cons) == 0):
            continue

        def holoCons(x,u):
            MU = 0.4
            dynFsol = dynF(x = x, u = u)
            return ca.vertcat(
                *[MU * dynFsol["F%d"%i][1] + dynFsol["F%d"%i][0] for i in range(sum(cons))],
                *[MU * dynFsol["F%d"%i][1] - dynFsol["F%d"%i][0] for i in range(sum(cons))],
                *[dynFsol["F%d"%i][1] - 10 for i in range(sum(cons))]
            )
        opt.addConstraint(
            holoCons, [0]*(sum(cons))*3, [np.inf]*(sum(cons))*3
        )

opt.addConstraint(lambda x,u: x - XDes, [0]*14, [0]*14)


if __name__ == "__main__" :

    with Session(__file__,terminalLog = True) as ss:
        opt.startSolve()
        
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "direct%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":opt._sol,
                "sol_x":opt.getSolX(),
                "sol_u":opt.getSolU()
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Note","Will a good U0 help?")