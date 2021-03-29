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
distance = 0.5
# X0 = np.array([0,0.25,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
#          0,0,0,0,    0,    0,    0])

# XDes = np.array([0.5,0.25,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
#          0,0,0,0,    0,    0,    0])

X0 = np.array([0,0.25,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([distance, 0.25 ,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])



xlim = [
    [-np.inf,np.inf],
    [0,np.inf],
    [-model.PI, model.PI],
    model.params["q1Lim"],
    model.params["q2Lim"],
    model.params["q1Lim"],
    model.params["q2Lim"],
    [-100,100],
    [-100,100],
    [-100,100],
    model.params["dq1Lim"],
    model.params["dq2Lim"],
    model.params["dq1Lim"],
    model.params["dq2Lim"]
]

DynFuncs = {
    (1,1): model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"]),
    (1,0): model.buildDynF([model.phbLeg2],"back_leg", ["btoe"]),
    (0,0): model.buildDynF([],"fly")
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 50, "start"),
    ((1,0), 50, "lift"),
    ((0,0), 50, "fly"),
    # ([model.phbLeg2], 3, "land"),
    # ((1,1), 50, "finish")
]

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
    lambda i:( # fly
        X0 + np.concatenate([np.array([distance/50*i, 0.2+i*(50-i)/625]), np.zeros(12)]),
        [0,0,0,0]
    ),
    # lambda i:( # finish
    #     XDes,
    #     [1,125,1,150]
    # )
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    (lambda x,u: x[1], [0], [np.inf]), # lift up body
    (lambda x,u: x[8], [0.5], [np.inf]), # have positive velocity
    # (lambda x,u: ca.vertcat(model.pFuncs["phbLeg2"](x)[1], model.pFuncs["phfLeg2"](x)[1]), 
    #                 [0]*2, [0]*2), # feet land
    (lambda x,u: (x - XDes)[:7], [0]*7, [0]*7) # arrive at desire state
]


# input dims: [ux4,Fbx2,Ffx2]
opt = DirectOptimizer(14, 4, xlim, [-200, 200], dT)


def rounge_Kutta(x,u,dynF):
    DT = dT
    k1 = dynF(x, u)
    k2 = dynF(x + DT/2 * k1, u)
    k3 = dynF(x + DT/2 * k2, u)
    k4 = dynF(x + DT * k3, u)
    x=x+DT/6*(k1 +2*k2 +2*k3 +k4)
    return x


opt.init([1,125,1,150],X0)

for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    dynF = DynFuncs[cons]
    for i in range(N):
        x_0, u_0 = R(i)
        opt.step(lambda x,u : dynF(x=x,u=u)["dx"],
                rounge_Kutta,
                np.array(u_0),x_0)
        opt.addCost(lambda x,u: 0.001*ca.dot(u,u))

        addAboveGoundConstraint(opt)

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

    opt.addConstraint(*FinalC)


if __name__ == "__main__" :

    # minutes_to_wait = 60*3
    # for i in range(minutes_to_wait):
    #     print("%d minutes to run the program"%(minutes_to_wait-i))
    #     time.sleep(60)


    with Session(__file__,terminalLog = True) as ss:
        opt.startSolve()
        
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "direct%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":opt._sol,
                "sol_x":opt.getSolX(),
                "sol_u":opt.getSolU(),
                "Scheme":Scheme
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("Note","Debugged the model!!!!!!")