import sys
sys.path.append("/home/ami/ycy/JyTo")

from trajOptimizer import *
import model, vis
import pickle as pkl
from trajOptimizerHelper import *

from ExperimentSecretary.Core import Session
import os
import time

"""
This file use dynamic constraint as dynamics, rather than dynF
"""


# input dims: [ux4,Fbx2,Ffx2]
PI = np.math.pi
dT = 0.01
distance = 0.1

X0 = np.array([0,0.25,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([distance, 0.25 ,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])


xlim = [
    [-np.inf,np.inf],
    [0,np.inf],
    [-np.inf, np.inf],
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

EoMFuncs = {
    (1,1): model.buildEOMF((1,1)),
    (1,0): model.buildEOMF((1,0)),
    (0,1): model.buildEOMF((0,1)),
    (0,0): model.buildEOMF((0,0))
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 20, "start"),
    ((1,0), 20, "lift"),
    ((0,0), 20, "fly"),
    ((0,1), 20, "step1"),
    ((0,0), 30, "fly1"),
    ((1,0), 20, "step2"),
    # ([model.phbLeg2], 3, "land"),
    # ((1,1), 50, "finish")
]

Xlift0 = X0.copy()
Xlift0[2] = np.math.pi/6

References = [
    lambda i:( # start
        X0,
        [1,125,1,125,0,100,0,100]
    ),
    lambda i:( # lift
        Xlift0,
        [0,100,0,0,0,100,0,0]
    ),
    lambda i:( # fly
        X0 + np.concatenate([np.array([distance/20*i, 0.2+i*(20-i)/100, -2*PI/5/20*i]), np.zeros(11)]),
        [0,0,0,0, 0,0,0,0]
    ),
    lambda i:( # step1
        X0 + np.concatenate([np.array([distance, 0.2, -2*PI/5- PI/4/20*i]), np.zeros(11)]),
        [0,0,0,0, 0,0,0,0]
    ),
    lambda i:( # fly1
        X0 + np.concatenate([np.array([distance + distance/30*i, 0.2+i*(30-i)/100, -2*PI/5- PI/4-2*PI/3/30*i]), np.zeros(11)]),
        [0,0,0,0, 0,0,0,0]
    ),    
    lambda i:( # step2
        X0 + np.concatenate([np.array([distance + distance, 0.2, -2*PI/5- PI/ -2*PI/3]), np.zeros(11)]),
        [0,0,0,0, 0,0,0,0]
    ),

    # lambda i:( # finish
    #     XDes,
    #     [1,125,1,150, 0,125,0,150]
    # )
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    None, #(lambda x,u: x[1], [0], [np.inf]), # lift up body # start
    None, #(lambda x,u: x[8], [0.5], [np.inf]),  # lift
    None, #(lambda x,u: x[8], [0.5], [np.inf]),  # fly
    (lambda x,u: (x - PI/2)[2], [-np.inf], [0]), #(lambda x,u: x[8], [0.5], [np.inf]),  # step1
    (lambda x,u: (x - PI)[2], [-np.inf], [0]),  # fly1
    (lambda x,u: (x - 3*PI/2)[2], [-np.inf], [0]),  # step2
    # (lambda x,u: (x - XDes)[:7], [0]*7, [0]*7) # arrive at desire state
]

opt = ycyCollocation(14, 8, xlim, [-200, 200], dT)

opt.init([1,125,1,125,0,100,0,100], X0)

DynF = model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"])

x_val = X0
for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    EOMF = EoMFuncs[cons]
    for i in range(N):
        x_0, u_0 = R(i)

        # forward simulation
        # Uk = np.array(u_0)[:4]
        # dynSol = DynF(x = x_val, u = Uk)
        # x_val += dT * dynSol["dx"]
        # opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        #         list(Uk) + list(dynSol["F0"].full().reshape(-1))+ list(dynSol["F1"].full().reshape(-1)), x_val.full().reshape(-1))

        initSol = model.solveCons(EOMF, [("x",x_0, 1e6), ("ddq", np.zeros(7), 1e3)])
        opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                ca.veccat(initSol["u"],initSol["F"]).full().reshape(-1), x_0)

        # opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        #         u_0, X0)

        opt.addCost(lambda x,u: 0.001*ca.dot(u[:4],u[:4]))
        # opt.addCost(lambda x,u: ca.dot(x - X0,x - X0))
        addAboveGoundConstraint(opt)

        def holoCons(x,u):
            MU = 0.4
            return ca.vertcat(
                *[MU * u[5+i*2] + u[4+i*2] for i in range(2) if cons[i]],
                *[MU * u[5+i*2] - u[4+i*2] for i in range(2) if cons[i]],
                *[u[5+i*2] - 0 for i in range(2) if cons[i]]
            )
        opt.addConstraint(
            holoCons, [0]*(sum(cons))*3, [np.inf]*(sum(cons))*3
        )

        if(i==0):
            if(cons[0]):
                opt.addConstraint(lambda x,u: model.pFuncs["phbLeg2"](x)[1], [0], [0])
            if(cons[1]):
                opt.addConstraint(lambda x,u: model.pFuncs["phfLeg2"](x)[1], [0], [0])

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

opt.step(lambda dx,x,u : EoMFuncs[(0,0)](x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
            [0,0,0,0, 0,0,0,0], XDes)

def rounge_Kutta(x,u,dynF):
    DT = dT
    k1 = dynF(x, u)
    k2 = dynF(x + DT/2 * k1, u)
    k3 = dynF(x + DT/2 * k2, u)
    k4 = dynF(x + DT * k3, u)
    x=x+DT/6*(k1 +2*k2 +2*k3 +k4)
    return x


if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    with Session(__file__,terminalLog = True) as ss:
    # if(True):
        opt.startSolve()

        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "ycyCollo%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":opt._sol,
                "sol_x":opt.getSolX(),
                "sol_u":opt.getSolU(),
                "Scheme":Scheme
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("Note","Da Bai Lun")