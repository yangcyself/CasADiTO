# from trajOptimizer import *
# from policyOptimizer import *
from optGen.trajOptimizer import *
from optGen.helpers import pointsTerrian2D
# import model, vis
from model.leggedRobot2D import LeggedRobot2D
from mathUtil import solveLinearCons
import pickle as pkl
from trajOptimizerHelper import *

from ExperimentSecretary.Core import Session
import os
import time

from optGen.util import caSubsti, caFuncSubsti, substiSX2MX

"""
This file use dynamic constraint as dynamics, rather than dynF
"""

model = LeggedRobot2D.fromYaml("data/robotConfigs/JYminiLitev2.yaml")
# input dims: [ux4,Fbx2,Ffx2]
dT0 = 0.01
initHeight = (model.params["legL2"] + model.params["legL1"])/2 # assume 30 angle of legs
distance = ca.SX.sym("distance",1)


X0 = np.array([0, initHeight,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])


XDes = np.array([distance, initHeight ,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

SchemeSteps = 50
height = 0

# XDes = substiSX2MX(XDes, [distance], [distance_mx])

xlim = [
    [-np.inf,np.inf],
    [0,np.inf],
    [-ca.pi, ca.pi],
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



ulim = [
    np.array([-1, 1])*model.params["tau2lim"],
    np.array([-1, 1])*model.params["tau3lim"],
    np.array([-1, 1])*model.params["tau2lim"],
    np.array([-1, 1])*model.params["tau3lim"]
]

EoMFuncs = {
    (1,1): model.buildEOMF((1,1)),
    (1,0): model.buildEOMF((1,0)),
    (0,0): model.buildEOMF((0,0))
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), SchemeSteps, "start"),
    ((1,0), SchemeSteps, "lift"),
    ((0,0), SchemeSteps, "fly"),
    # ([model.phbLeg2], 3, "land"),
    # ((1,1), SchemeSteps, "finish")
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
        X0 + np.concatenate([np.array([distance/SchemeSteps*i, 
        height/SchemeSteps*i + 0.2+i*(SchemeSteps-i)/(SchemeSteps**2/4)]), np.zeros(12)]),
        [0,0,0,0, 0,0,0,0]
    ),
    # lambda i:( # finish
    #     XDes,
    #     [1,125,1,150, 0,125,0,150]
    # )
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    None, #(lambda x,u: x[1], [0], [np.inf]), # lift up body
    None, #(lambda x,u: x[8], [0.5], [np.inf]), # have positive velocity
    #  #(lambda x,u: ca.vertcat(model.pFuncs["phbLeg2"](x)[1], model.pFuncs["phfLeg2"](x)[1],
    #             #  model.JacFuncs["Jbtoe"](x)@x[7:], model.JacFuncs["Jbtoe"](x)@x[7:]), 
    #             #     [0]*6, [0]*6), # feet land
    (lambda x,u: (x - XDes)[:7], [0]*7, [0]*7) # arrive at desire state
]


opt = TowrCollocationVTiming(14, 4, 4, xlim, ulim, [[-200, 200]]*4, dT0, [dT0/100, dT0])
opt.Xgen = xGenTerrianHoloCons(14, np.array(xlim), model.pFuncs.values(), lambda p: p[1],
                robustGap=[0 if p in ["phbLeg2", "phfLeg2"] else 0.02 for p in model.pFuncs.keys() ])
# opt.Xgen = xGenTerrianHoloCons(14, np.array(xlim), model.pFuncs.values(), lambda p: ca.if_else(p[0]>0.35, p[1],p[1] - 0.3))
# opt = TowrCollocationDefault(14, 4, 4, xlim, [[-100,100]]*4, [[-200, 200]]*4, dT0)
costU = opt.newhyperParam("costU")
costDDQ = opt.newhyperParam("costDDQ")
costQReg = opt.newhyperParam("costQReg")
opt.newhyperParam(distance)

opt.begin(x0=X0, u0=[1,125,1,125], F0=[0,100,0,100])

x_val = X0
# x_init = [X0]
for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    EOMF = EoMFuncs[cons]
    opt.dTgen.chMod(modName = name)
    for i in range(N):
        x_0, u_0 = R(i)
        # x_0 = caSubsti(x_0, opt.hyperParams.keys(), opt.hyperParams.values())

        initSol = solveLinearCons(caFuncSubsti(EOMF, {"x":x_0}), [("ddq", np.zeros(7), 1e3)])
        opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                x0 = x_0, u0 = initSol["u"],F0 = initSol["F"])
        # x_init.append(x_0)


        # opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        #         u_0, X0)

        opt.addCost(lambda x,u: costU*ca.dot(u[:4],u[:4]))
        opt.addCost(lambda ddq1: costDDQ * ca.dot(ddq1[-4:],ddq1[-4:]))
        opt.addCost(lambda x: costQReg * ca.dot((x - X0)[2:],(x - X0)[2:]))
        # opt.addCost(lambda x,u: 0.005*ca.dot(x[-4:],x[-4:]))

        # opt.addCost(lambda x,u: ca.dot(x - X0,x - X0))
        # addAboveGoundConstraint(opt)

        def holoCons(x,u,F):
            MU = 0.4
            return ca.vertcat(
                *[model.pFuncs[n](x)[1] +0.005 - 0.03*i*(N-i)*4/N**2 for j,n in enumerate(['phbLeg2', 'phfLeg2']) if not cons[j]],
                *[MU * F[1+i*2] + F[0+i*2] for i in range(2) if cons[i]],
                *[MU * F[1+i*2] - F[0+i*2] for i in range(2) if cons[i]],
                *[F[1+i*2] - 0 for i in range(2) if cons[i]]
            )
        opt.addConstraint(
            holoCons, [0]*(2 + sum(cons)*2), [np.inf]*(2+sum(cons)*2)
        )

        # Avoid the front leg collide with back leg
        opt.addConstraint(
            lambda x,u: x[5]+x[6], [-np.math.pi*1/2], [np.inf]
        )

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

opt.step(lambda dx,x,u : EoMFuncs[(0,0)](x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        x0 = caSubsti(XDes, opt.hyperParams.keys(), opt.hyperParams.values()), u0 = [0,0,0,0], F0=[0,0,0,0])


if __name__ == "__main__" :

    # opt.buildParseSolution("x_plot", lambda sol: sol["Xgen"]["x_plot"])
    # exit()

    # opt.cppGen("cppIpopt/generated/flatJump",parseFuncs=[
    #     ("x_plot", lambda sol: sol["Xgen"]["x_plot"]),
    #     ("u_plot", lambda sol: sol["Ugen"]["u_plot"]),
    #     ("t_plot", lambda sol: sol["dTgen"]["t_plot"]),
    #     ("terrain_plot", lambda sol: sol["Xgen"]["terrain_plot"])],
    #     cmakeOpt={'libName': 'nlpFltJmp'})
    # exit()

    import matplotlib.pyplot as plt
    with Session(__file__,terminalLog = True) as ss:
    # if True:
        opt.setHyperParamValue({"distance": 0.5, 
                                "costU":0.01,
                                "costDDQ":0.0001,
                                "costQReg":0.1})
    # if(True):
        res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 2000, # unkown option
                }
            })
        
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "ycyCollo%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":res,
                "Scheme":Scheme,
                # "x_init":x_init
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("Note","Generate CPP and Compare the Result!")