import os
import time

from optGen.trajOptimizer import *
from optGen.helpers import pointsTerrian2D
from optGen.util import caSubsti, caFuncSubsti, substiSX2MX
# from model.leggedRobotX_bak import LeggedRobotX
from model.leggedRobotX import LeggedRobotX
from mathUtil import solveLinearCons
import pickle as pkl

from ExperimentSecretary.Core import Session


model = LeggedRobotX.fromYaml("data/robotConfigs/JYminiLitev2.yaml")
# input dims: [ux4,Fbx2,Ffx2]
dT0 = 0.01
initHeight = (model.params["legL2"] + model.params["legL1"])/2 # assume 30 angle of legs
distance = ca.SX.sym("distance",1)
fleg_local_l = model.pLocalFuncs['pl2']
fleg_local_r = model.pLocalFuncs['pr2']
fleg_l = model.pLocalFuncs['pl2']
fleg_r = model.pLocalFuncs['pr2']

X0 = np.array([0, initHeight,0,  
                0, -np.math.pi*5/6,np.math.pi*2/3, 
                0, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0, 0,0,0, 0,0,0])


XDes = np.array([distance, initHeight , 2*np.math.pi,  
                0, -np.math.pi*5/6,np.math.pi*2/3,  
                0, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0, 0,0,0, 0,0,0])

local_x_0 = fleg_local_l(X0)[0]
contact_l = fleg_l(X0)
contact_r = fleg_r(X0)
# assert(local_x_0 == fleg_local_r(X0)[0])

SchemeSteps = 50
height = 0

# XDes = substiSX2MX(XDes, [distance], [distance_mx])

xlim = [
    [-np.inf,np.inf],
    [0,np.inf],
    [-np.inf, np.inf],
    model.params["q0Lim"],
    model.params["q1Lim"],
    model.params["q2Lim"],
    model.params["q0Lim"],
    model.params["q1Lim"],
    model.params["q2Lim"],
    [-100,100],
    [-100,100],
    [-100,100],
    model.params["dq0Lim"],
    model.params["dq1Lim"],
    model.params["dq2Lim"],
    model.params["dq0Lim"],
    model.params["dq1Lim"],
    model.params["dq2Lim"]
]


ulim = [
    np.array([-1, 1])*model.params["tau0lim"],
    np.array([-1, 1])*model.params["tau1lim"],
    np.array([-1, 1])*model.params["tau2lim"],
    np.array([-1, 1])*model.params["tau0lim"],
    np.array([-1, 1])*model.params["tau1lim"],
    np.array([-1, 1])*model.params["tau2lim"]
]

EoMFuncs = {
    (1,1): model.buildEOMF((1,1)),
    (1,0): model.buildEOMF((1,0)),
    (0,1): model.buildEOMF((0,1)),
    (0,0): model.buildEOMF((0,0))
}

Scheme = [ # list: (contact constaints, length)
    # ((0,1), SchemeSteps, "step"),
    ((1,1), SchemeSteps, "start"),
    ((1,0), SchemeSteps, "lift"),
    ((0,0), SchemeSteps, "fly"),
    # ([model.phbLeg2], 3, "land"),
    # ((1,1), SchemeSteps, "finish")
]


def rotaref(i,start,end):
    ret = np.zeros(18)
    ret[2] = start + (end - start) * i/SchemeSteps
    ret[11] = (end - start) / (dT0 * SchemeSteps)
    return ret

def heightref(i):
    ret = np.zeros(18)
    ret[1] = 0.3 * i*(SchemeSteps-i)/(SchemeSteps**2/4)
    return ret

# X0_copy = X0.copy()
# X0_copy[3] = model.params["q0Lim"][1]
# X0_copy[6] = model.params["q0Lim"][0]
# X0_copy[0] += 0.05
References = [
    # lambda i: X0 ,
    lambda i: X0, # start
    lambda i: X0 + rotaref(i, 0, ca.pi/8), # lift
    lambda i: X0 + rotaref(i, ca.pi/8, 2*ca.pi) + heightref(i) # fly       
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    # None,
    None, #(lambda x,u: x[1], [0], [np.inf]), # lift up body
    None, #(lambda x,u: x[8], [0.5], [np.inf]), # have positive velocity
    #  #(lambda x,u: ca.vertcat(model.pFuncs["phbLeg2"](x)[1], model.pFuncs["phfLeg2"](x)[1],
    #             #  model.JacFuncs["Jbtoe"](x)@x[7:], model.JacFuncs["Jbtoe"](x)@x[7:]), 
    #             #     [0]*6, [0]*6), # feet land
    (lambda x,u: (x - XDes)[1:9], [0]*8, [0]*8) # arrive at desire state
]


# opt = TowrCollocationDefault(18, 6, 4, xlim, ulim, [[-200, 200]]*4, dT0)
opt = TowrCollocationVTiming(18, 6, 4, xlim, ulim, [[-200, 200]]*4, dT0, [dT0/100, dT0])
opt.Xgen = xGenTerrianHoloCons(18, np.array(xlim), model.pFuncs.values(), lambda p: p[1],
                robustGap=[0 if p in ["pl2", "pr2"] else 0.02 for p in model.pFuncs.keys() ])

costU = opt.newhyperParam("costU")
costDDQ = opt.newhyperParam("costDDQ")
costQReg = opt.newhyperParam("costQReg")
opt.newhyperParam(distance)

opt.begin(x0=X0, u0=[0,1,125,0,1,125], F0=[0,100,0,100])

x_val = X0
x_init = [X0]
for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    EOMF = EoMFuncs[cons]
    opt.dTgen.chMod(modName = name)
    for i in range(N):
        x_0 = R(i)
        # x_0 = caSubsti(x_0, opt.hyperParams.keys(), opt.hyperParams.values())

        initSol = solveLinearCons(caFuncSubsti(EOMF, {"x":x_0}), [("ddq", np.zeros(9), 1e3)])
        opt.step(lambda dx,x,u : EOMF(x=x,u=u[:6],F=u[6:],ddq = dx[9:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                x0 = x_0, u0 = initSol["u"],F0 = initSol["F"])
        x_init.append(x_0)

        opt.addCost(lambda x,u: costU*ca.dot(u,u)) # these two lines need to be commented out
        opt.addCost(lambda ddq1: costDDQ * ca.dot(ddq1[-6:],ddq1[-6:])) # these two lines need to be commented out
        opt.addCost(lambda x: costQReg * ca.dot((x - X0)[4:],(x - X0)[4:]))
        # opt.addCost(lambda x,u: 0.005*ca.dot(x[-4:],x[-4:]))

        # opt.addCost(lambda x: ca.dot(x[0],x[0]))
        # addAboveGoundConstraint(opt)

        def holoCons(x,u,F):
            MU = 0.4
            return ca.vertcat(
                # *[model.pFuncs[n](x)[1] +0.005 - 0.03*i*(N-i)*4/N**2 for j,n in enumerate(['pl2', 'pr2']) if not cons[j]],
                *[MU * F[1+i*2] + F[0+i*2] for i in range(2) if cons[i]],
                *[MU * F[1+i*2] - F[0+i*2] for i in range(2) if cons[i]],
                *[F[1+i*2] - 0 for i in range(2) if cons[i]]
            )
        opt.addConstraint(
            # holoCons, [0]*(2 + sum(cons)*2), [np.inf]*(2+sum(cons)*2)
            holoCons, [0]*sum(cons)*3, [np.inf]*sum(cons)*3
        )

        opt.addConstraint(
            lambda x: ca.vertcat(fleg_local_l(x)[0]-local_x_0, fleg_local_r(x)[0]-local_x_0), [0,0], [0,0]
        )
        if(cons[0]):
            opt.addConstraint(
                lambda x: fleg_l(x)-contact_l, [0,0], [0,0]
            )
        if(cons[1]):
            opt.addConstraint(
                lambda x: fleg_r(x)-contact_r, [0,0], [0,0]
            )

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

opt.step(lambda dx,x,u : EoMFuncs[(0,0)](x=x,u=u[:6],F=u[6:],ddq = dx[9:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        x0 = X0, u0 = [0,0,0,0,0,0], F0=[0,0,0,0])



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
        opt.setHyperParamValue({"distance": 0, 
                                "costU":0.01,
                                "costDDQ":0.0001,
                                "costQReg":0.1})
    # if(True):
        x_init = opt.substHyperParam(ca.horzcat(*x_init))

        res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 20000, 
                "check_derivatives_for_naninf": "yes"
                }
            })
        
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "sideFlip%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":res,
                "Scheme":Scheme,
                "x_init":x_init
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
