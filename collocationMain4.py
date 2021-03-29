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

dT = 0.01
distance = 0.5

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

EoMFuncs = {
    (1,1): model.buildEOMF((1,1)),
    (1,0): model.buildEOMF((1,0)),
    (0,0): model.buildEOMF((0,0))
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
        [1,125,1,125,0,100,0,100]
    ),
    lambda i:( # lift
        Xlift0,
        [0,100,0,0,0,100,0,0]
    ),
    lambda i:( # fly
        X0 + np.concatenate([np.array([distance/50*i, 0.2+i*(50-i)/625]), np.zeros(12)]),
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
                ca.veccat(initSol["u"],initSol["F"]), x_0)

        opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                u_0, X0)

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
    if(FinalC is not None):
        opt.addConstraint(*FinalC)


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
        # DynF = model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"])
        # nextXk = None
        # lastXk = None
        # for i,(Xk, Uk) in enumerate( zip(opt.getSolX().T,opt.getSolU().T)):
        #     sol = DynF(x = Xk,u = Uk[:4])
        #     # print(EOMF(x = x_val,u = u,F=np.concatenate([sol["F0"],np.array([0,0]).reshape(sol["F0"].size())]),ddq = sol["ddq"]))
        #     eomf_check = EOMF(x = Xk,u = Uk[:4],F=Uk[4:],ddq = sol["ddq"])['EOM'].full()
        #     # print(Uk[4:].reshape(-1) - np.concatenate([sol["F0"], sol["F1"]]).reshape(-1))
        #     # print(eomf_check.reshape(-1))
        #     # # assert()
        #     # if(nextXk is not None):
        #     #     print("diff next:", Xk - nextXk)
        #     nextXk = rounge_Kutta(Xk,Uk[:4],lambda x,u : DynF(x=x,u=u)["dx"])

        #     if(lastXk is not None):
        #         print("laskXk:", lastXk)
        #         print("Xk:", Xk)
        #         clsol = opt.colloF(Xk = lastXk, Xk_puls_1 = Xk)
        #         print("collo:",clsol)

        #         if(i>-1 and not i%10):
        #             plt.figure()
        #             plotX = np.linspace(0,dT,100)
        #             ind = int(np.random.random()*7)
        #             plt.plot(plotX, (np.array(clsol["a3"])[ind,...]*plotX**3).T
        #                             +(np.array(clsol["a2"])[ind,...]*plotX**2).T
        #                             +(np.array(clsol["a1"])[ind,...]*plotX).T
        #                             +(np.array(clsol["a0"]))[ind,...].T)
        #             plt.show()
        #     lastXk = Xk

        
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
        ss.add_info("Note","U为0， Better Init value")