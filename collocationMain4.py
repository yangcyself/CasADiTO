from trajOptimizer import *
import model, vis
import pickle as pkl


from ExperimentSecretary.Core import Session
import os
import time


"""
This file use dynamic constraint as dynamics, rather than dynF
"""

# input dims: [ux4,Fbx2,Ffx2]

dT = 0.01

X0 = np.array([0,0.25,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([1, 0.25 ,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
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
    # # ([model.phbLeg2], 3, "land"),
    ((1,1), 50, "finish")
]

Xlift0 = X0.copy()
Xlift0[2] = np.math.pi/6

References = [
    lambda i:( # start
        X0,
        [1,125,1,150,0,125,0,150]
    ),
    lambda i:( # lift
        Xlift0,
        [100,200,0,0,100,200,0,0]
    ),
    lambda i:( # fly
        X0 + np.concatenate([np.array([0.5/50*i, 2*0.5/50*i*(0.5-0.5/50*i)]), np.zeros(12)]),
        [0,0,0,0, 0,0,0,0]
    ),
    lambda i:( # finish
        XDes,
        [1,125,1,150, 0,125,0,150]
    )
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    (lambda x,u: x[2], [0], [np.inf]), # lift up body
    (lambda x,u: ca.vertcat(x[7],x[8]), [0.5]*2, [np.inf]*2), # have positive velocity
    (lambda x,u: ca.vertcat(model.pFuncs["phbLeg2"](x)[1], model.pFuncs["phfLeg2"](x)[1]), 
                    [0]*2, [0]*2), # feet land
    (lambda x,u: (x - XDes)[:7], [0]*7, [0]*7) # arrive at desire state
]

opt = ycyCollocation(14, 8, xlim, [-100, 100], dT)

opt.init(X0)

for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    EOMF = EoMFuncs[cons]
    for i in range(N):
        x_0, u_0 = R(i)
        opt.step(lambda dx,x,u : EOMF(x=x,u=u[:4],F=u[4:],ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                np.array(u_0),x_0)
        opt.addCost(lambda x,u: 0.01*ca.dot(u[:4],u[:4]))

        def holoCons(x,u):
            MU = 0.4
            return ca.vertcat(
                *[MU * u[5+i*2] + u[4+i*2] for i in range(2) if cons[i]],
                *[MU * u[5+i*2] - u[4+i*2] for i in range(2) if cons[i]],
                *[u[5+i*2] - 10 for i in range(2) if cons[i]]
            )
        opt.addConstraint(
            holoCons, [0]*(sum(cons))*3, [np.inf]*(sum(cons))*3
        )

    opt.addConstraint(*FinalC)

if __name__ == "__main__" :

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

        # print("x",opt.getSolX())
        # print("u", opt.getSolU())
        # print(opt.dynChecker(x0 = opt.getSolX()[:,0], x1 = opt.getSolX()[:,1],
        #   u = opt.getSolU()[:,0]))
        # print(opt._dt,"VS",dT)
        # q0 = opt.getSolX()[:,0][:int(opt._xDim/2)]
        # dq0 = opt.getSolX()[:,0][int(opt._xDim/2):]
        # q1 = opt.getSolX()[:,1][:int(opt._xDim/2)]
        # dq1 = opt.getSolX()[:,1][int(opt._xDim/2):] 
        # ddq0 = (6 * q1 - 2*dq1*opt._dt - 6 * q0 - 4*dq0*opt._dt)/(opt._dt**2) # 6q1 - 2dq1dt - 6q0 - 4dq0dt
        # print("ddq0:",ddq0)
        # u_opt = opt.getSolU().T
        # x_opt = opt.getSolX().T

        # ddq = (6 * x_opt[1][:7] - 2*x_opt[1][7:]*dT 
        #     - 6 * x_opt[0][:7] - 4*x_opt[0][7:]*dT)/(dT**2)
        # print("ddq",ddq)

        # print("EoMFunc",EoMFuncs[(1,1)](x_opt[0], u_opt[0][:4],u_opt[0][4:],ddq))
        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("Note","I want to try ycycollocion algorithm")