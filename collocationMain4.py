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

XDes = np.array([0.5,0.5,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
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

opt = DirectOptimizer(14, 8, xlim, [-200, 200], dT)

opt.init(X0)

for i in range(10):
    opt.step(lambda dx,x,u : model.EOMfunc(x,dx[7:],u[:4],u[4:]), # EOMfunc:  [x,ddq,u,F]=>[EOM]) 
            np.array([1,125,1,125,0,150,0,150]),X0)
    opt.addCost(lambda x,u: ca.dot(x-XDes, x-XDes)+0.01*ca.dot(u[:4],u[:4]))

    def holoCons(x,u):
        MU = 0.4
        return ca.vertcat(
            *[MU * u[5+i*2] + u[4+i*2] for i in range(2)],
            *[MU * u[5+i*2] - u[4+i*2] for i in range(2)],
            *[u[5+i*2] - 10 for i in range(2)]
        )
    opt.addConstraint(
        holoCons, [0]*6, [np.inf]*6
    )

opt.addConstraint(lambda x,u: (x - XDes)[:14], [0]*14, [0]*14)

opt.startSolve()

if __name__ == "__main__" :

    with Session(__file__,terminalLog = True) as ss:
        opt.startSolve()
        
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "ycyCollo%d.pkl"%time.time()))

        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":opt._sol,
                "sol_x":opt.getSolX(),
                "sol_u":opt.getSolU()
            }, f)

        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Note","I want to try ycycollocion algorithm")