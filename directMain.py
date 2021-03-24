from trajOptimizer import *
import model, vis
import pickle as pkl

"""
use direct scription method, use ronge kuta
"""

DynFuncs = {
    (1,1): model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"]),
    (1,0): model.buildDynF([model.phbLeg2],"back_leg", ["btoe"]),
    (0,0): model.buildDynF([],"fly")
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 30, "start"),
    ((1,0), 30, "lift"),
    ((0,0), 30, "fly"),
    # ([model.phbLeg2], 3, "land"),
    ((1,1), 30, "finish")
]

# input dims: [ux4,Fbx2,Ffx2]
opt = DirectOptimizer(14, 4, [-100, 100], 0.005)

X0 = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([1.5,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

def rounge_Kutta(x,u,dynF):
    DT = 0.005
    k1 = dynF(x, u)
    k2 = dynF(x + DT/2 * k1, u)
    k3 = dynF(x + DT/2 * k2, u)
    k4 = dynF(x + DT * k3, u)
    x=x+DT/6*(k1 +2*k2 +2*k3 +k4)
    return x


opt.init(X0)

for cons, N, name in Scheme:
    dynF = DynFuncs[cons]
    for i in range(N):
        opt.step(lambda x,u : dynF(x=x,u=u)["dx"],
                rounge_Kutta,
                np.array([1,125,1,125]),X0)
        opt.addCost(lambda x,u: 0.01 * ca.dot(x-XDes, x-XDes)+0.0001*ca.dot(u,u))

        
        for pfunc in model.pFuncs.values():
            opt.addConstraint(
                lambda x,u : pfunc(x)[1], [0], [np.inf]
            )


        if(sum(cons) == 0):
            continue

        def holoCons(x,u):
            MU = 0.4
            dynFsol = dynF(x = x, u = u)
            return ca.vertcat(
                *[MU * dynFsol["F%d"%i][1] + dynFsol["F%d"%i][0] for i in range(sum(cons))],
                *[MU * dynFsol["F%d"%i][1] - dynFsol["F%d"%i][0] for i in range(sum(cons))]
            )
        opt.addConstraint(
            holoCons, [0]*(sum(cons))*2, [np.inf]*(sum(cons))*2
        )

opt.addConstraint(lambda x,u: x-XDes, [0]*14, [0]*14)


if __name__ == "__main__" :

    opt.startSolve()

    with open("directSol_with_init_large.pkl", "wb") as f:
        pkl.dump(opt._sol, f)

    with open("directSol_with_init_x_u_large.pkl", "wb") as f:
        pkl.dump((opt.getSolX(), opt.getSolU()), f)