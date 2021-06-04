from model.singleRigidBody import singleRigidBody
from optGen.trajOptimizer import *
from optGen.helpers import pointsTerrian2D

model = singleRigidBody({"m":10, "I": 1}, nc = 4)



opt = TowrCollocationVTiming(14, 4, 4, xlim, ulim, [[-200, 200]]*4, dT0, [dT0/100, dT0])
terrian = pointsTerrian2D(terrainPoints, 0.001) 
opt.Xgen = xGenTerrianHoloCons(14, np.array(xlim), model.pFuncs.values(), lambda p: p[1] - terrian(p[0]),
                                robustGap=[0 if p in ["phbLeg2", "phfLeg2"] else 0.02 for p in model.pFuncs.keys() ])


# opt = TowrCollocationDefault(14, 4, 4, xlim, [[-100,100]]*4, [[-200, 200]]*4, dT0)
costU = opt.newhyperParam("costU")
costDDQ = opt.newhyperParam("costDDQ")
costQReg = opt.newhyperParam("costQReg")
opt.newhyperParam(terrainPoints)

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
        opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                x0 = x_0, u0 = initSol["u"],F0 = initSol["F"])
        # x_init.append(x_0)


        # opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
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

        # opt.addConstraint(
        #     lambda x:ca.vertcat(
        #             *[model.vFuncs[n](x) for j,n in enumerate(['vtoeb', 'vtoef']) if cons[j]]),
        #     [0]*sum(cons)*2, [0]*sum(cons)*2
        # )

        # Avoid the front leg collide with back leg
        opt.addConstraint(
            lambda x,u: x[5]+x[6], [-np.math.pi*1/2], [np.inf]
        )

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

opt.step(lambda dx,x,u,F: EoMFuncs[(0,0)](x=x,u=u,F=F,ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        x0 = caSubsti(XDes, opt.hyperParams.keys(), opt.hyperParams.values()), u0 = [0,0,0,0], F0=[0,0,0,0])
