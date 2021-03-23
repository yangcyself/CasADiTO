from trajOptimizer import *
import model, vis
import pickle as pkl

"""
This file use dynamic constraint as dynamics, rather than dynF
"""

d = 3
tau_roots = np.append(0, ca.collocation_points(d, 'legendre'))

# input dims: [ux4,Fbx2,Ffx2]
opt = ColloOptimizer(14, 8, [-100, 100], 0.05, tau_roots)

X0 = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([0,0.5,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

opt.init(X0)

for i in range(10):
    opt.step(lambda dx,x,u : model.EOMfunc(x,dx[7:],u[:4],u[4:]), # EOMfunc:  [x,ddq,u,F]=>[EOM]) 
            np.array([1,125,1,125,0,150,0,150]),X0)
    opt.addCost(lambda x,u: ca.dot(x-XDes, x-XDes)+0.01*ca.dot(u[:4],u[:4]))

    def holoCons(x,u):
        MU = 0.4
        return ca.vertcat(
            *[MU * u[5+i*2] + u[4+i*2] for i in range(2)],
            *[MU * u[5+i*2] - u[4+i*2] for i in range(2)]
        )
    opt.addConstraint(
        holoCons, [0]*4, [np.inf]*4
    )

opt.startSolve()

with open("collo3Sol_with_init.pkl", "wb") as f:
    pkl.dump(opt._sol, f)