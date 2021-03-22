from trajOptimizer import *
import model, vis
import pickle as pkl

d = 3
tau_roots = np.append(0, ca.collocation_points(d, 'legendre'))

opt = ColloOptimizer(14, 4, [-100, 100], 0.05, tau_roots)

dynF = model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg")

X0 = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = np.array([0,0.5,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

opt.init(X0)

for i in range(30):
    opt.step(dynF, np.zeros(4),X0)
    opt.addCost(lambda x,u: ca.dot(x-XDes, x-XDes)+0.01*ca.dot(u,u))

    def holoCons(x,u):
        MU = 0.4
        dynFsol = dynF(x = x, u = u)
        return ca.vertcat(
            *[MU * dynFsol["F%d"%i][1] + dynFsol["F%d"%i][0] for i in range(2)],
            *[MU * dynFsol["F%d"%i][1] - dynFsol["F%d"%i][0] for i in range(2)]
        )
    opt.addConstraint(
        holoCons, [0]*4, [np.inf]*4
    )

opt.startSolve()

with open("collo2Sol.pkl", "wb") as f:
    pkl.dump(opt._sol, f)