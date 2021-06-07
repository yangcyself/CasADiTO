"""
Tricky tuning of the problem:
0a7695b7ea59a44e1bae2bb21990ac9df23cee13
    1. The dT should not be that short, otherwise the jac of dynamic constraint has too little gradient on u
    2. to move along x, set u0's force to 10,0,0 yeilds the solution, 0,0,0 cannot, 0,0,10 nether
"""

from model.singleRigidBody import singleRigidBody
from optGen.trajOptSRB import *
from optGen.helpers import pointsTerrian2D
import time
import pickle as pkl

model = singleRigidBody({"m":10, "I": 1}, nc = 4)

dT0 = 0.01
terrain = lambda p: p[2]
terrain_norm = lambda p: ca.DM([0,0,1])
distance = 1

X0 = ca.DM([0, 0, 0.2, 0, 0, 0,      0, 0, 0,  0, 0, 0])
XDes = ca.DM([distance, 0, 0.2, 0, 0, 0,      0, 0, 0,  0, 0, 0])

lx = 0.2
ly = 0.1
u0 = ca.DM([
     lx,  ly, 0,  10, 0, 0,
     lx, -ly, 0,  10, 0, 0,
    -lx,  ly, 0,  10, 0, 0,
    -lx, -ly, 0,  10, 0, 0,
])

optGen.VARTYPE = ca.SX

SchemeSteps = 20
Scheme = [ # list: (contact constaints, length)
    ((1,1,1,1), SchemeSteps, "start"),
    ((1,0,1,0), SchemeSteps, "step_l0"),
    ((0,1,0,1), SchemeSteps, "step_r0"),
    ((1,0,1,0), SchemeSteps, "step_l1"),
    ((0,1,0,1), SchemeSteps, "step_r1"),
    ((1,0,1,0), SchemeSteps, "step_l2"),
    ((0,1,0,1), SchemeSteps, "step_r2"),
    ((1,1,1,1), SchemeSteps, "stop")
]

References = [
    lambda i:( # start
        X0,
        u0    
    ),
    lambda i:( # step_l0
        X0+ca.DM([i*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([i*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # step_r0
        X0+ca.DM([(SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([(SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # step_l1
        X0+ca.DM([(2*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([(2*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # step_r1
        X0+ca.DM([(3*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([(3*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # step_l2
        X0+ca.DM([(4*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([(4*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # step_r2
        X0+ca.DM([(5*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0,      0, 0, 0,  0, 0, 0]), 
        u0+ca.vertcat(*[ca.DM([(5*SchemeSteps + i)*distance/(6*SchemeSteps), 0, 0, 0, 0, 0])]*4)
    ),
    lambda i:( # stop
        XDes,
        u0    
    )
]

stateFinalCons = [ # the constraints to enforce at the end of each state
    None, # start 
    None, # step_l0 
    None, # step_r0 
    None, # step_l1 
    None, # step_r1 
    None, # step_l2 
    None, # step_r2 
    (lambda x,u: (x - XDes)[:6], [0]*6, [0]*6) # arrive at desire state
]

opt = SRBoptDefault(12, [[-ca.inf, ca.inf]]*12 , 4, dT0, terrain, terrain_norm, 0.4)

opt.begin(x0=X0, u0=u0, F0=[])
DYNF = model.Dyn()

x_val = X0
# x_init = [X0]
for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    opt.dTgen.chMod(modName = name)
    opt.Ugen.chMod(modName = name, contactMap=cons)
    for i in range(N):
        x_0, u_0 = R(i)
        # x_0 = caSubsti(x_0, opt.hyperParams.keys(), opt.hyperParams.values())

        # initSol = solveLinearCons(caFuncSubsti(EOMF, {"x":x_0}), [("ddq", np.zeros(7), 1e3)])
        # opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[7:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
        #         x0 = x_0, u0 = initSol["u"],F0 = initSol["F"])
        # # x_init.append(x_0)


        opt.step(lambda x,u,F : DYNF(x,u),
                x_0, u_0, ca.DM([]))

        # opt.addCost(lambda u: 1*   ca.dot(u-u_0,u-u_0))
        # opt.addCost(lambda x: 0.0001 * ca.dot(x - x_0, x - x_0))
        opt.addCost(lambda x: 10 * ca.dot((x - x_0)[3:6], (x - x_0)[3:6])) # regularize on the orientation of x


        # leg to body distance
        opt.addConstraint(
            lambda x,u: ca.vertcat(*[ca.dot(u[6*i:6*i+3]-x[:3], u[6*i:6*i+3]-x[:3]) for i in range(4)]), 
                    [0.05**2]*4, [0.4**2] * 4
        )

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

# jac_g = ca.jacobian(ca.vertcat(*opt._g), opt.w)
# opt._parse.update({"g_jac": lambda: jac_g})


if __name__=="__main__":
    res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 1000, # unkown option
                }
            })
    dumpname = os.path.abspath(os.path.join("./data/nlpSol", "SRB%d.pkl"%time.time()))
    with open(dumpname, "wb") as f:
        pkl.dump({
            "sol":res,
            "Scheme":Scheme,
            # "x_init":x_init
        }, f)
