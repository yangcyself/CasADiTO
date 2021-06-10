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
from mathUtil import solveLinearCons
from optGen.util import dict_ca2np, caSubsti, caFuncSubsti, substiSX2MX
from ExperimentSecretary.Core import Session


model = singleRigidBody({"m":10, "I": 1}, nc = 4)

dT0 = 0.01
terrain = lambda p: p[2]
terrain_norm = lambda p: ca.DM([0,0,1])
distance = 1

X0 = ca.DM([0, 0, 0.2, 0, 0, 0,      0, 0, 0,  0, 0, 0])
XDes = ca.DM([distance, 0, 0.2, 0, 0, 0,      0, 0, 0,  0, 0, 0])

lx = 0.2
ly = 0.1
pc_B_norm = np.array([
    [lx,  ly, -0.2],
    [lx, -ly, -0.2],
    [-lx,  ly, -0.2],
    [-lx, -ly, -0.2]
])
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
    ((1,0,0,1), SchemeSteps, "step_l0"),
    ((0,1,1,0), SchemeSteps, "step_r0"),
    ((1,0,0,1), SchemeSteps, "step_l1"),
    ((0,1,1,0), SchemeSteps, "step_r1"),
    ((1,0,0,1), SchemeSteps, "step_l2"),
    ((0,1,1,0), SchemeSteps, "step_r2"),
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
    # (lambda x,u: (x - XDes)[:12], [0]*12, [0]*12) # arrive at desire state
    (lambda x,u: (x - XDes)[:6], [0]*6, [0]*6) # arrive at desire state
]

opt = SRBoptDefault(12, [[-ca.inf, ca.inf]]*12 , 4, dT0, terrain, terrain_norm, 0.4)
costPcNorm = opt.newhyperParam("costPcNorm")
costOri = opt.newhyperParam("costOri")

# this chMod is needed for uGenSRB2
opt.begin(x0=X0, u0=u0, F0=[],  contactMap=Scheme[0][0])
DYNF = model.Dyn()
EOMF = model.EOM_ufdx()

x_val = X0
# x_init = [X0]
for (cons, N, name),R,FinalC in zip(Scheme,References,stateFinalCons):
    opt.dTgen.chMod(modName = name)
    opt.Ugen.chMod(modName = name, contactMap=cons, pc0 = [ t[3:6] for t in ca.vertsplit( R(N/2)[1], 6)])
    for i in range(N):
        x_0, u_0 = R(i)
        # x_0 = caSubsti(x_0, opt.hyperParams.keys(), opt.hyperParams.values())

        initcons = {"x":x_0}
        initcons.update({
            "pc%d"%i: u_0[6*i:6*i+3] for i in range(4) 
        })
        initcons.update({
            "fc%d"%i: ca.DM([0,0,0]) for i,c in enumerate(cons) if not c
        })
        consDyn_ = caFuncSubsti(EOMF, initcons)
        initSol = solveLinearCons(consDyn_, [("dx", np.zeros(12), 1e3)])
        for i,c in enumerate(cons):
            if c: u_0[6*i+3:6*i+6] = ca.DM(initSol['fc%d'%i])

        opt.step(lambda x,u,F : DYNF(x,u),
                x_0, u_0, ca.DM([]))

        # adding this constraint increases the solving time, from 600 iter to 1100 iter
        # opt.addCost(lambda u: 0.1*   ca.norm_2(ca.vertcat(*[t[3:6] for t,c in zip(ca.vertsplit(u,6), cons) if c]) )**2)
        # add this constraint increase the solving time
        # opt.addCost(lambda x: 0.01 * ca.dot((x - x_0)[6:], (x - x_0)[6:])) # regularize on the velocity of x

        # adding this constraint reduces the solving time from 600 iter to 345 iter
        opt.addCost(lambda x,u: costPcNorm *   ca.norm_2(ca.vertcat(*[t[0:3]-x[:3] -B
                    for t,c,B in zip(ca.vertsplit(u,6), cons, pc_B_norm) if c]) )**2 )
        # adding this constraint prevents stuck at local infisible
        opt.addCost(lambda x: costOri * ca.dot((x - x_0)[3:6], (x - x_0)[3:6])) # regularize on the orientation of x


        # leg to body distance: this constraint improves the solution time
        opt.addConstraint(
            lambda x,u: ca.vertcat(*[ca.dot(u[6*i:6*i+3]-x[:3], u[6*i:6*i+3]-x[:3]) for i,c in enumerate(cons) if c]), 
                    [0.05**2]*np.sum(cons), [0.4**2] * np.sum(cons)
        )

    if(FinalC is not None):
        opt.addConstraint(*FinalC)

jac_g = ca.jacobian(ca.vertcat(*opt._g), opt.w)
opt._parse.update({"g_jac": lambda: jac_g})


if __name__=="__main__":
    with Session(__file__,terminalLog = False) as ss:
        opt.setHyperParamValue({"costPcNorm": 0.1, 
                                "costOri":10})
        res = opt.solve(options=
                {"calc_f" : True,
                "calc_g" : True,
                "calc_lam_x" : True,
                "calc_multipliers" : True,
                "expand" : True,
                    "verbose_init":True,
                    # "jac_g": gjacFunc
                "ipopt":{
                    "max_iter"  : 4000,
                    "mu_init"   : 1e-1, # default 1e-1 # single 1: 1236 iter Solved To Acceptable Level, single 1e-2: larger slack & memory
                    # "warm_start_init_point" : "yes", # default no: single yes 1349 iter
                    # "bound_frac": 0.4999, # default 0.01, set to 0.4999 needs 1307 iter
                    # "start_with_resto": "yes", # default no
                    # "mumps_mem_percent":2000, # default 1000
                    # "mumps_pivtol": 1e-4, #default 1e-6
                    "alpha_for_y": [ # 只有primal效果最好
                                    "primal",                   # 0 use primal step size
                                    "bound-mult",               # 1 use step size for the bound multipliers (good for LPs)
                                    "min",                      # 2 use the min of primal and bound multipliers
                                    "max",                      # 3 use the max of primal and bound multipliers
                                    "full",                     # 4 take a full step of size one
                                    "min-dual-infeas",          # 5 choose step size minimizing new dual infeasibility
                                    "safer-min-dual-infeas",    # 6 like "min_dual_infeas", but safeguarded by "min" and "max"
                                    "primal-and-full",          # 7 use the primal step size, and full step if delta_x <= alpha_for_y_tol
                                    "dual-and-full",            # 8 use the dual step size, and full step if delta_x <= alpha_for_y_tol
                                    "acceptor"][0]              # 9 Call LSAcceptor to get step size for y
                    }
                })
        dumpname = os.path.abspath(os.path.join("./data/nlpSol", "SRB%d.pkl"%time.time()))
        with open(dumpname, "wb") as f:
            pkl.dump({
                "sol":dict_ca2np(res),
                "Scheme":Scheme,
                # "x_init":x_init
            }, f,protocol=2)
        ss.add_info("solutionPkl",dumpname)
        ss.add_info("Scheme",Scheme)
        ss.add_info("sol_sec",res['exec_sec'])
        ss.add_info("Note","Robot Walk six steps")