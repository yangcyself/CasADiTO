"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")

from codogs.heavyRopeLoad import HeavyRopeLoad
from optGen.trajOptimizer import *
import pickle as pkl

NC = 3
xDim = 3
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 
STEPS = 50
model = HeavyRopeLoad(nc = NC)

class uGenXYmove(uGenDefault):
    def __init__(self, nc, eps):        
        """
        Args:
            nc (size_t): The number of contacts
            eps (double): The maximum difference between steps
        """
        super().__init__(uDim = nc*2, uLim=ca.DM([[-ca.inf, ca.inf]]*nc*2))
        self.nc = nc
        self.eps = eps
        
    def _begin(self, **kwargs):
        self._state.update({
            "u": None,
        })
    
    def step(self, step, u0, **kwargs):
        Uk = super().step(step, u0, **kwargs)
        Uk_ = self._state["u"]
        if(Uk_ is not None):
            g = ca.vertcat(*[ca.norm_2(d)**2 for d in ca.vertsplit(Uk_ - Uk)])
            self._g.append(g)
            self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([self.eps**2]*g.size(1)) #size(1): the dim of axis0
        self._state.update({
            "u": Uk,
        })
        return Uk


opt =  KKT_TO(
    Xgen = xGenDefault(xDim, xLim),
    Ugen = uGenXYmove(nc = NC, eps = 0.2),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(0) # there is no notion of dT in this problem
)


X0 = opt.newhyperParam("X0", (xDim,))
Xdes = opt.newhyperParam("Xdes", (xDim,))
pa0 = opt.newhyperParam("pa0", (2*NC,))
pc = opt.newhyperParam("pc", (2*NC,))
Q = opt.newhyperParam("Q", (3,3))
r = opt.newhyperParam("r", (NC,))


opt.begin(x0=X0, u0=pa0, F0=ca.DM([]))
opt.addConstraint(lambda u: u-pa0, ca.DM.zeros(pa0.size()), ca.DM.zeros(pa0.size()))
u_last = pa0

pfuncs = model.pcfunc
for i in range(STEPS):
    opt.step(model.integralFunc, 
    Func0 = lambda dx: model.Jfunc(dx, Q), 
    Func1 = lambda x,dx,u: model.gfunc(x,dx,pc,u, r), 
    Func2 = None, 
    x0 = X0, u0 = pa0, F0=ca.DM([]))
    opt.addCost(lambda x: ca.norm_2(x-Xdes)**2)

    def tmpf(x,u):
        normDirs = [-ca.vertcat(ca.cos(x[2]), ca.sin(x[2])),
                    ca.vertcat(-ca.sin(x[2]), ca.cos(x[2])),
                    ca.vertcat(ca.sin(x[2]), -ca.cos(x[2])),
                    ]
        return ca.vertcat(*[
            ca.dot(a - c.T, n)
            for a,c,n in zip(ca.vertsplit(u,2), ca.vertsplit(pfuncs(x,pc),1), normDirs)
        ])
    # constraint for rope not collide with box
    opt.addConstraint(tmpf, 
        ca.DM([0]*NC), ca.DM([ca.inf]*NC))

    opt.addConstraint(lambda dx: ca.norm_2(dx)**2, 
        ca.DM([-ca.inf]), ca.DM([0.3**2]))
    
    # opt.addCost(lambda u: 1e-3 * ca.norm_2(u-u_last)**2) # CANNOT ADD THIS COST
    # opt.addCost(lambda ml: 1e-3 * ml**2) # CANNOT ADD THIS COST
    u_last = opt._state['u']
    # if(i==40 and "ml" in opt._state.keys()):
        # opt.addConstraint(lambda ml: ml, ca.DM([1e-2]), ca.DM([1e-2])) # have solution
        # opt.addConstraint(lambda ml: ml, ca.DM([1e-2]), ca.DM([10])) # do not have solution

opt.addConstraint(lambda x: ca.norm_2((x-Xdes)[:3])**2, ca.DM([-ca.inf]), ca.DM([0]))

if __name__ == "__main__":

    X0 = ca.DM([0,0,0])
    Xdes = ca.DM([2,0,3.14])
    pa0 = ca.DM([-1,0,  0,1,  0,-1])
    pc = ca.DM([-1,0, 0,1, 0,-1])
    Q = np.diag([1,1,1])
    r = ca.DM([1,1,1])

    opt.setHyperParamValue({
        "X0" :X0,
        "Xdes" :Xdes,
        "pa0" :pa0,
        "pc" :pc,
        "Q" :Q,
        "r" :r
    })

    res = opt.solve(options=
        {"calc_f" : True,
        "calc_g" : True,
        "calc_lam_x" : True,
        "calc_multipliers" : True,
        "expand" : True,
            "verbose_init":True,
            # "jac_g": gjacFunc
        "ipopt":{
            "max_iter" : 10000, # unkown option
            }
        })
    
    print(res.keys())
    print("Uplot\n" ,res["Ugen"]["u_plot"].full().T)
    print("Xplot\n" ,res["Xgen"]["x_plot"].full().T)
    print(res['ml_plot'].full().T)
    # print(res['jacL_plot'].full().T)
    # print(res['comS_plot'].full().T)
    # print(res['g'])

    dumpname = os.path.abspath(os.path.join("./codogs/nlpSol", "dragplanner%d.pkl"%time.time()))

    with open(dumpname, "wb") as f:
        pkl.dump({
            "sol":res,
            "nc": NC,
            "X0" :X0,
            "Xdes" :Xdes,
            "pa0" :pa0,
            "pc" :pc,
            "Q" :Q,
            "r" :r
        }, f)
