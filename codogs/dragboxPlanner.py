"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")

from codogs.heavyRopeLoad import HeavyRopeLoad
from optGen.trajOptimizer import *
from mathUtil import normQuad
import pickle as pkl

NC = 3
Nobstacle_cylinder = 2
Nlinedivid = 3

xDim = 3
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 
STEPS = 15
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
            # g = ca.vertcat(*[ca.norm_2(d)**2 for d in ca.vertsplit(Uk_ - Uk,2)]) # this will have nan
            g = ca.vertcat(*[ca.dot(d,d) for d in ca.vertsplit(Uk_ - Uk,2)]) # this will not have nan
            self._g.append(g)
            self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([self.eps**2]*g.size(1)) #size(1): the dim of axis0
        self._state.update({
            "u": Uk,
        })
        return Uk

def lineCons(a,b,n, f):
    """Return the constraint on a line, the line is evenly distributed with n points

    Args:
        a (SX/MX): point
        b (SX/MX): point
        n (int): number of points on the line
        f ((SX[2])->g): The constraint, g will be constrainted to be less than zero
    """
    return ca.vertcat(*[ f(a+(i/(n-1)*(b-a)))for i in range(n)])

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
normAng = opt.newhyperParam("normAng", (NC,))
# each obstacle is represented by x,y,r
cylinderObstacles = opt.newhyperParam("cylinderObstacles", (Nobstacle_cylinder * 3, ))


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
    opt.addCost(lambda x: normQuad(x-Xdes))
    # opt.addCost(lambda u: 1e-3 * normQuad(u-u_last)) # CANNOT ADD THIS COST
    # opt.addCost(lambda ml: 1e-3 * ml**2) # CANNOT ADD THIS COST

    def tmpf(x,u):
        normDirs = [ca.vertcat(ca.cos(x[2]+normAng[0]), ca.sin(x[2]+normAng[0])),
                    ca.vertcat(ca.cos(x[2]+normAng[1]), ca.sin(x[2]+normAng[1])),
                    ca.vertcat(ca.cos(x[2]+normAng[2]), ca.sin(x[2]+normAng[2]))]
        return ca.vertcat(*[
            ca.dot(a - c.T, n)
            for a,c,n in zip(ca.vertsplit(u,2), ca.vertsplit(pfuncs(x,pc),1), normDirs)
        ])
    # constraint for rope not collide with box
    opt.addConstraint(tmpf, 
        ca.DM([0]*NC), ca.DM([ca.inf]*NC))

    opt.addConstraint(lambda dx: normQuad(dx), 
        ca.DM([-ca.inf]), ca.DM([0.3**2]))
    
    for obs in ca.vertsplit(cylinderObstacles,3):
        for i in range(NC):
            opt.addConstraint(lambda x,u: lineCons(
                pfuncs(x,pc)[i,:].T, u[2*i:2*i+2], Nlinedivid, lambda p: obs[2]**2 - normQuad(p-obs[:2])
            ), ca.DM([-ca.inf]*Nlinedivid), ca.DM([0]*Nlinedivid))
        

    # u_last = opt._state['u']
    # if(i==40 and "ml" in opt._state.keys()):
        # opt.addConstraint(lambda ml: ml, ca.DM([1e-2]), ca.DM([1e-2])) # have solution
        # opt.addConstraint(lambda ml: ml, ca.DM([1e-2]), ca.DM([10])) # do not have solution

opt.addConstraint(lambda x: normQuad((x-Xdes)[:3]), ca.DM([-ca.inf]), ca.DM([0]))

if __name__ == "__main__":

    opt.cppGen("codogs/localPlanner/generated", expand=True, parseFuncs=[
        ("x_plot", lambda sol: sol["Xgen"]["x_plot"]),
        ("u_plot", lambda sol: sol["Ugen"]["u_plot"])],
        cmakeOpt={'libName': 'localPlan', 'cxxflag':'"-O0 -fPIC"'})

    # exit()

    X0 = ca.DM([0,0,0])
    Xdes = ca.DM([1,0,0.1])
    pa0 = ca.DM([-1,0,  0,1,  0,-1])
    pc = ca.DM([-1,0, 0,1, 0,-1])
    Q = np.diag([1,1,1])
    r = ca.DM([1,1,1])
    normAng = ca.DM([ca.pi,ca.pi/2,-ca.pi/2])
    cylinderObstacles = [ # the x,y,r of obstacles
        4.5, 0, 1,
        0,0,0
    ]

    opt.setHyperParamValue({
        "X0" :X0,
        "Xdes" :Xdes,
        "pa0" :pa0,
        "pc" :pc,
        "Q" :Q,
        "r" :r,
        "normAng": normAng,
        "cylinderObstacles":cylinderObstacles
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
            "check_derivatives_for_naninf": "yes"
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
            "r" :r,
            "normAng": normAng,
            "obstacles":cylinderObstacles
        }, f)
