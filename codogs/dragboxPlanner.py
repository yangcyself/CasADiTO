"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")

from codogs.heavyRopeLoad import HeavyRopeLoad
from optGen.trajOptimizer import *
from optGen.helpers import addDisjunctivePositiveConstraint
from utils.mathUtil import normQuad, cross2d
import pickle as pkl

NC = 3
# Nobstacle_cylinder = 2
# Nobstacle_line = 4
# Nlinedivid = 3
Nobstacle_box = 0

Clineu_MIN = 0.8
RopeNormMin = 0.1

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
            g = Uk_ - Uk
            self._g.append(g)
            self._lbg.append([-self.eps]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([self.eps]*g.size(1)) #size(1): the dim of axis0
        self._state.update({
            "u": Uk,
        })
        return Uk


opt =  KKT_TO(
    Xgen = xGenDefault(xDim, xLim),
    Ugen = uGenXYmove(nc = NC, eps = 0.1),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(0), # there is no notion of dT in this problem
    # TransMethod="PF"
    TransMethod="NCP-FB"
)

X0 = opt.newhyperParam("X0", (xDim,))
Xdes = opt.newhyperParam("Xdes", (xDim,))
pa0 = opt.newhyperParam("pa0", (2*NC,))
pc = opt.newhyperParam("pc", (2*NC,))
Q = opt.newhyperParam("Q", (3,3))
r = opt.newhyperParam("r", (NC,))
normAng = opt.newhyperParam("normAng", (NC,))
boxObstacles = opt.newhyperParam("boxObstacles", (Nobstacle_box * 5, ))  # x,y,th, box_l, box_w

def makeClearanceAb(obstc):
    p = obstc[:2]
    th = obstc[2]
    box_l = obstc[3]
    box_w = obstc[4]
    A = ca.vertcat(
        ca.horzcat(ca.cos(th), ca.sin(th)),
        ca.horzcat(-ca.sin(th), ca.cos(th))
    )
    o = A@p
    AA = ca.vertcat(A,-A)
    b = ca.vertcat(box_l, box_w)/2
    bb = ca.vertcat(o+b, -o+b)
    return AA,bb


# Weights
Wboxfinal = opt.newhyperParam("Wboxfinal")
WropeNorm = opt.newhyperParam("WropeNorm")
Wboxstep = opt.newhyperParam("Wboxstep")

opt.begin(x0=X0, u0=pa0, F0=ca.DM([]))
opt.addConstraint(lambda u: u-pa0, ca.DM.zeros(pa0.size()), ca.DM.zeros(pa0.size()))
u_last = pa0

CC = opt.addNewVariable("C",ca.DM([-ca.inf]), ca.DM([ca.inf]), ca.DM([0])) # the slack variable for g
opt._state.update({"cc": CC})
opt.addCost(lambda cc: Wboxfinal * cc**2)

Clineu = opt.addNewVariable("Clineu",ca.DM([Clineu_MIN]), ca.DM([ca.inf]), ca.DM([1])) # the slack variable for g
opt._state.update({"clineu": Clineu})
opt.addCost(lambda clineu:  WropeNorm* clineu)

pfuncs = model.pcfunc
for i in range(STEPS-1):
    x_0 = X0 + (i+1)*(Xdes - X0)/STEPS
    normDirs = [ca.vertcat(ca.cos(x_0[2]+normAng[0]), ca.sin(x_0[2]+normAng[0])),
                ca.vertcat(ca.cos(x_0[2]+normAng[1]), ca.sin(x_0[2]+normAng[1])),
                ca.vertcat(ca.cos(x_0[2]+normAng[2]), ca.sin(x_0[2]+normAng[2]))]
    u_0 = ca.vertcat( *[p.T+n*rr  for p,n,rr in zip(ca.vertsplit(pfuncs(x_0,pc),1), normDirs, ca.vertsplit(r, 1))])
    opt.step(model.integralFunc, 
    Func0 = lambda dx: model.Jfunc(dx, Q), 
    Func1 = lambda x,dx,u: model.gfunc(x,dx,pc,u, r), 
    Func2 = None, 
    x0 = 0*x_0, u0 = 0*u_0, F0=ca.DM([]))
    opt.addCost(lambda x: Wboxstep * normQuad(x-Xdes))
    # opt.addCost(lambda u: 1e-1* normQuad(u-u_last)) # This Cost make dog as passive as posible, May not good for the next iter
    # opt.addCost(lambda ml: 1e-3 * ml**2) # CANNOT ADD THIS COST

    def tmpf(x,u):
        normDirs = [ca.vertcat(ca.cos(x[2]+normAng[0]), ca.sin(x[2]+normAng[0])),
                    ca.vertcat(ca.cos(x[2]+normAng[1]), ca.sin(x[2]+normAng[1])),
                    ca.vertcat(ca.cos(x[2]+normAng[2]), ca.sin(x[2]+normAng[2]))]
        return ca.vertcat(*[
            ca.dot(a - c.T, n)
            for a,c,n in zip(ca.vertsplit(u,2), ca.vertsplit(pfuncs(x,pc),1), normDirs)
        ])/r
    # constraint for rope not collide with box
    opt.addConstraint(tmpf, 
        ca.DM([RopeNormMin]*NC), ca.DM([ca.inf]*NC))

    opt.addConstraint(lambda x,u, clineu: tmpf(x,u) - clineu, 
        ca.DM([-ca.inf]*NC), ca.DM([0]*NC))

    # directly add the constraint of robustness: the diviation in dog position will not broke the rope
    # x_safe = opt.addNewVariable("x_safe", ca.DM([-ca.inf]*3), ca.DM([ca.inf]*3), x_0)
    # opt._state.update({"xsafe": x_safe})
    # opt.addConstraint(lambda xsafe, u: ca.vertcat(*[
    #     normQuad(a-c.T) - 0.8 * rr**2 # there exists a position that suits for smaller r
    #     for a,c, rr in zip(ca.vertsplit(u,2), ca.vertsplit(pfuncs(xsafe, pc),1), ca.vertsplit(r,1))
    # ]), ca.DM([-ca.inf]*3), ca.DM([0]*3) )

    ## Add convex obstacle avoidance
    for obs in ca.vertsplit(boxObstacles,5):
        A,b = makeClearanceAb(obs)
        avoidance = opt.calwithState(lambda u: ca.horzcat(*[ A@p-b for p in ca.vertsplit(u, 2)]))
        for g in ca.horzsplit(avoidance, 1):
            addDisjunctivePositiveConstraint(opt, g, 'eps%d'%opt._sc, 'disj%d'%opt._sc)
        



opt.addConstraint(lambda x, cc: (x-Xdes)-cc, ca.DM([-ca.inf]*3), ca.DM([0]*3)) # Note: Adding a slacked constraint is different to directly put it into cost
opt.addConstraint(lambda x, cc: (x-Xdes)+cc, ca.DM([0]*3), ca.DM([ca.inf]*3))  #       Because the Ipopt algorithm, using slacked constraint is better

if __name__ == "__main__":

    opt.cppGen("codogs/localPlanner/generated", expand=True, parseFuncs=[
        ("x_plot", lambda sol: sol["Xgen"]["x_plot"].T),
        ("u_plot", lambda sol: sol["Ugen"]["u_plot"].T)],
        cmakeOpt={'libName': 'localPlan', 'cxxflag':'"-O3 -fPIC"'})

    # exit()

    X0 = ca.DM([0,0,0])
    Xdes = ca.DM([0.5,0,1.2])
    pa0 = ca.DM([-1.2,0,  0,1.2,  0,-1.2])
    pc = ca.DM([-1,0, 0,1, 0,-1])
    Q = np.diag([1,1,3])
    r = ca.DM([1,1,1])
    normAng = ca.DM([ca.pi,ca.pi/2,-ca.pi/2])
    # boxObstacles = ca.DM([4,4,0,1,1, 0,0,0,0,0, 0,0,0,0,0])
    boxObstacles = ca.DM([])

    # Wboxfinal = 1e3
    # WropeNorm = 1e1
    # Wboxstep = 1e0
    Wboxfinal = 1e3
    WropeNorm = 0 # this cost improves the performance
    Wboxstep = 1e1


    opt.setHyperParamValue({
        "X0" :X0,
        "Xdes" :Xdes,
        "pa0" :pa0,
        "pc" :pc,
        "Q" :Q,
        "r" :r,
        "normAng": normAng,
        "boxObstacles":boxObstacles,
        "Wboxfinal" : Wboxfinal,
        "WropeNorm" : WropeNorm,
        "Wboxstep" : Wboxstep
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
            "max_iter" : 50000, # unkown option
            "check_derivatives_for_naninf": "yes"
            }
        })
    
    # print(res.keys())
    # print("Uplot\n" ,res["Ugen"]["u_plot"].full().T)
    # print("Xplot\n" ,res["Xgen"]["x_plot"].full().T)
    # print(res['ml_plot'].full().T)
    # print(res['jacL_plot'].full().T)
    # print(res['comS_plot'].full().T)
    # print(res['g'])
    print("EXECTIME:", res['exec_sec'])

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
            "boxObstacles":boxObstacles,
            "EXECTIME": res['exec_sec']
        }, f)
