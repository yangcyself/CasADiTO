"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")

from codogs.heavyRopeLoad import HeavyRopeLoad
from optGen.trajOptimizer import *

xDim = 3
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 
model = HeavyRopeLoad(nc = 1)

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
    Ugen = uGenXYmove(nc = 1, eps = 0.1),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(0) # there is no notion of dT in this problem
)


X0 = ca.DM([0,0,0])
Xdes = ca.DM([1,0,0])
pa0 = ca.DM([2.5,0])
pc_input = ca.DM([1,0])
Q = np.diag([1,1,1])
r = ca.DM([2])
STEPS = 10

opt.begin(x0=X0, u0=pa0, F0=ca.DM([]))
opt.addConstraint(lambda u: u-pa0, ca.DM([0,0]), ca.DM([0,0]))
for i in range(STEPS):
    opt.step(model.integralFunc, 
    Func0 = lambda dx: model.Jfunc(dx, Q), 
    Func1 = lambda x,dx,u: model.gfunc(x,dx,pc_input,u, r), 
    Func2 = None, 
    x0 = Xdes*i/STEPS, u0 = pa0+ca.DM([0.5,0])*i/STEPS, F0=ca.DM([]))
    opt.addCost(lambda x: ca.norm_2(x-Xdes)**2)
# opt.addConstraint(lambda x: ca.norm_2((x-Xdes)[:2])**2, ca.DM([-ca.inf]), ca.DM([0]))

if __name__ == "__main__":

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
    
    print(res.keys())
    print(res['_w'])
    print("Uplot\n" ,res["Ugen"]["u_plot"].full().T)
    print("Xplot\n" ,res["Xgen"]["x_plot"].full().T)
    print(res['ml_plot'].full().T)
    # print(res['jacL_plot'].full().T)
    # print(res['comS_plot'].full().T)
    # print(res['g'])