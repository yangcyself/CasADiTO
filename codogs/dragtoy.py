"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")
from optGen.trajOptimizer import *

xDim = 1
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 


opt =  KKT_TO(
    Xgen = xGenDefault(xDim, xLim),
    Ugen = uGenDefault(uDim=1, uLim=ca.DM([[-ca.inf, ca.inf]])),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(0) # there is no notion of dT in this problem
)


X0 = ca.DM([0])
Xdes = ca.DM([10])
p0 = ca.DM([0.5])
r = ca.DM([1])
STEPS = 20 # When using raw dual, if 9, then will not forward, if 10, then will forward

opt.begin(x0=X0, u0=p0, F0=ca.DM([]))
opt.addConstraint(lambda u: u-p0, ca.DM([0]), ca.DM([0]))
u_last = p0
for i in range(STEPS):
    opt.step(lambda x,dx: x+dx, 
    Func0 = lambda dx: dx**2, 
    Func1 = lambda x,dx,u: (x+dx - u)**2 - 1, 
    Func2 = None, 
    x0 = Xdes*i/STEPS, u0 = ca.DM([2])*i/STEPS, F0=ca.DM([]))
    opt.addCost(lambda x: ca.norm_2(x-Xdes)**2)
    opt.addConstraint(lambda u: u-u_last, ca.DM([-0.1]), ca.DM([0.1]))
    u_last = opt._state['u']
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