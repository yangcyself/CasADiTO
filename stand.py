from optGen.trajOptimizer import TowrCollocationDefault
from model.leggedRobot2D import LeggedRobot2D
from vis import saveSolution
import casadi as ca
import numpy as np
m = LeggedRobot2D.fromYaml("data/robotConfigs/JYminiLitev2.yaml")
dT0 = 0.01
legLength = m.params["legL2"]
opt = TowrCollocationDefault(2*m.dim, m.u_dim, m.F_dim, xLim = ca.DM([[-ca.inf, ca.inf]]*2*m.dim),
    uLim= ca.DM([[-100, 100]]*m.u_dim), FLim = ca.DM([[-ca.inf, ca.inf]]*m.F_dim), dt= dT0)
x0 = ca.DM([0, legLength,0,-np.math.pi*5/6,np.math.pi*2/3, -np.math.pi*5/6,np.math.pi*2/3,
          0,0,0,0,    0,    0,    0])
opt.begin(x0=x0, u0=[0]*4, F0=[0]*4)

EOMF = m.buildEOMF([1,1,1,1])

def desx0(i):
    x = x0[:]
    x[:2] += 0.3*np.cos(i/2/ca.pi*0.2), 0.3*np.sin(i/2/ca.pi*0.2)
    return x

print("EOMF built")
for i in range(100):
    opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[m.dim:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
            x0 = x0, u0=[0]*4, F0=[0]*4)
    opt.addCost(lambda x: ca.dot(x-desx0(i), x-desx0(i)))
    # opt.addCost(lambda x: 1e3*ca.dot(x[3:6], x[3:6]))
print("Opt Set")
res = opt.solve(options=
    {"calc_f" : True,
    "calc_g" : True,
    "calc_lam_x" : True,
    "calc_multipliers" : True,
    "expand" : True, 
        "verbose_init":True,
        # "jac_g": gjacFunc
    "ipopt":{
        "max_iter" : 2000, # unkown option
        }
    })


print(res["Xgen"]["x_plot"])
print(res["Ugen"]["u_plot"])
print(res["Fgen"]["F_plot"])
sol_x = res["Xgen"]["x_plot"].full().T
sol_u = res["Ugen"]["u_plot"].full().T
timeStamps = res["dTgen"]["t_plot"]
saveSolution("out.csv", sol_x, sol_u, timeStamps.full().reshape(-1), transform=True)