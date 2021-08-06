"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys
sys.path.append(".")

from optGen.trajOptimizer import *
from utils.mathUtil import normQuad, cross2d
import pickle as pkl


Nobstacle_box = 3
DT = 0.2
xDim = 6 # pos and vel
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 
uDim = 3 # acc x, acc y, acc turning. (in dog frame)
uLim = ca.DM([[-1, 1]]*uDim) 
STEPS = 10


opt =  EularCollocation(
    Xgen = xGenDefault(xDim, xLim),
    Ugen = uGenDefault(uDim, uLim),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(DT)
)

refTraj = opt.newhyperParam("refTraj", (2,STEPS))
gamma = opt.newhyperParam("gamma")
Wreference = opt.newhyperParam("Wreference")
Wacc = opt.newhyperParam("Wacc", (uDim,))
Wforward = opt.newhyperParam("Wforward")
x0 = opt.newhyperParam("x0", (xDim,))


opt.begin(x0=x0, u0=ca.DM.zeros(uDim), F0=ca.DM([]))
factor = 1

def dynF(Xk, Uk, Fk):
    px, py, pr = Xk[0], Xk[1], Xk[2]
    ux, uy, ur = Uk[0], Uk[1], Uk[2]
    s,c = ca.sin(pr), ca.cos(pr)
    return ca.vertcat(
        Xk[3:], # vel
        c*ux - s*uy,
        s*ux + c*uy,
        ur
    )

for ref in ca.horzsplit(refTraj):
    opt.step(dynF, 
        x0 = x0, u0 = ca.DM.zeros(uDim), F0=ca.DM([]))

    opt.addCost(lambda x: Wreference * factor * normQuad(x[:2]-ref))
    opt.addCost(lambda u: Wacc[0] * u[0]**2 + Wacc[1] * u[1]**2 + Wacc[2] * u[2]**2 )
    opt.addCost(lambda x: -x[3])
    factor *= gamma

if __name__ == "__main__":

    opt.cppGen("codogs/dogMPC/generated", expand=True, parseFuncs=[
        ("x_plot", lambda sol: sol["Xgen"]["x_plot"].T),
        ("u_plot", lambda sol: sol["Ugen"]["u_plot"].T)],
        cmakeOpt={'libName': 'localPlan', 'cxxflag':'"-O3 -fPIC"'})

    refTraj = np.linspace([2,0], [2, 5], STEPS).T
    opt.setHyperParamValue({
        "refTraj": refTraj,
        "gamma": 1,
        "Wreference" : 1e2,
        "Wacc" : [1,5,2],
        "Wforward" : 1,
        "x0": [0,0,0, 0,0,0]
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
    
    print("EXECTIME:", res['exec_sec'])

    # dumpname = os.path.abspath(os.path.join("./codogs/nlpSol", "dogmpc%d.pkl"%time.time()))

    # with open(dumpname, "wb") as f:
    #     pkl.dump({
    #         "sol":res,
    #         "EXECTIME": res['exec_sec']
    #     }, f)
    print(res.keys())

    print(res["Xgen"].keys())
    ######     ######     ######     #######
    ### ######     Animate      ######   ###
    ######     ######     ######     #######
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    sol_x= res['Xgen']['x_plot'].full().T
    sol_u= res['Ugen']['u_plot'].full().T
    print(sol_x)

    def animate(i):
        ind = i%len(sol_x)
        xsol = sol_x[ind]
        ax.clear()
        
        ax.plot(refTraj[0,:], refTraj[1,:], "o", label="ref")

        sth = np.sin(xsol[2])
        cth = np.cos(xsol[2])
        L,W = 2,1
        hl,hw = L/2, W/2

        box, = ax.plot(np.array([cth*hl-sth*hw, -cth*hl-sth*hw, -cth*hl+sth*hw, cth*hl+sth*hw, cth*hl-sth*hw])+xsol[0], 
                    np.array([sth*hl+cth*hw, -sth*hl+cth*hw, -sth*hl-cth*hw, sth*hl-cth*hw, sth*hl+cth*hw])+xsol[1],
                    label = "dog")
        
        ax.legend()
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)

        return box,

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, save_count=50)


    plt.show()
