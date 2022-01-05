"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""
import sys

from casadi.casadi import cos, forward, horzcat, mtimes, vertcat
from numpy import inf
sys.path.append(".")

from optGen.trajOptimizer import *
from utils.mathUtil import normQuad, cross2d
from optGen.helpers import addLinearClearanceConstraint
import pickle as pkl

optGen.VARTYPE = ca.SX

CVEL_FOWR = 0.35
CVEL_SIDE = 0.1

def getAccBound(V_M,N,T):
    # assume at time T the agent moving at full speed just cannot see the obstacle
    # Then calculate the accleration needed to ensure safety for the next loop
    aMin = V_M / (2*N*T)
    return aMin

Nobstacle_box = 3
DT = 0.1
xDim = 6 # pos and vel
xLim = ca.DM([[-ca.inf, ca.inf]]*xDim) 
uDim = 3 # acc x, acc y, acc turning. (in dog frame)

refLength = 1
STEPS = 8
NObstacles = 3
# uLim = ca.DM([getAccBound(CVEL_FOWR, STEPS, DT), getAccBound(CVEL_SIDE, STEPS, DT), [-1,1]]) 
uBound = [getAccBound(CVEL_FOWR, STEPS, DT), getAccBound(CVEL_SIDE, STEPS, DT)]
uLim = ca.DM([[-ca.inf, ca.inf], [-ca.inf, ca.inf], [-1,1]]) 

opt =  EularCollocation(
    Xgen = xGenDefault(xDim, xLim),
    Ugen = uGenDefault(uDim, uLim),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(DT)
)

if(refLength>1):
    ind0 = opt.addNewVariable("ind0", ca.DM([0]), ca.DM([refLength]), ca.DM([0]))
    opt._state.update({"ind0": ind0})
    opt._parse.update({"ind0": lambda :ind0})

x0 = opt.newhyperParam("x0", (xDim,))
refTraj = opt.newhyperParam("refTraj", (5,refLength))
dog_l = opt.newhyperParam("dog_l")
dog_w = opt.newhyperParam("dog_w")
obstacles = opt.newhyperParam("obstacles", (NObstacles * 5,))
gamma = opt.newhyperParam("gamma")
Cvel_forw = opt.newhyperParam("Cvel_forw") # forwarding velocity constraint
Cvel_side = opt.newhyperParam("Cvel_side") # side velocity constraint
Cacc_forw = opt.newhyperParam("Cacc_forw") # forwarding accleration constraint
Cacc_side = opt.newhyperParam("Cacc_side") # side accleration constraint
Wreference = opt.newhyperParam("Wreference")
Wvelref = opt.newhyperParam("Wvelref")
Wacc = opt.newhyperParam("Wacc", (uDim,))
Wrot = opt.newhyperParam("Wrot")

opt.begin(x0=x0, u0=ca.DM.zeros(uDim), F0=ca.DM([]))
factor = 1

def dynF(Xk, Uk, Fk):
    px, py, pr = Xk[0], Xk[1], Xk[2]
    ux, uy, ur = Uk[0], Uk[1], Uk[2]
    s,c = ca.sin(pr), ca.cos(pr)
    return ca.vertcat(
        Xk[3:], # vel
        c*ux - s*uy, # TODO: Change to global
        s*ux + c*uy,
        ur
    )

def boxShapeAB(bl, bw):
    box_shape_A = ca.vertcat(
            ca.horzcat(1,0),
            ca.horzcat(-1,0),
            ca.horzcat(0,1),
            ca.horzcat(0,-1),
    )
    return ca.horzcat(box_shape_A, ca.vertcat(bl/2, bl/2, bw/2, bw/2)) # box body

def rotation(r):
    s,c = ca.sin(r), ca.cos(r)
    return ca.vertcat(ca.horzcat(c, -s),
                      ca.horzcat(s,  c))

def boxObstacleAb(x,y,r,l,w):
    p = ca.vertcat(x,y)
    Ab = boxShapeAB(l, w)
    A,b = Ab[:,:2], Ab[:,2]
    T = rotation(r)
    A = ca.mtimes(A,T.T)
    o = ca.mtimes(A, p)
    b = b+o
    return A,b


obstacABs = [boxObstacleAb(a[0], a[1], a[2], a[3], a[4]) for a in ca.vertsplit(obstacles,5)]

for i in range(STEPS):
    opt.step(dynF, 
        x0 = x0, u0 = ca.DM.zeros(uDim), F0=ca.DM([]))

    # add forward and side speed constraint
    forw_speed_f = lambda x: x[3] * ca.cos(x[5]) + x[4] * ca.sin(x[5])
    side_speed_f = lambda x: - x[3] * ca.sin(x[5]) + x[4] * ca.cos(x[5])
    w0 = ca.DM.ones(1)
    w = opt.addNewVariable("slack%d"%i, 0*w0, ca.inf*w0, 0*w0)
    opt._state.update({"slack_w":w})
    opt.addConstraint(lambda x, slack_w: 1 - (forw_speed_f(x)/Cvel_forw)**2 - (side_speed_f(x)/Cvel_side)**2  + slack_w, ca.DM([0]), ca.DM([ca.inf])) 
    opt.addCost(lambda slack_w: 1e6 * slack_w)

    # add forward and side acc constraint
    opt.addConstraint( lambda u: (u[0]/Cacc_forw)**2+(u[1]/Cacc_side)**2, ca.DM([-ca.inf]), ca.DM([1]))
    ## constraint on the yaw angle of the dog
    opt.addCost(lambda x: Wreference*Wrot*((ca.cos(x[2])-ca.cos(refTraj[2]))**2 
                            +(ca.sin(x[2])-ca.sin(refTraj[2]))**2))
    opt.addConstraint(lambda x:  ca.sin((x[2]-refTraj[2])/2), 
                                    ca.DM([-0.5]), ca.DM([0.5])) 
    _x = opt._state["x"]
    dogA, dogb = boxObstacleAb(_x[0],_x[1],_x[2], dog_l, dog_w)
    for A,b in obstacABs:
        addLinearClearanceConstraint(opt, ca.vertcat(dogA, A), ca.vertcat(dogb, b), slackWeight=1e6)
    
    opt.addCost(lambda u: Wacc[0] * u[0]**2 + Wacc[1] * u[1]**2 + Wacc[2] * u[2]**2 )
    factor *= gamma

if(refLength==1): # this is the final target
    opt.addCost(lambda x: Wreference * (normQuad(x[:2]-refTraj[:2]))
                        + Wvelref * (normQuad(x[3:5]-refTraj[3:5]) ))

def run_a_loop(x0, reftraj, obslist):
    dog_l = 0.65
    dog_w = 0.35
    print("x0      ", x0)
    print("reftraj ", reftraj)
    opt.setHyperParamValue({
        "refTraj": refTraj,
        "gamma": 1,
        "Wreference" : 1e3,
        "Wvelref": 1e1,
        "Wacc" : [1e3,1e6,10],
        "Wrot" : 3e-1,
        "Cvel_forw": CVEL_FOWR,
        "Cvel_side": CVEL_SIDE,
        "Cacc_forw": uBound[0],
        "Cacc_side": uBound[1],
        "x0": x0,
        "obstacles":[i for a in obstacleList for i in a],
        "dog_l" : dog_l,
        "dog_w" : dog_w
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
            "max_iter" : 3000, # unkown option
            "check_derivatives_for_naninf": "yes",
            "print_level": 0 #5
            }
        })
    
    print("EXECTIME:", res['exec_sec'])
    sol_x= res['Xgen']['x_plot'].full().T
    sol_u= res['Ugen']['u_plot'].full().T
    return sol_x,sol_u



if __name__ == "__main__":

    opt.cppGen("codogs/dogMPC/generated", expand=True, parseFuncs=[
        ("x_plot", lambda sol: sol["Xgen"]["x_plot"].T),
        ("u_plot", lambda sol: sol["Ugen"]["u_plot"].T)],
        cmakeOpt={'libName': 'localPlan', 'cxxflag':'"-O3 -fPIC"'})

    # refTraj = np.linspace([2,-5], [2, 2], refLength).T
    dog_l = 0.65
    dog_w = 0.35
    x0 = ca.DM([0, 0, ca.pi, 0., 0, 0])
    refTraj = ca.DM([ -2, 0.1, -ca.pi/6*7, 0, 0])
    # obstacleList = [(1.5, -0.000000, .5, 1.5, 0.2),
    obstacleList = [(0,0,0,0,0),
                    (0,0,0,0,0),
                    (0,0,0,0,0)]
    for a in obstacleList:
        print(boxObstacleAb(*a))

    sol_x,sol_u = run_a_loop(x0,refTraj,obstacleList)
    x_list = [np.array(sol_x)]
    u_list = [np.array(sol_u)]
    for i in range(50):
        sol_x,sol_u = run_a_loop(x_list[-1][1], refTraj, obstacleList)
        x_list.append(sol_x)
        u_list.append(sol_u)
    sol_x = [x[0] for x in x_list]
    sol_u = [u[0] for u in u_list]
    # dumpname = os.path.abspath(os.path.join("./codogs/nlpSol", "dogmpc%d.pkl"%time.time()))

    # with open(dumpname, "wb") as f:
    #     pkl.dump({
    #         "sol":res,
    #         "EXECTIME": res['exec_sec']
    #     }, f)

    ######     ######     ######     #######
    ### ######     Animate      ######   ###
    ######     ######     ######     #######
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    # sol_x= res['Xgen']['x_plot'].full().T
    # sol_u= res['Ugen']['u_plot'].full().T
    # print(sol_x)
    print(sol_x)
    def animate(i):
        ind = i%len(sol_x)
        xsol = sol_x[ind]
        ax.clear()
        
        ax.plot(refTraj[0,:], refTraj[1,:], "o", label="ref")

        sth = np.sin(xsol[2])
        cth = np.cos(xsol[2])
        hl,hw = dog_l/2, dog_w/2

        box, = ax.plot(np.array([cth*hl-sth*hw, -cth*hl-sth*hw, -cth*hl+sth*hw, cth*hl+sth*hw, cth*hl-sth*hw])+xsol[0], 
                    np.array([sth*hl+cth*hw, -sth*hl+cth*hw, -sth*hl-cth*hw, sth*hl-cth*hw, sth*hl+cth*hw])+xsol[1],
                    label = "dog")
        
        for ii,(x, y, th, bl, bw) in enumerate(obstacleList):
            bl = bl/2
            bw = bw/2
            c = np.cos(th)
            s = np.sin(th)
            ax.plot(x+ np.array([c*bl-s*bw, -c*bl-s*bw, -c*bl+s*bw, c*bl+s*bw, c*bl-s*bw]),
                    y+ np.array([s*bl+c*bw, -s*bl+c*bw, -s*bl-c*bw, s*bl-c*bw, s*bl+c*bw]), label = "obstacle%d"%ii)

        ax.legend()
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)

        return box,

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, save_count=50)

    
    try:
        plt.figure()
        for i, x in enumerate(x_list):
            r_arr = [a[2] for a in x]
            plt.plot(np.arange(i,i+len(r_arr)), r_arr)
        plt.title("r")
    except Exception as E:
        print("MPC rotation array failed to plot!!")
        print(E)
    try:
        plt.figure()
        for i, x in enumerate(x_list):
            r_arr = [a[3] for a in x]
            plt.plot(np.arange(i,i+len(r_arr)), r_arr)
        plt.title("vx")
    except Exception as E:
        print("MPC VX failed to plot!!")
        print(E)
    try:
        plt.figure()
        for i, x in enumerate(x_list):
            r_arr = [a[4] for a in x]
            plt.plot(np.arange(i,i+len(r_arr)), r_arr)
        plt.title("vy")
    except Exception as E:
        print("MPC VY failed to plot!!")
        print(E)
    plt.show()