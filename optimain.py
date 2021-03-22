from casadi import *
import model, vis
import pickle as pkl

T = 10. # Time horizon
N = 20 # number of control intervals

MU = 0.4


def Runge_kutta_builder(dynF,dT):
# # Formulate discrete time dynamics
# if False:
#    # CVODES from the SUNDIALS suite
#    dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
#    opts = {'tf':T/N}
#    F = integrator('F', 'cvodes', dae, opts)
# else:
   # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = dT/M
    x = MX.sym("x",14)
    u = MX.sym("u",4)
    xdot = dynF(x = x, u = u)["dx"]
    L = dot(u, u)*0.001

    f = Function('f', [x, u], [xdot, L])
    X0 = MX.sym('X0', 14)
    U = MX.sym('U',4)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])
    return F

#endif 

# # Evaluate at a test point
# Fk = F(x0=[0.2,0.3],p=0.4)
# print(Fk['xf'])
# print(Fk['qf'])

DynFuncs = {
    (1,1): model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"]),
    (1,0): model.buildDynF([model.phbLeg2],"back_leg", ["btoe"]),
    (0,0): model.buildDynF([],"fly")
}

Fs = {k:Runge_kutta_builder(f, 0.01) for k,f in DynFuncs.items()}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 3, "start"),
    ((1,0), 3, "lift"),
    ((0,0), 3, "fly"),
    # ([model.phbLeg2], 3, "land"),
    ((1,1), 3, "finish")
]


# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# Formulate the NLP
# Xk = MX([0, 1])
Xinit = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])
Xk = MX(Xinit)

XDes = MX([2,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

if __name__ == "__main__":

    for cons, N, name in Scheme:
        DynF = DynFuncs[cons]
        F = Fs[cons]

        if(cons[0]):
            g += [model.pFuncs["phbLeg2"](Xk)[1] ]
            lbg += [0] 
            ubg += [0]
        if(cons[1]):
            g += [model.pFuncs["phfLeg2"](Xk)[1]]
            lbg += [0] 
            ubg += [0]
        
        for k in range(N):
            # New NLP variable for the control
            Uk = MX.sym('U_%s_%d'%(name,k),4)
            w += [Uk]
            lbw += [-100]*4
            ubw += [100]*4
            # w0 += [0]*4
            w0 += list(np.random.random(4))

            # add friction cone constaint
            dynFk = DynF(x = Xk, u = Uk)
            g += [MU * dynFk["F%d"%i][1] + dynFk["F%d"%i][0] for i in range(sum(cons))]
            lbg += [0] * sum(cons)
            ubg += [inf] * sum(cons)
            g += [MU * dynFk["F%d"%i][1] - dynFk["F%d"%i][0] for i in range(sum(cons))]
            lbg += [0] * sum(cons)
            ubg += [inf] * sum(cons)

            # add higher than ground constraint
            g += [pfunc(Xk)[1] for i,pfunc in enumerate(model.pFuncs.values())
                            if not(i+k)%6]
            lbg += [0 for i in range(len(model.pFuncs)) if not (i+k)%6]
            ubg += [inf for i in range(len(model.pFuncs)) if not (i+k)%6]

            # Integrate till the end of the interval
            Fk = F(x0=Xk, p=Uk)
            Xk = Fk['xf']
            J=J+Fk['qf']

            # Add inequality constraint
            # g += [Xk[0]]
            # lbg += [-.25]
            # ubg += [inf]

    J += dot((Xk - XDes),(Xk - XDes))

    print("problem defined")

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob);

    print("solve the NLP")

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']



    with open("solution.pkl","wb") as f:
        pkl.dump(sol,f)

    # # Plot the solution
    # u_opt = w_opt
    # x_opt = [[0, 1]]
    # for k in range(N):
    #     Fk = F(x0=x_opt[-1], p=u_opt[k])
    #     x_opt += [Fk['xf'].full()]
    # x1_opt = [r[0] for r in x_opt]
    # x2_opt = [r[1] for r in x_opt]

    # tgrid = [T/N*k for k in range(N+1)]
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.clf()
    # plt.plot(tgrid, x1_opt, '--')
    # plt.plot(tgrid, x2_opt, '-')
    # plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
    # plt.xlabel('t')
    # plt.legend(['x1','x2','u'])
    # plt.grid()
    # plt.show()
