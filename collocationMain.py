#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
import casadi as ca
import numpy as np
import model
import matplotlib.pyplot as plt
import matplotlib

def visPoly(p):
    plt.figure()
    x = np.linspace(-0.5,1.5)
    plt.plot(x,p(x))
    plt.grid()



# Degree of interpolating polynomial
d = 3
MU = 0.4

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
print("tau_root",tau_root)
# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

    # VIS
    # visPoly(p)

# plt.show()

# Time horizon
# T = 10.
# # Control discretization
# N = 20 # number of control intervals
# h = T/N

DynFuncs = {
    (1,1): model.buildDynF([model.phbLeg2, model.phfLeg2],"all_leg", ["btoe","ftoe"]),
    (1,0): model.buildDynF([model.phbLeg2],"back_leg", ["btoe"]),
    (0,0): model.buildDynF([],"fly")
}

Scheme = [ # list: (contact constaints, length)
    ((1,1), 3, "start"),
    ((1,0), 3, "lift"),
    ((0,0), 3, "fly"),
    # ([model.phbLeg2], 3, "land"),
    ((1,1), 3, "finish")
]
h = 0.05

# # Objective term
# L = x1**2 + x2**2 + u**2

# # Continuous time dynamics
# f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])


# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

X0 = np.array([0,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])

XDes = ca.MX([2,1,0,-np.math.pi/6,-np.math.pi*2/3, -np.math.pi/6,-np.math.pi*2/3,
         0,0,0,0,    0,    0,    0])


# "Lift" initial conditions
Xk = ca.MX.sym('X0', 14)
w.append(Xk)
lbw.append(X0)
ubw.append(X0)
w0.append(X0)
x_plot.append(Xk)

# Formulate the NLP
for cons, N, name in Scheme:
    DynF = DynFuncs[cons]
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_%s_%d'%(name,k), 4)
        w.append(Uk)
        lbw.append([-100]*4)
        ubw.append([100]*4)
        w0.append(list(np.random.random(4)))
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_%s_%d_%d'%(name,k,j), 14)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf]*14)
            ubw.append([np.inf]*14)
            w0.append(list(np.random.random(14)))

        # Loop over collocation points
        Xk_end = D[0]*Xk # Xk_end is the next break point D[0]*collocation[0] + D[0]*collocation[1] ... 
        # the collocation[0] needs to be the previous Xk, so Xc only contains the collocations[1...d]
        # p[0] is the polynomial that p(0)=1, and p[j!=0](0) = 0
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            # fj, qj = f(Xc[j-1],Uk)
            dynFsol = DynF(x=Xc[j-1],u = Uk)
            q = ca.dot(Uk,Uk) * 0.0001


            g.append(h*dynFsol["dx"] - xp) #??YCY?? Why need h here?
            #    g.append(fj - xp) #ycytmp
            lbg.append([0, 0])
            ubg.append([0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1];

            # Add contribution to quadrature function
            J = J + B[j]*q*h

            ## add collocation constraints

            g += [MU * dynFsol["F%d"%i][1] + dynFsol["F%d"%i][0] for i in range(sum(cons))]
            lbg.append( [0] * sum(cons))
            ubg.append( [np.inf] * sum(cons))
            g += [MU * dynFsol["F%d"%i][1] - dynFsol["F%d"%i][0] for i in range(sum(cons))]
            lbg.append([0] * sum(cons))
            ubg.append([np.inf] * sum(cons))

            # add higher than ground constraint
            g += [pfunc(Xk)[1] for i,pfunc in enumerate(model.pFuncs.values())]
            lbg.append([0 for i in range(len(model.pFuncs))])
            ubg.append([np.inf for i in range(len(model.pFuncs))])




        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X__%s_%d'%(name,k+1), 14)
        w.append(Xk)
        lbw.append([-np.inf]*14)
        ubw.append([np.inf]*14)
        w0.append([0, 0])
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0, 0])
        ubg.append([0, 0])

J += ca.dot((Xk - XDes),(Xk - XDes))

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob);

# # Function to get x and u trajectories from w
# trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
# x_opt, u_opt = trajectories(sol['x'])
# x_opt = x_opt.full() # to numpy array
# u_opt = u_opt.full() # to numpy array

# # Plot the result
# tgrid = np.linspace(0, T, N+1)
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x_opt[0], '--')
# plt.plot(tgrid, x_opt[1], '-')
# plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
# plt.xlabel('t')
# plt.legend(['x1','x2','u'])
# plt.grid()
# plt.show()

# print(sol.keys())
# print("optimal cost",sol['f'])

# print(x_opt[0][3],u_opt[0][2])
# print(f(x_opt[0][3],u_opt[0][2]))


import pickle as pkl

with open("collocationSolution.pkl","wb") as f:
    pkl.dump(sol,f)
