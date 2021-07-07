import os
import time
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import casadi as ca

class BoxModel:

    def __init__(self, nc):
        """The BoxModel model is a discrete static model, whose dynamics solved by ipopt
            Assume it is a box on the groud, draged via a rope
                governed mostly by friction, thus momentum is negligible.
            state X: [px, py, r]
            trajsition: X_k+1 = f(X, ...)

            The underlying problem is:
            min         || dx ||
            s.t.    ||pc(x+dx) - pa|| < r for each rope and dog

        Args:
            pc ([[x,y]]): The position of rope connections. 
                In the load's body frame.
            r ([double]): the length of all ropes
            Q: The matrix representing the frictional effects. 
                The model min || x-x_ ||Q
        """
        self.nc = nc
        self.x = ca.SX.sym("xold", 3)
        self.Q = ca.SX.sym("Q",3,3)  # representing friction 
        self.r = ca.SX.sym("r", nc)  # rope length
        self.pc = [ca.SX.sym("pc%d", 2) for i in range(self.nc)] # contact point on box
        self.pa = [ca.SX.sym("pa%d", 2) for i in range(self.nc)] # contact point on dog
        self.pc_vec = ca.vertcat(*self.pc)
        self.pa_vec = ca.vertcat(*self.pa)

        self.dx = ca.SX.sym("dx", 3)

        self._w = [self.dx]
        
        self._J = self.dx.T @ self.Q @ self.dx
        self._g = ca.vertcat(*[self._lenCons(self.x, self.dx, pa ,pc) -self.r[i]**2
                for i,(pc,pa) in enumerate(zip(self.pc, self.pa))]) # < 0
        
        x_w = (self.T_WB(self.x)@ca.vertcat(self.dx[:2], 1))[:2] # calculate the new_x from old_x and dx
        self.newx = ca.vertcat(x_w, self.dx[2]+self.x[2])
    

    @property
    def Jfunc(self):
        return ca.Function("J", [self.dx, self.Q], [self._J], ["dx", "Q"], ["J"])

    @property
    def gfunc(self):
        return ca.Function("g", [self.x, self.dx, self.pc_vec, self.pa_vec, self.r], [self._g], 
                                ["oldx", "dx", "pc", "pa", "r"], ["g"])

    @property
    def Jfunc_jac(self):
        return ca.Function("J_jac", [self.dx, self.Q], [ca.gradient(self._J, self.dx)], ["dx", "Q"], ["J"])

    @property
    def gfunc_jac(self):
        return ca.Function("g", [self.x, self.dx, self.pc_vec, self.pa_vec, self.r], 
                                [ca.jacobian(self._g, self.dx)], 
                                ["oldx", "dx", "pc", "pa", "r"], ["g"])

    @property
    def integralFunc(self):
        """An integral function for x+dx = newx
        """
        return ca.Function("dyn", [self.x, self.dx], [self.newx], 
                                ["x", "dx"], ["newx"])

    @property
    def pcfunc(self):
        """compute the pc given current position
        """
        p = ca.horzcat(*[ (self.T_WB(self.x)@ca.vertcat(pc, 1))[:2] 
                for pc in self.pc]).T
        return ca.Function("pcFunc", [self.x, self.pc_vec], [p])

    def T_WB(self, x):
        """The transition matrix from body frame to world frame
        """
        r = x[2]
        s,c = ca.sin(r), ca.cos(r)
        return ca.vertcat(ca.horzcat(c, -s, x[0]),
                          ca.horzcat(s,  c, x[1]),
                          ca.horzcat(0,  0, 1))

    def _lenCons(self, x, dx, pa, pc):
        """Return the squared length from pa to pc in new body state

        Args:
            dx (x,y,r): the new body state in the body frame
            pa (x,y): the position of the end of the rop in world frame
            pc (x,y): the position of the contact point, in the body frame
        """
        pa_B = (ca.inv(self.T_WB(x))@ca.vertcat(pa, 1))[:2]
        pc_B = (self.T_WB(dx)@ca.vertcat(pc, 1))[:2]
        return ca.dot(pa_B - pc_B, pa_B - pc_B)

boxModel = BoxModel(3)
pc_w_func = boxModel.pcfunc
class LocalPlanner:
    def __init__(self):
        self.NC = 3         # number of dogs
        self.Nobstacle_box = 3
        self.STEPS = 15
        self.xDim = 3       # x,y,r

        self.normAng = ca.DM([ca.pi,ca.pi/2,-ca.pi/2]) # the norm direction of each dog
        self.pc = ca.DM([-1,0, 0,1, 0,-1]) # the contact point on box
        self.r = ca.DM([1,1,1]) # the rope length
        self.Q = np.diag([1,1,3])
    
    def reset(self):
        self.opti = ca.Opti()
        self.opti.solver("ipopt", {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
                "verbose_init":True,
            "ipopt":{
                "max_iter" : 50000, # unkown option
                "check_derivatives_for_naninf": "yes"
                }})
        self.Xtraj = self.opti.variable(self.xDim, self.STEPS) # box trajectory, each point x,y,r
        self.dXtraj = self.opti.variable(self.xDim, self.STEPS)
        self.Ptraj = self.opti.variable(2*self.NC, self.STEPS) # dog trajectory, each dog x,y

    def solve(self, xinit, xdes, pa0):
        self.reset()
        # set first and last state constraint
        totalCost = 0
        self.opti.subject_to(self.Xtraj[:,0] == xinit)
        self.opti.subject_to(self.Ptraj[:,0] == pa0)
        slack_xdes = self.opti.variable()
        totalCost += 1e3 * slack_xdes
        self.opti.subject_to(slack_xdes >= 0)
        self.opti.subject_to( ca.dot((self.Xtraj[:,-1] - xdes), (self.Xtraj[:,-1] - xdes))-slack_xdes  <= 0)
        self.opti.subject_to( ca.dot((self.Xtraj[:,-1] - xdes), (self.Xtraj[:,-1] - xdes))+slack_xdes >= 0)
        # set constraint for each step
        for i in range(self.STEPS):
            x = self.Xtraj[:,i]
            dx = self.dXtraj[:,i]
            p = self.Ptraj[:,i]

            # Limit the diff of dog position
            eps_p = 0.1
            if(i>0):
                self.opti.subject_to((p - self.Ptraj[:,i-1])-eps_p <= 0)
                self.opti.subject_to((p - self.Ptraj[:,i-1])+eps_p >= 0)

            ####
            # Add the constraint that dx is the optimal of optimization defined in boxModel
            intF = boxModel.integralFunc # x0,dx0 -> x1
            f0 = boxModel.Jfunc(dx, self.Q)
            f1 = boxModel.gfunc(x, dx, self.pc, p, self.r)
            f0_jac = boxModel.Jfunc_jac(dx, self.Q)
            f1_jac = boxModel.gfunc_jac(x, dx, self.pc, p, self.r)
            lam = self.opti.variable(f1.size(1)) # Add lagrange multiplier

            self.opti.subject_to(f1<=0) # Add primal feasibility
            self.opti.subject_to(lam>=0) # Add dual feasibility

            jacL = f0_jac + f1_jac.T @ lam # stationary constraint
            self.opti.subject_to(jacL==0)

            # Add complementary slackness
            # reference: https://kilthub.cmu.edu/articles/thesis/Advances_in_Newton-based_Barrier_Methods_for_Nonlinear_Programming)
            tmpf = lambda y,z: y+z-ca.sqrt(y**2 + z**2 + 1e-7) 
            self.opti.subject_to( tmpf(lam, -f1) == 0)


            # Add integral constraint: x,dx -> x_
            if(i<self.STEPS-1):
                self.opti.subject_to( intF(x, dx) - self.Xtraj[:, i+1] == 0)

            # Add constraint that dog is at the norm side of box
            normDirs = [ca.vertcat(ca.cos(x[2]+self.normAng[0]), ca.sin(x[2]+self.normAng[0])),
                    ca.vertcat(ca.cos(x[2]+self.normAng[1]), ca.sin(x[2]+self.normAng[1])),
                    ca.vertcat(ca.cos(x[2]+self.normAng[2]), ca.sin(x[2]+self.normAng[2]))]
            for pp,c,n in zip(ca.vertsplit(p,2), ca.vertsplit(pc_w_func(x,self.pc),1), normDirs):
                self.opti.subject_to( 0 <= ca.dot(pp - c.T, n))
            
            # Add costs
            totalCost += ca.dot(x-xdes, x-xdes)

        self.opti.minimize(totalCost)
        start_time = time.time()
        sol = self.opti.solve()
        exec_seconds = time.time() - start_time
        print(sol.value(self.Xtraj).T)
        print(sol.value(self.Ptraj).T)
        print("time consumption: ", exec_seconds)

        return sol

if __name__ == "__main__":
    nlp = LocalPlanner()
    sol = nlp.solve(
        xinit = ca.DM([0,0,0]),
        xdes = ca.DM([0.5, 0.2, 0.1]),
        pa0 = ca.DM([-1.2,0,  0,1.2,  0,-1.2])
    )

    sol_x = sol.value(nlp.Xtraj).T
    sol_p = sol.value(nlp.Ptraj).T
    
    ######     ######     ######     #######
    ### ######     Animate      ######   ###
    ######     ######     ######     #######
    fig, ax = plt.subplots()
    def animate(i):
        ind = i%len(sol_x)
        xsol = sol_x[ind]
        psol = sol_p[ind]
        ax.clear()
        
        sth = np.sin(xsol[2])
        cth = np.cos(xsol[2])
        L,W = 2,2
        hl,hw = L/2, W/2

        box, = ax.plot(np.array([cth*hl-sth*hw, -cth*hl-sth*hw, -cth*hl+sth*hw, cth*hl+sth*hw, cth*hl-sth*hw])+xsol[0], 
                    np.array([sth*hl+cth*hw, -sth*hl+cth*hw, -sth*hl-cth*hw, sth*hl-cth*hw, sth*hl+cth*hw])+xsol[1])
        pcs = pc_w_func(xsol, nlp.pc)
        lines=[
            ax.plot([pcs[i,0], psol[2*i]],[pcs[i,1], psol[2*i+1]], label="rope%d"%i)
            for i in range(nlp.NC)
        ]

        ax.legend()
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)

        return (box,*[l[0] for l in lines])

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, save_count=50)

    fig = plt.figure()

    plt.plot(np.array([
        [np.linalg.norm([pc_w_func(x_, nlp.pc)[i,0] - u[2*i], pc_w_func(x_, nlp.pc)[i,1] - u[2*i+1]  ]) for i in range(nlp.NC)]
        for x_,u in zip(sol_x[1:], sol_p) # Note: the length cons enforces on u0 and x1
    ]) )
    plt.legend( ["rope%d"%i for i in range(nlp.NC)])

    plt.figure()
    plt.plot(sol_x, ".")

    plt.show()
