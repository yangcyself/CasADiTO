import numpy as np
import casadi as ca


class TrajOptimizer:
    def __init__(self, xDim, uDim, uLim, dt):
        self._xDim = xDim
        self._uDim = uDim
        uLim = np.array(uLim)
        self._uLim = uLim if uLim.ndim == 2 else np.tile(uLim,(uDim,1))
        assert(self._uLim.shape == (self._uDim,2))
        self._w = []
        self._w0 = []
        self._lbw = []
        self._ubw = []
        self._J = 0
        self._g=[]
        self._lbg = []
        self._ubg = []
        self._sol = None
        self._stepCount = 0
        self._lastStep = {}
        self._dt = dt
    
    def getSolU(self):
        raise NotImplementedError

    def getSolX(self):
        raise NotImplementedError
    
    def startSolve(self):
        raise NotImplementedError
        
    def init(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
    
    # Add constriant of the state of last step
    def addConstraint(self):
        raise NotImplementedError
    
    # Add constriant of the state of last step
    def addCost(self):
        raise NotImplementedError

    def loadSol(self, sol):
        self._sol = sol

class HaveNotRunOptimizerError(Exception):
    def __init__(self):
        super().__init__("optimization must be runned before this")


class ColloOptimizer(TrajOptimizer):
    def __init__(self, xDim, uDim, uLim, dt, colloRoots):
        super().__init__(xDim, uDim, uLim, dt)
        self._x_plot = []
        self._u_plot = []
        self._parseSol = None

        self._tau_root = colloRoots
        self._d = len(colloRoots) - 1
        self._C = np.zeros((self._d+1,self._d+1))
        # Coefficients of the continuity equation
        self._D = np.zeros(self._d+1)

        # Coefficients of the quadrature function
        self._B = np.zeros(self._d+1)
        # Construct polynomial basis
        for j in range(self._d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self._d+1):
                if r != j:
                    p *= np.poly1d([1, -self._tau_root[r]]) / (self._tau_root[j]-self._tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            self._D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(self._d+1):
                self._C[j,r] = pder(self._tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            self._B[j] = pint(1.0)

    def getSolU(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        
        x_opt, u_opt = self._parseSol(self._sol['x'])
        u_opt = u_opt.full() # to numpy array
        return u_opt

    def getSolX(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        
        x_opt, u_opt = self._parseSol(self._sol['x'])
        x_opt = x_opt.full() # to numpy array
        return x_opt
    
            
    def init(self, x0):
        Xk = ca.MX.sym('X0', self._xDim)
        self._w.append(Xk)
        self._lbw.append(x0)
        self._ubw.append(x0)
        self._w0.append(x0)
        self._x_plot.append(Xk)
        self._lastStep = {
            "Xk": Xk,
        }
        
    """
    dynF: (dx, x, u) -> g
    """
    def step(self,dynF,u0,x0):
        Uk = ca.MX.sym('U_%d'%(self._stepCount), self._uDim)
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk)

        Xc = []
        for j in range(self._d):
            Xkj = ca.MX.sym('X_%d_%d'%(self._stepCount,j), self._xDim)
            Xc.append(Xkj)
            self._w.append(Xkj)
            self._lbw.append([-np.inf]*self._xDim)
            self._ubw.append([np.inf]*self._xDim)
            self._w0.append(x0)
        
        Xk = self._lastStep["Xk"]

        # Loop over collocation points
        Xk_end = self._D[0]*Xk # Xk_end is the next break point D[0]*collocation[0] + D[0]*collocation[1] ... 
        # the collocation[0] needs to be the previous Xk, so Xc only contains the collocations[1...d]
        # p[0] is the polynomial that p(0)=1, and p[j!=0](0) = 0
        for j in range(1,self._d+1):
            # Expression for the state derivative at the collocation point
            xp = self._C[0,j]*Xk
            for r in range(self._d): xp = xp + self._C[r+1,j]*Xc[r]

            # Append collocation equations
            # fj, qj = f(Xc[j-1],Uk)
            g = dynF(xp/self._dt, Xc[j-1], Uk)
            # dynFsol = dynF(x=Xc[j-1],u = Uk)

            self._g.append(g)
            self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([0]*g.size(1)) #size(1): the dim of axis0

            # Add contribution to the end state
            Xk_end = Xk_end + self._D[j]*Xc[j-1];


        self._stepCount += 1
        self._lastStep = {
            "Uk":Uk,
            "Xk": Xk_end,
            "Xc": Xc
        }

    # Add constriant of the state of last step
    # func: x,u -> g
    def addConstraint(self, func, lb, ub):
        for c in self._lastStep["Xc"]:
            self._g.append(func(c, self._lastStep["Uk"]))
            self._lbg.append(lb)
            self._ubg.append(ub)
    
    # Add constriant of the state of last step
    def addCost(self,func):
        self._J += func(self._lastStep["Xk"], self._lastStep["Uk"])

        
    def startSolve(self, solver = 'ipopt'):
        w = ca.vertcat(*self._w)
        g = ca.vertcat(*self._g)
        x_plot = ca.horzcat(*self._x_plot)
        u_plot = ca.horzcat(*self._u_plot)
        w0 = np.concatenate(self._w0)
        lbw = np.concatenate(self._lbw)
        ubw = np.concatenate(self._ubw)
        lbg = np.concatenate(self._lbg)
        ubg = np.concatenate(self._ubg)
        # Create an NLP solver
        prob = {'f':self._J, 'x': w, 'g': g}
        print("begin setting up solver")
        solver = ca.nlpsol('solver', solver, prob)

        print("Finished setting up solver")

        self._sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        self._parseSol = ca.Function('solutionParse', [w], [x_plot, u_plot], ['w'], ['x', 'u'])



class DirectOptimizer(TrajOptimizer):
    def __init__(self, xDim, uDim, uLim, dt):
        super().__init__(xDim, uDim, uLim, dt)
        self._x_plot = []
        self._u_plot = []
        self._parseSol = None

    def getSolU(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        if(self._parseSol is None):
            self._parseSol = ca.Function('solutionParse', [ca.vertcat(*self._w)],
                                 [ca.horzcat(*self._x_plot), ca.horzcat(*self._u_plot)],
                                  ['w'], ['x', 'u'])

        x_opt, u_opt = self._parseSol(self._sol['x'])
        u_opt = u_opt.full() # to numpy array
        return u_opt

    def getSolX(self):        
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        if(self._parseSol is None):
            self._parseSol = ca.Function('solutionParse', [ca.vertcat(*self._w)],
                                 [ca.horzcat(*self._x_plot), ca.horzcat(*self._u_plot)],
                                  ['w'], ['x', 'u'])

        x_opt, u_opt = self._parseSol(self._sol['x'])
        x_opt = x_opt.full() # to numpy array
        return x_opt
    
            
    def init(self, x0):
        Xk = ca.MX.sym('X0', self._xDim)
        self._w.append(Xk)
        self._lbw.append(x0)
        self._ubw.append(x0)
        self._w0.append(x0)
        self._x_plot.append(Xk)
        self._lastStep = {
            "Xk": Xk,
        }
        
    """
    dynF: (x, u) -> dx
    intF: (x, u, dynF) -> x
    """
    def step(self, dynF, intF, u0,x0):
        Uk = ca.MX.sym('U_%d'%(self._stepCount), self._uDim)
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk)


        Xk = self._lastStep["Xk"]

        Xnew = intF(Xk, Uk, dynF)

        Xk_puls_1 = ca.MX.sym('X_%d'%(self._stepCount), self._xDim)
        self._w.append(Xk_puls_1)
        self._lbw.append([-np.inf] * self._xDim)
        self._ubw.append([+np.inf] * self._xDim)
        self._w0.append(x0)
        self._x_plot.append(Xk_puls_1)

        self._g.append(Xk_puls_1 - Xnew)
        self._lbg.append([0]*self._xDim) #size(1): the dim of axis0
        self._ubg.append([0]*self._xDim) #size(1): the dim of axis0


        # Loop over collocation points
        self._stepCount += 1
        self._lastStep = {
            "Uk":Uk,
            "Xk": Xk_puls_1,
        }

    # Add constriant of the state of last step
    # func: x,u -> g
    def addConstraint(self, func, lb, ub):
        Xk = self._lastStep["Xk"]
        self._g.append(func(Xk, self._lastStep["Uk"]))
        self._lbg.append(lb)
        self._ubg.append(ub)
    
    # Add constriant of the state of last step
    def addCost(self,func):
        self._J += func(self._lastStep["Xk"], self._lastStep["Uk"])

    def startSolve(self, solver = 'ipopt'):
        w = ca.vertcat(*self._w)
        g = ca.vertcat(*self._g)
        x_plot = ca.horzcat(*self._x_plot)
        u_plot = ca.horzcat(*self._u_plot)
        w0 = np.concatenate(self._w0)
        lbw = np.concatenate(self._lbw)
        ubw = np.concatenate(self._ubw)
        lbg = np.concatenate(self._lbg)
        ubg = np.concatenate(self._ubg)

        # ## hand in the jacobian of constraint
        gjac = ca.simplify(ca.jacobian(g,w)).sparsity() 
        # passing this sparsity makes it useful
        p = ca.MX.sym("p")
        gjacFunc = ca.Function("gjacFunc", [w,p], [g, gjac])
        # print(" generated the sparse jacobian")

        # Create an NLP solver
        prob = {'f':self._J, 'x': w, 'g': g}
        print("begin setting up solver")
        solver = ca.nlpsol('solver', solver, prob, 
        {"calc_f" : True,
         "calc_g" : True,
         "calc_lam_x" : True,
         "calc_multipliers" : True,
        # "expand" : True,
            "verbose_init":True,
            # "max_iter" : 3, # unkown option
            # "jac_g": gjacFunc
             })

        print("Finished setting up solver")

        self._sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        self._parseSol = ca.Function('solutionParse', [w], [x_plot, u_plot], ['w'], ['x', 'u'])
