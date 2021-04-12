from trajOptimizer import *
import numpy as np


class ConstPDPolicy(TrajOptimizer):
    """ConstPDPolicy
        calculates the input u = kp(q_ref - q) + kd(dq_ref - dq) + FF
                where kp, kd, and x_ref are variables.
    """
    def __init__(self, xDim, uDim, xLim, uLim, dt):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        
        self.Kp = None
        self.Kd = None
        self._FF_plot = []
        self._xref_plot = []
        self._qDim = int(self._xDim/2)

    def init(self):
        self.Kp = ca.SX.sym('Kp')
        self.Kd = ca.SX.sym('Kd')
        self.Kpd = ca.vertcat(self.Kp, self.Kd)

        self._w.append(self.Kpd)
        self._lbw.append([0, 0])
        self._ubw.append([np.inf,np.inf])
        self._w0.append([10,1])

        self._J += ca.dot(self.Kpd, self.Kpd)

    def step(self, step, x, x_ref0, FF0):
        """define policy variable and return the control signals

        Args:
            step (int): the step number k
            x (ca.SX): the state input of the controller
            x_ref0 (narray/ca.SX): the init value of the variable: x_ref
            FF0 (narray/ca.SX): the init value of the feedforward

        Returns:
            ca.SX: The control signal
        """
        Xrefk = ca.SX.sym('Xref%d'%step, self._xDim)
        self._w.append(Xrefk)
        self._lbw.append(self._xLim[:,0])
        self._ubw.append(self._xLim[:,1])
        self._w0.append(x_ref0)
        self._xref_plot.append(Xrefk)


        # the u
        FFk = ca.SX.sym('FF%d'%step, 4)
        self._w.append(FFk)
        self._lbw.append([-np.inf]*4)
        self._ubw.append([np.inf]*4)
        self._w0.append(FF0[:4])
        self._FF_plot.append(FFk)

        u =  self.Kp * (Xrefk[3:self._qDim] - x[3:self._qDim]) + \
                self.Kd * (Xrefk[self._qDim+3:] - x[self._qDim+3:]) + FFk
        
        # the reaction force F
        F = ca.SX.sym('F%d'%step, 4)
        self._w.append(F)
        self._lbw.append(self._uLim[4:,0])
        self._ubw.append(self._uLim[4:,1])
        self._w0.append(FF0[4:])

        return ca.veccat(u,F)

    def start(self):
        self._parseSol = ca.Function('policyParse', [ca.vertcat(*self._w)], 
                    [ca.horzcat(*self._xref_plot), ca.horzcat(*self._FF_plot), self.Kp, self.Kd], 
                    ['w'], ['xref', 'FF', 'Kp', 'Kd'])


class ycyConstPD(TrajOptimizer):
    """ycyConstPD
        The class mimics the parameterization of ycycollocation. 
            i.e. using third order polynomial fitting and towr like costraint on ddq
        However, the control input u is generated from a PD controller u = kp(q_ref - q) + kd(dq_ref - dq) + FF
                where kp, kd, and x_ref are variables.
        What's more, the forward simulation assumes a gausian noise term on the q and dq at each time step.
    """
    def __init__(self, xDim, uDim, xLim, uLim,dt, sol_x, sol_u):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        self.policy = ConstPDPolicy(xDim, uDim, xLim, uLim, dt)
        self._x_plot = []
        self._u_plot = []
        self._qDim = int(self._xDim/2)
        self._parseSol = None
        self._dUlim = (np.max(uLim)**2 * uDim)*self._dt*10
        self.SolXRef = sol_x
        self.SolFFRef = sol_u


    def getSolX(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        if(self._parseSol is None):
            self._parseSol = ca.Function('solutionParse', [ca.vertcat(*self._w)], 
                                [ca.horzcat(*self._x_plot), ca.horzcat(*self._u_plot), ca.horzcat(*self.policy._w)], 
                                ['w'], ['x', 'u', 'policy'])
        x_opt, u_opt, policy = self._parseSol(self._sol['x'])
        x_opt = x_opt.full() # to numpy array
        return x_opt
    
    def getSolU(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        if(self._parseSol is None):
            self._parseSol = ca.Function('solutionParse', [ca.vertcat(*self._w)], 
                                [ca.horzcat(*self._x_plot), ca.horzcat(*self._u_plot), ca.horzcat(*self.policy._w)], 
                                ['w'], ['x', 'u', 'policy'])

        x_opt, u_opt,policy = self._parseSol(self._sol['x'])
        u_opt = u_opt.full() # to numpy array
        return u_opt
    
    def getSolPolicy(self):
        if(self._sol is None):
            raise HaveNotRunOptimizerError
        if(self._parseSol is None):
            self._parseSol = ca.Function('solutionParse', [ca.vertcat(*self._w)], 
                                [ca.horzcat(*self._x_plot), ca.horzcat(*self._u_plot), ca.horzcat(*self.policy._w)], 
                                ['w'], ['x', 'u', 'policy'])
        x_opt, u_opt, p_opt = self._parseSol(self._sol['x'])
        p_opt = p_opt.full() # to numpy array
        return self.policy._parseSol(w = p_opt)

    
    def init(self, u0, x0):
        Xk = ca.SX.sym('X0', self._xDim)
        self._w.append(Xk)
        self._lbw.append(x0)
        self._ubw.append(x0)
        self._w0.append(x0)
        self._x_plot.append(Xk)

        self.policy.init()
        Uk = self.policy.step(0, Xk, self.SolXRef[0], self.SolFFRef[0])

        self._g.append(Uk)
        self._lbg.append(self._uLim[:,0])
        self._ubg.append(self._uLim[:,1])
        self._u_plot.append(Uk)

        self._lastStep = {
            "Xk": Xk,
            "ddQk":None,
            "Uk" : Uk
        }
        
    """
    dynF: (dx, x, u) -> g
    dynF_g_lim: (lbg, ubg)
    """
    def step(self,dynF,u0,x0, dynF_g_lim = (0,0)):

        Xk_puls_1 = ca.SX.sym('X_%d'%(self._stepCount+1), self._xDim)
        self._w.append(Xk_puls_1)
        self._lbw.append(self._xLim[:,0])
        self._ubw.append(self._xLim[:,1])
        self._w0.append(x0)
        self._x_plot.append(Xk_puls_1)

        Uk_puls_1 = self.policy.step(0, Xk_puls_1, self.SolXRef[self._stepCount+1], self.SolFFRef[self._stepCount+1])
        self._g.append(Uk_puls_1)
        self._lbg.append(self._uLim[:,0])
        self._ubg.append(self._uLim[:,1])
        self._u_plot.append(Uk_puls_1)

        Xk = self._lastStep["Xk"]
        Uk = self._lastStep["Uk"]

        q0 = Xk[:self._qDim]
        dq0 = Xk[self._qDim:]
        q1 = Xk_puls_1[:self._qDim]
        dq1 = Xk_puls_1[self._qDim:]
        
        # retrieve the param of a3t^3 + a2t^2 + a1t + a0 fitted by the q0 and q1
        a0 = q0
        a1 = dq0
        a2 = -(3*(q0 - q1) + self._dt*(2*dq0 + dq1))/(self._dt**2)
        a3 = (2*(q0 - q1) + self._dt*(dq0 + dq1))/(self._dt**3)

        ddq0 = 2*a2 # (6 * q1 - 2*dq1*self._dt - 6 * q0 - 4*dq0*self._dt)/(self._dt**2) # 6q1 - 2dq1dt - 6q0 - 4dq0dt

        Xk += 0.001*np.random.random(self._xDim) - 0.001*np.random.random(self._xDim)
        g = dynF(ca.vertcat(dq0,ddq0),Xk, Uk) 

        self._g.append(g)
        self._lbg.append([dynF_g_lim[0]]*g.size(1)) #size(1): the dim of axis0
        self._ubg.append([dynF_g_lim[1]]*g.size(1)) #size(1): the dim of axis0

        if(self._lastStep["ddQk"] is not None):
            self._g.append(ddq0 - self._lastStep["ddQk"])
            self._lbg.append([0]*self._qDim) 
            self._ubg.append([0]*self._qDim)

        ddq1 = 6 * a3 * self._dt + 2*a2
        self._stepCount += 1
        self._lastStep = {
            "Uk": Uk_puls_1, # assume the U is first order spline
            "Xk": Xk_puls_1,
            "ddQk":ddq1
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
        w = ca.vertcat(*self._w, *self.policy._w)
        g = ca.vertcat(*self._g, *self.policy._g)
        x_plot = ca.horzcat(*self._x_plot)
        u_plot = ca.horzcat(*self._u_plot)
        p_plot = ca.veccat(*self.policy._w)

        w0 = np.concatenate([*self._w0, *self.policy._w0])
        lbw = np.concatenate([*self._lbw, *self.policy._lbw])
        ubw = np.concatenate([*self._ubw, *self.policy._ubw])
        lbg = np.concatenate([*self._lbg, *self.policy._lbg])
        ubg = np.concatenate([*self._ubg, *self.policy._ubg])
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
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 10000, # unkown option
                }
            })
        print("Finished setting up solver")

        self._sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        self._parseSol = ca.Function('solutionParse', [w], [x_plot, u_plot, p_plot], ['w'], ['x', 'u', 'policy'])
        self.policy.start()


