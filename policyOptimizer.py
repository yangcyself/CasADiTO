from trajOptimizer import *
import numpy as np


class ConstPDPolicy(TrajOptimizer):
    """ConstPDPolicy
        calculates the input u = kp(q_ref - q) + kd(dq_ref - dq) + FF
                where kp, kd, and x_ref are variables.
    """
    def __init__(self, xDim, uDim, xLim, uLim, dt, kpd, Wkpd, Wxref, Wff):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        
        self.Kp = None
        self.Kd = None
        self._FF_plot = []
        self._xref_plot = []
        self._qDim = int(self._xDim/2)
        self._Wkpd = Wkpd
        self._Wxref = Wxref
        self._Wff = Wff
        self._kpd0 = kpd

    def init(self):
        self.Kp = ca.SX.sym('Kp')
        self.Kd = ca.SX.sym('Kd')
        self.Kpd = ca.vertcat(self.Kp, self.Kd)

        self._w.append(self.Kpd)
        self._lbw.append([0, 0])
        self._ubw.append([np.inf,np.inf])
        self._w0.append(self._kpd0)

        self._J += ca.dot((self.Kpd - self._kpd0), self._Wkpd @ (self.Kpd - self._kpd0))

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


        self._J += ca.dot(Xrefk - x_ref0, self._Wxref @ Xrefk - x_ref0)
        self._J += ca.dot(FFk[:4] - FF0[:4], self._Wff @ FFk[:4] - FF0[:4])

        return ca.veccat(u,F)

    def start(self):
        self._parseSol = ca.Function('policyParse', [ca.vertcat(*self._w)], 
                    [ca.horzcat(*self._xref_plot), ca.horzcat(*self._FF_plot), self.Kp, self.Kd], 
                    ['w'], ['xref', 'FF', 'Kp', 'Kd'])

class DefaultPolicy(TrajOptimizer):
    def __init__(self, xDim, uDim, xLim, uLim, dt):
        super().__init__(xDim, uDim, xLim, uLim, dt)

        self._U_plot = []
        self._F_plot = []
        self._qDim = int(self._xDim/2)

    def init(self):
        pass

    def step(self, step, x, **kwarg):
        """define policy variable and return the control signals

        Args:
            step (int): the step number k
            x (ca.SX): the state input of the controller

        Returns:
            ca.SX: The control signal
        """

        # the u
        U = ca.SX.sym('U%d'%step, 4)
        self._w.append(U)
        self._lbw.append([-np.inf]*4)
        self._ubw.append([np.inf]*4)
        self._w0.append(kwarg["U0"])
        self._U_plot.append(U)

        # the reaction force F
        F = ca.SX.sym('F%d'%step, 4)
        self._w.append(F)
        self._lbw.append(self._uLim[4:,0])
        self._ubw.append(self._uLim[4:,1])
        self._w0.append(kwarg["F0"])
        self._F_plot.append(F)

        return ca.veccat(U,F)

    def start(self):
        self._parseSol = ca.Function('policyParse', [ca.vertcat(*self._w)], 
                    [ca.horzcat(*self._U_plot), ca.horzcat(*self._F_plot)], 
                    ['w'], ['U', 'F'])


class DefaultTiming(TrajOptimizer):
    def __init__(self, xDim, uDim, xLim, uLim, dt):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        self.DT = dt
        self._parseSol = lambda *args, **kwargs: self.DT
        
    def init(self):
        pass

    def step(self,**kwarg):
        """
        Return a constant (not variable) dt
        """
        return self.DT

    def start(self):
        pass

class VariableTiming(TrajOptimizer):
    def __init__(self, xDim, uDim, xLim, uLim, dt, tLim):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        self.DT = dt
        self.tLim = tLim
        self._dt_plot = []
        self.curent_dt = None
        
    def init(self, **kwargs):
        T = ca.SX.sym('dT%d'%0, 1)
        self._w.append(T)
        self._lbw.append([self.tLim[0]])
        self._ubw.append([self.tLim[1]])
        self._w0.append(self.DT)
        self.curent_dt = T

    def chStat(self, **kwargs):
        """[summary] Change the inner state of the object
        Currently is just generating a new time length variable for the next
        """
        T = ca.SX.sym('dT_%s'%kwargs["name"], 1)
        self._w.append(T)
        self._lbw.append([self.tLim[0]])
        self._ubw.append([self.tLim[1]])
        self._w0.append(self.DT)
        self.curent_dt = T


    def step(self,**kwarg):
        """
        Return a constant (not variable) dt
        """
        self._dt_plot.append(self.curent_dt)
        return self.curent_dt

    def start(self):
        self._parseSol = ca.Function('timingParse', [ca.vertcat(*self._w)], 
                    [ca.vertcat(*self._dt_plot),  ca.cumsum(0,ca.vertcat(*self._dt_plot),0), ca.vertcat(*self._w)], 
                    ['w'], ['dt', 't' ,'T'])

class ycyConstPD(TrajOptimizer):
    """ycyConstPD
        The class mimics the parameterization of ycycollocation. 
            i.e. using third order polynomial fitting and towr like costraint on ddq
        However, the control input u is generated from a PD controller u = kp(q_ref - q) + kd(dq_ref - dq) + FF
                where kp, kd, and x_ref are variables.
        What's more, the forward simulation assumes a gausian noise term on the q and dq at each time step.
    """
    def __init__(self, xDim, uDim, xLim, uLim,dt, sol_x, sol_u,  kpd, Wkpd, Wxref, Wff):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        self.policy = ConstPDPolicy(xDim, uDim, xLim, uLim, dt, kpd, Wkpd, Wxref, Wff)
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

        Xk += 0.01*np.random.random(self._xDim) - 0.01*np.random.random(self._xDim)
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


class ComponentsNotSetError(Exception):
    def __init__(self, msg):
        super().__init__("Please set all Components of the collocation algorithm"+msg)


class TimingCollocation(TrajOptimizer):
    """TimingCollocation
        The collocation method follows ycyCollocation. However, the dt of each time step is a variable one.
    """
    def __init__(self, xDim, uDim, xLim, uLim,dt):
        super().__init__(xDim, uDim, xLim, uLim, dt)
        self._x_plot = []
        self._u_plot = []
        self._qDim = int(self._xDim/2)
        self._parseSol = None

        self.timingGen = None

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
    
    def getSolT(self):
        x_opt, u_opt, t_opt = self._parseSol(self._sol['x'])
        return self.timingGen(w = t_opt)
    

    def init(self, u0, x0):
        if(self.timingGen is None):
            raise ComponentsNotSetError(": timingGen")

        self.timingGen.init()

        Xk = ca.SX.sym('X0', self._xDim)
        self._w.append(Xk)
        self._lbw.append(x0)
        self._ubw.append(x0)
        self._w0.append(x0)
        self._x_plot.append(Xk)

        Uk = ca.SX.sym('U_%d'%(self._stepCount), self._uDim)
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
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
        Uk_puls_1 = ca.SX.sym('U_%d'%(self._stepCount), self._uDim)
        self._w.append(Uk_puls_1)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk_puls_1)

        Xk_puls_1 = ca.SX.sym('X_%d'%(self._stepCount+1), self._xDim)
        self._w.append(Xk_puls_1)
        self._lbw.append(self._xLim[:,0])
        self._ubw.append(self._xLim[:,1])
        self._w0.append(x0)
        self._x_plot.append(Xk_puls_1)

        dt = self.timingGen.step()

        Xk = self._lastStep["Xk"]
        Uk = self._lastStep["Uk"]

        q0 = Xk[:self._qDim]
        dq0 = Xk[self._qDim:]
        q1 = Xk_puls_1[:self._qDim]
        dq1 = Xk_puls_1[self._qDim:]
        
        # retrieve the param of a3t^3 + a2t^2 + a1t + a0 fitted by the q0 and q1
        a0 = q0
        a1 = dq0
        a2 = -(3*(q0 - q1) + dt*(2*dq0 + dq1))/(dt**2)
        a3 = (2*(q0 - q1) + dt*(dq0 + dq1))/(dt**3)

        ddq0 = 2*a2 # (6 * q1 - 2*dq1*dt - 6 * q0 - 4*dq0*dt)/(dt**2) # 6q1 - 2dq1dt - 6q0 - 4dq0dt

        g = dynF(ca.vertcat(dq0,ddq0),Xk, Uk)

        self._g.append(g)
        self._lbg.append([dynF_g_lim[0]]*g.size(1)) #size(1): the dim of axis0
        self._ubg.append([dynF_g_lim[1]]*g.size(1)) #size(1): the dim of axis0

        if(self._lastStep["ddQk"] is not None):
            self._g.append(ddq0 - self._lastStep["ddQk"])
            self._lbg.append([0]*self._qDim) 
            self._ubg.append([0]*self._qDim)

        ddq1 = 6 * a3 * dt + 2*a2
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
        w = ca.vertcat(*self._w, *self.timingGen._w)
        g = ca.vertcat(*self._g, *self.timingGen._g)
        x_plot = ca.horzcat(*self._x_plot, *self.timingGen._x_plot)
        u_plot = ca.horzcat(*self._u_plot, *self.timingGen._u_plot)
        w0 = np.concatenate(self._w0 + self.timingGen._w0)
        lbw = np.concatenate(self._lbw + self.timingGen._lbw)
        ubw = np.concatenate(self._ubw + self.timingGen._ubw)
        lbg = np.concatenate(self._lbg + self.timingGen._lbg)
        ubg = np.concatenate(self._ubg + self.timingGen._ubg)
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

        self.timingGen.start()
        self._parseSol = ca.Function('solutionParse', [w], [x_plot, u_plot, ca.vertcat(*self.timingGen._w)], ['w'], ['x', 'u', 'dTgen'])

