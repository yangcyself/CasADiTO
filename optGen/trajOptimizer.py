from optGen.optGen import *
from optGen.util import substiSX2MX
class Collocation(optGen):
    """The traj optimization builder using towr like integration algorithm

    The collocation method assumes the x is second order. x = [q,dq]
    Then ddq0 can be represented as 6q1 - 2dq1dt - 6q0 - 4dq0dt, 
        assuming three order function aq^3 + bq^2 + cq + d = 0
    The collocation constraint adds on dynamic f(q,dq,ddq) 
        and smooth ddq: ddq_{k}(T_k) = ddq_{k+1}(T_k)
    
    This class have 4 kids(variable factories): 
        - Xgen: state (might contain terrian holonomic constraints)
        - Ugen: input signal (might involve parameterizable control variable)
        - Fgen: external force (might contain friction cone information)
        - dTgen: step length
    """

    def __init__(self, Xgen, Ugen, Fgen, dTgen):
        super().__init__()
        
        self._Xgen = Xgen
        self._Ugen = Ugen
        self._Fgen = Fgen
        self._dTgen = dTgen

        self._xDim = self.Xgen._xDim
        self._uDim = self.Ugen._uDim
        self._FDim = self.Fgen._FDim
        self._qDim = int(self._xDim/2)

        self._child.update({
            "Xgen": self._Xgen,
            "Ugen": self._Ugen,
            "Fgen": self._Fgen,
            "dTgen": self._dTgen
        })
        self._sc = 0 # step count

    @property
    def Xgen(self):
        return self._Xgen
    @property
    def Ugen(self):
        return self._Ugen
    @property
    def Fgen(self):
        return self._Fgen
    @property
    def dTgen(self):
        return self._dTgen
    
    @Xgen.setter
    def Xgen(self, xg):
        self._Xgen = xg
        self._child["Xgen"] = xg    
    @Ugen.setter
    def Ugen(self, ug):
        self._Ugen = ug
        self._child["Ugen"] = ug
    @Fgen.setter
    def Fgen(self, Fg):
        self._Fgen = Fg
        self._child["Fgen"] = Fg
    @dTgen.setter
    def dTgen(self, dtg):
        self._dTgen = dtg
        self._child["dTgen"] = dtg


    def _begin(self, x0, u0, F0, **kwargs):
        """Set up the first step
           calls Xgen, Ugen, Fgen with x0, xk, u0, F0
        """
        self._sc = 0
        Xk = self._Xgen.step(step = self._sc, x0 = x0, **kwargs)
        Uk = self._Ugen.step(step = self._sc, u0 = u0, xk = Xk, **kwargs)
        Fk = self._Fgen.step(step = self._sc, F0 = F0, **kwargs)
        
        self._state.update({
            "x": Xk,        # x at step K
            "u" : Uk,       # u at step K
            "F" : Fk,       # F at step K
            "ddq1":None     # ddq at step K+1, fitted by Xs at step K and K+1
        })
        return Xk, Uk, Fk
    
    def step(self, x0, u0, F0, **kwargs):
        self._sc += 1
        Xk_puls_1 = self._Xgen.step(step = self._sc, x0 = x0, **kwargs)
        Uk_puls_1 = self._Ugen.step(step = self._sc, u0 = u0, xk = Xk_puls_1, **kwargs)
        Fk_puls_1 = self._Fgen.step(step = self._sc, F0 = F0, **kwargs)

        dt = self._dTgen.step(step = self._sc)

        Xk = self._state["x"]
        Uk = self._state["u"]
        Fk = self._state["F"]
        return Xk_puls_1, Uk_puls_1, Fk_puls_1, Xk, Uk, Fk, dt


class TowrCollocation(Collocation):
    def __init__(self, Xgen, Ugen, Fgen, dTgen):
        super().__init__(Xgen, Ugen, Fgen, dTgen)
    
    def _begin(self, x0, u0, F0, **kwargs):
        Xk, Uk, Fk =  super()._begin(x0, u0, F0, **kwargs)
        self._state.update({
            "ddq1":None     # ddq at step K+1, fitted by Xs at step K and K+1
        })
        return Xk, Uk, Fk
    
    def step(self, eomF, x0, u0, F0, **kwargs):
        """
        Args:
            eomF (Equation of Motion Function): (dx, x, u, F) -> g
        """
        Xk_puls_1, Uk_puls_1, Fk_puls_1, Xk, Uk, Fk, dt = super().step(x0, u0, F0, **kwargs)

        q0 = Xk[:self._qDim]
        dq0 = Xk[self._qDim:]
        q1 = Xk_puls_1[:self._qDim]
        dq1 = Xk_puls_1[self._qDim:]

        # retrieve the param of a3t^3 + a2t^2 + a1t + a0 fitted by the q0 and q1
        a0 = q0
        a1 = dq0
        a2 = -(3*(q0 - q1) + dt*(2*dq0 + dq1))/(dt**2)
        a3 = (2*(q0 - q1) + dt*(dq0 + dq1))/(dt**3)

        # dynamic constraint
        ddq0 = 2*a2 
        g = eomF(ca.vertcat(dq0,ddq0), Xk, Uk, Fk)
        self._g.append(g)
        self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*g.size(1)) #size(1): the dim of axis0

        # smooth constraint
        if(self._state["ddq1"] is not None):
            self._g.append(ddq0 - self._state["ddq1"])
            self._lbg.append([0]*self._qDim) 
            self._ubg.append([0]*self._qDim)
        
        ddq1 = 6 * a3 * dt + 2*a2
        self._state.update({
            "x": Xk_puls_1,
            "u" : Uk_puls_1,
            "F" : Fk_puls_1,
            "ddq1":ddq1
        })

class EularCollocation(Collocation):
    def __init__(self, Xgen, Ugen, Fgen, dTgen):
        super().__init__(Xgen, Ugen, Fgen, dTgen)
    

    def step(self, dynF, x0, u0, F0, **kwargs):
        """
        Args:
            eomF (Dynamics Function): (x, u, F) -> dx
        """
        Xk_puls_1, Uk_puls_1, Fk_puls_1, Xk, Uk, Fk, dt = super().step(x0, u0, F0, **kwargs)

        # dynamic constraint
        dxk = dynF(Xk, Uk, Fk)
        g = Xk_puls_1 - (Xk + dxk * dt)
        self._g.append(g)
        self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*g.size(1)) #size(1): the dim of axis0

        self._state.update({
            "x": Xk_puls_1,
            "u" : Uk_puls_1,
            "F" : Fk_puls_1,
        })


class KKT_TO(Collocation):
    """The trajectory optimization that enforces a KKT condition every step
    """
    def __init__(self, Xgen, Ugen, Fgen, dTgen):
        super().__init__(Xgen, Ugen, Fgen, dTgen)
        self._ml_plot=[]
        self._mu_plot=[]
        self._jacL_plot=[]
        self._comS_plot=[]
        self._parse.update({
            "ml_plot": lambda: ca.horzcat(*self._ml_plot),
            "mu_plot": lambda: ca.horzcat(*self._mu_plot),
            "jacL_plot": lambda: ca.horzcat(*self._jacL_plot),
            "comS_plot": lambda: ca.horzcat(*self._comS_plot)
        })


    def step(self, intF, Func0, Func1, Func2, x0, u0, F0, **kwargs):
        """
        Args:
            intF (integral Function): (x, dx) -> newx
            Func0 (KKT cost function for dx)
            Func1 (KKT neq constraints for dx)
            Func2 (KKT eq constraints for dx)
        """
        Xk_puls_1, Uk_puls_1, Fk_puls_1, Xk, Uk, Fk, dt = super().step(x0, u0, F0, **kwargs)
        ## Add variable dx
        dxk = optGen.VARTYPE.sym('dx%d'%self._sc, self.Xgen._xDim)
        self._w.append(dxk)
        self._lbw.append([-ca.inf]*self.Xgen._xDim)
        self._ubw.append([ca.inf]*self.Xgen._xDim)
        self._w0.append(x0)
        self._state["dx"] = dxk

        # constraints on step K
        f0 = kwargFunc(Func0)(**self._state)
        f1 = kwargFunc(Func1)(**self._state) if Func1 is not None else ca.DM([0])
        f2 = kwargFunc(Func2)(**self._state) if Func2 is not None else ca.DM([0])

        # Add primal conditions
        self._g.append(f1)
        self._lbg.append([-ca.inf]*f1.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*f1.size(1)) #size(1): the dim of axis0

        self._g.append(f2)
        self._lbg.append([0]*f2.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*f2.size(1)) #size(1): the dim of axis0


        ## Add variable Lambda: dual variable for neq cons
        #>>> Exp Lambda
        # ml_ = optGen.VARTYPE.sym('lam%d'%self._sc, f1.size(1))
        # self._w.append(ml_)
        # self._lbw.append([-1e2]*f1.size(1))
        # self._ubw.append([ca.inf]*f1.size(1))
        # self._w0.append([0]*f1.size(1))
        # ml = ca.exp(ml_)
        #--- 1/x Lambda
        # ml_ = optGen.VARTYPE.sym('lam%d'%self._sc, f1.size(1))
        # self._w.append(ml_)
        # self._lbw.append([1e-2]*f1.size(1))
        # self._ubw.append([ca.inf]*f1.size(1))
        # self._w0.append([1e-2]*f1.size(1))
        # ml = 1/ml_
        #--- raw Lambda
        ml = optGen.VARTYPE.sym('lam%d'%self._sc, f1.size(1))
        self._w.append(ml)
        self._lbw.append([0]*f1.size(1))
        self._ubw.append([ca.inf]*f1.size(1))
        self._w0.append([0]*f1.size(1))
        #---
        # self._J+= ca.sum1(1/(ml+0.001))
        #<<<
        self._ml_plot.append(ml)


        ## Add variable mu: dual variable for eq cons
        mu = optGen.VARTYPE.sym('mu%d'%self._sc, f2.size(1))
        self._w.append(mu)
        self._lbw.append([-ca.inf]*f2.size(1))
        self._ubw.append([ca.inf]*f2.size(1))
        self._w0.append([0]*f2.size(1))
        self._mu_plot.append(mu)


        ## Add Stationarity Constraint
        jacL = ca.gradient(f0, dxk) + ca.jtimes(f1, dxk, ml, True) + ca.jtimes(f2, dxk, mu, True)
        self._g.append(jacL)
        self._lbg.append([0]*jacL.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*jacL.size(1)) #size(1): the dim of axis0
        self._jacL_plot.append(jacL)


        ## Add Complementary Slackness
        comS = ml*f1
        self._g.append(comS)
        self._lbg.append([0]*comS.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*comS.size(1)) #size(1): the dim of axis0
        self._comS_plot.append(comS)


        g = Xk_puls_1 - intF(Xk, dxk)
        self._g.append(g)
        self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
        self._ubg.append([0]*g.size(1)) #size(1): the dim of axis0


        self._state.update({
            "x": Xk_puls_1,
            "u" : Uk_puls_1,
            "F" : Fk_puls_1,
            "ml": ml
        })


class xGenDefault(optGen):
    def __init__(self, xDim, xLim):
        super().__init__()
        self._xDim = xDim
        self._xLim = xLim

        self._x_plot = [] # a matrix used for plotting the x of each step
        self._parse.update({
            "x_plot": lambda: ca.horzcat(*self._x_plot)
        })
    
    def _begin(self, **kwargs):
        pass
    
    def step(self, step, x0, **kwargs):
        Xk = optGen.VARTYPE.sym('X%d'%step, self._xDim)
        self._w.append(Xk)
        if(step == 0): # init state
            self._lbw.append(x0)
            self._ubw.append(x0)
        else:
            self._lbw.append(self._xLim[:,0])
            self._ubw.append(self._xLim[:,1])
        self._w0.append(x0)
        self._x_plot.append(Xk)
        return Xk


class xGenTerrianHoloCons(xGenDefault):
    def __init__(self, xDim, xLim, ptrFuncs, terrian, terrianLim = (-2,2), robustGap = None):
        """xGen that adds the terrian constraints of points

        Args:
            ptrFuncs ([(x)=> p]): list of functions that calculates interested points given current state
            terrian ([g(p)]): the constraint function on points, the constraint is: g(p)>0
            robustGap([double]): the gap for the ptr functions. The ptrs should be higher than the gap. Same size as ptrFunctions
        """
        super().__init__(xDim, xLim)
        self.ptrFuncs = ptrFuncs
        self.terrain = terrian
        self.terrainLim = terrianLim
        terrianX = np.linspace(*terrianLim, 100)
        self._parse.update({
            "x_plot": lambda: ca.horzcat(*self._x_plot),
            "terrain_plot": lambda: ca.horzcat(terrianX, substiSX2MX( 
                ca.vertcat(*[- self.terrain([x,0]) for x in terrianX]), 
                self.hyperParamList(ca.SX),
                self.hyperParamList(ca.MX) ) )
        })
        self.robustGap = [0] * len(self.ptrFuncs) if robustGap is None else robustGap
        assert(len(self.robustGap) == len(self.ptrFuncs))

    def _begin(self, **kwargs):
        pass
    
    def step(self, step, x0, **kwargs):
        Xk = super().step(step,x0,**kwargs)

        for pF, r in zip(self.ptrFuncs, self.robustGap):
            g = self.tryCallWithHyperParam(self.terrain, {"p" : pF(Xk)} )
            self._g.append(g - r)
            self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([np.infty]*g.size(1)) #size(1): the dim of axis0

        return Xk


class uGenDefault(optGen):
    def __init__(self, uDim, uLim):
        super().__init__()
        self._uDim = uDim
        self._uLim = uLim

        self._u_plot = [] # a matrix used for plotting the x of each step
        self._parse.update({
            "u_plot": lambda: ca.horzcat(*self._u_plot)
        })
    
    def _begin(self, **kwargs):
        pass
    
    def step(self, step, u0, **kwargs):
        Uk = optGen.VARTYPE.sym('U%d'%step, self._uDim)
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk)
        return Uk


class FGenDefault(optGen):
    def __init__(self, FDim, FLim):
        super().__init__()
        self._FDim = FDim
        self._FLim = FLim

        self._F_plot = [] # a matrix used for plotting the x of each step
        self._parse.update({
            "F_plot": lambda: ca.horzcat(*self._F_plot)
        })
    
    def _begin(self, **kwargs):
        pass
    
    def step(self, step, F0, **kwargs):
        if(not self._FDim):
            return ca.DM([])
        Fk = optGen.VARTYPE.sym('F%d'%step, self._FDim)
        self._w.append(Fk)
        self._lbw.append(self._FLim[:,0])
        self._ubw.append(self._FLim[:,1])
        self._w0.append(F0)
        self._F_plot.append(Fk)
        return Fk


class dTGenDefault(optGen):
    def __init__(self, dT):
        super().__init__()
        self._dT = dT
        self._t_plot = []
        self._parse.update({
            "t_plot": lambda: ca.vertcat(*self._t_plot)
        })
    
    def chMod(self, modName, *args, **kwargs):
        pass
    
    def _begin(self, **kwargs):
        self._t_plot.append(0)
    
    def step(self, step):
        self._t_plot.append(self._t_plot[-1] + self._dT)
        return self._dT

class dTGenVariable(dTGenDefault):
    def __init__(self, dT, dtLim):
        super().__init__(dT)
        self._dtLim = dtLim
        self.curent_dt = None
    
    def chMod(self, modName, *args, **kwargs):
        T = optGen.VARTYPE.sym('dT_%s'%modName, 1)
        self._w.append(T)
        self._lbw.append([self._dtLim[0]])
        self._ubw.append([self._dtLim[1]])
        self._w0.append([self._dT])
        self.curent_dt = T

    def step(self, step):
        self._t_plot.append(self._t_plot[-1] + self.curent_dt)
        return self.curent_dt


def TowrCollocationDefault(xDim, uDim, FDim, xLim, uLim, FLim, dt):
    return TowrCollocation(
        Xgen = xGenDefault(xDim, np.array(xLim)),
        Ugen = uGenDefault(uDim, np.array(uLim)),
        Fgen = FGenDefault(FDim, np.array(FLim)),
        dTgen= dTGenDefault(dt)
    )

def TowrCollocationVTiming(xDim, uDim, FDim, xLim, uLim, FLim, dt, dtLim):
    return TowrCollocation(
        Xgen = xGenDefault(xDim, np.array(xLim)),
        Ugen = uGenDefault(uDim, np.array(uLim)),
        Fgen = FGenDefault(FDim, np.array(FLim)),
        dTgen= dTGenVariable(dt, np.array(dtLim))
    )

