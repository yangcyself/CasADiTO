from optGen.optGen import *

class TowrCollocation(optGen):
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
    
    def step(self, dynF, x0, u0, F0, **kwargs):
        self._sc += 1
        Xk_puls_1 = self._Xgen.step(step = self._sc, x0 = x0, **kwargs)
        Uk_puls_1 = self._Ugen.step(step = self._sc, u0 = u0, xk = Xk_puls_1, **kwargs)
        Fk_puls_1 = self._Fgen.step(step = self._sc, F0 = F0, **kwargs)

        dt = self._dTgen.step(step = self._sc)

        Xk = self._state["x"]
        Uk = self._state["u"]
        Fk = self._state["F"]

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
        g = dynF(ca.vertcat(dq0,ddq0), Xk, ca.vertcat(Uk, Fk))
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
        Xk = ca.SX.sym('X%d'%step, self._xDim)
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
    def __init__(self, xDim, xLim, ptrFuncs, terrian):
        """xGen that adds the terrian constraints of points

        Args:
            ptrFuncs ([(x)=> p]): list of functions that calculates interested points given current state
            terrian ([g(p)]): the constraint function on points, the constraint is: g(p)>0
        """
        super().__init__(xDim, xLim)
        self.ptrFuncs = ptrFuncs
        self.terrain = terrian
    
    def _begin(self, **kwargs):
        pass
    
    def step(self, step, x0, **kwargs):
        Xk = super().step(step,x0,**kwargs)

        for pF in self.ptrFuncs:
            g = self.terrain( pF(Xk) )
            self._g.append(g)
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
        Uk = ca.SX.sym('U%d'%step, self._uDim)
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
        Fk = ca.SX.sym('F%d'%step, self._FDim)
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
        T = ca.SX.sym('dT_%s'%modName, 1)
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
