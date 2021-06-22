"""A trajectory optimizer for dragging the box to a target position
The box is defined in `heavyRopeLoad`
"""

from codogs.heavyRopeLoad import HeavyRopeLoad
from optGen.trajOptimizer import *

xDim = 3
xLim = [[-ca.inf, ca.inf]] *xDim


class uGenXYmove(uGenDefault):
    def __init__(self, nc, eps):        
        """
        Args:
            nc (size_t): The number of contacts
            eps (double): The maximum difference between steps
        """
        super().__init__(uDim = nc*2, uLim=[[-ca.inf, ca.inf]]*nc*2)
        self.nc = nc
        self.eps = eps
        
    def _begin(self, **kwargs):
        self._state.update({
            "u": None,
        })
    
    def step(self, step, u0, **kwargs):
        Uk = super().step(step, u0, **kwargs)
        Uk_ = self._state["u"]
        if(Uk_ is not None):
            g = ca.vertcat(*[ca.norm_2(d)**2 for d in ca.vertsplit(Uk_ - Uk)])
            self._g.append(g)
            self._lbg.append([0]*g.size(1)) #size(1): the dim of axis0
            self._ubg.append([self.eps**2]*g.size(1)) #size(1): the dim of axis0
        return Uk


opt =  KKT_TO(
    Xgen = xGenDefault(xDim, np.array(xLim)),
    Ugen = uGenXYmove(nc = 3, eps = 0.1),
    Fgen = FGenDefault(0, np.array([])),
    dTgen= dTGenDefault(0) # there is no notion of dT in this problem
)

