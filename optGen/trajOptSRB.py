"""
traj optimize modules for Single Rigid Body traj optimization
"""
from optGen.trajOptimizer import *


class uGenSRB(uGenDefault):
    def __init__(self, nc, terrain, terrain_norm, mu):
        """the input generater for Single Rigid Body traj opt
            The u of SRB problem is [p_c, F_c]
        Args:
            nc (int): number of contact points
            terrain (function: (p:vec3)->(h:float) ): return the height of a point to ground
            terrain_norm (function: (P:vec3)->(n:vec3) ): return the norm vector of terrain
        """
        uDim = 6 * nc
        super().__init__(uDim, ca.DM([-ca.inf, ca.inf]*uDim))
        self.nc = nc
        self.terrain = terrain
        self.terrain_norm = terrain_norm
        self.mu = mu
        self.contactMap = [True] * nc
        self._pc_plot = [] # for plotting contact points
        self._fc_plot = [] # for plotting contact force
        self._parse.update({
            "pc_plot": lambda: ca.vertcat(*self._pc_plot),
            "fc_plot": lambda: ca.vertcat(*self._pc_plot)
        })

    def _begin(self, **kwargs):
        self._state.update({
            "pclist": None,
            "fclist": None
        })

    def chMod(self, modName, contactMap, *args, **kwargs):
        assert(len(contactMap) == self.nc)
        self.contactMap = contactMap

    def step(self, step, u0, **kwargs):
        """
        Args:
            u0 ([DM]): a vector (p_0, f_0, p_1, f_1, ...)
        """
        pcK = [ca.SX.sym("pc%d_%d"%(step, i), 3) for i in range(self.nc)]
        fcK = [ca.SX.sym("fc%d_%d"%(step, i), 3) for i in range(self.nc)]
        Uk = ca.vertcat(*[ ca.vertcat(pc, fc) for pc, fc in zip(pcK, fcK)])
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk)
        self._pc_plot+= pcK
        self._fc_plot+= fcK

        # add constraint: contact points does not move
        if(self._state["pclist"] is not None):
            pcK_1 = self._state["pclist"]
            g = ca.vertcat(ca.SX([]), *[p - q for p,q,c in zip(pcK, pcK_1,self.contactMap) if c])
            self._g.append(g)
            self._lbg.append([0]*g.size(1))
            self._ubg.append([0]*g.size(1))

        # add equality constraints:
        g = ca.vertcat(*[f for f,c in zip(fcK, self.contactMap) 
                        if not c], # zero force float 
                       *[self.tryCallWithHyperParam(self.terrain, {"p" : p} )
                        for p,c in zip(pcK, self.contactMap) if c] # contact on ground constraint
                        )
        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([0]*g.size(1))

        # add inequality constraints:
        tnorms = [self.tryCallWithHyperParam(self.terrain_norm, {"p": p}) for p in pcK]
        g = ca.vertcat(*[(self.mu * ca.dot(f,n)) - ca.norm_2(f - n * ca.dot(f,n))
                        for f,n,c in zip(fcK, tnorms, self.contactMap) if c], # zero force float 
                       *[self.tryCallWithHyperParam(self.terrain, {"p" : p} )
                        for p,c in zip(pcK, self.contactMap) if not c] # float above ground constraint
                        )

        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([ca.inf]*g.size(1))

        self._state.update({
            "pclist": pcK,
            "fclist": fcK
        })

        return Uk

def SRBoptDefault(xDim, xLim, nc, dt, terrain, terrain_norm, mu):
    return EularCollocation(
        Xgen = xGenDefault(xDim, np.array(xLim)),
        Ugen = uGenSRB(nc, terrain, terrain_norm, mu),
        Fgen = FGenDefault(0, []),
        dTgen= dTGenDefault(dt)
    )
