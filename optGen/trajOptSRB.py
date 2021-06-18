"""
traj optimize modules for Single Rigid Body traj optimization
"""
from optGen.trajOptimizer import *


class uGenSRB(uGenDefault):
    def __init__(self, nc, terrain, terrain_norm, mu, debug=True):
        """the input generater for Single Rigid Body traj opt
            The u of SRB problem is [p_c, F_c]
        Args:
            nc (int): number of contact points
            terrain (function: (p:vec3)->(h:float) ): return the height of a point to ground
            terrain_norm (function: (P:vec3)->(n:vec3) ): return the norm vector of terrain
        """
        uDim = 6 * nc
        uLim = ca.DM([[-ca.inf, ca.inf], [-ca.inf, ca.inf], [-ca.inf, ca.inf], 
                      [-200, 200], [-200, 200], [-200, 200]]*nc)
        super().__init__(uDim, uLim)
        self.nc = nc
        self.terrain = terrain
        self.terrain_norm = terrain_norm
        self.mu = mu
        self.contactMap = [True] * nc
        self._pc_plot = [] # for plotting contact points
        self._fc_plot = [] # for plotting contact force
        self._parse.update({
            "pc_plot": lambda: ca.horzcat(*self._pc_plot),
            "fc_plot": lambda: ca.horzcat(*self._fc_plot)
        })

        self.debug = debug
        if(self.debug):
            self._gdb0_plot = [] # collecting the `g` debug information 0
            self._gdb1_plot = [] # collecting the `g` debug information 1
            self._gdb2_plot = [] # collecting the `g` debug information 2
            # self._gdb3_plot = [] # collecting the `g` debug information 3

            self._parse.update({
                "gdb0_plot": lambda: ca.vertcat(*self._gdb0_plot),
                "gdb1_plot": lambda: ca.vertcat(*self._gdb1_plot),
                "gdb2_plot": lambda: ca.vertcat(*self._gdb2_plot)
                # "gdb3_plot": lambda: ca.vertcat(*self._gdb3_plot),
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
        pcK = [optGen.VARTYPE.sym("pc%d_%d"%(step, i), 3) for i in range(self.nc)]
        fcK = [optGen.VARTYPE.sym("fc%d_%d"%(step, i), 3) for i in range(self.nc)]
        Uk = ca.vertcat(*[ ca.vertcat(pc, fc) for pc, fc in zip(pcK, fcK)])
        self._w.append(Uk)
        self._lbw.append(self._uLim[:,0])
        self._ubw.append(self._uLim[:,1])
        self._w0.append(u0)
        self._u_plot.append(Uk)
        self._pc_plot.append(ca.vertcat(*pcK))
        self._fc_plot.append(ca.vertcat(*fcK))

        # add constraint: contact points does not move
        if(self._state["pclist"] is not None):
            pcK_1 = self._state["pclist"]
            g = ca.vertcat(optGen.VARTYPE([]), *[p - q for p,q,c in zip(pcK, pcK_1,self.contactMap) if c])
            self._g.append(g)
            self._lbg.append([0]*g.size(1))
            self._ubg.append([0]*g.size(1))
            if self.debug: self._gdb0_plot.append(g)

        # add equality constraints:
        g = ca.vertcat(*[f for f,c in zip(fcK, self.contactMap) 
                        if not c], # zero force float 
                       *[self.tryCallWithHyperParam(self.terrain, {"p" : p} )
                        for p,c in zip(pcK, self.contactMap) if c] # contact on ground constraint
                        )
        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([0]*g.size(1))
        if self.debug: self._gdb1_plot.append(g)

        # add inequality constraints:
        tnorms = [self.tryCallWithHyperParam(self.terrain_norm, {"p": p}) for p in pcK]
        g = ca.vertcat(*[(self.mu * ca.dot(f,n))**2 - ca.norm_2(f - n * ca.dot(f,n))**2
                        for f,n,c in zip(fcK, tnorms, self.contactMap) if c], # friction cone square
                       *[ca.dot(f,n) 
                        for f,n,c in zip(fcK, tnorms, self.contactMap) if c], # support force direction
                       *[self.tryCallWithHyperParam(self.terrain, {"p" : p} )
                        for p,c in zip(pcK, self.contactMap) if not c] # float above ground constraint
                        )
        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([ca.inf]*g.size(1))
        if self.debug: self._gdb2_plot.append(g)

        self._state.update({
            "pclist": pcK,
            "fclist": fcK
        })

        return Uk


class uGenSRB2(uGenDefault):
    def __init__(self, nc, terrain, terrain_norm, mu, debug=True):
        """Similar to uGenSRB, but the number of variables is set to the minimum
        NOTE: This uGen assumes that there is no constraints set on swing legs
        """
        uDim = 6 * nc
        uLim = ca.DM([[-ca.inf, ca.inf]]*uDim)
        super().__init__(uDim, uLim)
        self.flim = ca.DM([[-200, 200]]*3)

        self.nc = nc
        self.terrain = terrain
        self.terrain_norm = terrain_norm
        self.mu = mu
        self.contactMap = [True] * nc
        self._pc_plot = [] # for plotting contact points
        self._fc_plot = [] # for plotting contact force
        self._parse.update({
            "pc_plot": lambda: ca.horzcat(*self._pc_plot),
            "fc_plot": lambda: ca.horzcat(*self._fc_plot)
        })

    def _begin(self, contactMap, u0, **kwargs):
        self.contactMap = [False] * self.nc
        self.contactPos = [ca.DM([0,0,0])] * self.nc # DM 0,0,0 for the positions that are not contacted
        self.chMod("begin", contactMap, [ t[:3] for t in ca.vertsplit(u0, 6)], fix=False)

    def chMod(self, modName, contactMap, pc0, fix=False, *args, **kwargs):
        assert(len(contactMap) == self.nc)
        # generate new variables according to the change of the contact map
        self.contactPos = [
            p if c0 == c1 else # If the contactmap does not change, then the pos keeps the same
            optGen.VARTYPE.sym("pc_%s_%d"%(modName, i), 3) if c1 else
            ca.DM([0,0,0])
            for i,(p,c0,c1) in enumerate(zip(self.contactPos, self.contactMap, contactMap))
        ]
        pp0 = [(p,p0) for p,p0,c0,c1 in zip(self.contactPos, pc0, self.contactMap, contactMap) if c1 and not c0]
        var = ca.vertcat(* [p for p,p0 in pp0])
        self._w.append(var)
        if(fix):
            self._lbw.append(ca.vertcat(* [p0 for p,p0 in pp0]))
            self._ubw.append(ca.vertcat(* [p0 for p,p0 in pp0]))
        else:
            self._lbw.append([-ca.inf] * var.size(1))
            self._ubw.append([ca.inf] * var.size(1))
        self._w0.append(ca.vertcat(* [p0 for p,p0 in pp0]))
        self.contactMap = contactMap

        # add equality constraints:
        g = ca.vertcat(*[self.tryCallWithHyperParam(self.terrain, {"p" : p} )
                        for p,p0 in pp0] # contact on ground constraint
                        )
        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([0]*g.size(1))


    def step(self, step, u0, **kwargs):
        """
        Args:
            u0 ([DM]): a vector (p_0, f_0, p_1, f_1, ...)
        """
        pcK = self.contactPos
        fcK = [optGen.VARTYPE.sym("fc%d_%d"%(step, i), 3) if c else ca.DM([0,0,0]) 
                    for i,c in enumerate(self.contactMap)]
        Uk = ca.vertcat(*[ ca.vertcat(pc, fc) for pc, fc in zip(pcK, fcK)])
        self._w.append(ca.vertcat(*[f for f,c in zip(fcK,self.contactMap) if c]))
        self._lbw.append(ca.vertcat(*[self.flim[:,0] for c in self.contactMap if c]))
        self._ubw.append(ca.vertcat(*[self.flim[:,1] for c in self.contactMap if c]))
        f0 = [t[3:6] for t,c in zip(ca.vertsplit(u0, 6), self.contactMap) if c]
        self._w0.append(ca.vertcat(*f0))

        self._u_plot.append(Uk)
        self._pc_plot.append(ca.vertcat(*pcK))
        self._fc_plot.append(ca.vertcat(*fcK))

        # add inequality constraints:
        tnorms = [self.tryCallWithHyperParam(self.terrain_norm, {"p": p}) for p in pcK]
        g = ca.vertcat(*[(self.mu * ca.dot(f,n))**2 - ca.norm_2(f - n * ca.dot(f,n))**2
                        for f,n,c in zip(fcK, tnorms, self.contactMap) if c], # friction cone square
                       *[ca.dot(f,n) 
                        for f,n,c in zip(fcK, tnorms, self.contactMap) if c], # support force direction
                        )
        self._g.append(g)
        self._lbg.append([0]*g.size(1))
        self._ubg.append([ca.inf]*g.size(1))
        return Uk


def SRBoptDefault(xDim, xLim, nc, dt, terrain, terrain_norm, mu):
    return EularCollocation(
        Xgen = xGenDefault(xDim, np.array(xLim)),
        # Ugen = uGenSRB(nc, terrain, terrain_norm, mu),
        Ugen = uGenSRB2(nc, terrain, terrain_norm, mu),
        Fgen = FGenDefault(0, []),
        dTgen= dTGenDefault(dt)
    )
