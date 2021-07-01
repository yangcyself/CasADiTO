from model.articulateBody import *
from model.articulateFullBody import *
"""The Articulate system of full body
using x,y,z, rx,ry,rz , q1, q2, ... as the state
"""
class FullDynamicsBody(ArticulateSystem):
    def __init__(self, urdfFile, eularMod = "ZYX", toeList = None):
        """[summary]

        Args:
            urdfFile ([type]): [description]
            eularMod (str, optional): [description]. Defaults to "ZYX".
            toeList ((linkName:str, pose_in_link_frame: vec3), optional): The list of position of toes
        """
        root = urdfWrap3D.FloatBaseUrdf('base', urdfFile, eularMod)
        super().__init__(root)
        self.urdfBase = self.root.child[0]

        # add the toes, used for buildEOMF
        self.toeList = [] if toeList is None else toeList
        self.toePoses = [
            self.getGlobalPos(n, p)
            for n,p in self.toeList
        ]

    @property
    def u_dim(self):
        return self.urdfBase.x.size(1)

    @property
    def F_dim(self):
        return 3 * len(self.toePoses)

    @property
    def B(self):
        return ca.jacobian(self.root.x, self.urdfBase.x)

    def getGlobalPos(self, linkName, localPos):
        return (self.urdfBase._linkdict[linkName].Bp @ ca.vertcat(localPos,1))[:3]
    
    def buildEOMF(self, consMap, name=""):
        """Build the equation of Motion and constraint. Return g(x,u,F,ddq)

        Args:
            consMap ((bool,bool)): The contact condition of the toes, the same order as toelist
            name (str, optional): The name of the function. Defaults to "EOMF".

        Returns:
            g(x,u,F,ddq)->[SX]: The equation that should be zero
                g contains: 1. dynamic constriant, 2. contact point fix, 3. float point no force
        """
        ddq = ca.SX.sym("ddq",self.dim)
        F = ca.SX.sym("F",self.F_dim) #Fb, Ff
        u = ca.SX.sym("u", self.u_dim)

        cons = self.toePoses
        consJ = [ca.jacobian(c,self.q) for c in cons]
        toeJac = ca.vertcat(*consJ)


        g = self.EOM_func(self.q, self.dq, ddq, self.B @ u+toeJac.T @ F) # the EOM
        g = ca.vertcat(g, *[ cJ @ ddq + ca.jtimes(cJ,self.q,self.dq)@self.dq for cJ,cm in zip(consJ,consMap) if cm])
        g = ca.vertcat(g, *[ F[i*2:i*2+2] for i,cm in enumerate(consMap) if not cm])
        g = ca.simplify(g)
        return ca.Function("%sEOMF"%name, [self.x,u,F,ddq], [g], ["x","u","F","ddq"], ["%sEOM"%name])
    
    
if __name__ == "__main__":

    m =FullDynamicsBody('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf',
        toeList=[("FL_SHANK", ca.DM([0,0,-0.2])),  ("FR_SHANK", ca.DM([0,0,-0.2])),
                 ("HL_SHANK", ca.DM([0,0,-0.2])), ("HR_SHANK", ca.DM([0,0,-0.2]))])

    # print(m.buildEOMF([1,1,1,1]))
    # print(ca.DM(m.B).full())

    ## Do a simple simulation using towr collocation form, enforce u to be zero
    from optGen.trajOptimizer import TowrCollocationDefault
    dT0 = 0.01
    opt = TowrCollocationDefault(2*m.dim, m.u_dim, m.F_dim, xLim = ca.DM([[-ca.inf, ca.inf]]*2*m.dim),
        uLim= ca.DM([[0, 0]]*m.u_dim), FLim = ca.DM([[-ca.inf, ca.inf]]*m.F_dim), dt= dT0)
    x0 = ca.DM([0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]+[0]*18)
    opt.begin(x0=x0, u0=[0]*12, F0=[0]*12)

    EOMF = m.buildEOMF([1,1,1,1])
    print("EOMF built")
    for i in range(10):
        opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[m.dim:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                x0 = x0, u0=[0]*12, F0=[0]*12)
    print("Opt Set")
    res = opt.solve(options=
        {"calc_f" : True,
        "calc_g" : True,
        "calc_lam_x" : True,
        "calc_multipliers" : True,
        "expand" : True, 
            "verbose_init":True,
            # "jac_g": gjacFunc
        "ipopt":{
            "max_iter" : 2000, # unkown option
            }
        })

    print(res["Xgen"]["x_plot"])