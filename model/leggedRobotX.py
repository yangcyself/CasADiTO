from model.articulateBody import *
import yaml
import matplotlib.pyplot as plt
class LeggedRobotX(ArticulateSystem):
    def __init__(self, params):
        self.params = params

        root = Base2D.Freebase("Base", 0, 0)
        super().__init__(root) # this will set self.root

        self.torso = root.addChild(
            Link2D.Rot, name = "Torso",
            la = - params["torW"]/2, lb = params["torW"]/2, lc = params["torLc"],
            M = params["torM"], I = params["torI"], fix = True
        )

        def getAleg(name):
            thigh = Link2D.Rot(name = "%sThigh"%name, Fp = ca.vertcat(0,0,0),
                la = 0, lb = params["legL1"], lc = params["legLc1"],
                M = params["legM1"], I = params["legI1"])
            shank = thigh.addChild(
                Link2D.Rot, params["legL1"], name = "%sShank"%name,
                la = 0, lb = params["legL2"], lc = params["legLc2"],
                M = params["legM2"], I = params["legI2"]
            )
            return [thigh, shank]

        lleg = getAleg("L")
        rleg = getAleg("R")

        self.lhip = self.torso.addChild(
            Proj2dRot, - params["torW"]/2, name = "LHip",
            bdy = lleg[0], rotdir = ca.vertcat(0,1,0),  
        ) 

        self.rhip = self.torso.addChild(
            Proj2dRot,   params["torW"]/2, name = "RHip",
            bdy = rleg[0], rotdir = ca.vertcat(0,1,0),  
        )
        self.l1, self.l2 = lleg
        self.r1, self.r2 = rleg
        
        print(self.root.x)


    @property
    def u_dim(self):
        return 6

    @property
    def F_dim(self):
        return 4

    @property
    def B(self):
        return ca.jacobian(self.root.x, ca.vertcat(self.lhip._q, self.l1._q, self.l2._q, self.rhip._q, self.r1._q, self.r2._q))
    
    @property
    def pFuncs(self):
        return {
            "plTor" : ca.Function("plTor", [self.x], [self.torso.points["a"]]),
            "prTor" : ca.Function("prTor", [self.x], [self.torso.points["b"]]),
            "pl1" : ca.Function("pl1", [self.x], [self.lhip._p_proj(self.l1.points["b"])]),
            "pl2" : ca.Function("pl2", [self.x], [self.lhip._p_proj(self.l2.points["b"])]),
            "pr1" : ca.Function("pr1", [self.x], [self.rhip._p_proj(self.r1.points["b"])]),
            "pr2" : ca.Function("pr2", [self.x], [self.rhip._p_proj(self.r2.points["b"])])
        }

    @property
    def pLocalFuncs(self):
        """The position functions in the local frames (x,z frame) of each leg, where as the frame of root is y,z frame
        """
        return {
            "pl1" : ca.Function("pl1", [self.x], [self.l1.points["b"]]),
            "pl2" : ca.Function("pl2", [self.x], [self.l2.points["b"]]),
            "pr1" : ca.Function("pr1", [self.x], [self.r1.points["b"]]),
            "pr2" : ca.Function("pr2", [self.x], [self.r2.points["b"]]),
        }

    def buildEOMF(self, consMap, name=""):
        """Build the equation of Motion and constraint. Return g(x,u,F,ddq)
            Note: this EOM function returned is insufficient for the full dynamics. 
                This is because there is no constrait for the x direction of the contact

        Args:
            consMap ((bool,bool)): The contact condition of the back and front legs
            name (str, optional): The name of the function. Defaults to "EOMF".

        Returns:
            g(x,u,F,ddq)->[SX]: The equation that should be zero
                g contains: 1. dynamic constriant, 2. contact point fix, 3. float point no force
        """
        ddq = ca.SX.sym("ddq",self.dim)
        F = ca.SX.sym("F",self.F_dim) #Fb, Ff
        u = ca.SX.sym("u", self.u_dim)

        cons = [self.lhip._p_proj(self.l2.points["b"]), self.rhip._p_proj(self.r2.points["b"])]
        consJ = [ca.jacobian(c,self.q) for c in cons]
        toeJac = ca.vertcat(*consJ)


        g = self.EOM_func(self.q, self.dq, ddq, self.B @ u+toeJac.T @ F) # the EOM
        g = ca.vertcat(g, *[ cJ @ ddq + ca.jtimes(cJ,self.q,self.dq)@self.dq for cJ,cm in zip(consJ,consMap) if cm])
        g = ca.vertcat(g, *[ F[i*2:i*2+2] for i,cm in enumerate(consMap) if not cm])
        g = ca.simplify(g)
        return ca.Function("%sEOMF"%name, [self.x,u,F,ddq], [g], ["x","u","F","ddq"], ["%sEOM"%name])
    

    def visulize(self, x, ax = None):
        ax = plt.gca() if ax is None else ax
        x = x[:self.dim]
        try:
            linkLine = self.linkLine_func_cache(x)
            # np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        except AttributeError:
            # xsym = ca.SX.sym("xsym", len(x))
            linkLine = ca.vertcat(self.lhip._p_proj(self.l2.points["b"]).T, 
                                 self.lhip._p_proj(self.l1.points["b"]).T,
                                 self.torso.points["a"].T,
                                 self.torso.points["b"].T,
                                 self.rhip._p_proj(self.r1.points["b"]).T, 
                                 self.rhip._p_proj(self.r2.points["b"]).T)
            self.linkLine_func_cache = ca.Function("linkLine_func", [self.q], [linkLine], ["xsym"], ["linkLine"])
            linkLine = self.linkLine_func_cache(x)

        line, = ax.plot(linkLine[:,0], linkLine[:,1], marker = '.', ms = 5)
        return line

    def visulizeLocal(self, x, ax = None):
        ax = plt.gca() if ax is None else ax
        x = x[:self.dim]
        try:
            linkLinel, linkLiner = self.linkLine_local_func_cache(x)
            # np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        except AttributeError:
            # xsym = ca.SX.sym("xsym", len(x))
            linkLinel = ca.vertcat( self.l1.points["a"].T,
                                    self.l1.points["b"].T, 
                                    self.l2.points["b"].T)
            linkLiner = ca.vertcat( self.r1.points["a"].T,
                                    self.r1.points["b"].T, 
                                    self.r2.points["b"].T)
            self.linkLine_local_func_cache = ca.Function("linkLine_local_func", [self.q], [linkLinel, linkLiner], ["xsym"], ["linkLinel", "linkLiner"])
            linkLinel, linkLiner = self.linkLine_local_func_cache(x)

        linel, = ax.plot(linkLinel[:,0], linkLinel[:,1])
        liner, = ax.plot(linkLiner[:,0], linkLiner[:,1])
        return linel, liner


    @staticmethod
    def fromYaml(yamlFilePath):
        ConfigFile = yamlFilePath
        with open(ConfigFile, 'r') as stream:
            try:
                robotParam = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        params = {
            "legL1":robotParam["l1"],
            "legL2":robotParam["l2"],
            "legLc1":robotParam["lc1"],
            "legLc2":robotParam["lc2"],
            "torL":robotParam["L"],
            "torW": robotParam["LW"],
            "torLL":robotParam["LL"],
            "torLc":robotParam.get("Lc", 0),
            "legM1":robotParam["m1"],
            "legM2":robotParam["m2"],
            "torM": robotParam["M"],
            "legI1":robotParam["j1"],
            "legI2":robotParam["j2"],
            "torI":robotParam["J"],
            "q1Lim": [-ca.pi/2 - robotParam["ang1max"], -ca.pi/2 - robotParam["ang1min"]],
            "q2Lim": [-robotParam["ang2max"], -robotParam["ang2min"]],
            "dq1Lim": [-robotParam["dang1lim"], robotParam["dang1lim"]],
            "dq2Lim": [-robotParam["dang2lim"], robotParam["dang2lim"]],
            "tau1lim": robotParam["tau1lim"],
            "tau2lim": robotParam["tau2lim"],
            "tau3lim": robotParam["tau3lim"],
            "G":9.81,
        }
        return LeggedRobotX(params)

