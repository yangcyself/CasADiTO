from model.articulateBody import *
import yaml
import matplotlib.pyplot as plt
class LeggedRobot2D(ArticulateSystem):
    def __init__(self, params):
        self.params = params

        root = Base2D.Freebase("Base", 0, 0)
        super().__init__(root) # this will set self.root

        self.torso = root.addChild(
            Link2D.Rot, name = "Torso",
            la = - params["torLL"]/2, lb = params["torLL"]/2, lc = 0,
            M = params["torM"], I = params["torI"], fix = True
        )

        self.b1 = self.torso.addChild(
            Link2D.Rot, - params["torL"]/2,  name = "Hthigh",
            la = 0, lb = params["legL1"], lc = params["legLc1"],
            M = params["legM1"], I = params["legI1"]
        )# back leg 1

        self.b2 = self.b1.addChild(
            Link2D.Rot, params["legL1"], name = "Hshank",
            la = 0, lb = params["legL2"], lc = params["legLc2"],
            M = params["legM2"], I = params["legI2"]
        )# back leg 2
        
        self.f1 = self.torso.addChild(
            Link2D.Rot, params["torL"]/2, name = "Fthigh",
            la = 0, lb = params["legL1"], lc = params["legLc1"],
            M = params["legM1"], I = params["legI1"]
        )# front leg 1

        self.f2 = self.f1.addChild(
            Link2D.Rot, params["legL1"], name = "Fshank",
            la = 0, lb = params["legL2"], lc = params["legLc2"],
            M = params["legM2"], I = params["legI2"]
        )# front leg 2

    @property
    def u_dim(self):
        return 4

    @property
    def F_dim(self):
        return 4

    @property
    def B(self):
        return ca.jacobian(self.root.x, ca.vertcat(self.b1._q, self.b2._q, self.f1._q, self.f2._q))
    
    @property
    def pFuncs(self):
        return {
            "prTor" : ca.Function("prTor", [self.x], [self.torso.points["a"]]),
            "phTor" : ca.Function("phTor", [self.x], [self.torso.points["b"]]),
            "phbLeg1" : ca.Function("phbLeg1", [self.x], [self.b1.points["b"]]),
            "phbLeg2" : ca.Function("phbLeg2", [self.x], [self.b2.points["b"]]),
            "phfLeg1" : ca.Function("phfLeg1", [self.x], [self.f1.points["b"]]),
            "phfLeg2" : ca.Function("phfLeg2", [self.x], [self.f2.points["b"]])
        }

    def buildEOMF(self, consMap, name=""):
        """Build the equation of Motion and constraint. Return g(x,u,F,ddq)

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

        cons = [self.b2.points["b"], self.f2.points["b"]]
        consJ = [ca.jacobian(c,self.q) for c in cons]
        toeJac = ca.vertcat(*consJ)


        g = self.EOM_func(self.q, self.dq, ddq, self.B @ u+toeJac.T @ F) # the EOM
        g = ca.vertcat(g, *[ cJ @ ddq + ca.jtimes(cJ,self.q,self.dq)@self.dq for cJ,cm in zip(consJ,consMap) if cm])
        g = ca.vertcat(g, *[ F[i*2:i*2+2] for i,cm in enumerate(consMap) if not cm])
        g = ca.simplify(g)
        return ca.Function("%sEOMF"%name, [self.x,u,F,ddq], [g], ["x","u","F","ddq"], ["%sEOM"%name])
    

    def visulize(self, x):
        x = x[:self.dim]
        try:
            linkLine = self.linkLine_func_cache(x)
            # np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        except AttributeError:
            # xsym = ca.SX.sym("xsym", len(x))
            linkLine = ca.vertcat(self.b2.points["b"].T, 
                                 self.b1.points["b"].T,
                                 self.b1.points["a"].T,
                                 self.torso.points["a"].T,
                                 self.torso.points["b"].T,
                                 self.f1.points["a"].T, 
                                 self.f1.points["b"].T,
                                 self.f2.points["b"].T)
            self.linkLine_func_cache = ca.Function("linkLine_func", [self.q], [linkLine], ["xsym"], ["linkLine"])
            linkLine = self.linkLine_func_cache(x)

        line, = plt.plot(linkLine[:,0], linkLine[:,1])
        return line

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
            "torLL":robotParam["LL"],
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
        return LeggedRobot2D(params)

