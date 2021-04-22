from articulateBody import *

class LeggedRobot2D(ArticulateSystem):
    def __init__(self, params):
        root = FreeBase2D("Base", 0, 0)
        super().__init__(root) # this will set self.root

        self.torso = root.addChild(
            Link2D.Rot, name = "Torso",
            la = - params["torLL"]/2, lb = - params["torLL"]/2, lc = 0,
            M = params["torM"], I = params["torI"]
        )
        self.torso.fix()

        print(self.torso.q)
        
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
            M = params["legI2"], I = params["legI2"]
        )# front leg 2
