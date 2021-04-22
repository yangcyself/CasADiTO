from model.articulateBody import *

class LeggedRobot2D(ArticulateSystem):
    def __init__(self, params):
        root = FreeBase2D("Base", params["torM"], params["torI"])
        super().__init__(root)

        self.b1 = root.addChild(
            Link2D.Rot, name = "Hthigh",
            la = 0, lb = params["legL1"], lc = params["legLc1"],
            M = params["legL1"], I = params["legL2"]
        )# back leg 1

        self.b2 = root.addChild(
            Link2D.Rot, name = "Hshank",
            la = 0, lb = params["legL1"], lc = params["legLc2"],
            M = params["torM"], I = params["torI"]
        )# back leg 1
        