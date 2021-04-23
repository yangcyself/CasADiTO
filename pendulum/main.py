import sys
import matplotlib.pyplot as plt
sys.path.append(".")

from model.articulateBody import *

class Pendulum(ArticulateSystem):
    def __init__(self):
        root = Base2D.FixedBase("Base")
        super().__init__(root) # this will set self.root

        self.cart = root.addChild(
            Link2D.Prisma, name = "cart",
            la = - 0.5, lb = 0.5, lc = 0,
            M = 1, I = 10000
        )

        self.link1 = self.cart.addChild(
            Link2D.Rot, 0, name = "link1",
            la = 0, lb = 1, lc = 0.5,
            M = 1, I = 1/12
        )

        self.link2 = self.link1.addChild(
            Link2D.Rot, 1, name = "link2",
            la = 0, lb = 1, lc = 0.5,
            M = 1, I = 1/12
        )

        self.link3 = self.link2.addChild(
            Link2D.Rot, 1, name = "link3",
            la = 0, lb = 1, lc = 0.5,
            M = 1, I = 1/12
        )

        self.link4 = self.link3.addChild(
            Link2D.Rot, 1, name = "link3",
            la = 0, lb = 1, lc = 0.5,
            M = 1, I = 1/12
        )
        self.links = [self.link1, self.link2, self.link3, self.link4]

    @property
    def dyn_func(self):
        # self.D ddq + self.Cg = 0
        ddq = ca.mldivide(self.D ,-self.Cg)
        return ca.Function("dyn", [self.x], [ddq], ["x"], ["ddq"])

    def visulize(self, x):
        import inspect
        print(inspect.getargspec(self.cart.visPoints))
        x = x[:self.dim]

        linkLine = np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        plt.plot(linkLine[:,0], linkLine[:,1], marker = 'o', markersize = 5)


import numpy as np
pend = Pendulum()
x0 = np.array([0.,0,0,0,0] * 2).reshape(10,1)

dyn = pend.dyn_func

for i in range(0):
    ddq = dyn(x0)
    x0 += ca.vertcat(x0[5:], ddq).full() * 0.001

plt.figure()

pend.visulize(x0)
plt.show()
