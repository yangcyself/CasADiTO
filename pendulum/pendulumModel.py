import sys
import matplotlib.pyplot as plt
sys.path.append(".")

from model.articulateBody import *
import matplotlib.animation as animation


class Pendulum(ArticulateSystem):
    def __init__(self, symbolWeight = False):
        root = Base2D.FixedBase("Base")
        super().__init__(root) # this will set self.root

        Masses = [ca.SX.sym("M%d"%i) for i in range(4)] if symbolWeight else [1,1,1,1]
        self._syms = Masses if symbolWeight else []

        self.cart = root.addChild(
            Link2D.Prisma, name = "cart",
            la = - 0.5, lb = 0.5, lc = 0,
            M = Masses[0], I = 0
        )
        link1 = self.cart.addChild(
            Link2D.Rot, 0, name = "link1",
            la = 0, lb = 1, lc = 0.5,
            M = Masses[1], I = 1/12
        )
        self.links = [link1]

        for i in range(2):
            self.links.append(
                self.links[-1].addChild(
                    Link2D.Rot, 1, name = "link%d"%(i+2),
                    la = 0, lb = 1, lc = 0.5,
                    M = Masses[2+i], I = 1/12
                )
            )

    @property
    def B(self):
        return ca.jacobian(self.root.x, self.cart._q)

    @property
    def dyn_func(self):
        # self.D ddq + self.Cg = 0
        # ddq = ca.mldivide(self.D ,-self.Cg)
        u = ca.SX.sym("u", 1)
        ddq = ca.inv(self.D) @ (- self.Cg + self.B @ u)
        # ddq = - ca.inv(self.D) @ (self.C @ self.root.dx + self.G)
        return ca.Function("dyn", [self.x, u], [ddq], ["x", "u"], ["ddq"])

    def visulize(self, x):
        x = x[:self.dim]
        try:
            linkLine = self.linkLine_func_cache(x)
            # np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        except AttributeError:
            xsym = ca.SX.sym("xsym", len(x))
            linkLine = ca.vertcat(*[c.visPoints(self.root.x, xsym) for c in [self.cart] + self.links])
            self.linkLine_func_cache = ca.Function("linkLine_func", [xsym], [linkLine], ["xsym"], ["linkLine"])
            linkLine = self.linkLine_func_cache(x)

        # linkLine = np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        line, = plt.plot(linkLine[:,0], linkLine[:,1], marker = 'o', markersize = 5)
        return line


    def CoMposValue(self, x):
        """
        return the position of CoM's of each link
        """
        x = x[:self.dim]
        try:
            return self.CoMposValue_func_cache(x)
            # np.concatenate([c.visPoints(self.root.x, x) for c in [self.cart] + self.links])
        except AttributeError:
            CoMpos = ca.vertcat(*[c.points['c'].T for c in [self.cart] + self.links])
            self.CoMposValue_func_cache = ca.Function("linkLine_func", [self.root.x], [CoMpos], ["x"], ["linkLine"])
            return self.CoMposValue_func_cache(x)


if __name__ == "__main__":

    import numpy as np
    pend = Pendulum()
    dim = pend.dim
    x0 = np.zeros((dim*2, 1))

    dyn = pend.dyn_func


    print(pend.CoMposValue(x0)[1,1])

    fig, ax = plt.subplots()

    Xs = []

    # line, = ax.plot(robotLines[0][:,0], robotLines[0][:,1])
    EoM = pend.EOM_func
    for i in range(50000):
        ddq = dyn(x0, np.random.random(1) - np.random.random(1))
        x0 = x0 + ca.vertcat(x0[dim:], ddq).full() * 0.001
        Xs.append(x0)
        # print(np.mean(pend.CoMposValue(x0), axis = 0))
        # print(EoM(x0[:5], x0[5:], ddq, np.zeros(5) ))

    def animate(i):
        i = (i*10)%len(Xs)
        # line.set_xdata(robotLines[i][:,0])  # update the data.
        # line.set_ydata(robotLines[i][:,1])  # update the data.
        ax.clear()
        line = pend.visulize(Xs[i])
        # ax.set_xlim(-5+Xs[i][0],5+Xs[i][0])
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, save_count=5000)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()
