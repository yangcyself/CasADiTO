import sys
sys.path.append(".")

import casadi as ca
from optGen.optGen import optGen

class HeavyRopeLoad(optGen):

    def __init__(self, nc):
        """The HeavyLoad model is a discrete static model, whose dynamics solved by ipopt
            Assume it is a box on the groud, draged via a rope
                governed mostly by friction, thus momentum is negligible.
            state X: [px, py, r]
            trajsition: X_k+1 = f(X, ...)

        Args:
            pc ([[x,y]]): The position of rope connections. 
                In the load's body frame.
            r ([double]): the length of all ropes
            Q: The matrix representing the frictional effects. 
                The model min || x-x_ ||Q
        """
        super().__init__()
        self.nc = nc
        self.x = ca.SX.sym("xold", 3)
        self.Q = ca.SX.sym("Q",3,3)
        self.r = ca.SX.sym("r", nc)
        self.pc = [ca.SX.sym("pc%d", 2) for i in range(self.nc)]
        self.pa = [ca.SX.sym("pa%d", 2) for i in range(self.nc)]

        self.newhyperParam(self.r, name="r")
        self.pc_input = self.newhyperParam(ca.vertcat(*self.pc), name="pc")
        self.newhyperParam(self.Q, name="Q")
        self.newhyperParam(self.x, name="xold")
        self.pa_input = self.newhyperParam(ca.vertcat(*self.pa), name="pa")
    
        self.dx = ca.SX.sym("dx", 3)

        self._w = [self.dx]
        self._w0 = [ca.DM([0,0,0])]
        self._lbw = [ca.DM([-0.1,-0.1,-0.1])]
        self._ubw = [ca.DM([0.1,0.1,0.1])]
        self._J = self.dx.T @ self.Q @ self.dx
        self._g = [ca.vertcat(*[self._lenCons(self.x, self.dx, pa ,pc)**2 -self.r[i]**2
                for i,(pc,pa) in enumerate(zip(self.pc, self.pa))])]
        self._lbg = [ca.DM([-ca.inf]*self.nc)]
        self._ubg = [ca.DM([0]*self.nc)]
        
        x_w = (self.T_WB(self.x)@ca.vertcat(self.dx[:2], 1))[:2]
        x_w = ca.vertcat(x_w, self.dx[2]+self.x[2])
        self._parse.update({
            "newx": lambda: x_w
        })

    @property
    def Jfunc(self):
        return ca.Function("J", [self.dx, self.Q], [self._J], ["dx", "Q"], ["J"])

    @property
    def gfunc(self):
        return ca.Function("g", [self.x, self.dx, self.pc_input, self.pa_input], [self._g], 
                                ["oldx", "dx", "pc_input", "pa_input"], ["g"])

    @property
    def intefunc(self):
        """An integral function for x+dx = newx
        """
        return ca.Function("dyn", [self.x, self.dx], [self.x_w], 
                                ["oldx", "dx"], ["newx"])


    def T_WB(self, x):
        """The transition matrix from body frame to world frame
        """
        r = x[2]
        s,c = ca.sin(r), ca.cos(r)
        return ca.vertcat(ca.horzcat(c, -s, x[0]),
                          ca.horzcat(s,  c, x[1]),
                          ca.horzcat(0,  0, 1))

    def _lenCons(self, x, dx, pa, pc):
        """Return the length from pa to pc in new body state

        Args:
            dx (x,y,r): the new body state in the body frame
            pa (x,y): the position of the end of the rop in world frame
            pc (x,y): the position of the contact point, in the body frame
        """
        pa_B = (ca.inv(self.T_WB(x))@ca.vertcat(pa, 1))[:2]
        pc_B = (self.T_WB(dx)@ca.vertcat(pc, 1))[:2]
        return ca.norm_2(pa_B - pc_B)

    def dynam(self, x, pa):
        """Return the new state x given current state x and pa
            call it after hyper param value are setted
        """
        x,pa = ca.DM(x), ca.DM(pa)
        self.setHyperParamValue({
            "pa": pa, 
            "xold": x
        })
        res = self.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
            "verbose_init":False,
            "verbose" : False ,
            "print_time": False,
            "error_on_fail":True,
            "ipopt":{
                "max_iter" : 2000,
                "print_level": 0
                }
            })
        dx = res["x"]
        return res['newx']
        

    @property
    def pFuncs(self):
        p = ca.horzcat(*[ (self.T_WB(self.x)@ca.vertcat(pc, 1))[:2] 
                for pc in self.pc]).T
        # ["pc"][3] returns the value of the symbol
        p = ca.substitute([p], [self.pc_input], [self._hyperParams["pc"][3]] )[0]
        return ca.Function("pcFunc", [self.x], [p])

if __name__ == "__main__":
    import numpy as np
    opt = HeavyRopeLoad(nc=3)

    opt.cppGen("codogs/cpp/generated", expand=True, parseFuncs=[
        ("newx", lambda sol: sol["newx"])],
        cmakeOpt={'libName': 'hrl', 'cxxflag':'"-O3 -fPIC"'})

    # opt.setHyperParamValue({
    #     "r": ca.DM([1]),
    #     "pc": ca.DM([[-1,1]]), 
    #     "pa": ca.DM([[1,0]]), 
    #     "Q": np.diag([1,1,1]), 
    #     "xold": ca.DM([0,0,0])
    # })

    # res = opt.solve(options=
    #         {"calc_f" : True,
    #         "calc_g" : True,
    #         "calc_lam_x" : True,
    #         "calc_multipliers" : True,
    #         "expand" : True,
    #             "verbose_init":True,
    #             # "jac_g": gjacFunc
    #         "ipopt":{
    #             "max_iter" : 2000, # unkown option
    #             }
    #         })

    # x = [0,0,0]
    # p = np.array([1,0])
    # x = opt.dynam(x, [p])
    # pfunc = opt.pFuncs
    # print(pfunc(x))
    # print(ca.norm_2(pfunc(x).full() - p))

    # def T_WB(x):
    #     """The transition matrix from body frame to world frame
    #     """
    #     r = x[2]
    #     s,c = ca.sin(r), ca.cos(r)
    #     return ca.vertcat(ca.horzcat(c, -s, x[0]),
    #                       ca.horzcat(s,  c, x[1]),
    #                       ca.horzcat(0,  0, 1))

    # pa = [0.1,4]
    # x = [-2, 3, -ca.pi/2]
    # print((ca.inv(T_WB(x))@ca.vertcat(pa, 1))[:2])
    # x = opt.dynam(x, [pa])
    # print(x)