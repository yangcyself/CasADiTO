import casadi as ca
from optGen import optGen
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
        self.nc = nc
        self.r = ca.SX.sym("r", nc)
        self.pc = [ca.SX.sym("pc%d", 3) for i in range(self.nc)]
        self.Q = ca.SX.sym("Q",3,3)
    
        self.x = ca.SX.sym("x", 3)
        self.dx = ca.SX.sym("dx", 3)
        self.pa = [ca.SX.sym("pa%d", 3) for i in range(self.nc)]

        self._w = dx
        self._w0 = ca.DM([0,0,0])
        self._lbw = ca.DM([-1,-1,-1]) # arbitrary
        self._ubw = ca.DM([1,1,1])

        self._J = dx.T @ self.Q @ dx
        self._g = ca.vertcat(*[self._lenCons(self.x, self.dx, pa ,pc) -r  for r,pc,pa in zip(self.r, self.pc, self.pa)])
        self._lbg = ca.DM([-ca.inf]*self.nc)
        self._ubg = ca.DM([0]*self.nc)
        
    def T_WB(self, x):
        """The transition matrix from body frame to world frame
        """
        r = x[2]
        s,c = ca.sin(r), ca.cos(r)
        return ca.DM([[c, -s, x[0]],
                      [s,  c, x[1]],
                      [0,  0, 1]])

    def _lenCons(self, x, dx, pa, pc):
        """Return the length from pa to pc in new body state

        Args:
            dx (x,y,r): the new body state in the body frame
            pa (x,y): the position of the end of the rop in world frame
            pc (x,y): the position of the contact point, in the body frame
        """
        pa_B = ca.inv(T_WB(x))@ca.vertcat(pa, 1)
        pc_B = T_WB(dx)@ca.vertcat(pc, 1)
        return ca.norm_2(pa_B - pc_B)

    def dynamic(self):
        """Return the opt problem for the dynamics
        """




