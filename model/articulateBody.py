import casadi as ca

class Body:
    def __init__(self, name, freeD):
        self.name = name
        self.freeD = freeD
        self._q = ca.SX.sym("%s_q"%name,freeD)
        self._dq = ca.SX.sym("%s_dq"%name,freeD)
        self.child = []
        self.parent = None

    def fix(self, q = None, dq = None):
        q = ca.DM.zeros(self._q.size()) if q is None else ca.DM(q)
        dq = ca.DM.zeros(self._dq.size()) if dq is None else ca.DM(dq)
        if(not (q.size() == self._q.size() and dq.size()==self._dq.size()) ):
            raise ValueError("The size of q and dq should be equal to the original")
        self._q = q
        self._dq = dq
        self.freeD = 0


    def _KE(self):
        raise NotImplementedError

    def _PE(self):
        raise NotImplementedError

    @property
    def q(self):
        q = ca.DM([]) if not self.freeD else self._q
        return (q if self.parent is None 
                    else ca.vertcat(self.parent.q,  q))

    @property
    def dq(self):
        dq = ca.DM([]) if not self.freeD else self._dq
        return (dq if self.parent is None 
                    else ca.vertcat(self.parent.dq,  dq))

    @property
    def x(self):
        """The pos of the whole system, including all the children"""
        q = ca.DM([]) if not self.freeD else self._q
        return ca.vertcat(q, *[c.x for c in self.child])

    @property
    def dx(self):
        """The vel of the whole system, including all the children"""
        dq = ca.DM([]) if not self.freeD else self._dq
        return ca.vertcat(dq, *[c.dx for c in self.child])

    @property
    def KE(self):
        return ca.sum1(ca.vertcat(self._KE(), *[c.KE for c in self.child]))

    @property
    def PE(self):
        return ca.sum1(ca.vertcat(self._PE(), *[c.PE for c in self.child]))

class Body2D(Body):
    defaultG = ca.vertcat(0, -9.81)

    def __init__(self, name, freeD, M, I,  Fp = None, g = None):
        """[summary]

        Args:
            name (string): The name of the body
            freeD (int): dim of freedom
            Fp (SX[3x1]): The position of the parent frame, defaut to DX(0,0,0)
            Fdp (SX[3x1]): The velocity of the parent frame, defaut to DX(0,0,0)
            g ([type], optional): [description]. Defaults to Body2D.defaultG.
        """
        super().__init__(name, freeD)            
        self.g = Body2D.defaultG if g is None else g #2D: the gravtition accleration
        self.Fp = ca.vertcat(0,0,0) if Fp is None else Fp
        self.M = M
        self.I = I

    def _Mp(self):
        raise NotImplementedError

    def _Mdp(self):
        if(self.q.size(1)==0):
            return ca.DM.zeros(3)
        return ca.jtimes(self.Mp, self.q, self.dq)

    @property
    def Mp(self):
        """3D verctor: px, py, pr
            Note: px, py is the CoM position"""
        return self._Mp()

    @property
    def Mdp(self):
        """3D verctor: dpx, dpy, dpr
            Note: px, py is the CoM position"""
        return self._Mdp()
    
    def _Bp(self):
        raise NotImplementedError

    @property
    def Bp(self):
        """3D verctor: px, py, pr of the Body's Frame"""
        return self._Bp()

    def _KE(self):
        return (0.5*self.M * ca.dot(self.Mdp[:2], self.Mdp[:2]) # velocity
                + 0.5*self.I*self.Mdp[2]**2) # rotation
    
    def _PE(self):
        return - ca.dot(self.g, self.Mp[:2])


    def addChild(self, ChildType, **kwargs):
        """add a Child `ChildType(**kwargs)`

        Args:
            ChildType ([Body2D]): A object constructor that returns a child Body2D
        """
        c = ChildType(**kwargs)
        if(not isinstance(c,Body2D)):
            raise TypeError("Body2D must add Children of type Body2D, but get %s"%type(c).__name__)
        self.child.append(c)
        c.parent = self
        return c

    def _visFunc(self, xr):
        """ ca.Function: build an function for visPoints"""
        raise NotImplementedError

    def visPoints(self, xr, xv):
        """return the points for visulization. The xr is often the state of root, 

        Args:
            xr (ca.SX): root state the symbol of the xv passed in.
            xv (narray/DM): the value of the xr

        Returns:
            [ptrs][nx2]: list of points for used for visulization
        """
        try:
            return self._visFunc_cache(xv).full()
        except AttributeError:
            self._visFunc_cache = self._visFunc(xr)
            return self._visFunc_cache(xv).full()


class Base2D(Body2D):
    def __init__(self, name, M, I, g = None):
        super().__init__(name, 3, M, I,  g = g)

        
    def _Mp(self):
        return self._q
    
    def _Bp(self):
        return self._q

    def addChild(self, ChildType, **kwargs):
        return super().addChild(ChildType, Fp= self.Bp, **kwargs)
    
    @staticmethod
    def Freebase(name, M, I, g = None):
        return Base2D(M, I, g)
    
    @staticmethod
    def FixedBase(name, q = None, dq = None, g = None):
        b = Base2D(name, 0,0,g)
        b.fix(q, dq)
        return b
        
    def _visFunc(self, xr):
        return ca.Function("basePoints", [xr], [self._Mp[:2].T], "x", "ps" )

class Link2D(Body2D):

    @staticmethod
    def Rot(name, Fp, lc, la, lb, M, I, g = None):
        Ax = ca.vertcat(0,0,1)
        return Link2D(name, Fp, lc, la, lb, M, I, Ax, g)

    @staticmethod
    def Prisma(name, Fp, lc, la, lb, M, I, g = None):
        Ax = ca.vertcat(1,0,0)
        return Link2D(name, Fp, lc, la, lb, M, I, Ax, g)

    def __init__(self, name, Fp, lc, la, lb, M, I, Ax, g = None):
        """Link2D: 2D link. The pos direction of the direction of it's angle

        Args:
            g ([SX[2x1]], optional): Gravity accleration. Defaults to (0, -9.81).
            lc (double): the length from the frame to the CoM
            la (double): the length from the frame to one of the end point 
            lb (double): the length from the frame to the other end point
            M (double): the mass
            I (double): the moment
            Ax (DX[3x1]):  Ax * _q is the of B frame in F frame.
                e.g. Ax = (0,0,1) means this is a rotation link
        """
        
        super().__init__(name, 1, M, I, Fp = Fp, g=g)
        self.lc = lc
        self.la = la
        self.lb = lb
        self.Ax = Ax * self._q

        self.points = {
            "a": self.move_X_p(self.la),
            "b": self.move_X_p(self.lb),
            "f": self.Fp[:2],
            "c": self.move_X_p(self.lc) # the center of mass
        }

    def move_X_p(self, l):
        """return the position. The pos is (l,0,0) in Bframe
        Args:
            l (float): length
        """
        rB = self.Bp[2]
        return self.Bp + l * ca.vertcat(ca.cos(rB), ca.sin(rB), 0)

    def _Bp(self):
        r = self.Fp[2]
        cr = ca.cos(r)
        sr = ca.sin(r)
        return ca.vertcat(
           ca.horzcat( cr, -sr, 0), 
           ca.horzcat( sr, cr, 0),
           ca.horzcat( 0, 0, 1) 
        )@ self.Ax + self.Fp

    def _Mp(self):
        return self.move_X_p(self.lc)

    def addChild(self, ChildType, lp, **kwargs):
        """add a Child that at the lp of the link

        Args:
            ChildType ([type]): [description]
            lp (float): the pos of the child's frame

        Returns:
            [type]: [description]
        """
        rB = self.Bp[2]
        return super().addChild(ChildType, Fp = self.move_X_p(lp), **kwargs)

    def _visFunc(self, xr):
        return ca.Function("%sPoints"%self.name, [xr], 
            [ca.vertcat(self.points["a"][:2].T,
                        self.points["b"][:2].T)], ["x"], ["ps"] )
        

class ArticulateSystem:
    def __init__(self, root):
        """The base class of Articulate System

        Args:
            root (Body): The Root Body of the system (have no parents)
        """
        self.root = root
    
    @property
    def x(self):
        """the q and dq of the system"""
        return ca.vertcat(self.root.x, self.root.dx)
    
    @property
    def L(self):
        """the lagrange"""
        return self.root.KE - self.root.PE

    @property
    def dim(self):
        return self.root.x.size(1)

    @property
    def EOM_func(self):
        """The Equation of Motion (x, dx, ddx, Q)
        """
        L = self.L
        ddq = ca.SX.sym("ddq",7)
        dq = self.root.dx
        q = self.root.x
        Q = ca.SX.sym("Q",7)
        EOM = (ca.jtimes(ca.jacobian(L,dq).T, dq, ddq) 
            + ca.jtimes(ca.jacobian(L,dq).T, q, dq) 
            - ca.jacobian(L,q).T - Q) # equation of motion
        EOM = ca.simplify(EOM)
        return ca.Function("EOM_func", [q, dq, ddq, Q], [EOM], 
                    ["q", "dq", "ddq", "Q"], ["EOM"])

    @property
    def D(self):
        KE = self.root.KE
        dq = self.root.dx
        return ca.simplify(ca.jacobian(ca.jacobian(KE,dq)    ,dq))

    @property
    def C(self):
        L = self.L
        dq = self.root.dx
        q = self.root.x
        return ca.simplify(ca.jacobian(ca.jacobian(L,dq).T, q))

    @property
    def G(self):
        PE = self.root.PE
        q = self.root.x
        return ca.simplify(ca.jacobian(PE,q)).T
    
    @property
    def B(self):
        raise NotImplementedError

    @property
    def Cg(self):
        """C+G, models the nonlinearity and gravity together
        """
        L = self.L
        dq = self.root.dx
        q = self.root.x
        return ca.jtimes(ca.jacobian(L,dq).T, q, dq) - ca.jacobian(L,q).T

    
if __name__ == "__main__":
    Body2D.defaultG = ca.vertcat(0,-9.888)
    fb = Base2D.FreeBase("fb", 2, 0.5)
    print(fb.g)

    print(fb.KE)
    print(fb.PE)

