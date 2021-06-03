"""
Implement a single rigid body model. The derivation of dynamics does not involve lagrange
Parameterized following "Trajectory Optimization for dynamic Aerial Motions of Legged robots" Matthew Chignoli


x = [p, theta, v, omega_B]

\dot{x} = [v, B(theta) omega_B,  1/m f-g, I^{-1}_B(R_b^T tau - omega_B I_B omega_B )]

theta is the Z-Y-X euler angle, and omega_B is the angular velocity of the Body
"""

import yaml
import matplotlib.pyplot as plt
import casadi as ca
import mathUtil

class singleRigidBody:
    def __init__(self, params, nc):
        self.params = params
        self.nc = nc

        # Variables
        self.p = ca.SX.sym("p", 3)
        self.th = ca.SX.sym("Theta", 3)
        self.v = ca.SX.sym("v", 3)
        self.w = ca.SX.sym("Omega_B", 3)
        self.x = ca.vertcat(self.p, self.th, self.v, self.w)

        # Config of the model
        self.m = params["m"]
        self.I = params["I"]
        self.g = params.get("g", ca.DM([0,0,-9.81]))

    @property
    def dim(self):
        return 6

    @property
    def u_dim(self):
        # [p_c, f_c] * nc
        return 6 * self.nc

    @property
    def F_dim(self):
        return 0 # this is because F is the external force to enforcing the constraints

    @property
    def pFuncs(self):
        return {"COM": ca.Function('com', [self.x], [self.p])}

    def LTH(self):
        """the transition matrix from eular theta rate to self.w
        """
        drx = ca.SX.sym("drx")
        dry = ca.SX.sym("dry")
        drz = ca.SX.sym("drz")
        rx = self.th[0]
        ry = self.th[1]
        rz = self.th[2]

        omega = ca.DM.eye(3) @ (ca.vertcat(0,0,drz) 
            + mathUtil.Rot(rz, ca.DM([0,0,1])) @ (ca.vertcat(0,dry,0)
            + mathUtil.Rot(ry, ca.DM([0,1,0])) @ ca.vertcat(drx,0,0)))
        return ca.jacobian(omega, ca.vertcat(drx, dry, drz))

    def Dyn(self):
        u = ca.SX.sym("u", self.u_dim)
        pcfc = [[a for a in ca.vertsplit(t,3)] for t in ca.vertsplit(u, 6)]
        f = ca.sum2(ca.horzcat([0,0,0], *[ fc for pc,fc in pcfc]))
        tau_w = ca.sum2(ca.horzcat([0,0,0], *[ca.cross((pc - self.p), fc) for pc,fc in pcfc]))
        B = ca.inv(self.LTH())
        dx = ca.vertcat( self.v,
                        B @ self.w,
                        f/self.m + self.g, 
                        ca.inv(self.I) @ (mathUtil.ZYXRot(self.th).T @  tau_w - ca.skew(self.w) @ self.I @ self.w)
                        )
        return ca.Function('EOMF', [self.x, u], [dx], ["x", "u"], ['dx'])

    def visulize(self, x, ax = None):
        pass

    @staticmethod
    def fromYaml(yamlFilePath):
        pass