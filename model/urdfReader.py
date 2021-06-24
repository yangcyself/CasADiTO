# I modified "collada/material.py", line 600, in load 
#   inorder to run the urdfReader for urdf with poor dae info
# I removed float conversion in "urdfpy/urdf.py" line 2306

from urdfpy import URDF
import casadi as ca
import numpy as np
import sympy as sp

# TODO Move it into util
def list2ca(a):
    return ca.vertcat(*[ca.horzcat(*[c for c in r]) for r in a])

"""
Read the urdf and export the forward kinematics as casadi functions
"""
class urdfReader:
    def __init__(self, filepath):
        self.robot_urdf = URDF.load(filepath)
        joint_sp_sym = [sp.symbols(j.name) for j in self.robot_urdf.joints if j.joint_type!="fixed"] # TODO: assume here each joint is only 1dim
        fk = self.robot_urdf.link_fk(cfg = {k.name:v for k,v in zip(self.robot_urdf.joints, joint_sp_sym)})
        forwardKin = { # the forward kinematic functions of lambdify
            l.name: sp.lambdify(joint_sp_sym, fk[l], {'sin': ca.sin, 'cos': ca.cos})
            for l in self.robot_urdf.links
        }
        joint_ca_sym = [ca.SX.sym(j.name) for j in self.robot_urdf.joints if j.joint_type!="fixed"]
        self._dim = len(joint_ca_sym)

        self.forwardKin_cfg = { # the functions passing variable separately
            k: ca.Function(k, joint_ca_sym, [list2ca(v(*joint_ca_sym))],
                [n.name()for n in joint_ca_sym], ["R"] )
            for k,v in forwardKin.items()
        }

        self.forwardKin = {  # the functions passing variable as a vecter
            k: ca.Function(k, [ca.vertcat(*joint_ca_sym)], [list2ca(v(*joint_ca_sym))])
            for k,v in forwardKin.items()
        }

    @property
    def joints(self):
        return self.robot_urdf.joints
        
    @property
    def links(self):
        return self.robot_urdf.links
    
    @property
    def dim(self):
        return self._dim
    
if __name__ == "__main__":

    urdfmodel = urdfReader('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf')

    for joint in urdfmodel.joints:
        print('{} connects {} to {}'.format(
            joint.name, joint.parent, joint.child
        ))

    print("inertia", urdfmodel.links[3].inertial.inertia)

    print(urdfmodel.links[3].name)
    print(urdfmodel.forwardKin_cfg[urdfmodel.links[3].name]( 0,0,0,0,0,0,0,0,0,0,0,0))
    print(urdfmodel.forwardKin[urdfmodel.links[3].name]( ca.DM.rand(urdfmodel.dim)))
