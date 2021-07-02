# I modified "collada/material.py", line 600, in load 
#   inorder to run the urdfReader for urdf with poor dae info
# I removed float conversion in "urdfpy/urdf.py" line 2306

# I made several changes to the urdf_reader: please use the modified version: https://github.com/yangcyself/urdfpy.git
from urdfpy import URDF
import casadi as ca
import numpy as np
import sympy as sp
from utils.caUtil import list2ca

"""
Read the urdf and export the forward kinematics as casadi functions
"""
class urdfReader:
    def __init__(self, filepath):
        self.robot_urdf = URDF.load(filepath)
        joint_sp_sym = [sp.symbols(j.name) for j in self.robot_urdf.joints if j.joint_type!="fixed"] # TODO: assume here each joint is only 1dim
        joint_sp_name = [j.name for j in self.robot_urdf.joints if j.joint_type!="fixed"]
        fk = self.robot_urdf.link_fk(cfg = {k:v for k,v in zip(joint_sp_name, joint_sp_sym)})
        forwardKin = { # the forward kinematic functions of lambdify
            l.name: sp.lambdify(joint_sp_sym, fk[l], {'sin': ca.sin, 'cos': ca.cos})
            for l in self.robot_urdf.links
        }
        joint_ca_name_sym = [(j.name, ca.SX.sym(j.name)) for j in self.robot_urdf.joints if j.joint_type!="fixed"]
        self._joint_ca_sym = [a[1] for a in joint_ca_name_sym]
        self._joint_ca_map = {k:v for k,v in joint_ca_name_sym}
        self._dim = len(self._joint_ca_sym)

        self._fk_dict = {
            k: list2ca(v(*self._joint_ca_sym))
            for k,v in forwardKin.items()
        }

    @property
    def joints(self):
        return self.robot_urdf.joints
        
    @property
    def links(self):
        return self.robot_urdf.links
    
    @property
    def linkDict(self):
        d = {}
        def defaultItemDictFac(n):
            iner = self.robot_urdf.link_map[n].inertial
            return {"childs":[], "parent":None, "joint":None, "fixed":None, # build a default dict
            "inertia":iner.inertia, "mass":iner.mass, "origin": iner.origin} 

        for joint in self.joints:
            parentdict = d.get(joint.parent, defaultItemDictFac(joint.parent))
            parentdict["childs"].append(joint.child)
            d[joint.parent] = parentdict

            childdict = d.get(joint.child, defaultItemDictFac(joint.child))
            childdict["parent"]=joint.parent
            childdict["joint"] =joint.name
            childdict["fixed"] =joint.joint_type=="fixed"
            d[joint.child] = childdict
        return d

    @property
    def dim(self):
        return self._dim

    def getFk(self, link_name, joint_arg_names):
        """return a function for forward kinematics of a link
        """
        f = ca.Function("%s_fk"%link_name, [ca.vertcat(*[ self._joint_ca_map[n] for n in joint_arg_names])],
                [self._fk_dict[link_name]])
        assert not f.has_free(), "get FK should not have free variabels"
        return f

    
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
