from urdfpy import URDF
robot = URDF.load('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf')
# robot = URDF.load('/home/ami/git_source/urdfpy/tests/data/ur5/ur5.urdf')




for joint in robot.joints:
    print('{} connects {} to {}'.format(
        joint.name, joint.parent, joint.child
    ))

# print(fk = robot.link_fk(cfg={'shoulder_pan_joint' : 1.0}))