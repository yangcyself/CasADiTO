
import yaml
from model.leggedRobotX import LeggedRobotX as Model1
from model.leggedRobotX_bak import LeggedRobotX as Model2
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


model1 = Model1.fromYaml_SX("data/robotConfigs/JYminiLitev2.yaml")
model2 = Model2.fromYaml_SX("data/robotConfigs/JYminiLitev2.yaml")
mode1p = [ p for p in model1.params.values() if isinstance(p, ca.SX)]
print("mode1p:", [ k for k,p in model1.params.items() if isinstance(p, ca.SX)])
mode2p = [ p for p in model2.params.values() if isinstance(p, ca.SX)]
print("mode2p:", [ k for k,p in model2.params.items() if isinstance(p, ca.SX)])


t12_9 = np.zeros(9)
t12_9[3] = t12_9[6] = ca.pi/2
t12_18 = np.zeros(18)
t12_18[3] = t12_18[6] = ca.pi/2



x_val = ca.DM([0,1,0, 0, -np.math.pi*5/6,np.math.pi*2/3, 0, -np.math.pi*5/6,np.math.pi*2/3 ]+[0]*9)
# def x_f(i):
#     add = ca.DM.zeros(18)
#     add[4] = i/100
#     return x_val + add

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))

# def animate(i):
#     x = x_f(i)

#     ax1.clear()
#     vis1 = model1.visulize(x,ax1)
#     ax1.set_xlim(-0.5,1.5)
#     ax1.set_ylim(-0.5,1.5)


#     ax2.clear()
#     vis2 = model2.visulize(x+t12_18,ax2)
#     ax2.set_xlim(-0.5,1.5)
#     ax2.set_ylim(-0.5,1.5)


#     ax3.clear()
#     legl1, legr1 = model1.visulizeLocal(x,ax3)
#     ax3.set_xlim(-0.3,0.15)
#     ax3.set_ylim(-0.4,0.05)


#     ax4.clear()
#     legl2, legr2 = model2.visulizeLocal(x+t12_18,ax4)
#     ax4.set_xlim(-0.3,0.15)
#     ax4.set_ylim(-0.4,0.05)


#     return vis1,legl1, legr1,vis2, legl2, legr2
#     # return linesol,til

# ani = animation.FuncAnimation(
#     fig, animate, interval=25, blit=True, save_count=50)

# plt.show()

x_rand = ca.DM.rand(18)
x_rand[9:12] *= 0

print("COMPARE TOTAL KE and PE")
KEfunc1 = ca.Function('f1', [model1.x], [model1.root.KE])
KEfunc2 = ca.Function('f2', [model2.x], [model2.root.KE])
PEfunc1 = ca.Function('f1', [model1.x], [model1.root.PE])
PEfunc2 = ca.Function('f2', [model2.x], [model2.root.PE])
# print(KEfunc1(x_rand))
# print(KEfunc2(x_rand+t12_18))
# print(PEfunc1(x_rand))
# print(PEfunc2(x_rand+t12_18))


print("COMPARE LOCAL L2 KE and PE")
KElocall2func1 = ca.Function('f1', [model1.x] ,[model1.l2.bdy.KE])
KElocall2func2 = ca.Function('f2', [model2.x] ,[model2.l2.KE])
# print(KElocall2func1(x_rand))
# print(KElocall2func2(x_rand+t12_18))

print("COMPARE HIP KE")
KEhipfunc1 = ca.Function('f1', [model1.x] ,[model1.lhip.KE])
KEhipfunc2 = ca.Function('f2', [model2.x] ,[model2.lhip.KE])
# print(KEhipfunc1(x_rand))
# print(KEhipfunc2(x_rand+t12_18))

mode1p = [ p for p in model1.params.values() if isinstance(p, ca.SX)]
print("mode1p:", [ k for k,p in model1.params.items() if isinstance(p, ca.SX)])
mode2p = [ p for p in model2.params.values() if isinstance(p, ca.SX)]
print("mode2p:", [ k for k,p in model2.params.items() if isinstance(p, ca.SX)])
KEhipJac1 = ca.Function('f1', [model1.x] ,
            [ca.jacobian(model1.root.KE, ca.vertcat(*mode1p))])
KEhipJac2 = ca.Function('f2', [model2.x] ,
            [ca.jacobian(model2.root.KE, ca.vertcat(*mode2p))])

print(KEhipJac1(x_rand))
print(KEhipJac2(x_rand+t12_18))

print("COMPARE THE VELOCITY")
velthighfunc1 = ca.Function('f1', [model1.x] ,[model1.l1.bdy.Mdp, model1.l1.Mdp])
velthighfunc2 = ca.Function('f2', [model2.x] ,[model2.l1.Mdp[:2],  model2.lhip._Mdp_perp(model2.l1)])
v1_0, v1_1 = velthighfunc1(x_rand)
v2_1, v2_2 = velthighfunc2(x_rand+t12_18)
print(v1_0, v1_1)
print(v2_1, v2_2)
print(ca.dot(v1_1,v1_1))
print(ca.dot(v2_1,v2_1)+ca.dot(v2_2,v2_2))

print("THE FP")
FPfunc = ca.Function('f1', [model1.x] ,[model1.l1.Fp])
fp = FPfunc(x_rand)
print(fp)
print(np.linalg.det(fp[:3,:3]))
