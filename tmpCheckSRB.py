from model.singleRigidBody import singleRigidBody as model
import casadi as ca
import numpy as np
import mathUtil
m = model({"m":10, "I": 1}, nc = 0)

print(m.LTH().size())

dynf = m.Dyn()

# print(dynf(ca.DM([0,0,0, ca.pi/2,0,0, 0,0,0, 0,0,0]), 
#             ca.DM([0,0,0, 0,0,0,
#                    1,0,0, 0,0,1])))

axis = ca.DM([1,1,1])
axis = axis/ca.norm_2(axis)

T = 1
n = 1000
dt = T/n
x = ca.DM([0,0,0, 0,0,0, 0,0,0])
x = ca.vertcat(x,
    axis * ca.pi/3*2
)

for i in range(n):
    x +=  dynf( x, ca.SX([])) * dt
print(ca.DM(x).full().T)

print(ca.DM(mathUtil.Rot(ca.pi/3*2, axis )).full())
print(mathUtil.ZYXRot(x[3:6] ))