from model.singleRigidBody import singleRigidBody as model
import casadi as ca
import numpy as np
import mathUtil

from mathUtil import solveLinearCons
from optGen.util import caSubsti, caFuncSubsti, substiSX2MX
m = model({"m":10, "I": 1}, nc = 4)

eom = m.EOM_ufdx()

cons = {"x":ca.DM.rand(12)}
cons.update({
    "fc%d"%i: ca.DM.rand(3) for i in range(4) 
})
# cons.update({
#     "fc%i": ca.DM.rand(3) for i in range(4) 
# })
consDyn_ = caFuncSubsti(eom, cons)
res = solveLinearCons(consDyn_, [("dx", np.zeros(12), 1e3)])
print(res)
# print(ca.simplify( res["u"][0]))
exit()

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