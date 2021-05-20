
from model.articulateBody import *
import yaml


ConfigFile = "data/robotConfigs/JYminiLitev2.yaml"
with open(ConfigFile, 'r') as stream:
    try:
        robotParam = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()
params = {
    "legL1":robotParam["l1"],
    "legL2":robotParam["l2"],
    "legLc1":robotParam["lc1"],
    "legLc2":robotParam["lc2"],
    "torW":robotParam["LW"],
    "torLL":robotParam["LLW"],
    "torLc":robotParam.get("LWc"),
    "legM1":robotParam["m1"],
    "legM2":robotParam["m2"],
    "torM": robotParam["M"],
    "legI1":robotParam["j1xx"],
    "legI2":robotParam["j2xx"],
    "torI": robotParam["Jxx"], 
    "q0Lim": [ca.pi/2 + robotParam["ang0min"], ca.pi/2 + robotParam["ang0max"]], 
    "q1Lim": [-ca.pi/2 - robotParam["ang1max"], -ca.pi/2 - robotParam["ang1min"]],
    "q2Lim": [-robotParam["ang2max"], -robotParam["ang2min"]],
    "dq0Lim": [-robotParam["dang0lim"], robotParam["dang0lim"]],
    "dq1Lim": [-robotParam["dang1lim"], robotParam["dang1lim"]],
    "dq2Lim": [-robotParam["dang2lim"], robotParam["dang2lim"]],
    "tau0lim": robotParam["tau0lim"],
    "tau1lim": robotParam["tau1lim"],
    "tau2lim": robotParam["tau2lim"],
    "G":9.81,
}

def getAleg(name):
    thigh = Link2D.Rot(name = "%sThigh"%name, Fp = ca.vertcat(0,0,0),
        la = 0, lb = params["legL1"], lc = params["legLc1"],
        M = params["legM1"], I = params["legI1"])
    # shank = thigh.addChild(
    #     Link2D.Rot, params["legL1"], name = "%sShank"%name,
    #     la = 0, lb = params["legL2"], lc = params["legLc2"],
    #     M = params["legM2"], I = params["legI2"]
    # )
    return [thigh,]# shank]

llegs1 = getAleg("L")
llegs2 = getAleg("L")

hipx1 = Link2D.Rot("hip",ca.DM([0,0,0]),0,0,0,0,0)
vertLeg1 = hipx1.addChild(planeWrap3D.from2D, 0, bdy = llegs1[0], name="", 
T = ca.DM([[0,1,0],
           [0,0,1],
           [1,0,0]]))


hipx2 = Proj2dRot("", llegs2[0],ca.DM([0,1,0]),ca.DM([0,0,0]))


# Tab = ca.SX([[1,0,0,0],
#              [0,0,-1,0],
#              [0,1,0,0],
#              [0,0,0,1]])

# vertLeg = planeWrap3D(llegs[0],"",Tab)

# print(vertLeg.child[0].name)
# print(vertLeg.x)
# print(llegs[0].x)
# print(vertLeg.KE)
# print(llegs[0].KE)

KEfunc1 = ca.Function('f1', [hipx1.x, hipx1.dx] ,[hipx1.KE])
KEfunc2 = ca.Function('f2', [hipx2.x, hipx2.dx] ,[hipx2.KE])

randx_val = ca.DM.rand(2)
randdx_val = ca.DM.rand(2)
print(KEfunc1(randx_val, randdx_val))
print(KEfunc2(randx_val, randdx_val))

mdpfunc1 = ca.Function('mdp1', [hipx1.x, hipx1.dx], [vertLeg1.Mdp])
omgfunc1 = ca.Function('omg1', [hipx1.x, hipx1.dx], [vertLeg1.omega_b])
rotfunc1 = ca.Function('omg2', [hipx1.x, hipx1.dx], [0.5* vertLeg1.omega_b.T @ vertLeg1.I @ vertLeg1.omega_b])
mdpfunc2 = ca.Function('mdp2', [hipx2.x, hipx2.dx], [hipx2.bdy.Mdp, hipx2._Mdp_perp(hipx2.bdy)])
rotfunc2 = ca.Function('omg2', [hipx2.x, hipx2.dx], [0.5 * hipx2._I_perp(hipx2.bdy) * hipx2.Mdp[2]**2
                                                    ,0.5 * hipx2.bdy.I * hipx2.bdy.Mdp[2]**2])

# mdpfunc2 = ca.Function('mdp2', [hipx2.x, hipx2.dx], [hipx2._Mdp_perp])

a = mdpfunc1(randx_val, randdx_val)
b = omgfunc1(randx_val, randdx_val)
c1,c2 = mdpfunc2(randx_val, randdx_val)
e1 = rotfunc1(randx_val, randdx_val)
e21, e22 = rotfunc2(randx_val, randdx_val)


print(a)
print(b)
print(c1,c2)


print("\n\ncompare rotational")
print(e1)
print(e21 + e22)


print("\n\nCompare transitional")
print(0.5 * vertLeg1.M * ca.dot(a,a))
print(0.5* hipx2.bdy.M* ca.dot(c1[:2],c1[:2]) + 0.5 * hipx2.bdy.M* c2**2)

print("\n\ntotal")
print(e1+0.5* vertLeg1.M * ca.dot(a,a))
print(e21+e22+0.5* hipx2.bdy.M * ca.dot(c1[:2],c1[:2]) + 0.5 * hipx2.bdy.M* c2**2)
print()
print(e21)
print(e22)
print(0.5* hipx2.bdy.M * ca.dot(c1[:2],c1[:2]))
print(0.5 * hipx2.bdy.M* c2**2)
print()
print(e22 + 0.5* hipx2.bdy.M * ca.dot(c1[:2],c1[:2]))
# print(vertLeg.Bdp)

