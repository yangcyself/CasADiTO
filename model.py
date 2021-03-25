
from casadi import *
import yaml


ConfigFile = "data/robotConfigs/robot1.yaml"
PI = np.math.pi

with open(ConfigFile, 'r') as stream:
    try:
        robotParam = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

# Declare model variables
px = SX.sym('px')
py = SX.sym('py')
th = SX.sym('th')
bq1 = SX.sym('bq1')
bq2 = SX.sym('bq2')
fq1 = SX.sym('fq1')
fq2 = SX.sym('fq2')

dpx = SX.sym('dpx')
dpy = SX.sym('dpy')
dth = SX.sym('dth')
dbq1 = SX.sym('dbq1')
dbq2 = SX.sym('dbq2')
dfq1 = SX.sym('dfq1')
dfq2 = SX.sym('dfq2')

params = {
    "legL1":robotParam["l1"],
    "legL2":robotParam["l2"],
    "legLc1":robotParam["lc1"],
    "legLc2":robotParam["lc2"],
    "torL":robotParam["L"],
    "legM1":robotParam["m1"],
    "legM2":robotParam["m2"],
    "torM":robotParam["M"],
    "legI1":robotParam["j1"],
    "legI2":robotParam["j2"],
    "torI":robotParam["J"],
    "q1Lim": [-PI/2 - robotParam["ang1max"], -PI/2 - robotParam["ang1min"]],
    "q2Lim": [-robotParam["ang2max"], -robotParam["ang2min"]],
    "dq1Lim": [-robotParam["dang1lim"], robotParam["dang1lim"]],
    "dq2Lim": [-robotParam["dang2lim"], robotParam["dang2lim"]],
    "G":9.81,
}

q = vertcat(px, py, th, bq1, bq2, fq1, fq2)
dq = vertcat(dpx, dpy, dth, dbq1, dbq2, dfq1, dfq2)
x = vertcat(q, dq)
u = SX.sym('u',4)

# Build Model using the euler methods

pMtor = vertcat(
    px + params["torL"]/2 * cos(th),
    py + params["torL"]/2 * sin(th),
)#torsor center of position

pMbLeg1 = vertcat(
    px + params["legLc1"] * cos(th+bq1),
    py + params["legLc1"] * sin(th+bq1),
)#back leg thigh center of position

pMbLeg2 = vertcat(
    px + params["legL1"] * cos(th+bq1) + params["legLc2"] * cos(th+bq1+bq2),
    py + params["legL1"] * sin(th+bq1) + params["legLc1"] * sin(th+bq1+bq2),
)#back leg center of position

pMfLeg1 = vertcat(
    px + params["torL"] * cos(th) + params["legLc1"] * cos(th+fq1),
    py + params["torL"] * sin(th) + params["legLc1"] * sin(th+fq1),
)#front leg thigh center of position

pMfLeg2 = vertcat(
    px + params["torL"] * cos(th) + params["legL1"] * cos(th+fq1) + params["legLc2"] * cos(th+fq1+fq2),
    py + params["torL"] * sin(th) + params["legL1"] * sin(th+fq1) + params["legLc2"] * sin(th+fq1+fq2),
)#front leg center of position

prTor = vertcat(px,py) #rear of the torso
phTor = vertcat(px+params["torL"] * cos(th),py+params["torL"] * sin(th)) #head of the torso
phbLeg1 = vertcat(px + params["legL1"] * cos(th+bq1),
    py + params["legL1"] * sin(th+bq1)) #rear of the back leg thigh
phbLeg2 = vertcat(px + params["legL1"] * cos(th+bq1) + params["legL2"] * cos(th+bq1+bq2),
    py + params["legL1"] * sin(th+bq1) + params["legL2"] * sin(th+bq1+bq2),) #rear of the back leg 
phfLeg1 = vertcat(px + params["torL"] * cos(th) + params["legL1"] * cos(th+fq1),
    py + params["torL"] * sin(th) + params["legL1"] * sin(th+fq1))#front leg thigh center of position
phfLeg2 = vertcat(px + params["torL"] * cos(th) + params["legL1"] * cos(th+fq1) + params["legL2"] * cos(th+fq1+fq2),
    py + params["torL"] * sin(th) + params["legL1"] * sin(th+fq1) + params["legL2"] * sin(th+fq1+fq2))#front leg

#jtimes: Jacobian-times-vector
dpMtor = jtimes(pMtor, q, dq)
dpMbLeg1 = jtimes(pMbLeg1, q, dq)
dpMbLeg2 = jtimes(pMbLeg2, q, dq)
dpMfLeg1 = jtimes(pMfLeg1, q, dq)
dpMfLeg2 = jtimes(pMfLeg2, q, dq)

pComFuncs = {
"pMtor" : Function("pMtor", [x], [pMtor]),
"pMbLeg1" : Function("pMbLeg1", [x], [pMbLeg1]),
"pMbLeg2" : Function("pMbLeg2", [x], [pMbLeg2]),
"pMfLeg1" : Function("pMfLeg1", [x], [pMfLeg1]),
"pMfLeg2" : Function("pMfLeg2", [x], [pMfLeg2])
}

dpComFuncs = {
"dpMtor" : Function("dpMtor", [x], [dpMtor]),
"dpMbLeg1" : Function("dpMbLeg1", [x], [dpMbLeg1]),
"dpMbLeg2" : Function("dpMbLeg2", [x], [dpMbLeg2]),
"dpMfLeg1" : Function("dpMfLeg1", [x], [dpMfLeg1]),
"dpMfLeg2" : Function("dpMfLeg2", [x], [dpMfLeg2])
}

pFuncs = {
"prTor" : Function("prTor", [x], [prTor]),
"phTor" : Function("phTor", [x], [phTor]),
"phbLeg1" : Function("phbLeg1", [x], [phbLeg1]),
"phbLeg2" : Function("phbLeg2", [x], [phbLeg2]),
"phfLeg1" : Function("phfLeg1", [x], [phfLeg1]),
"phfLeg2" : Function("phfLeg2", [x], [phfLeg2])
}

JacFuncs = {
    "Jbtoe": Function( "Jbtoe",  [x], [simplify(jacobian(phbLeg2, q))] ),
    "Jftoe": Function( "Jftoe",  [x], [simplify(jacobian(phfLeg2, q))] )
}

# Kinetic energy
KEtor = 0.5*params["torM"] * dot(dpMtor, dpMtor) + 0.5*params["torI"]*dth**2
KEbLeg1 = 0.5*params["legM1"] * dot(dpMbLeg1, dpMbLeg1) + 0.5*params["legI1"]*(dth+dbq1)**2
KEbLeg2 = 0.5*params["legM2"] * dot(dpMbLeg2, dpMbLeg2) + 0.5*params["legI2"]*(dth+dbq1+dbq2)**2
KEfLeg1 = 0.5*params["legM1"] * dot(dpMfLeg1, dpMfLeg1) + 0.5*params["legI1"]*(dth+dfq1)**2
KEfLeg2 = 0.5*params["legM2"] * dot(dpMfLeg2, dpMfLeg2) + 0.5*params["legI2"]*(dth+dfq1+dfq2)**2
KE = KEtor + KEbLeg1 + KEbLeg2 + KEfLeg1 + KEfLeg2

# Potential energy
PEtor = params["G"] * params["torM"] * pMtor[1]
PEbLeg1 = params["G"] * params["legM1"] * pMbLeg1[1]
PEbLeg2 = params["G"] * params["legM2"] * pMbLeg2[1]
PEfLeg1 = params["G"] * params["legM1"] * pMfLeg1[1]
PEfLeg2 = params["G"] * params["legM2"] * pMfLeg2[1]
PE = PEtor + PEbLeg1 + PEbLeg2 + PEfLeg1 + PEfLeg2

L = KE - PE #ycytmp I think this should be plus, but in ME192 it is -

ddq = SX.sym("ddq",7)
Q = SX.sym("Q",7)
EOM0 = jtimes(jacobian(L,dq).T, dq, ddq) - jacobian(L,q).T - Q # equation of motion
EOM0 = simplify(EOM0)
# print(EOM)
EOM_func0 = Function("EOM0_func", [x, ddq, Q], [EOM0])
# print("EOM",EOM_func(x_val))

MD = simplify(jacobian(jacobian(KE,dq)    ,dq))
MC = SX(7,7)

for k in range(7):
    for j in range(7):
        MC[k,j] = 0
        for i in range(7):
            MC[k,j] += 0.5*(gradient(MD[k,j], q[i]) + gradient(MD[k,j], q[j]) - gradient(MD[k,j], q[k])) * dq[i]
MC = simplify(MC)
MG = simplify(jacobian(PE,q)).T
MB = simplify(jacobian(veccat(bq1, bq2, fq1, fq2), q)).T

# print(pMtor)

# print(dpMtor)

# print(MD.size())
# print(MC.size())

# MD * ddq + MC * dq + MG = MB J

# construct the matrix for solving dynamics FA x = FB

selectionM = np.zeros((3,7))
selectionM[:3,:3] = np.eye(3)

FA = vertcat(MD,selectionM)
FA = simplify( horzcat(FA, vertcat(selectionM.T, np.zeros((3,3)))))

Fb = simplify(vertcat(- (MC@dq) - MG , 0,0,0))

Fsol = solve(FA,Fb)

# MC_func = Function("MC_func",[x], [MC])
# MD_func = Function("MD_func",[x], [MD])
# MG_func = Function("MG_func",[x], [MG])
# MB_func = Function("MB_func",[x], [MB])
# FA_func = Function("FA_func",[x], [FA])
# Fb_func = Function("Fb_func",[x], [Fb])

# print("MC", MC_func(x_val))
# print("MD", MD_func(x_val))
# print("MG", MG_func(x_val))
# print("MB", MB_func(x_val))
# print("FA", FA_func(x_val))
# print("Fb", Fb_func(x_val))

dx = vertcat(dq, Fsol[:7])

# DynF = Function('Dynf', [x], [dx])


ddq = SX.sym("ddq", 7)
btoeF = SX.sym("btoeF",2)
ftoeF = SX.sym("ftoeF",2)
F = vertcat(btoeF,ftoeF)
dx = veccat(dq,ddq)
toeJac = jacobian(vertcat(phbLeg2, phfLeg2), q)
EOM = MD @ ddq + MC@dq + MG - MB @ u - toeJac.T @ F
EOMfunc = Function("EOM",[x,ddq,u,F],[EOM])


# TODO: can try avoid constraint float
# The cache to reuse jacobians of each constraints
# This cache together with other Rounge caches saves 1/3 of the memory cost!
buildJacobian_Cache = {}

def updateJacobianCache(cons, name):
    if(name is None):
        return jacobian(cons,q) 
    if(name in buildJacobian_Cache.keys()):
        return buildJacobian_Cache[name]
    else:
        res = jacobian(cons,q) 
        buildJacobian_Cache[name] = res
        return res

def buildDynF(constraints,name="",consNames = None):
    consNames = [None]*len(constraints) if consNames is None else consNames

    Flist_format = [SX.sym("F%d"%i, c.size()[0]) for i,c in enumerate(constraints)]
    Flist_Names = ["F%d"%i for i in range(len(constraints))]
    ddq_format = SX.sym("ddq",7)
    Sol_format = vertcat(ddq_format, *Flist_format)
    tmp = SX([0])# add tmp as an output to ensure that the function always returns a tuple
    solParse = Function("solParse",[Sol_format], [tmp, ddq_format, *Flist_format], ["sol"], ["0","ddq", *Flist_Names])
    print("name out:",solParse.name_out())
    consJs = [updateJacobianCache(c,n) for c,n in zip(constraints, consNames)]

    consA = vertcat(* consJs ) # constraint matrix
    consb = vertcat(* [- jtimes(j,q,dq)@dq for j in consJs]) # consA ddq = consb
    consl = consA.size()[0]

    FA = vertcat(MD,consA)
    FA = simplify( horzcat(FA, -vertcat(consA.T, np.zeros((consl, consl) ))))
    Fb = simplify(vertcat(- (MC@dq) - MG + MB @ u, consb))
    Fsol = solve(FA,Fb)

    ddq = Fsol[:7]
    dx = vertcat(dq,ddq)

    solS = solParse(Fsol)[1:]
    return Function("DynF_%s"%name, [x,u], [dx, *solS], ["x","u"], ["dx", *(solParse.name_out()[1:])])

def buildValF(v,name = ""):
    return Function("%s_val"%name, [x], [v],["x"], [name+"v"])



def buildInvF(constraints,name="",consNames = None):
    """build dynamic function (x,F)-> (dx, u, ...)
       [MD,  MB] x [ddq, u]^T = [....]

    Args:
        constraints (MX): postion of syms of joints to be fixed, e.g.: phbleg1
        name (str, optional): name of the built function. Defaults to "".
        consNames ([str], optional): name of the jacobians. This is used to reuse Jac calculation. Defaults to None.

    Returns:
        Function
    """    
    consNames = [None]*len(constraints) if consNames is None else consNames

    Flist_format = [SX.sym("F%d"%i, c.size()[0]) for i,c in enumerate(constraints)]
    Flist_Names = ["F%d"%i for i in range(len(constraints))]
    ddq_format = SX.sym("ddq",7)
    Fvec = veccat(*Flist_format)

    consJs = [updateJacobianCache(c,n) for c,n in zip(constraints, consNames)]

    consA = vertcat(* consJs ) # constraint matrix
    consb = vertcat(* [- jtimes(j,q,dq)@dq for j in consJs]) # consA ddq = consb
    consl = consA.size()[0]
    ul = 4

    FA = vertcat(MD,consA)
    FA = simplify( horzcat(FA, -vertcat(MB, np.zeros((consl, ul) ))))
    Fb = simplify(vertcat(- (MC@dq) - MG + consA.T @ Fvec, consb))
    Fsol = pinv(FA) @ Fb 

    ddq = Fsol[:7]
    u = Fsol[7:]
    dx = vertcat(dq,ddq)

    return Function("DynF_%s"%name, [x,*Flist_format], [dx, u], ["x",*Flist_Names], ["dx", "u"])

def buildValF(v,name = ""):
    return Function("%s_val"%name, [x], [v],["x"], [name+"v"])


x_val = np.array([0,0,0,-0.3,-2.5, -0.3, -2.5,
                  0,0,0,0,    0,    0,    0])
# u_val = np.array([20, 60, 32, 45])
# DynF = buildDynF([phbLeg2, phfLeg2])
# # DynF = buildDynF([])
# # DynF = buildDynF([phbLeg2])

# sol = DynF(x = x_val, u = u_val)

# print(sol["F0"],sol["F1"])
# print(EOMfunc(x_val, sol["ddq"], u_val, vertcat(sol["F0"],sol["F1"])))
# # print(EOMfunc(x_val, sol["ddq"], u_val, vertcat(0,0,0,0)))
# # print(EOM_func0(x_val, sol["ddq"], vertcat(0,0,0,u_val)))
# # print(EOM_func0(x_val, sol["ddq"], vertcat(0,0,0,u_val)))
# # print(EOM_func0(x_val, sol["ddq"], vertcat(0,0,0,u_val) - JacFuncs["Jbtoe"](x_val).T@sol["F0"]))

IdynF = buildInvF([phbLeg2, phfLeg2])
print(IdynF(x = x_val, F0 = np.array([0,150]), F1 = np.array([0,150])))