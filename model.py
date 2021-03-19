
from casadi import *


# Declare model variables
px = MX.sym('px')
py = MX.sym('py')
th = MX.sym('th')
bq1 = MX.sym('bq1')
bq2 = MX.sym('bq2')
fq1 = MX.sym('fq1')
fq2 = MX.sym('fq2')

dpx = MX.sym('dpx')
dpy = MX.sym('dpy')
dth = MX.sym('dth')
dbq1 = MX.sym('dbq1')
dbq2 = MX.sym('dbq2')
dfq1 = MX.sym('dfq1')
dfq2 = MX.sym('dfq2')

params = {
    "legL":1,
    "torL":1,
    "legM":5,
    "torM":10,
    "legI":0.5,
    "torI":1,
    "G":9.81,
}

q = vertcat(px, py, th, bq1, bq2, fq1, fq2)
dq = vertcat(dpx, dpy, dth, dbq1, dbq2, dfq1, dfq2)
x = vertcat(q, dq)
u = MX.sym('u',4)

# Build Model using the euler methods

pMtor = vertcat(
    px + params["torL"]/2 * cos(th),
    py + params["torL"]/2 * sin(th),
)#torsor center of position

pMbLeg1 = vertcat(
    px + params["legL"]/2 * cos(th+bq1),
    py + params["legL"]/2 * sin(th+bq1),
)#back leg thigh center of position

pMbLeg2 = vertcat(
    px + params["legL"] * cos(th+bq1) + params["legL"]/2 * cos(th+bq1+bq2),
    py + params["legL"] * sin(th+bq1) + params["legL"]/2 * sin(th+bq1+bq2),
)#back leg center of position

pMfLeg1 = vertcat(
    px + params["torL"] * cos(th) + params["legL"]/2 * cos(th+fq1),
    py + params["torL"] * sin(th) + params["legL"]/2 * sin(th+fq1),
)#front leg thigh center of position

pMfLeg2 = vertcat(
    px + params["torL"] * cos(th) + params["legL"] * cos(th+fq1) + params["legL"]/2 * cos(th+fq1+fq2),
    py + params["torL"] * sin(th) + params["legL"] * sin(th+fq1) + params["legL"]/2 * sin(th+fq1+fq2),
)#front leg center of position

prTor = vertcat(px,py) #rear of the torso
phTor = vertcat(px+params["torL"] * cos(th),py+params["torL"] * sin(th)) #head of the torso
phbLeg1 = vertcat(px + params["legL"] * cos(th+bq1),
    py + params["legL"] * sin(th+bq1)) #rear of the back leg thigh
phbLeg2 = vertcat(px + params["legL"] * cos(th+bq1) + params["legL"] * cos(th+bq1+bq2),
    py + params["legL"] * sin(th+bq1) + params["legL"] * sin(th+bq1+bq2),) #rear of the back leg 
phfLeg1 = vertcat(px + params["torL"] * cos(th) + params["legL"] * cos(th+fq1),
    py + params["torL"] * sin(th) + params["legL"] * sin(th+fq1))#front leg thigh center of position
phfLeg2 = vertcat(px + params["torL"] * cos(th) + params["legL"] * cos(th+fq1) + params["legL"] * cos(th+fq1+fq2),
    py + params["torL"] * sin(th) + params["legL"] * sin(th+fq1) + params["legL"] * sin(th+fq1+fq2))#front leg

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
KEbLeg1 = 0.5*params["legM"] * dot(dpMbLeg1, dpMbLeg1) + 0.5*params["legI"]*(dth+dbq1)**2
KEbLeg2 = 0.5*params["legM"] * dot(dpMbLeg2, dpMbLeg2) + 0.5*params["legI"]*(dth+dbq1+dbq2)**2
KEfLeg1 = 0.5*params["legM"] * dot(dpMfLeg1, dpMfLeg1) + 0.5*params["legI"]*(dth+dfq1)**2
KEfLeg2 = 0.5*params["legM"] * dot(dpMfLeg2, dpMfLeg2) + 0.5*params["legI"]*(dth+dfq1+dfq2)**2
KE = KEtor + KEbLeg1 + KEbLeg2 + KEfLeg1 + KEfLeg2

# Potential energy
PEtor = params["G"] * params["torM"] * pMtor[1]
PEbLeg1 = params["G"] * params["legM"] * pMbLeg1[1]
PEbLeg2 = params["G"] * params["legM"] * pMbLeg2[1]
PEfLeg1 = params["G"] * params["legM"] * pMfLeg1[1]
PEfLeg2 = params["G"] * params["legM"] * pMfLeg2[1]
PE = PEtor + PEbLeg1 + PEbLeg2 + PEfLeg1 + PEfLeg2

L = KE - PE #ycytmp I think this should be plus, but in ME192 it is -

# EOM = jacobian(jacobian(L,dq), q)@dq - jacobian(L,q).T # equation of motion
# EOM = simplify(EOM)
# print(EOM)
# EOM_func = Function("EOM_func", [x], [EOM])
# # print("EOM",EOM_func(x_val))

MD = simplify(jacobian(jacobian(KE,dq)    ,dq))
MC = MX(7,7)

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

    Flist_format = [MX.sym("F%d"%i, c.size()[0]) for i,c in enumerate(constraints)]
    ddq_format = MX.sym("ddq",7)
    Sol_format = vertcat(ddq_format, *Flist_format)
    tmp = MX([0])# add tmp as an output to ensure that the function always returns a tuple
    solParse = Function("solParse",[Sol_format], [tmp, ddq_format, *Flist_format], ["sol"], ["0","ddq", *[s.name() for s in Flist_format]])
    print("name out:",solParse.name_out())
    consJs = [updateJacobianCache(c,n) for c,n in zip(constraints, consNames)]

    consA = vertcat(* consJs ) # constraint matrix
    consb = vertcat(* [- jtimes(j,q,dq)@dq for j in consJs]) # consA ddq = consb
    consl = consA.size()[0]

    FA = vertcat(MD,consA)
    FA = simplify( horzcat(FA, vertcat(consA.T, np.zeros((consl, consl) ))))
    Fb = simplify(vertcat(- (MC@dq) - MG + MB @ u, consb))
    Fsol = solve(FA,Fb)

    ddq = Fsol[:7]
    dx = vertcat(dq,ddq)

    solS = solParse(Fsol)[1:]
    return Function("DynF_%s"%name, [x,u], [dx, *solS], ["x","u"], ["dx", *(solParse.name_out()[1:])])

def buildValF(v,name = ""):
    return Function("%s_val"%name, [x], [v],[x], [name+"v"])

DynF = buildDynF([phTor, phbLeg2])