import casadi as ca

def normQuad(d):
    """
    Use this to replace `ca.norm_2(d)**2`, The later one still generates nan
    """
    return ca.dot(d,d)

def solveLinearCons(consFunc, targets, eps = 1e-6):
    """Find the most proximate value meeting a constraint
        min_x sum alpha_i(x_i-x_refi)^2
            s.t. g(x_0, x_1 ...) = 0
    Args:
        consFunc ([Function]): Calculates a constraint given several variables(needs to be linear to variables)
        targets ([tuples]): (name, value, factor): the name of variables in consFunc, its reference value
        eps (float): the epsilon value for the other variables
    """

    X_dict = {n: ca.SX.sym(n,consFunc.size_in(n)) for n in consFunc.name_in()}
    X_vec = ca.veccat(*list(X_dict.values()))
    solParse = ca.Function("parse", [X_vec], list(X_dict.values()), ["sol"], list(X_dict.keys()))
    Q = ca.diag([eps] * X_vec.size(1))
    X_ref = ca.DM.zeros(X_vec.size())
    for tn, tv, tf in targets:
        locJac = ca.jacobian(X_vec, X_dict[tn])
        Q += ca.diag(locJac
            @ca.DM([tf] * X_dict[tn].size(1)))
        X_ref += locJac @ tv

    g = consFunc(**X_dict)[consFunc.name_out()[0]]
    # represent g as A(x+x_ref) = b
    A = ca.jacobian(g, X_vec)
    b = -consFunc(**solParse(sol = X_ref))[consFunc.name_out()[0]]

    # By using KKT on lagrange, we get A(x_ref+x)=b; 2Qx+A^Tlambda = 0
    lml = g.size(1) # length of lambda
    QPA = ca.vertcat(
        ca.horzcat(A, ca.DM.zeros(A.size(1), lml)),
        ca.horzcat(Q, A.T)
    )
    QPb = ca.vertcat(b, ca.DM.zeros(X_vec.size(1)))
    res = ca.simplify(ca.mldivide(QPA, QPb))
    return solParse(sol = X_ref + res[:X_vec.size(1)])

def solveCons(consFunc, targets, eps = 1e-6):
    """Use QP to find the most proximate value meeting a constraint

    Args:
        consFunc ([Function]): Calculates a constraint given several variables(needs to be all vectors)
        targets ([tuples]): (name, value, factor)
        eps (float): the epsilon value for the other variables
    """
    X_dict = {n: ca.SX.sym(n,consFunc.size_in(n)) for n in consFunc.name_in()}
    X_vec = ca.veccat(*list(X_dict.values()))
    f = 0
    for tn, tv, tf in targets:
        f += tf * ca.dot(X_dict[tn] - tv, X_dict[tn] - tv)

    for x in X_dict.values():
        f += eps * ca.dot(x,x)
    # print(X_dict)
    g = consFunc(**X_dict)[consFunc.name_out()[0]]

    solParse = ca.Function("parse", [X_vec], list(X_dict.values()), ["sol"], list(X_dict.keys()))

    qp = {'x':X_vec, 'f':f, 'g':g}

    solver = ca.nlpsol('solver', 'ipopt', qp, {"verbose" : False ,"print_time": False,"verbose_init":False,
        "ipopt":{
            "print_level": 0
            # "verbose":False,
            # "print_header": False,
            # "print_iteration": False,
            # "print_status": False,
        }
    }
            )# , {'sparse':True})
    # Get the optimal solution
    sol = solver(lbx=[-ca.inf] * X_vec.size(1), ubx=[ca.inf] * X_vec.size(1), 
                 lbg=[0] * g.size(1), ubg=[0] * g.size(1))
    
    return solParse(sol = sol["x"])

def cross2d(a,b):
    return a[0]*b[1] - a[1]*b[0]

def Rot(a,d):
    """The rotation matrix of dimension d

    Args:
        a (SX): the angle
        d (SX/DM): the omega direction. E.g. (0,0,1) for rot in z dir
    """
    d = ca.SX(d)
    sk = ca.skew(d)
    sk2 = d @ d.T - ca.DM.eye(3)
    return ca.DM.eye(3) + sk * ca.sin(a) + sk2 * (1- ca.cos(a))

######      ######      ######      ######
###      Rotation Transformations      ###
######      ######      ######      ######
_et,_de = ca.SX.sym('e', 3), ca.SX.sym('de', 3)

# Tait–Bryan angles, (extrinsic, the X,Y,Z is always the first second third element)
# Generate XYZRot ... ZYXRot
_execmap =  globals().copy()
_execmap.update({"rotMap":{"X": Rot(_et[0],[1,0,0]), "Y": Rot(_et[1],[0,1,0]), "Z": Rot(_et[2],[0,0,1])}})
for d in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",]:
    exec("{dr}Rot = ca.Function('{dr}Rot', [_et], [rotMap['{rot0}'] @ rotMap['{rot1}'] @ rotMap['{rot2}']])".format(dr=d, rot0=d[0], rot1=d[1], rot2=d[2]), _execmap)
XYZRot = _execmap["XYZRot"]
XZYRot = _execmap["XZYRot"]
YXZRot = _execmap["YXZRot"]
YZXRot = _execmap["YZXRot"]
ZXYRot = _execmap["ZXYRot"]
ZYXRot = _execmap["ZYXRot"]

_R,_dR = ca.SX.sym('R',3,3), ca.SX.sym('dR',3,3)
_SO2vec = ca.Function('SO2vec', [_R], [ca.vertcat(_R[2,1], _R[0,2], _R[0,1] )])
R2omega_B = ca.Function("R2omega_B", [_R, _dR], [_SO2vec(_R.T @ _dR)])
R2omega_W = ca.Function("R2omega_W", [_R, _dR], [_SO2vec(_dR @ _R.T)])

def _buildeular2Omega(et, de, OmgF, rotF, name):
    R = rotF(et)
    dR = ca.jtimes(R, et, de)
    return ca.Function(name, [et, de], [OmgF(R, dR)])
_execmap =  globals().copy()
for d in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",]:
    exec("d{dr}2Omega_B = _buildeular2Omega(_et,_de, R2omega_B ,{dr}Rot, 'd{dr}2Omega_B')".format(dr=d), _execmap)
    exec("d{dr}2Omega_W = _buildeular2Omega(_et,_de, R2omega_W ,{dr}Rot, 'd{dr}2Omega_W')".format(dr=d), _execmap)

dXYZ2Omega_B = _execmap["dXYZ2Omega_B"]
dXZY2Omega_B = _execmap["dXZY2Omega_B"]
dYXZ2Omega_B = _execmap["dYXZ2Omega_B"]
dYZX2Omega_B = _execmap["dYZX2Omega_B"]
dZXY2Omega_B = _execmap["dZXY2Omega_B"]
dZYX2Omega_B = _execmap["dZYX2Omega_B"]
dXYZ2Omega_W = _execmap["dXYZ2Omega_W"]
dXZY2Omega_W = _execmap["dXZY2Omega_W"]
dYXZ2Omega_W = _execmap["dYXZ2Omega_W"]
dYZX2Omega_W = _execmap["dYZX2Omega_W"]
dZXY2Omega_W = _execmap["dZXY2Omega_W"]
dZYX2Omega_W = _execmap["dZYX2Omega_W"]


if __name__ == "__main__":
    a = ca.SX.sym("a", 3)
    b = ca.SX.sym("b", 3)
    f = ca.DM([[3,2,1],[1,4,1]])@a + ca.DM([[0,1,0],[1,1,1]])@b +3
    g = ca.Function("g", [a,b],[f], ["a","b"], ["f"])
    res = solveLinearCons(g, [
        ("a", ca.DM([1,0,1]), 1e2)
    ])
    print(res)
    print(g(**res))

    ## Test rotation transformations
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    theta = np.random.rand(3) * ca.pi * 2
    r = R.from_euler('ZYX', theta) # scipy use xyz to represent extrinsic, XYZ for intrinsic, But I think they get it misplaced
    r_ = ZYXRot(ca.vertcat(theta[2], theta[1], theta[0]))
    # r_ = ZYXRot(ca.vertcat(theta))
    print(r.as_matrix())
    print(r_)

    print(dZYX2Omega_B(ca.DM.rand(3), ca.DM.rand(3)))