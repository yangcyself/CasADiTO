import casadi as ca

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
    X_dict = {n: SX.sym(n,consFunc.size_in(n)) for n in consFunc.name_in()}
    X_vec = veccat(*list(X_dict.values()))
    f = 0
    for tn, tv, tf in targets:
        f += tf * dot(X_dict[tn] - tv, X_dict[tn] - tv)

    for x in X_dict.values():
        f += eps * dot(x,x)
    # print(X_dict)
    g = consFunc(**X_dict)[consFunc.name_out()[0]]

    solParse = Function("parse", [X_vec], list(X_dict.values()), ["sol"], list(X_dict.keys()))

    qp = {'x':X_vec, 'f':f, 'g':g}

    solver = nlpsol('solver', 'ipopt', qp, {"verbose" : False ,"print_time": False,"verbose_init":False,
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
    sol = solver(lbx=[-np.inf] * X_vec.size(1), ubx=[np.inf] * X_vec.size(1), 
                 lbg=[0] * g.size(1), ubg=[0] * g.size(1))
    
    return solParse(sol = sol["x"])


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

et = ca.SX.sym('e', 3)
ZYXRot = ca.Function("ZYXRot", [et], [Rot(et[2],[0,0,1]) @ Rot(et[1],[0,1,0]) @ Rot(et[0],[1,0,0])])


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