import numpy as np
import casadi as ca

def pointsTerrian2D(points, sigm = 0.001):
    """generate a terrian function x->y. Given a list of [x,y] points
        The terrian is calculated as a convolute of the points using a gaussian kernel

    Args:
        points (narray/SX): a list of points
        sigm (float, optional): The sigma of gaussian kernal. Defaults to 0.001.
            Note: this sigm parameter is valuable for giving some gradient to the solver.
                E.g. when solving for jump down, sigm = 0.001 uses 54 iter, sigm = 0.0005 takes 250+ iter,  
                sigm = 0.0002 yields invalid number, sigm = 0.01 yields no solution as the ground is not flat enough,
    """
    K = lambda x: np.exp( - x**2 / sigm )
    # return lambda p: np.sum([y * K(x-p) for x,y in points]) / np.sum([K(x-p) for x,y in points])
    return lambda p: ca.dot(points[:,1], K(points[:,0] - p)) / ca.sum1(K(points[:,0] - p))

def addDisjunctivePositiveConstraint(opt, x, epsName="eps", consName="disj"):
    """Add the constraint that ensure at least one element in x is greater than or equal to zero
    By introducing eps>0 and add the constriant that x+eps y > 0
    """

    eps0 = ca.DM.ones(x.size(1)) # the first element set to 1
    eps = opt.addNewVariable(epsName, 0*eps0[1:], ca.inf*eps0[1:], eps0[1:])

    g = ca.dot(x, ca.vertcat(1, eps))
    opt._g.append(g)
    opt._lbg.append([0])
    opt._ubg.append([ca.inf])
    
    ### Add plot and parse information
    try:
        opt._disj_x_plot.append(x)
        opt._disj_eps_plot.append(eps)
        opt._disj_g_plot.append(g)
    except AttributeError:
        opt._disj_x_plot=[x]
        opt._disj_eps_plot=[eps]
        opt._disj_g_plot=[g]
        opt._parse.update({
            "%s_x"%consName: lambda : ca.horzcat(*opt._disj_x_plot),
            "%s_eps"%consName: lambda : ca.horzcat(*opt._disj_eps_plot),
            "%s_g"%consName: lambda : ca.horzcat(*opt._disj_g_plot)
        })


def addLinearClearanceConstraint(opt, A,b, wName="w", consName="linearClearance", slackWeight=None):
    """Add the constraint that Ax<b for all x in Rn has no solution
        for the set {Ax - b < 0} be empty, there exists a w>0 s.t. ATw=0, wTb <= 0

    Args:
        opt (optGen): The optimization object to add constraint to
        A (SX/MX): 
        b (SX/MX): 
        wName (str, optional): the name of the added variablle. Defaults to "w".
        consName (str, optional): the name of the constraint. Defaults to "linearClearance".
    """
    sz_x = A.size(2)
    sz_w = A.size(1)
    if(A.size(1) != b.size(1)):
        raise ValueError("A and b should have number of rows, but got {}, {}".format(A.size(1), b.size(1)))

    w0 = ca.DM.ones(sz_w)
    w = opt.addNewVariable(wName, 0*w0, ca.inf*w0, w0)

    g0 = A.T @ w # ATw=0
    opt._g.append(g0)
    opt._lbg.append(ca.DM.zeros(g0.size()))
    opt._ubg.append(ca.DM.zeros(g0.size()))

    if(slackWeight is None):
        g1 = ca.dot(w, b) # wTb<=0
    else:
        c = opt.addNewVariable("c", ca.DM([0]), ca.DM([ca.inf]), ca.DM([0]))
        g1 = ca.dot(w,b)-c
        opt._J+=slackWeight*c*c
    opt._g.append(g1)
    opt._lbg.append(ca.DM([-ca.inf]))
    opt._ubg.append(ca.DM([0]))

    g2 = ca.dot(w,w) # w neq 0
    opt._g.append(g2)
    opt._lbg.append(ca.DM([1]))
    opt._ubg.append(ca.DM([ca.inf]))


    

