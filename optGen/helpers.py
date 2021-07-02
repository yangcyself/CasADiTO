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

def addDisjunctivePositiveConstraint(opt, x,y, epsName="eps", consName="disj"):
    """Add the constraint that ensure eigher x or y is greater than zero
    By introducing eps>0 and add the constriant that x+eps y > 0
    """
    if(not x.size()==y.size()):
        raise ValueError("x and y should have same size, but got {}, {}".format(x.size(), y.size()))
    eps0 = ca.DM.ones(x.size())
    eps = opt.addNewVariable(epsName, 0*eps0, ca.inf*eps0, eps0)

    g = x + eps * y
    opt._g.append(g)
    opt._lbg.append(0*eps0)
    opt._ubg.append(ca.inf*eps0)
    
    ### Add plot and parse information
    try:
        opt._disj_x_plot.append(x)
        opt._disj_y_plot.append(y)
        opt._disj_eps_plot.append(eps)
        opt._disj_g_plot.append(g)
    except AttributeError:
        opt._disj_x_plot=[x]
        opt._disj_y_plot=[y]
        opt._disj_eps_plot=[eps]
        opt._disj_g_plot=[g]
        opt._parse.update({
            "%s_x"%consName: lambda : ca.horzcat(*opt._disj_x_plot),
            "%s_y"%consName: lambda : ca.horzcat(*opt._disj_y_plot),
            "%s_eps"%consName: lambda : ca.horzcat(*opt._disj_eps_plot),
            "%s_g"%consName: lambda : ca.horzcat(*opt._disj_g_plot)
        })
