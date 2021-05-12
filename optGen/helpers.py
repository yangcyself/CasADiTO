import numpy as np
import casadi as ca

def pointsTerrian2D(points, sigm = 0.01):
    """generate a terrian function x->y. Given a list of [x,y] points
        The terrian is calculated as a convolute of the points using a gaussian kernel

    Args:
        points (narray/SX): a list of points
        sigm (float, optional): The sigma of gaussian kernal. Defaults to 0.01.
    """
    K = lambda x: np.exp( - x**2 / sigm )
    # return lambda p: np.sum([y * K(x-p) for x,y in points]) / np.sum([K(x-p) for x,y in points])
    return lambda p: ca.dot(points[:,1], K(points[:,0] - p)) / ca.sum1(K(points[:,0] - p))
