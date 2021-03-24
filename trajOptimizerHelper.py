import model
import numpy as np

def addAboveGoundConstraint(opt):
    for pfunc in model.pFuncs.values():
        opt.addConstraint(
            lambda x,u : pfunc(x)[1], [0], [np.inf]
        )
