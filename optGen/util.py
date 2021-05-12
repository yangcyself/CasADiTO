import casadi as ca

def notImplementedFunc():
    raise NotImplementedError

class LazyFunc:
    """
    Lazy function that only implent the function when it is first called
    """
    def __init__(self, funcGen = lambda: notImplementedFunc):
        self.funcGen = funcGen

    @property
    def func(self):
        try:
            return self._func
        except AttributeError:
            self._func = self.funcGen()
        return self._func
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

import inspect

def kwargFunc(f):
    """Decorator: Make the f accepts only kwarg,
        and the keywords can have more than f's signature
    Args:
        f (function(kwarg)): a function 
            (should not declare *args or **kwarg in its signature)
    """
    sig = inspect.getfullargspec(f)
    # retunrs sth like (['a', 'b', 'c'], 'arglist', 'keywords', (4,))
    if(not (sig[1] is None and 
            sig[2] is None and
            sig[3] is None)):
        raise ValueError("kwargFunc should contain no *args and **kwargs or default values")
    def func(**kwargs):
        return f(**{k:kwargs[k] for k in sig[0]})
    return func

def getName(a):
    return a[0,0].name()[:-2]

def caSubsti(a, sym, val):
    """Substitute the symbols by values and calculate the value of a

    Args:
        a (MX/SX): The target to calculate
        sym ([MX/SX]): The symbols to be substituded
        val (DM/narray): The value of the substituded symbol
    """
    if(len(sym) == 0 or isinstance(a, ca.DM)):
        return a
    sameType = [type(s) == type(a) for s in sym]
    sym = [s for s,i in zip(sym, sameType) if i]
    val = [v for v,i in zip(val, sameType) if i]
    res = ca.substitute([a], sym, val)[0]
    return ca.DM(res) if res.is_constant() else res

def caFuncSubsti(f, kwargs):
    """Substitute the symbols by values and calculate the value of a

    Args:
        a (MX/SX): The target to calculate
        kwargs {name, val} the name and value to substitute in the function
    """
    caType = ca.MX if isinstance (next(iter(kwargs.values())), ca.MX) else ca.SX
    X_dict = {n: caType.sym(n,f.size_in(n)) 
        for n in f.name_in() if n not in kwargs.keys()
    }
    fres = f(**X_dict, **kwargs)
    return ca.Function(
        f.name(),
        list(X_dict.values()),
        list(fres.values()),
        list(X_dict.keys()),
        list(fres.keys())
    )

def substiSX2MX(a, SX, MX):
    """given a SX expression, make it a function and call with MX

    Args:
        a ([type]): [description]
        SX ([type]): [description]
        MX ([type]): [description]
    """
    return ca.Function("TMPsubstiSX2MX", SX, [a])(*MX)

def MXinSXop(op, a, SXandMX):
    """conduct the SX operation with MX

    Args:
        op ([type]): [description]
        a ({name: MX}): a dict of MX variables
        SXandMX([(name, SX, MX)])
    """
    asx = {k:ca.SX.sym(k, v.size(1), v.size(2)) for k,v in a.items() if v is not None}
    opsx = op(**asx)
    opsx_func = ca.Function("op", list(asx.values()) + [s for n,s,m in SXandMX], [opsx], 
                            list(asx.keys()) + [n for n,s,m in SXandMX], ["out"])
    return opsx_func(*[v for v in a.values() if v is not None], *[m for n,s,m in SXandMX])

