
def notImplementedFunc():
    raise NotImplementedError

class LazyFunc:
    """
    Lazy function that only implent the function when it is first called
    """
    def __init__(self, funcGen = lambda: notImplementedFunc):
        self.funcGen = funcGen
    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except AttributeError:
            self.func = self.funcGen()
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

