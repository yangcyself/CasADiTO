import numpy as np


"""
This file implements an initializer that use second order polynomial to interpolate the points
The PolynomInit stores the time and state of each key state. 
The state must be q, dq. So that the second order is calculate
"""
class PolynomInit:
    
    def __init__(self):
        # [ [t,x]]
        self._timeStateArray = np.array([[]])
    
    def addKeyState(self, t, x):
        
        if(self._timeStateArray.shape[-1]==0):
            self._timeStateArray = np.column_stack([t,x.reshape(1,-1)]).astype(float)
        else:
            ind = np.searchsorted(self._timeStateArray[:,0], t, "right")
            self._timeStateArray = np.insert(self._timeStateArray, ind, np.column_stack([t,x.reshape(1,-1)]), axis = 0)
        
    def itp(self,t):
        """interpolate using time stamp t
        """
        # ind = 0
        ind = np.searchsorted(self._timeStateArray[:,0], t, "right")
        qdim = (self._timeStateArray.shape[1]-1)//2

        q0 = self._timeStateArray[ind-1][1:-qdim]
        dq0 = self._timeStateArray[ind-1][-qdim:]
        q1 = self._timeStateArray[ind][1:-qdim]
        dq1 = self._timeStateArray[ind][-qdim:]

        dt = self._timeStateArray[ind,0] - self._timeStateArray[ind-1,0]
        a0 = q0
        a1 = dq0
        a2 = -(3*(q0 - q1) + dt*(2*dq0 + dq1))/(dt**2)
        a3 = (2*(q0 - q1) + dt*(dq0 + dq1))/(dt**3)

        tt = t - self._timeStateArray[ind-1,0]
        return {
            "t": t,
            "q": a0 + a1*tt + a2*tt**2 + a3 * tt**3,
            "dq": a1 + 2*a2*tt + 3*a3 * tt**2,
            "ddq": 2*a2 + 6 * a3*tt
        }
        

if __name__=="__main__":
    ini = PolynomInit()

    ini.addKeyState(3, np.array([7,2]))
    ini.addKeyState(1, np.array([0,0]))
    ini.addKeyState(2, np.array([1,0]))
    print(ini.itp(1))