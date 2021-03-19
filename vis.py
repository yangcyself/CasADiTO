from model import *


visFunc = Function('visFunc', [q], [vertcat(phbLeg2.T, phbLeg1.T, prTor.T, phTor.T, phfLeg1.T, phfLeg2.T )])

def visState(q):
    line = visFunc(q)
    plt.figure()
    plt.plot(line[:,0],line[:,1])
    print("Line:",line)
    plt.show()
    


