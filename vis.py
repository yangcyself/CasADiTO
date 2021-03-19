from model import *



prTor = vertcat(px,py) #rear of the torso
phTor = vertcat(px+params["torL"] * cos(th),py+params["torL"] * sin(th)) #head of the torso
phbLeg1 = vertcat(px + params["legL"] * cos(th+bq1),
    py + params["legL"] * sin(th+bq1)) #rear of the back leg thigh
phbLeg2 = vertcat(px + params["legL"] * cos(th+bq1) + params["legL"] * cos(th+bq1+bq2),
    py + params["legL"] * sin(th+bq1) + params["legL"] * sin(th+bq1+bq2),) #rear of the back leg 
phfLeg1 = vertcat(px + params["torL"] * cos(th) + params["legL"] * cos(th+fq1),
    py + params["torL"] * sin(th) + params["legL"] * sin(th+fq1))#front leg thigh center of position
phfLeg2 = vertcat(px + params["torL"] * cos(th) + params["legL"] * cos(th+fq1) + params["legL"] * cos(th+fq1+fq2),
    py + params["torL"] * sin(th) + params["legL"] * sin(th+fq1) + params["legL"] * sin(th+fq1+fq2))#front leg

visFunc = Function('visFunc', [q], [vertcat(phbLeg2.T, phbLeg1.T, prTor.T, phTor.T, phfLeg1.T, phfLeg2.T )])

def visState(q):
    line = visFunc(q)
    plt.figure()
    plt.plot(line[:,0],line[:,1])
    print("Line:",line)
    plt.show()
    


