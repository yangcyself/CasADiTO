from model import *
import pandas as pd

visFunc = Function('visFunc', [q], [vertcat(phbLeg2.T, phbLeg1.T, prTor.T, pRTor.T, pHTor.T, phTor.T, phfLeg1.T, phfLeg2.T )])

def visState(q):
    line = visFunc(q)
    plt.figure()
    plt.plot(line[:,0],line[:,1])
    print("Line:",line)
    plt.show()
    

def saveSolution(filename, x_opt, u_opt, t):
    df_x = pd.DataFrame(x_opt, 
        columns = ["x", "y", "r", "bh", "bt", "fh", "ft",
                "dx", "dy", "dr", "dbh", "dbt", "dfh", "dft"], 
        index = t
    )

    df_u = pd.DataFrame(u_opt, 
        columns = ["ubh", "ubt", "ufh", "uft"], 
        index = t
    )

    df = pd.concat([df_x,df_u],axis = 1)
    print(df.head())

    # Frame shift

    df["bh"] = df["bh"] + np.math.pi/2
    df["fh"] = df["fh"] + np.math.pi/2
    df.to_csv(filename, index_label = "t", 
            columns = ["x", "y", "r", "bh", "bt", "fh", "ft", 
            "dx", "dy", "dr", "dbh", "dbt", "dfh", "dft", 
            "ubh", "ubt", "ufh", "uft"])

