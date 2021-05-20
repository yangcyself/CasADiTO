import pandas as pd
import numpy as np


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

def saveXdirSolution(filename, x_opt, u_opt, t):
    df_x = pd.DataFrame(x_opt, 
        columns = ["x", "y", "r", "lhx", "lhy", "lk", "rhx", "rhy", "rk",
                   "dx", "dy", "dr", "dlhx", "dlhy", "dlk", "drhx", "drhy", "drk"], 
        index = t
    )

    df_u = pd.DataFrame(u_opt, 
        columns = ["ulhx", "ulhy", "ulk", "urhx", "urhy", "urk"], 
        index = t
    )

    df = pd.concat([df_x,df_u],axis = 1)
    print(df.head())

    # Frame shift
    df["lhx"] = df["lhx"] - np.math.pi/2
    df["rhx"] = df["rhx"] - np.math.pi/2
    df["lhy"] = df["lhy"] + np.math.pi/2
    df["rhy"] = df["rhy"] + np.math.pi/2

    df["lhx"] = -df["lhx"]
    df["dlhx"] = -df["dlhx"]
    df["ulhx"] = -df["ulhx"]

    df.to_csv(filename, index_label = "t", 
            columns = ["x", "y", "r", "lhx", "lhy", "lk", "rhx", "rhy", "rk",
            "dx", "dy", "dr", "dlhx", "dlhy", "dlk", "drhx", "drhy", "drk",
            "ulhx", "ulhy", "ulk", "urhx", "urhy", "urk"])


def save3Dsolution(filename, x_opt, u_opt, t):
    target_columns = [
        "time", "state", "x", "y", "z", "qw", "qx", "qy", "qz", "Vx", "Vy", "Vz", "wr", "wp", "wy", "LF_HAA",
        "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE", "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE", "w_LF_HAA",
        "w_LF_HFE", "w_LF_KFE", "w_RF_HAA", "w_RF_HFE", "w_RF_KFE", "w_LH_HAA", "w_LH_HFE", "w_LH_KFE", "w_RH_HAA", "w_RH_HFE", "w_RH_KFE", "tau_LF_HAA",
        "tau_LF_HFE", "tau_LF_KFE", "tau_RF_HAA", "tau_RF_HFE", "tau_RF_KFE", "tau_LH_HAA", "tau_LH_HFE", "tau_LH_KFE", "tau_RH_HAA", "tau_RH_HFE", "tau_RH_KFE"
    ]


    df_x = pd.DataFrame(x_opt, 
        columns = ["x", "y", "r", "bh", "bt", "fh", "ft",
                "dx", "dy", "dr", "dbh", "dbt", "dfh", "dft"], 
        index = t
    )

    df_u = pd.DataFrame(u_opt, 
        columns = ["ubh", "ubt", "ufh", "uft"], 
        index = t
    )
