import casadi as ca
import numpy as np
import pandas as pd

from model.fullDynamicsBody import FullDynamicsBody
from optGen.optGen import optGen

model = FullDynamicsBody('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf', 
        toeList=[("FL_SHANK", ca.DM([0,0,-0.19])),  ("FR_SHANK", ca.DM([0,0,-0.19])),
                ("HL_SHANK", ca.DM([0,0,-0.19])), ("HR_SHANK", ca.DM([0,0,-0.19]))], symbolize=True)

optGen.VARTYPE = ca.SX
opt = optGen()
m0 = ca.DM.ones(model.root.confsym.size())
m = opt.addNewVariable("Masses", 0*m0, ca.inf*m0, 1*m0)
print(model.x)

data = pd.read_csv("data/runlog/JY-S2-107 2021-07-08 13-27/JY-S2-107 2021-07-08 13-27.csv", header=2)

print(list(data.columns))
timestamps = data['Timestamp (s)']

q_traj = data[[
        "Roll (deg)" , 'Pitch (deg)', 'Yaw (deg)',
        "IN_FL_HipX_Ang (rad)", 'IN_FL_HipY_Ang (rad)', 'IN_FL_Knee_Ang (rad)', 
        'IN_FR_HipX_Ang (rad)', 'IN_FR_HipY_Ang (rad)', 'IN_FR_Knee_Ang (rad)',
        'IN_HL_HipX_Ang (rad)', 'IN_HL_HipY_Ang (rad)', 'IN_HL_Knee_Ang (rad)',
        'IN_HR_HipX_Ang (rad)', 'IN_HR_HipY_Ang (rad)', 'IN_HR_Knee_Ang (rad)'
]]

dq_traj = data[[
        'Roll_Vel (rad/s)', 'Pitch_Vel (rad/s)', 'Yaw_Vel (rad/s)',
        'IN_FL_HipX_Vel (rad/s)', 'IN_FL_HipY_Vel (rad/s)', 'IN_FL_Knee_Vel (rad/s)',
        'IN_FR_HipX_Vel (rad/s)', 'IN_FR_HipY_Vel (rad/s)', 'IN_FR_Knee_Vel (rad/s)',
        'IN_HL_HipX_Vel (rad/s)', 'IN_HL_HipY_Vel (rad/s)', 'IN_HL_Knee_Vel (rad/s)',
        'IN_HR_HipX_Vel (rad/s)', 'IN_HR_HipY_Vel (rad/s)', 'IN_HR_Knee_Vel (rad/s)'
]]

ddq_traj = data[[
    'X_Acc (m/s^2)', 'Y_Acc (m/s^2)', 'Z_Acc (m/s^2)'
]]

u_traj = data[[
'IN_FL_HipX_Tor (Nm)', 'IN_FL_HipY_Tor (Nm)', 'IN_FL_Knee_Tor (Nm)',
'IN_FR_HipX_Tor (Nm)', 'IN_FR_HipY_Tor (Nm)', 'IN_FR_Knee_Tor (Nm)',
'IN_HL_HipX_Tor (Nm)', 'IN_HL_HipY_Tor (Nm)', 'IN_HL_Knee_Tor (Nm)',
'IN_HR_HipX_Tor (Nm)', 'IN_HR_HipY_Tor (Nm)', 'IN_HR_Knee_Tor (Nm)']]

print(q_traj.to_numpy().shape)

EPISODS = 10
EPILENGTH = 10
EPISTEP = 100
for I in range(EPISODS):
    for i in range(EPISTEP*I, EPISTEP*I + EPILENGTH):
        p = opt.addNewVariable("p%d_%d"%(I,i), 3) # x,y,z
        dp = opt.addNewVariable("dp%d_%d"%(I,i), 3) # dx,dy,dz
        ddq = opt.addNewVariable("ddq%d_%d"%(I,i), 15) # ddr ... dd HR
        F = opt.addNewVariable("dp%d_%d"%(I,i), 12)
