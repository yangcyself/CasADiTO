import casadi as ca
import numpy as np
import pandas as pd
import os
import pickle as pkl
import time

from model.fullDynamicsBody import FullDynamicsBody
from optGen.optGen import optGen
from utils.mathUtil import normQuad

model = FullDynamicsBody('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf', 
        toeList=[("FL_SHANK", ca.DM([0,0,-0.19])),  ("FR_SHANK", ca.DM([0,0,-0.19])),
                ("HL_SHANK", ca.DM([0,0,-0.19])), ("HR_SHANK", ca.DM([0,0,-0.19]))], symbolize=True)

optGen.VARTYPE = ca.SX
opt = optGen()
m0 = ca.DM.ones(model.root.confsym.size())
# [TORSO_M, INERTIA_M, FL_HIP_M, FL_THIGH_M, FL_SHANK_M, FR_HIP_M, FR_THIGH_M, FR_SHANK_M, HL_HIP_M, HL_THIGH_M, HL_SHANK_M, HR_HIP_M, HR_THIGH_M, HR_SHANK_M]
m_prior = ca.DM([0, 5.298, 0.428, 0.61, 0.145, 0.428, 0.61, 0.145, 0.428, 0.61, 0.145, 0.428, 0.61, 0.145])
m = opt.addNewVariable("Masses", 0*m0, ca.inf*m0, m_prior)
opt._parse.update({"mass": lambda: m})
print(model.x)
print(model.root.confsym)

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

ddp_traj = data[[
    'X_Acc (m/s^2)', 'Y_Acc (m/s^2)', 'Z_Acc (m/s^2)'
]]

u_traj = data[[
    'IN_FL_HipX_Tor (Nm)', 'IN_FL_HipY_Tor (Nm)', 'IN_FL_Knee_Tor (Nm)',
    'IN_FR_HipX_Tor (Nm)', 'IN_FR_HipY_Tor (Nm)', 'IN_FR_Knee_Tor (Nm)',
    'IN_HL_HipX_Tor (Nm)', 'IN_HL_HipY_Tor (Nm)', 'IN_HL_Knee_Tor (Nm)',
    'IN_HR_HipX_Tor (Nm)', 'IN_HR_HipY_Tor (Nm)', 'IN_HR_Knee_Tor (Nm)'
]]

# print(q_traj.to_numpy().shape)

q_array = q_traj.to_numpy()
dq_array = dq_traj.to_numpy()
ddp_array = ddp_traj.to_numpy()
u_array = u_traj.to_numpy()
t_array = timestamps.to_numpy()

EPISODS = 20
EPILENGTH = 5
EPISTEP = 200
EOMF = model.EOM_sym_func # "q", "dq", "ddq", "Q", "confsym"
cons = model.toePoses
consJ = [ca.jacobian(c,model.q) for c in cons]
toeJac = ca.vertcat(*consJ)
dtoeJac = ca.jtimes(toeJac, model.q, model.dq)
toeJacFunc = ca.Function("toeJac", [model.q], [toeJac])
dtoeJacFunc = ca.Function("dtoeJac", [model.q, model.dq], [dtoeJac])

x_plot = []
ddq_plot = []
F_plot = []
u_plot = []
t_plot = []
opt._parse.update({
    "x": lambda: ca.horzcat(*x_plot),
    "ddq": lambda: ca.horzcat(*ddq_plot),
    "F": lambda: ca.horzcat(*F_plot),
    "u": lambda: ca.horzcat(*u_plot),
    "t": lambda: ca.horzcat(*t_plot)
})


for I in range(EPISODS):
    for i in range(EPISTEP*I, EPISTEP*I + EPILENGTH):
        p = opt.addNewVariable("p%d_%d"%(I,i), -ca.inf*ca.DM.ones(3), ca.inf*ca.DM.ones(3), ca.DM.zeros(3)) # x,y,z
        dp = opt.addNewVariable("dp%d_%d"%(I,i), -ca.inf*ca.DM.ones(3), ca.inf*ca.DM.ones(3), ca.DM.zeros(3)) # dx,dy,dz
        ddq = opt.addNewVariable("ddq%d_%d"%(I,i), -ca.inf*ca.DM.ones(15), ca.inf*ca.DM.ones(15), ca.DM.zeros(15)) # ddr ... dd HR
        F = opt.addNewVariable("dp%d_%d"%(I,i), ca.DM([-ca.inf, -ca.inf, 0]*4), ca.inf*ca.DM.ones(12), ca.DM.zeros(12)) # [ASSUME] the ground is always flat

        x = ca.vertcat(p, q_array[i])
        dx = ca.vertcat(dp, dq_array[i])
        ddx = ca.vertcat(ddp_array[i], ddq)
        u = u_array[i]
        _toeJac = toeJacFunc(x)
        _dtoeJac = dtoeJacFunc(x, dx)
        Q = model.B @ u + _toeJac.T @ F
        t = t_array[i]

        eom_slack = opt.addNewVariable("eom%d_%d"%(I,i), ca.DM([-ca.inf]*18), ca.DM([ca.inf]*18), ca.DM([0]*18))
        int_slack = opt.addNewVariable("eom%d_%d"%(I,i), ca.DM([-ca.inf]*36), ca.DM([ca.inf]*36), ca.DM([0]*36))
        opt.addCost(lambda: 1e10 * ca.dot(eom_slack, eom_slack))
        opt.addCost(lambda: 1e10 * ca.dot(int_slack, int_slack))

        ## EOM constraint
        eom = EOMF( x, dx, ddx, Q, m)
        # opt.addConstraint(lambda: eom, ca.DM.zeros(eom.size()), ca.DM.zeros(eom.size()))
        # opt.addConstraint(lambda: eom, ca.DM.zeros(eom.size()), ca.DM.zeros(eom.size()))
        opt.addConstraint(lambda: eom - eom_slack, ca.DM([-ca.inf]*18), ca.DM([0]*18))
        opt.addConstraint(lambda: eom + eom_slack, ca.DM([0]*18), ca.DM([ca.inf]*18))

        ## Integration 
        if(opt._state.get("x",None) is not None):
            _x = opt._state["x"]
            _dx = opt._state["dx"]
            _ddx = opt._state["ddx"]
            _t = opt._state["t"]
            # opt.addCost(lambda: 1e2 * normQuad( x - _x - _dx * (t - _t)) )
            # opt.addCost(lambda: 1e2 * normQuad( dx - _dx - _ddx * (t - _t)) )
            opt.addConstraint(lambda: x - _x - _dx * (t - _t) - int_slack[:18], ca.DM([-ca.inf]*18), ca.DM([0]*18))
            opt.addConstraint(lambda: x - _x - _dx * (t - _t) + int_slack[:18], ca.DM([0]*18), ca.DM([ca.inf]*18))
            opt.addConstraint(lambda: dx - _dx - _ddx * (t - _t) - int_slack[18:], ca.DM([-ca.inf]*18), ca.DM([0]*18))
            opt.addConstraint(lambda: dx - _dx - _ddx * (t - _t) + int_slack[18:], ca.DM([0]*18), ca.DM([ca.inf]*18))

        ## Contact force complement
        ## [ASSUME] no relative translation of contact
        dpc = _toeJac @ dx # the velocity of contact points
        ddpc = _dtoeJac @ dx + _toeJac @ ddx
        # complementSlack = lambda y,z: y+z-ca.sqrt(y**2 + z**2 + 1e-7)
        for _dpc, _ddpc, _f in zip(ca.vertsplit(dpc, 3), ca.vertsplit(ddpc, 3), ca.vertsplit(F, 3)):
            opt.addCost(lambda: 1e5 * normQuad(_dpc) * normQuad(_f))
            opt.addCost(lambda: 1e5 * normQuad(_ddpc) * normQuad(_f))
            # opt.addConstraint(lambda: complementSlack(normQuad(_dpc), normQuad(_f) ) , ca.DM([0]), ca.DM([0]) )
            # opt.addConstraint(lambda: complementSlack(normQuad(_ddpc), normQuad(_f) ), ca.DM([0]), ca.DM([0])  )

        opt._state.update({"x":x, "dx":dx, "ddx":ddx, "t":t}) 
        x_plot.append(ca.vertcat(x,dx))
        ddq_plot.append(ddq)
        F_plot.append(F)
        u_plot.append(u)
        t_plot.append(t)  


    opt._state.update({"x":None, "dx":None, "ddx":None, "t":None})

res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True,
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 20000, # unkown option
                }
            })
print(res['x'][:10])
print(res['mass'])
dumpname = os.path.abspath(os.path.join("./data/nlpSol", "modelID%d.pkl"%time.time()))

with open(dumpname, "wb") as f:
    pkl.dump({
        "sol":res
    }, f)
