import sys
sys.path.append(".")

import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import sys
import pandas as pd
import numpy as np
from codogs.heavyRopeLoad import HeavyRopeLoad
import casadi as ca
if(len(sys.argv)<2):
    print("please input the solution file name")
    exit()

solFile = sys.argv[1]

with open(solFile, "rb") as f:
    solraw = pkl.load(f)
    sol = solraw["sol"]
    nc = solraw["nc"]
    r = solraw["r"]
    obs_list = solraw.get('obstacles',[])
    lineobs_list = solraw.get('lineObstacles',ca.DM([])).full().reshape(-1)
    boxobs_list = solraw.get('boxObstacles',ca.DM([])).full().reshape(-1)
    pc = solraw["pc"]
    # sol_ddq = sol["ddq_plot"].full().T
    sol_x= sol['Xgen']['x_plot'].full().T
    sol_u= sol['Ugen']['u_plot'].full().T
    sol_lam = sol["ml_plot"].full().T
    sol_lam = np.vstack([np.zeros([1,nc]),sol_lam])
    print("SOLVE TIME:", solraw['EXECTIME'])
    # print(np.all(0<sol["comS_plot"].full()+1e-6))
    # print(np.all(0>sol["comS_plot"].full()-1e-6))
    # print(np.all(sol["_gb"][:,0].full()<sol["_g"].full()+1e-6))
    # print(np.all(sol["_g"].full()<sol["_gb"][:,1].full()+1e-6))

    # print("\ndisj_x", sol["disj_x"])
    # print("\ndisj_y", sol["disj_y"])
    # print("\ndisj_eps", sol["disj_eps"])
    # print("\ndisj_g", sol["disj_g"])
    # print(np.all(sol["_g"].full()<sol["_gb"][:,1].full()+1e-6))

model = HeavyRopeLoad(nc)
model.setHyperParamValue({
    "r": r,
    "pc": pc,

})

pfuncs = model.pFuncs

fig, ax = plt.subplots()
def animate(i):
    ind = i%len(sol_x)
    xsol = sol_x[ind]
    usol = sol_u[ind]
    mlsol = sol_lam[ind]
    ax.clear()
    
    sth = np.sin(xsol[2])
    cth = np.cos(xsol[2])
    L,W = 2,2
    hl,hw = L/2, W/2

    box, = ax.plot(np.array([cth*hl-sth*hw, -cth*hl-sth*hw, -cth*hl+sth*hw, cth*hl+sth*hw, cth*hl-sth*hw])+xsol[0], 
                  np.array([sth*hl+cth*hw, -sth*hl+cth*hw, -sth*hl-cth*hw, sth*hl-cth*hw, sth*hl+cth*hw])+xsol[1])
    pcs = pfuncs(xsol)
    lines=[
        ax.plot([pcs[i,0], usol[2*i]],[pcs[i,1], usol[2*i+1]], label="rope%d"%i, lw=(1+ml/(ml+1e-2)))
        for i,ml in enumerate(mlsol)
    ]
    if(len(obs_list)):
        for obs_x, obs_y, obs_r in np.hsplit(np.array(obs_list), len(obs_list)//3):
            ax.plot(obs_x+obs_r*np.array([np.cos(2*ca.pi*i/19) for i in range(20)]),
                    obs_y+obs_r*np.array([np.sin(2*ca.pi*i/19) for i in range(20)]))
    
    if(len(lineobs_list)):
        for flag, p1x, p1y, p2x, p2y in np.hsplit(np.array(lineobs_list), len(lineobs_list)//5):
            if(np.abs(flag)>1e-6):
                ax.plot([p1x, p2x],
                        [p1y, p2y])

    if(len(boxobs_list)):
        for x, y, th, bl, bw in np.hsplit(np.array(boxobs_list), len(boxobs_list)//5):
            bl = bl/2
            bw = bw/2
            c = np.cos(th)
            s = np.sin(th)
            ax.plot(x+ np.array([c*bl-s*bw, -c*bl-s*bw, -c*bl+s*bw, c*bl+s*bw, c*bl-s*bw]),
                    y+ np.array([s*bl+c*bw, -s*bl+c*bw, -s*bl-c*bw, s*bl-c*bw, s*bl+c*bw]))


    ax.legend()

    # Obstacles


    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)

    return (box,*[l[0] for l in lines])
    # return linesol,til

ani = animation.FuncAnimation(
    fig, animate, interval=100, blit=True, save_count=50)

fig = plt.figure()

plt.plot(np.array([
    [np.linalg.norm([pfuncs(x_)[i,0] - u[2*i], pfuncs(x_)[i,1] - u[2*i+1]  ]) for i in range(nc)]
    for x_,u in zip(sol_x[1:], sol_u) # Note: the length cons enforces on u0 and x1
 ]) )
plt.legend( ["rope%d"%i for i in range(nc)])

plt.figure()
plt.plot(sol_x, ".")

plt.show()

