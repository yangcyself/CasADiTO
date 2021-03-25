
from casadi import *
import yaml


ConfigFile = "data/robotConfigs/robot1.yaml"
PI = np.math.pi

with open(ConfigFile, 'r') as stream:
    try:
        robotParam = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

# Declare model variables
px = SX.sym('px')
py = SX.sym('py')
th = SX.sym('th')

dpx = SX.sym('dpx')
dpy = SX.sym('dpy')
dth = SX.sym('dth')

def addToe(name = ""):
    ptoex = SX.sym("ptoe%sx"%name)
    ptoey = SX.sym("ptoe%sy"%name)

    dptoex = dpx - (ptoey - py) * dth
    dptoey = dpy + (ptoex - px) * dth
    

params = {
    "legL1":robotParam["l1"],
    "legL2":robotParam["l2"],
    "legLc1":robotParam["lc1"],
    "legLc2":robotParam["lc2"],
    "torL":robotParam["L"],
    "legM1":robotParam["m1"],
    "legM2":robotParam["m2"],
    "torM":robotParam["M"],
    "legI1":robotParam["j1"],
    "legI2":robotParam["j2"],
    "torI":robotParam["J"],
    "q1Lim": [-PI/2 - robotParam["ang1max"], -PI/2 - robotParam["ang1min"]],
    "q2Lim": [-robotParam["ang2max"], -robotParam["ang2min"]],
    "dq1Lim": [-robotParam["dang1lim"], robotParam["dang1lim"]],
    "dq2Lim": [-robotParam["dang2lim"], robotParam["dang2lim"]],
    "G":9.81,
}