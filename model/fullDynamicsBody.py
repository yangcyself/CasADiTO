from model.articulateBody import *
from model.articulateFullBody import *
"""The Articulate system of full body
using x,y,z, rx,ry,rz , q1, q2, ... as the state
"""
class FullDynamicsBody(ArticulateSystem):
    def __init__(self, urdfFile, eularMod = "ZYX", toeList = None):
        """[summary]

        Args:
            urdfFile ([type]): [description]
            eularMod (str, optional): [description]. Defaults to "ZYX".
            toeList ((linkName:str, pose_in_link_frame: vec3), optional): The list of position of toes
        """
        root = urdfWrap3D.FloatBaseUrdf('base', urdfFile, eularMod)
        super().__init__(root)
        self.urdfBase = self.root.child[0]

        # add the toes, used for buildEOMF
        self.toeList = [] if toeList is None else toeList
        self.toePoses = [
            self.getGlobalPos(n, p)
            for n,p in self.toeList
        ]

    @property
    def u_dim(self):
        return self.urdfBase.x.size(1)

    @property
    def F_dim(self):
        return 3 * len(self.toePoses)

    @property
    def B(self):
        return ca.jacobian(self.root.x, self.urdfBase.x)

    def getGlobalPos(self, linkName, localPos):
        return (self.urdfBase._linkdict[linkName].Bp @ ca.vertcat(localPos,1))[:3]
    
    def buildEOMF(self, consMap, name=""):
        """Build the equation of Motion and constraint. Return g(x,u,F,ddq)

        Args:
            consMap ((bool,bool)): The contact condition of the toes, the same order as toelist
            name (str, optional): The name of the function. Defaults to "EOMF".

        Returns:
            g(x,u,F,ddq)->[SX]: The equation that should be zero
                g contains: 1. dynamic constriant, 2. contact point fix, 3. float point no force
        """
        ddq = ca.SX.sym("ddq",self.dim)
        F = ca.SX.sym("F",self.F_dim) #Fb, Ff
        u = ca.SX.sym("u", self.u_dim)

        cons = self.toePoses
        consJ = [ca.jacobian(c,self.q) for c in cons]
        toeJac = ca.vertcat(*consJ)


        g = self.EOM_func(self.q, self.dq, ddq, self.B @ u+toeJac.T @ F) # the EOM
        g = ca.vertcat(g, *[ cJ @ ddq + ca.jtimes(cJ,self.q,self.dq)@self.dq for cJ,cm in zip(consJ,consMap) if cm])
        g = ca.vertcat(g, *[ F[i*3:i*3+3] for i,cm in enumerate(consMap) if not cm])
        g = ca.simplify(g)
        return ca.Function("%sEOMF"%name, [self.x,u,F,ddq], [g], ["x","u","F","ddq"], ["%sEOM"%name])
    
    
if __name__ == "__main__":

    from scipy.spatial.transform import Rotation as R
    import numpy as np
    np.set_printoptions(precision = 2,linewidth=200,threshold=99999)

    m =FullDynamicsBody('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf',
        toeList=[("FL_SHANK", ca.DM([0,0,-0.2])),  ("FR_SHANK", ca.DM([0,0,-0.2])),
                 ("HL_SHANK", ca.DM([0,0,-0.2])), ("HR_SHANK", ca.DM([0,0,-0.2]))])

    # x0 = ca.DM([0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]+[0]*18)
    toeFunc = ca.Function('toe', [m.x], [m.toePoses[0]] )
    # # print(m.toePoses[0])
    # print(toeFunc(x0))
    # x0[8] = 0.1
    # print(toeFunc(x0))
    # MpFunc = ca.Function('mp', [m.x], [m.urdfBase._linkdict["INERTIA"].Mp])
    # # print(MpFunc(x0))

    # print(m.buildEOMF([1,1,1,1]))
    # print(ca.DM(m.B).full())
    x0 = ca.DM([0, 0, 0.190013, 0, 0, ca.pi/2, 0, -1, 2.1, 0, -1., 2.1, 0, -1.0, 2.1, 0, -1.0, 2.1]+[0]*18)
    JacF = ca.Function("f", [m.x], [ca.jacobian(m.toePoses[0] ,m.q)] )
    print(JacF(x0).full())
    print(toeFunc(x0))
    def quat_to_ZYX(q):
        # NOTE: The quat of scipy is xyzw norm. However, quat from raisim is wxyz
        w,x,y,z = q
        r = R.from_quat( np.array([x,y,z,w]) )
        Z,Y,X = r.as_euler('ZYX', degrees=False) # the order of the angles is the first, second, third
        return np.array([X,Y,Z])

    #######       #######       #######       #######       #######       #######
    ## Do a simple simulation using towr collocation form, enforce u to be zero
    if(False):
        from vis import saveSolution
        from optGen.trajOptimizer import TowrCollocationDefault
        dT0 = 0.01
        opt = TowrCollocationDefault(2*m.dim, m.u_dim, m.F_dim, xLim = ca.DM([[-ca.inf, ca.inf]]*2*m.dim),
            uLim= ca.DM([[-100, 100]]*m.u_dim), FLim = ca.DM([[-ca.inf, ca.inf]]*m.F_dim), dt= dT0)
        x0 = ca.DM([0, 0, 0.190013, 0, 0, 0, 0, -1, 2.1, 0, -1., 2.1, 0, -1.0, 2.1, 0, -1.0, 2.1]+[0]*18)
        opt.begin(x0=x0, u0=[0]*12, F0=[0]*12)

        EOMF = m.buildEOMF([1,1,1,1])
        print("EOMF built")
        for i in range(100):
            opt.step(lambda dx,x,u,F : EOMF(x=x,u=u,F=F,ddq = dx[m.dim:])["EOM"], # EOMfunc:  [x,u,F,ddq]=>[EOM]) 
                    x0 = x0, u0=[0]*12, F0=[0]*12)
            opt.addCost(lambda x: ca.dot(x-x0, x-x0))
            opt.addCost(lambda x: 1e3*ca.dot(x[3:6], x[3:6]))
        print("Opt Set")
        res = opt.solve(options=
            {"calc_f" : True,
            "calc_g" : True,
            "calc_lam_x" : True,
            "calc_multipliers" : True,
            "expand" : True, 
                "verbose_init":True,
                # "jac_g": gjacFunc
            "ipopt":{
                "max_iter" : 2000, # unkown option
                }
            })

        print(res["Xgen"]["x_plot"])
        print(res["Ugen"]["u_plot"])
        print(res["Fgen"]["F_plot"])
        sol_x = res["Xgen"]["x_plot"].full().T
        # get corres cols x,y,r,bh,bt,fh,ft,dx,dy,dr,dbh,dbt,dfh,dft
        sol_x = sol_x[:, [0,2,4,13,14, 7, 8,18,20,22, 31, 32, 25, 26]]
        sol_u = res["Ugen"]["u_plot"].full().T
        sol_u = sol_u[:, [7, 8, 1, 2]]
        timeStamps = res["dTgen"]["t_plot"]
        saveSolution("out.csv", sol_x, sol_u, timeStamps.full().reshape(-1), transform=False)
        # exit()
    #######       #######       #######       #######       #######       #######       
    ## Generate a cpp version forward dynamics function to be compared with raisim
    # result: the Cg have large diviation, but KE, PE, D mat all similar
    if(True):
        print("LINKS:", m.x)
        # C = ca.CodeGenerator("dyn", {"cpp": True, "with_header": True,"verbose":False})
        D_func = ca.Function("Dfunc", [m.x], [m.D])
        Cg_func = ca.Function("Cgfunc", [m.x], [m.Cg])
        KEPE_func = ca.Function("KEPEfunc", [m.x], [ca.vertcat(m.root.KE, m.root.PE)])
        FLpos_func = ca.Function("FLposfunc", [m.x], [m.urdfBase._linkdict["FL_SHANK"].Bp])
        toeFunc = ca.Function("toeFunc", [m.x], [m.toePoses[0]])

        for i in range(18):

            state = np.genfromtxt("data/dynMatrix/%d_state.csv"%i, delimiter=',')
            print("State:", state)
            state = list(state)
            state[3:7] = quat_to_ZYX(state[3:7]) # change rotation
            state = ca.DM(state)          
            state[21:24] = mathUtil.omega_W2dZYX(state[3:6], state[21:24])
            print("State:", state.full().reshape(-1))
            print("rotationVelocity", state[21:24])
            D = np.genfromtxt("data/dynMatrix/%d_Massmat.csv"%i, delimiter=',')
            Cg = np.genfromtxt("data/dynMatrix/%d_Nonlinearity.csv"%i, delimiter=',')
            E = np.genfromtxt("data/dynMatrix/%d_Energy.csv"%i, delimiter=',')
            FLpos = np.genfromtxt("data/dynMatrix/%d_FL_pos.csv"%i, delimiter=',')
            # print(state)
            print("\nD_func Error")
            print(np.array(D_func(state)- D))
            # print("\nD_func")
            # print(D_func(state)[:5,:5])
            print("\nCg_func Error")
            print(np.array(Cg_func(state) - Cg).reshape(-1))
            print(np.array(Cg_func(state)).reshape(-1))
            print(np.array(Cg).reshape(-1))

            print("\nE")
            print(E)
            print(KEPE_func(state).full().T)

            print("\nFL_pos")
            print(FLpos)
            print(FLpos_func(state).full())
            print(toeFunc(state).full().T)
            exit()
            # print(Cg_func(state)- Cg)
    ######
    # Test the EOMF
    if(True):
        from utils.mathUtil import solveLinearCons
        from utils.caUtil import caSubsti, caFuncSubsti
        last_state = None
        EOMF = m.buildEOMF([1,1,1,1])
        for i in range(20):
            state = np.genfromtxt("data/dynMatrix/%d_state.csv"%i, delimiter=',')
            force = np.genfromtxt("data/dynMatrix/%d_Force.csv"%i, delimiter=',')[6:]

            state = list(state)
            state[3:7] = quat_to_ZYX(state[3:7]) # change rotation
            state = ca.DM(state)
            state[21:24] = mathUtil.omega_W2dZYX(state[3:6], state[21:24])

            if(last_state is not None):
                ddx = (state - last_state)/1e-3
                # print("\nstate:", state)
                # print("dx:", ddx[:18])
                # print("ddx", ddx[18:])
                # print("force:", force)
                x0 = ca.DM([0, 0, 0.190013, 0, 0, ca.pi/2, 0, -1, 2.1, 0, -1., 2.1, 0, -1.0, 2.1, 0, -1.0, 2.1]+[0]*18)

                initSol = solveLinearCons(caFuncSubsti(EOMF, {"x":x0}), [("ddq", np.zeros(18), 1e3)])
                print("ddq", initSol['ddq'])
                print("F", initSol['F'])
                print("u", initSol['u'])
                # print(EOMF(state, force, initSol['F'],  ddx[18:] ))
                break
            last_state = state
        # print(D_func(ca.DM.rand(m.x.size())))
        # print(D_func(ca.DM.rand(m.x.size())).size())
        # C.add(D_func)
        # C.generate()


