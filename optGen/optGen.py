from optGen.util import LazyFunc, kwargFunc, caSubsti, MXinSXop, getName
import casadi as ca
import numpy as np
import os
import shutil
import time

"""[optGen]: optimization problem generator
Contains cost, variable, and constraints. 
Forms a tree structure for building opt problem and parsing its solution

Each derived class needs to define all the following:
    - `_state`
    - `_child`
    - `_parse`
    - `_begin`
    - `step`
    - `chMod`

Two steps for building an opt problem:
    - `begin`
    - `step`
    derivative classes should implement their `_begin`, and is called automatically
    derivative classes should call the `step` function of their children manuly

`addConstraint` and `addCost` acts on the current state. 
    These two methods call a factory function with `_state` and get returned J or g
    Note: the calling of the factory func depends on argument names of the func, rather than its order
        e.g. if _state has keys x,u,F. 
            Then the function with argument (x,F,u) or (u) are all valid, 
            but ones with argument (x0, u) is not valid

hyperParams (dict: {name: (shape, SX, MX)}): e.g. the distance for the robot to jump is a user defined value, but 
    appears in w0 and g as a free MX symbol. 
    
"""
class optGen:
    VARTYPE = ca.MX
    def __init__(self):
        self._w = []    # variable
        self._w0 = []
        self._lbw = []
        self._ubw = []
        self._J = 0     # cost
        self._g = []    # constraint
        self._lbg = []
        self._ubg = []
        self._sol = None

        """ "current state" of the optimization building
            these state is used for adding constraints and costs
            should be updated in _begin and step
        """
        self._state = {} 

        """
            children are all optGens
            NOTE: depends on the unity of the order of values() and keys() etc.
            "In Python 3.7.0 the insertion-order preservation nature of 
                dict objects has been declared to be an official part of 
                the Python language spec"
        """
        self._child = {} 

        """
            The name and values for parsing the solution
                Note: The value should be functions, 
                as the class is only complete once the problem is built
                e.g. x_plot and u_plot
        """
        self._parse = {
            "_w": lambda: ca.vertcat(*self._w),
            "_g": lambda: ca.vertcat(*self._g),
            "_gb": lambda: ca.horzcat(ca.vertcat(*self._lbg), ca.vertcat(*self._ubg)),
            "_J": lambda: self._J
        }

        """_parseSol: the solution parser
        should have the signature
            w -> [sol1, sol2, <childname1>, <childname2> ...]
        """
        self._parseSol = LazyFunc(
            lambda: ca.Function('%sParse'%self.__class__.__name__, 
                [self.w]+self.hyperParamList(self.w), 
                    [c() for c in self._parse.values()] + [c.w for c in self._child.values()], 
                ['w']+self.hyperParamList("name"), 
                    [n for n in self._parse.keys()] + [n for n in self._child.keys()])
        )

        """_hyperParams:  free variables and their values. Defaults to None.
            Those free variables might involves building the w0 or the cost function etc.
            {name: (shape, SX, MX, value)}. We need SX and MX together as the traj cal(e.g. w) uses MX 
                                    where the init value cal (e.g. w0) uses SX
        """
        self._hyperParams = {}

    @property
    def w(self):
        """[MX,SX]: the variables of this and child"""
        return ca.vertcat(*self._w, *[c.w for c in self._child.values()])

    @property
    def w0(self):
        """narray: the init value of w"""
        return ca.vertcat(
                *self._w0, 
                *[c.w0 for c in self._child.values()])

    @property
    def lbw(self):
        """narray: the lower bound of w"""
        return ca.vertcat(
                *self._lbw,
                *[c.lbw for c in self._child.values()])

    @property
    def ubw(self):
        """narray: the upper bound of w"""
        return ca.vertcat(
            *self._ubw,
            *[c.ubw for c in self._child.values()])

    @property
    def g(self):
        """[MX,SX]: the variables of this and child"""
        return ca.vertcat(
                *self._g, *[c.g for c in self._child.values()])


    @property
    def lbg(self):
        """narray: the lower bound of g"""
        return ca.vertcat(
                *self._lbg,
                *[c.lbg for c in self._child.values()])

    @property
    def ubg(self):
        """narray: the upper bound of g"""
        return ca.vertcat(
                *self._ubg,
                *[c.ubg for c in self._child.values()])

    @property
    def J(self):
        """[SX,MX]: the cost function"""
        return ca.sum1(ca.vertcat(self._J, *[c.J for c in self._child.values()]))

    @property
    def hyperParams(self):
        return {**self._hyperParams}

    def updateHyperParams(self, hp = None):
        hp ={} if hp is None else hp
        self._hyperParams.update(hp)
        [c.updateHyperParams(self._hyperParams) for c in self._child.values()]

    def newhyperParam(self, hp, shape=None, name=None):
        """Add a new hyper parameter to the system. Return the SX version

        Args:
            hp (string or ca.SX): the hp of the hyperparam
            shape (tuple, optional): the shape. Defaults to (1,1).
            name: (only when hp is ca.SX)the name of the hyper parameter.
        Returns:
            [SX]: SX version of the hyper parameter
        """
        if(isinstance(hp, str)):
            assert(hp not in self._hyperParams.keys())
            shape = (1,1) if shape is None else shape
            if(len(shape)==1): shape += (1,)
            self._hyperParams[hp] = (shape, ca.SX.sym(hp, *shape), ca.MX.sym("%s_mx"%hp, *shape), None)
            ret = self._hyperParams[hp][1]
        elif(isinstance(hp, ca.SX)):
            name = getName(hp) if name is None else name
            self._hyperParams[name] = (hp.size(), hp, ca.MX.sym("%s_mx"%name, *hp.size()), None)
            ret = hp
        else:
            raise TypeError("the hp should be eitehr a string or a SX, but got %s"%str(type(hp)))
        self.updateHyperParams()
        return ret

    
    def hyperParamList(self, t):
        """get a list of hyperParameters. e.g. MX list, SX list, or all names

        Args:
            t ({string; MX; SX}): "name", "value" or ca.MX or ca.SX
        """
        if(t=="name"):
            return list(self._hyperParams.keys())
        if(t=="value"):
            return [(m if v is None else v) for z,s,m,v in self._hyperParams.values()]
        elif(isinstance(t, ca.MX) or t is ca.MX):
            return [m for z,s,m,v in self._hyperParams.values()]
        elif(isinstance(t, ca.SX) or t is ca.SX or isinstance(t, ca.DM) or t is ca.DM):
            return [s for z,s,m,v in self._hyperParams.values()]
        if(t.is_empty()):
            return [m for z,s,m,v in self._hyperParams.values()]
        else:
            raise ValueError("t should be 'name' or SX MX class or objects, but get %s"%(type(t)))
        
    def setHyperParamValue(self, hpdict):
        for k,v in hpdict.items():
            self._hyperParams[k] = (*(self._hyperParams[k][:3]),v )
        self.updateHyperParams()

    def tryCallWithHyperParam(self, func, kwargs):
        try:
            res = func(**kwargs)
        except TypeError as e:
            if("operand type(s)" not in str(e) and "Wrong number or type of arguments" not in str(e)):
                raise e
            res = MXinSXop(func, kwargs, 
                list(zip(self.hyperParamList("name"),
                         self.hyperParamList(ca.SX),
                         self.hyperParamList(ca.MX))))
        return res

    def buildParseSolution(self, name, solEx):
        """Build a casadi function that extract the target value given optimized result

        Args:
            name (string): the name of the function
            solEx (solDict->MX): A function that takes in solDict and output the target MX, solDict is what `parseSol` generates

        Returns:
            [casadi.Function]: [x] -> [targetMX]
        """
        solDict = self.parseSol({"x": self.w})
        target = solEx(solDict)
        return ca.Function("parse_%s"%name, 
                self.hyperParamList(self.w) + [ self.w ], [target], 
                self.hyperParamList("name") + ["x"], [name])

    def substHyperParam(self, target):
        target = ca.DM(target) if isinstance(target, np.ndarray) else target
        target = ca.DM(target) if isinstance(target, ca.SX) and target.is_constant() else target
        return caSubsti(target, self.hyperParamList(target), self.hyperParamList("value"))

    def loadSol(self, sol):
        """[loadSol]: load the solution and pass the corresponding 
            solution to child
        Args:
            sol ([narray, MX]): the solution of the optimization problem
        """
        self._sol = sol
        parseRes = self._parseSol(w = sol['x'], **{k:v for k,v in zip(self.hyperParamList("name"), self.hyperParamList("value"))})
        for n,c in self._child.items():
            c.loadSol({'x':parseRes[n]})
    
    def parseSol(self, sol):
        parseType = sol['x'] if(isinstance(sol['x'], ca.MX) or isinstance(sol['x'], ca.SX)) else "value"
        return { k: 
                v if k not in self._child.keys()
                  else self._child[k].parseSol({'x':v})
            for k,v in self._parseSol(w=sol['x'], **{k:v for k,v in zip(self.hyperParamList("name"), self.hyperParamList(parseType))}).items()
            }
    
    def buildSolver(self, solver = "ipopt", options = None):
        """build the nlp solver for solving the problem

        Args:
            solver (str, optional): the algorithm to use. Defaults to "ipopt".
            options (dict, optional): the options for the nlp solver.

        Returns:
            ca.Function: the solver function
        """
        options = {} if options is None else options
        prob = {'f':self.substHyperParam(self.J),
                'x': self.substHyperParam(self.w),
                'g': self.substHyperParam(self.g)}

        solver = ca.nlpsol('solver', solver, prob, options)
        return solver

    def begin(self, **kwargs):
        for c in self._child.values():
            c.begin(**kwargs)
        self._begin(**kwargs)

    def solve(self, options = None):
        solver = self.buildSolver(options = options)
        start_time = time.time()
        self._sol = solver( x0=self.substHyperParam(self.w0),
                            lbx=self.substHyperParam(self.lbw),
                            ubx=self.substHyperParam(self.ubw),
                            lbg=self.substHyperParam(self.lbg),
                            ubg=self.substHyperParam(self.ubg))
        exec_seconds = time.time() - start_time
        self._sol.update(self.parseSol(self._sol))
        self._sol["exec_sec"]= exec_seconds
        return self._sol

    # Add constriant of the state of last step
    def addConstraint(self, func, lb, ub):
        g = self.tryCallWithHyperParam(kwargFunc(func), self._state)
        assert g.size(1)==lb.size(1)==ub.size(1), "constraint's dim is not the same with bound"
        self._g.append(g)
        self._lbg.append(lb)
        self._ubg.append(ub)
    
    # Add constriant of the state of last step
    def addCost(self, func):
        self._J += self.tryCallWithHyperParam(kwargFunc(func), self._state)

    def _begin(self,**kwargs):
        raise NotImplementedError
        
    def step(self, **kwargs):
        raise NotImplementedError
    
    def chMod(self, modName, *args, **kwargs):
        """ change mode
            a method used for receive "mode change signales" from user
            e.g. in dTGenVariable, this method is called whenever the 
                traj reaches a new mode and needs a new dT
        Args:
            modName (str): the name of the new mod
        """
        raise NotImplementedError

    def cppGen(self, cppname, expand=True, parseFuncs = None, genFolder = True, cmakeOpt = None):
        """Generate cpp files of the optimization problem. Specifically, the `_info` `f` `g` `grad_` `hessian` and prse functions

        Args:
            cppname (string): The name of the cpp file. If `genFolder` is True, then this is the folder of the cpp  
            parseFuncs ([(string:name, sol->MX : extractFunction ) ], optional): A list of functions to parse the solution. See `buildParseSolution` for details  
            genFolder (bool, optional): Whether to generate functions separetly into a folder. Defaults to True.  
        """
        parseFuncs = [] if parseFuncs is None else parseFuncs
        cppOptions = {"cpp": True, "with_header": True,"verbose":False}
        
        # Build different policies according to whether to generate the functions in 
        # a single file or in a folder
        if(not genFolder):
            C = ca.CodeGenerator(cppname, cppOptions) # comments in the generated file
            def addFunction(f):
                f = f.expand() if expand else f
                C.add(f)
            def outputFile():
                C.generate()
        else:
            hfiles = []
            try:
                os.makedirs(cppname)
            except FileExistsError:
                pass
            def addFunction(f):
                f = f.expand() if expand else f
                f.generate(f.name(), cppOptions)# casadi seems like not able to generate in subdir
                shutil.move("%s.cpp"%f.name(), os.path.join(cppname, "%s.cpp"%f.name())) # this will overwrite if exist
                shutil.move("%s.h"%f.name(), os.path.join(cppname, "%s.h"%f.name()))
                os.path.join(cppname, f.name())
                hfiles.append("%s.h"%f.name())
            def outputFile():
                with open(os.path.join(cppname, "interface.h"), "w") as f:
                    for h in hfiles:
                        f.write('#include "%s"\n'%h)
                    f.write('struct hyperParameters\n{\n')
                    for k,v in self._hyperParams.items():
                        f.write('\ttypedef double %s[%d];\n'%(k,v[0][0]*v[0][1]))
                    f.write("};\n")
                if(cmakeOpt is not None):
                    cmakeOption = {"projName": "casaditoGen",
                                "cxxflag": '"-O1"',
                                "libName": "nlpgen"}
                    cmakeOption.update(cmakeOpt)
                    with open(os.path.join(cppname, "CMakeLists.txt"), "w") as f:
                        f.write(
                        'project({projName})\n'\
                        'cmake_minimum_required(VERSION 3.7)\n'\
                        'set(CMAKE_BUILD_TYPE Release)\n'\
                        'SET(CMAKE_CXX_FLAGS_RELEASE {cxxflag})\n'\
                        'link_directories(/usr/local/lib)\n'\
                        'file(GLOB NlpGenFiles "*.cpp")\n'\
                        'add_library({libName} ${{NlpGenFiles}})\n'\
                        .format(**cmakeOption))
            
        glen = self.g.size(1)
        jacg = ca.jacobian(self.g, self.w)
        lams = type(self.w).sym("lambda", self.g.size(1))
        sigm = type(self.w).sym("sigma_f") # the objective factor for calculating the hessian of the lagrange
        hLs = [ca.hessian(self.g[i], self.w ) for i in range(glen)] # the result contains hessian and gradient of each g
    
        hessL = sigm * ca.hessian(self.J, self.w)[0] + sum([lams[i] * H for i, (H,j) in enumerate(hLs)])

        nlp_info = ca.Function("nlp_info",[],
        [ca.DM(self.w.size(1)),        # Storage for the number of variables x
         ca.DM(self.g.size(1)),        # Storage for the number of constraints g(x)
        #  ca.DM(len(jacg.nonzeros())),    # Storage for the number of nonzero entries in the Jacobian
        #  ca.DM(len(hessL.sparsity().get_lower()))], # Storage for the number of nonzero entries in the Hessian
         ca.DM(jacg.sparsity().nnz()),    # Storage for the number of nonzero entries in the Jacobian
         ca.DM(hessL.sparsity().nnz_lower())], # Storage for the number of nonzero entries in the Hessian

        [],["n", "m", "nnz_jac_g", "nnz_h_lag"])

        addFunction(nlp_info)

        bounds_info = ca.Function("bounds_info", self.hyperParamList(self.lbw),
        [self.lbw,        # the lower bounds xL for the variables  x
         self.ubw,        # the upper bounds xU for the variables 
         self.lbg,    # the lower bounds gL for the constraints
         self.ubg], # the upper bounds gU for the constraints 
        self.hyperParamList('name'),["x_l", "x_u", "g_l", "g_u"]
        )

        addFunction(bounds_info)

        starting_point = ca.Function("starting_point", self.hyperParamList(self.w0),
        [self.w0],	#the initial values for the primal variables x
        self.hyperParamList('name'),["x"]
        )

        addFunction(starting_point)

        # The parameter conversion is hyper parameter first, and the interface parameter follows
        # So that feeding the hyper parameters makes in becomes a interface method for IPOPT
        hyperAndWsym = self.hyperParamList(self.w) + [ self.w ]
        hyperAndWname = self.hyperParamList('name')+["x"]
        
        eval_f = ca.Function("nlp_f", hyperAndWsym,
        [self.J],
        hyperAndWname, ["f"])
        
        addFunction(eval_f)


        eval_grad_f = ca.Function("nlp_grad_f", hyperAndWsym,
        [ca.gradient(self.J, self.w)],
        hyperAndWname, ["grad_f"])
                
        addFunction(eval_grad_f)
        

        eval_g = ca.Function("nlp_g", hyperAndWsym,
        [self.g],
        hyperAndWname, ["g"])

        addFunction(eval_g)


        eval_jac_g = ca.Function("nlp_grad_g", hyperAndWsym, # note: this method only return the Jg
        [jacg],
        hyperAndWname, ["grad_g"])

        addFunction(eval_jac_g)

        maskLower = ca.DM.ones(ca.Sparsity.lower(hessL.size(1)))
        eval_h = ca.Function("nlp_h", hyperAndWsym+[sigm, lams], # note: this method only return the h
        [hessL * maskLower],
        hyperAndWname+["ms", "ml"], ["h"])

        addFunction(eval_h)

        # for f in self.allSolutionParse:
        #     addFunction(f)

        for n,t in parseFuncs:
            addFunction(self.buildParseSolution(n,t))

        outputFile()