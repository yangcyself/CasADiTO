"""This file follows `articulateBody.py`, uses `urdfReader.py`
defines the 3Dbody and articulated system from urdf
"""

from model.articulateBody import *
from model.urdfReader import urdfReader

class urdfWrap3D(Body3D):
    """The wrapper of the urdf parse result from urdfReader
    """
    @staticmethod
    def FloatBaseUrdf(name, urdffile, eularMod = "ZYX"):
        """Build a floating base body, whose body is represented by 6 varialbes
        """
        root = Base3D(name, 0, 0, eularMod)
        root.addChild(urdfWrap3D.fromUrdf, urdffile = urdffile)
        return root

    @staticmethod
    def fromUrdf(urdffile, Fp=None, baseName = None, workingDict = None):
        """Generate urdfWrap3D bodies from a urdffile
        Args:
            urdffile (string): The file to load
            Fp (The parent frame): The parent frame
            baseName(string): The link to add, if default None, the return the base_link.
                    What's more, if the baseName is None, it will calculate the `_Bpurdf` for all decendents
            workingDict (dict, optional): The dictionary to working on recursively. 
                Default to init a new empty dict
        """
        workingDict  = {} if workingDict is None else workingDict

        urdfmodel = workingDict.get("urdfmodel", None)
        if(urdfmodel is None): urdfmodel = urdfReader(urdffile) # must use if, otherwise urdfReader still got validated
        workingDict['urdfmodel'] = urdfmodel

        linkdict = workingDict.get("linkdict", None) # the information of links from urdfmodel
        if(linkdict is None): linkdict = urdfmodel.linkDict
        workingDict['linkdict'] = linkdict

        bodydict = workingDict.get("bodydict", {})
        workingDict['bodydict'] = bodydict
        urdfBase = workingDict.get("urdfBase", None)

        baseName = urdfmodel.robot_urdf.base_link.name if baseName is None else baseName
        baselinkdict = linkdict[baseName]
        freeD = 1 if baselinkdict['joint'] is not None and not baselinkdict['fixed'] else 0
        baselink = urdfWrap3D(baseName, freeD, baselinkdict['mass'], baselinkdict['inertia'], baselinkdict['origin'],
            Fp = Fp,urdfBase=urdfBase)
        bodydict.update({baseName: baselink})
        if(urdfBase is None):
            workingDict['urdfBase'] = baselink
        newchilds = [urdfWrap3D.fromUrdf(urdffile, Fp, cn, workingDict) for cn in baselinkdict['childs']]
        for c in newchilds:
            c.parent = baselink
        baselink.child += newchilds
        if(baselinkdict['fixed']): baselink.fix()

        if baseName != urdfmodel.robot_urdf.base_link.name: 
            return baselink
        #### Update the _Bp and Fp of all the decendents
        dofvars = baselink.x
        def getdofNames(rootName):
            tmpl =  [linkdict[rootName]['joint'] if not linkdict[rootName]['fixed'] else None] \
                + [jn for cn in linkdict[rootName]['childs'] for jn in getdofNames(cn)]
            return list(filter(lambda a: a is not None,tmpl))
        dofNames = getdofNames(baseName)

        def childUpdateworker(root,rootName):
            fkfunc = urdfmodel.getFk(rootName, dofNames)
            root._Bpurdf = fkfunc(dofvars)
            [childUpdateworker(c,cn) for c,cn in zip(root.child, linkdict[rootName]['childs'])]
        childUpdateworker(baselink, baseName)
        baselink._linkdict = bodydict
        return baselink


    def __init__(self, name, freeD, M, I, MO, Bpurdf = None, Fp=None, urdfBase=None, g=None):
        RMo = MO[:3,:3]
        I_b = RMo @ I @ RMo.T # the inertia tensor in Bp
        super().__init__(name, freeD, M, I_b, Fp=Fp, g=g)
        self._Bpurdf = Bpurdf # the transimition in urdf 
        self._MO = MO # the origin of CoM
        self._urdfBase = urdfBase #[urdfWrap3D] the baselink of the urdf
    def _Bp(self):
        return self.Fp @ self._Bpurdf
    def _Mp(self):
        return (self.Bp @ self._MO[:,3])[:3]

if __name__ == "__main__":
    # m = urdfWrap3D.fromUrdf('/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf')
    # print(m.child)
    # print(m._Bpurdf)
    # print(m.child[1].child[0].name, m.child[1].child[0].Bp)
    # # print(m.child[1].child[0]._urdfBase.name)
    # print(m.child[1].child[0].child[0].name, m.child[1].child[0].child[0].Bp)
    # print(m.child[1].child[0].Mp)
    # print(m.child[1].child[0].KE)
    # print(m.child[1].child[0].PE)

    m = urdfWrap3D.FloatBaseUrdf('Lite', '/home/ami/jy_models/JueyingMiniLiteV2/urdf/MiniLiteV2_Rsm.urdf')

    print([n.name for n in m.child])
    print(m.child[0].child[1].child[0].Bp)
    print(m.child[0].child[1].child[0].M)
    print(ca.symvar( m.child[0].child[1].child[0].KE))
