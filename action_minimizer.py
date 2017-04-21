from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
import reparam as rep
import bspline as bs
from matplotlib import pyplot as plt
import itertools

def main(nelems = 42, degree=3, interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'Sines', save = True, File = 'save', plotting = True, ltol = 1e-7, btol = 1e20):
    
    knots = [ut.knot_object(0,1,nelems+1)]*2
    domain, geom = mesh.rectilinear([n.knots for n in knots])
    basis = domain.basis('spline', degree = degree)
    ischeme = 5
    go = ut.grid_object(domain.boundary['bottom'], geom, basis, degree, ischeme, knots = knots[0])
    
    func = lambda g: function.stack([2*g, -0.6*function.sin(g*numpy.pi*4)])
    func_ = lambda g: function.stack([2*g, 0.6*function.sin(g*numpy.pi*4) + 2])
    ref = 0
    opt, pol = rep.minimize_action_bspline(go, func, ref = ref)
    
    print(opt.x)
    kv = np.concatenate(([0]*degree, go.knots.ref(ref).knots, [1]*degree))
    vec = np.concatenate(([0], opt.x, [1]))
    vec_ = np.repeat(vec,len(vec))
    #bas = bs.Bspline(kv,degree)
    #def resh_(g):
    #    return np.asarray([vec*bas(g[i]) for i in range(len(g))]).T
    #print(resh_([0,0.5,1]).ndim)
    #resh = ut.nutils_function(resh_)#, derivative = lambda g_: sum(vec*bas.d(g_)))
    resh = go.basis.dot(vec_)
    
    cont = domain.refine(2).elem_eval( (1 - geom[1])*func(geom[0]) + geom[1]*func_(geom[0]), ischeme='vtk', separate=True )
    with plot.VTKFile('before') as vtu:
        vtu.unstructuredgrid( cont, npars=2 )
        
    
    cont = domain.refine(2).elem_eval( (1 - geom[1])*func(resh) + geom[1]*func_(resh), ischeme='vtk', separate=True )
    with plot.VTKFile('after') as vtu:
        vtu.unstructuredgrid( cont, npars=2 )
    



if __name__ == '__main__':
    cli.run(main)
