from nutils import *
import os
import sys
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
import reparam as rep
from auxilliary_classes import *
import os, sys, pickle


def main(nelems = [12,8], degree=3, basis_type = 'bspline', interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'middle', save = True, ltol = 1e-7, btol = 0.01, name = 'rotor'):
   
    assert len(nelems) == 2
    
    ischeme = 8
    
    if basis_type == 'spline':
        knots = [ut.uniform_kv(0,1,nelems[i]+1) for i in range(2)]
        domain, geom = mesh.rectilinear([n.knots for n in knots])
        go = ut.make_go(basis_type, domain, geom, degree, ischeme = ischeme, knots = knots)
        
    elif basis_type == 'bspline':
        knots = numpy.prod([ut.nonuniform_kv(numpy.linspace(0,1,nelems[i] + 1)) for i in range(2)])
        go = ut.make_go(basis_type, degree, ischeme = ischeme, knots = knots)
    
    if problem == 'middle':
        goal_boundaries, corners = pl.middle(go)
        
    elif problem == 'bottom':
        go, goal_boundaries, corners = pl.bottom(go)
        for i in range(0):
            l = len(knots[1].knots())
            go = go.ref_by([[],[l-4, l-3, l-2]])
            
            
    goal_boundaries = prep.preproc_dict(goal_boundaries)
    
    mgo = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)

    for i in range(len(mgo)):
        go_ = mgo[i] if i == 0 else mgo[i] | mgo[i-1]  ## take mg_prolongation after first iteration
        solver = Solver.Solver(go_, corners, go_.cons)   
        initial_guess = solver.one_d_laplace() if i == 0 else go_.s
        go_.s = solver.solve(initial_guess)
        mgo[i] = go_
        
    a, b = [len(mgo[-1].knots[i]) + mgo[-1].degree - 1 for i in range(2)]
    
    if save:
        output = open(sys.path[0] + '/saves/' + name + '_' + problem + '_' + basis_type +'_%i_%i_%i.pkl' %(degree ,a, b), 'wb')
        pickle.dump(mgo, output)
        output.close()




if __name__ == '__main__':
    cli.run(main)
