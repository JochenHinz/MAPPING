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


def main(nelems = [10,10], degree=3, basis_type = 'bspline', interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'nrw', save = True, ltol = 1e-7, btol = 0.01, name = 'rotor'):
   
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
        
    elif problem == 'bottom':  ## THE REFINEMENT NEEDS TO BE CARRIED OUT BEFORE pl.bottom is called otherwise we get nonsense
        ## This is prolly due to goal_boundaries being faulty or so
        for i in range(4):
            l = len(go.knots[1])
            go = go.ref_by([[],[l-4, l-3, l-2]])
        go, goal_boundaries, corners = pl.bottom(go)
        ## carrying out refinement after calling pl.bottom does work on go but then prep.boundary projection give crap
        ## the culprit is prolly goal_boundaries
        ## goal_boundaries then contains a non-refined go, which is probably the problem
        
    elif problem == 'single_female_casing':
        for i in range(2):
            go = go.ref_by([[0,1,2], []])
        goal_boundaries, corners = pl.single_female_casing(go, radius = 36.030884376335685)
        
    elif problem == 'single_male_casing':
        for i in range(2):
            l = len(go.knots[0])
            go = go.ref_by([[l-4,l-3, l-2], []])
        goal_boundaries, corners = pl.single_male_casing(go, radius = 36.030884376335685)
            
    elif problem == 'nrw':
        goal_boundaries, corners = pl.nrw(go)
                   
    #goal_boundaries = prep.preproc_dict(goal_boundaries)
    
    mgo = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)
    
    mgo[-1].quick_plot_boundary()
    #goal_boundaries.plot(mgo[-1], 'contours', ref = 3)
    start = 0
    for i in range(start,len(mgo)):
        go_ = mgo[i] if i == start else mgo[i] | mgo[i-1]  ## take mg_prolongation after first iteration
        if i > len(mgo) - 3:
            for j in range(4):
                l1, l2 = len(go_.knots[0]), len(go_.knots[1])
                go_ = go_.ref_by([[0,1,2,l1-4,l1-3, l1-2], [0,1,2,l2-4,l2-3, l2-2]])
                print(go_.s)
        solver = Solver.Solver(go_, corners, go_.cons)   
        #initial_guess = solver.one_d_laplace() if i == 0 else go_.s
        initial_guess = solver.transfinite_interpolation(goal_boundaries, corners) if i == 0 else go_.s
        go_.s = solver.solve(initial_guess)
        mgo[i] = go_
        
    a, b = [len(mgo[-1].knots[i]) + mgo[-1].degree - 1 for i in range(2)]
    
    if save:
        output = open(sys.path[0] + '/saves/' + name + '_' + problem + '_' + basis_type +'_%i_%i_%i.pkl' %(degree ,a, b), 'wb')
        pickle.dump(mgo, output)
        output.close()




if __name__ == '__main__':
    cli.run(main)
