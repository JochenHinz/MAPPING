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


def main(nelems = [[10, False],[10, False]], degree=3, basis_type = 'bspline', interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'separator', save = True, ltol = 1e-7, btol = 0.1, name = 'Elliptic'):
   
    assert len(nelems) == 2
    
    ischeme = 8
    rep_dict = None
    
    if basis_type == 'spline':
        raise NotImplementedError
        
    elif basis_type == 'bspline':
        knots = numpy.prod([ut.nonuniform_kv(degree, knotvalues = numpy.linspace(0,1,nelems[i][0] + 1), periodic = nelems[i][1]) for i in range(2)])
        go = ut.make_go(basis_type, ischeme = ischeme, knots = knots)
    
    if problem == 'middle':
        go, goal_boundaries, corners = pl.middle(go, c0 = False)
        
    elif problem == 'wedge':
        go, goal_boundaries, corners = pl.wedge(go)
        go = go.add_knots([[0.55],[]])
        go = go.raise_multiplicities([go.degree[1]-1,0], knotvalues = [[0.55],[]])
        print(go.knotmultiplicities)
        
    elif problem == 'separator':
        angle = 0.25*np.pi
        go, goal_boundaries, corners = pl.separator(go,angle)
        
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
        for i in range(0):
            l = len(go.knots[0])
            go = go.ref_by([[l-4,l-3, l-2], []])
        go, goal_boundaries, corners = pl.single_female_casing(go)
        #rep_func = rep.minimize_angle(go, goal_boundaries, 'left')
        #rep_dict = {'left': rep_func, 'right': None}
        
    elif problem == 'single_male_casing':
        for i in range(2):
            go = go.ref_by([[0,1,2], []])
        go, goal_boundaries, corners = pl.single_male_casing(go)
        
    elif problem == 'single_left_snail':
        #for i in range(0):
        #    go = go.ref_by([[0,1,2], []])
        go, goal_boundaries, corners = pl.single_left_snail(go)
            
    elif problem == 'nrw':
        goal_boundaries, corners = pl.nrw(go)
        
    elif problem == 'isoline':
        goal_boundaries, corners = pl.isoline(go)
    
    mgo = prep.boundary_projection(go, goal_boundaries, corners, btol = btol, rep_dict = rep_dict)
    
    mgo[-1].quick_plot_boundary()

    start = 0
    for i in range(start,len(mgo)):
        go_ = mgo[i] if i == start else mgo[i-1].elast(mgo[i]) #mgo[i] | mgo[i-1] ##take mg_prolongation after first iteration
        solver = Solver.Solver(go_, go_.cons)   
        #initial_guess = solver.one_d_laplace(0) if i == 0 else go_.s
        initial_guess = solver.transfinite_interpolation(goal_boundaries, corners = corners) if i == start else go_.s
        dummy_go = ut.tensor_grid_object.with_mapping(initial_guess, go_.cons, **go_.instantiation_lib)
        dummy_go.quick_plot()
        go_.s = solver.solve(initial_guess, method = 'Elliptic', solutionmethod = 'Newton')
        mgo[i] = go_
        mgo[i].quick_plot()
        
    a, b = [len(mgo[-1].knots[i]) + mgo[-1].degree[i] - 1 for i in range(2)]
    
    if save:
        output = open(sys.path[0] + '/saves/' + name + '_' + problem + '_' + basis_type +'_%i_%i_%i.pkl' %(degree ,a, b), 'wb')
        pickle.dump(mgo, output)
        output.close()




if __name__ == '__main__':
    cli.run(main)
