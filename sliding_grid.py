from nutils import *
import os,sys,copy
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
import reparam as rep
from auxilliary_classes import *
import os, sys, pickle


def main(nelems = [20,40], degree=3, basis_type = 'bspline', interp_degree = 5, repair_dual = False, save = False, ltol = 1e-7, btol = 4, name = 'sliding_grid_small_timestep', endtheta = 0.2, dtheta = 0.003):
   
    assert len(nelems) == 2        
    
    ischeme = 8
    
    if basis_type == 'spline':
        raise NotImplementedError
        
    elif basis_type == 'bspline':
        knots = numpy.prod([ut.nonuniform_kv(numpy.linspace(0,1,nelems[i] + 1)) for i in range(2)])
        go = ut.make_go(basis_type, degree, ischeme = ischeme, knots = knots)
        
    for i in range(2):
        go = go.ref_by([[0,1,2], []])
        goal_boundaries, corners = pl.single_female_casing(go, radius = 36.030884376335685)
        
    mgo = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)
    
    
    def rotate(go, goal_boundaries_, corners, angle, initial_guess):
        spl_ = goal_boundaries_['left'](go.geom)
        spl_._rotate(angle)
        spl_._reparam(initial_guess = initial_guess)  ## iteratively reparam s.t. spline(0)[1] = 0
        corners_ = {(0,0): tuple(spl_.evalf(np.array([0]))[0]), (0,1): tuple(spl_.evalf(np.array([0]))[0]), (1,0): corners[(1,0)], (1,1): corners[(1,1)]}
        goal_boundaries = dict(
            bottom = lambda g: corners_[0,0]*(1-g[0]) + corners_[1,0]*g[0],
            top = lambda g: corners_[0,1]*(1-g[0]) + corners_[1,1]*g[0],
            right = goal_boundaries_['right'],
            left = lambda g: spl_(g[1])
        )
        return goal_boundaries, corners_
    
    def quick_plot(go, func, name, side, ref = 0):
        points = go.domain.boundary[side].refine(ref).elem_eval( func, ischeme='vtk', separate=True )
        with plot.VTKFile(name) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
    
    mgo[-1].quick_plot_boundary()

    
    ## Multigrid to get started at theta = 0
    start = 0
    theta = 0
    j = 0
    while j < endtheta/dtheta:
        if j == 0:
            for i in range(start,len(mgo)):
                go_ = mgo[i] if i == start else mgo[i] | mgo[i-1]  ## take mg_prolongation after first iteration
                solver = Solver.Solver(go_, go_.cons)   
                initial_guess = solver.transfinite_interpolation(goal_boundaries, corners) if i == 0 else go_.s
                #initial_guess = solver.linear_spring() if i == 0 else go_.s
                if i == 0:
                    _go_ = copy.deepcopy(go_)
                    _go_.s = initial_guess
                    _go_.quick_plot()
                go_.s = solver.solve(initial_guess, method = 'Elliptic', solutionmethod = 'Newton')
                mgo[i] = go_  
        else:
            goal_boundaries, corners = rotate(mgo[-1], goal_boundaries, corners, dtheta, 0)
            go = ut.make_go(basis_type, degree, ischeme = ischeme, knots = mgo[1 if len(mgo) > 1 else 0]._knots)
            mgo_ = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)
            go_ = mgo_[-1] % mgo[-1]
            solver = Solver.Solver(go_, go_.cons)   
            go_.s = solver.solve(go_.s, method = 'Elliptic', solutionmethod = 'Newton')
            mgo_[-1] = go_
            mgo = mgo_
        
        if save:
            output = open(sys.path[0] + '/saves/sliding_grid/' + name + '_' + basis_type +'_%i_%i.pkl' %(degree,j), 'wb')
            pickle.dump(go_, output)
            output.close()
            
        theta += dtheta
        j += 1




if __name__ == '__main__':
    cli.run(main)
