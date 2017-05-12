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


def main(nelems = [20,60], degree=3, basis_type = 'bspline', interp_degree = 5, repair_dual = False, save = True, ltol = 1e-7, btol = 5, name = 'sliding_grid_newgo', endtheta = 2*np.pi, dtheta = 0.03):
   
    assert len(nelems) == 2        
    
    ischeme = 8
    
    if basis_type == 'spline':
        raise NotImplementedError
        
    elif basis_type == 'bspline':
        knots = numpy.prod([ut.nonuniform_kv(degree, knotvalues = numpy.linspace(0,1,nelems[i] + 1)) for i in range(2)])
        go = ut.make_go(basis_type, ischeme = ischeme, knots = knots)
        
    for i in range(2):
        go = go.ref_by([[0,1,2], []])
        goal_boundaries, corners = pl.single_female_casing(go, radius = 37)#36.061810867369296)
            
    
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
    
    
    theta = 0
        
    mgo = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)
    
    def quick_plot(go, func, name, side, ref = 0):
        points = go.domain.boundary[side].refine(ref).elem_eval( func, ischeme='vtk', separate=True )
        with plot.VTKFile(name) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
    
    mgo[-1].quick_plot_boundary()

    
    ## Multigrid to get started at theta = 0
    go_list = []
    verts = []
    j = 0
    while j < endtheta/dtheta:
        if j == 0:
            for i in range(len(mgo)):
                go_ = mgo[i] if i == 0 else mgo[i] | mgo[i-1]  ## take mg_prolongation after first iteration
                solver = Solver.Solver(go_, go_.cons)   
                initial_guess = solver.transfinite_interpolation(goal_boundaries, corners) if i == 0 else go_.s
                #initial_guess = solver.linear_spring() if i == 0 else go_.s
                go_.s = solver.solve(initial_guess, method = 'Elliptic', solutionmethod = 'Newton')
                mgo[i] = go_  
        else:
            goal_boundaries, corners = rotate(go_, goal_boundaries, corners, dtheta, 0)
            go = ut.make_go(basis_type, ischeme = ischeme, knots = mgo[1 if len(mgo) > 1 else 0]._knots)
            mgo_ = prep.boundary_projection(go, goal_boundaries, corners, btol = btol)
            go_ = mgo_[-1] % go_
            solver = Solver.Solver(go_, go_.cons)
            verts_, gos_ = [verts, go_list] if j <= 5 else [verts[-5:], go_list[-5:]]
            initial_guess = go_.s if j < 3 else go_.cons | ut.tensor_grid_object.grid_interpolation(verts_,gos_)(theta)
            go_.s = solver.solve(initial_guess, method = 'Elliptic', solutionmethod = 'Newton')
            
        go_list.append(go_)
        verts.append(theta)
        
        if save:
            output = open(sys.path[0] + '/saves/sliding_grid/' + name + '_dtheta_%.3f' %dtheta + '_degree_%i_%i.pkl' %(degree,j), 'wb')
            pickle.dump(go_, output)
            output.close()
            
        theta += dtheta
        j += 1




if __name__ == '__main__':
    cli.run(main)
