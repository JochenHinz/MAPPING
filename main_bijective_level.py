from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
import reparam as rep
from auxilliary_classes import *


def main(nelems = 450, degree=3, basis_type = 'spline', interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'Benchmark', save = True, File = 'save', plotting = True, ltol = 1e-7, btol = 1e-0):
    
    nelems = nelems, 20
    knots = [ut.knot_object(0,1,nelems[i]+1) for i in range(2)]
    domain, geom = mesh.rectilinear([n.knots for n in knots])
    basis = domain.basis('spline', degree = degree)
    ischeme = 5
    go = ut.grid_object(domain, geom, basis, degree, ischeme, knots = knots)
    
    if problem == 'Benchmark':
        goal_boundaries, corners = pl.benchmark(go)
    elif problem == 'Sines':
        goal_boundaries, corners = pl.sines(go)
    else:
        raise ValueError('unknown problem: ' + problem)
        
    keys = ['left', 'right', 'bottom', 'top']
    reparam_keys = []

        goal_boundaries[key] = goal_boundaries[key](go.geom)

    prep_object, mgo = prep.boundary_projection(go, goal_boundaries, corners, btol)
    
    for i in range(len(mgo)):
        cont = mgo[i].domain.refine(0).elem_eval( geom, ischeme='vtk', separate=True )
        with plot.VTKFile('grid_%i' %i) as vtu:
            vtu.unstructuredgrid( cont, npars=2 )
    
    solver = Solver.Solver(mgo[-1], corners, prep_object[0][-1])

    for side in keys:
        cont = go.domain.boundary.refine(1).elem_eval(goal_boundaries[side], ischeme='vtk', separate=True )
        with plot.VTKFile('contours'+side) as vtu:
            vtu.unstructuredgrid( cont, npars=2 )
    
    cont, det = go.domain.refine(1).elem_eval( [solver.go.basis.vector(2).dot(lhs), function.determinant(solver.go.basis.vector(2).dot(lhs).grad(go.geom))], ischeme='vtk', separate=True )
    with plot.VTKFile('Final') as vtu:
        vtu.unstructuredgrid( cont, npars=2 )
        vtu.pointdataarray( 'det', det )




if __name__ == '__main__':
    cli.run(main)
