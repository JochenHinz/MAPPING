from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
gauss = 'gauss{:d}'.format

def main(nelems = 12, degree=5, interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'Sines', save = True, File = 'save', plotting = True, ltol = 1e-7, btol = 1e20):
    
    knots = [ut.knot_object(0,1,nelems+1), ut.knot_object(0,1,(nelems+1)//1)]
    domain, geom = mesh.rectilinear([n.knots for n in knots])
    #km = [[degree+1] + [1]*(len(knots[i].knots) - 2) + [degree+1] for i in range(len(knots))]
    #basis = domain.basis_bspline(degree, knotmultiplicities = km)
    basis = domain.basis('spline', degree = degree)
    ischeme = 10
    go = ut.grid_object(domain, geom, basis, degree, ischeme, knots = knots)
    
    if problem == 'Benchmark':
        goal_boundaries, corners = pl.benchmark(go)
    elif problem == 'Sines':
        goal_boundaries, corners = pl.sines(go)      
    else:
        raise ValueError('unknown problem: ' + problem)
        
    for key in ['left', 'right', 'bottom', 'top']:
            goal_boundaries[key] = goal_boundaries[key](go.geom)

    po, mgo = prep.boundary_projection(go, goal_boundaries, corners, btol)
    
    i = numpy.max(list(po.cons_lib.keys()))
    solver = Solver.Solver(go, corners, po.cons_lib[i])
    
    initial_guess = solver.one_d_laplace()
    #initial_guess[12] += 0.6
    
    determinant = function.determinant(go.basis.vector(2).dot(initial_guess).grad(go.geom))

    print(type(go.domain), 'go.domain before')
    d = ut.defect_check(go, determinant, 'discont', ref = 1)
    
    print(d, 'defective basis functions')

    cont, det = po.domain_lib[i].levels[-1].refine(2).elem_eval( [go.basis.vector(2).dot(initial_guess), determinant - jac_basis.dot(d)], ischeme='vtk', separate=True )
    with plot.VTKFile('Initial') as vtu:
        vtu.unstructuredgrid( cont, npars=2 )
        vtu.pointdataarray( 'det', det )
        #vtu.pointdataarray( 'det_jac', det_jac )
        
    print(len(basis), len(jac_basis))




if __name__ == '__main__':
    cli.run(main)
