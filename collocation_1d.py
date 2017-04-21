from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
from matplotlib import pyplot as plt
gauss = 'gauss{:d}'.format

def main(nelems = 12, degree=3, interp_degree = 5, preproc = True, multigrid = 1, repair_ordinary = False, local_repair = False, repair_dual = False, problem = 'Sines', save = True, File = 'save', plotting = True, ltol = 1e-7, btol = 1e20):
    
    knots = [ut.knot_object(0,1,nelems+1)]
    domain, geom = mesh.rectilinear([n.knots for n in knots])
    #km = [[degree+1] + [1]*(len(knots[i].knots) - 2) + [degree+1] for i in range(len(knots))]
    #basis = domain.basis_bspline(degree, knotmultiplicities = km)
    basis = domain.basis('spline', degree = degree)
    basis = (basis*basis.grad(geom).T).T
    ischeme = 20
    go = ut.grid_object(domain, geom, basis, degree, ischeme, knots = knots)

    initial_guess = np.zeros(len(basis))
    initial_guess[6] = 1
    
    func = sum(basis.dot(initial_guess))
    #func = sum(func*func.grad(geom))
    print(func)
    
    jac_basis, jac_knots = ut.make_jac_basis(go.pull_to_finest())
    jac_basis = jac_basis
    
    #d = ut.collocate_greville(go.pull_to_finest(), func, jac_basis, 2*go.degree - 1, onto_knots = jac_knots)
    d = domain.project(func, onto = jac_basis, geometry = go.geom, ischeme = gauss(go.ischeme))
    
    #det_vec = ut.collocate_greville(go, determinant, jac_basis.vector(2), onto_knots = jac_knots, onto_p = 2*degree - 1)
    print(d)
    #d = ut.basis_functions_on_defective_elements(go, determinant,ref = 0)

    cont = domain.refine(2).elem_eval( func - jac_basis.dot(d), ischeme='bezier1', separate=True )
    print(cont)
    print(len(basis), len(jac_basis))




if __name__ == '__main__':
    cli.run(main)
