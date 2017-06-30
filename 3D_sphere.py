from nutils import *
import numpy as np
import scipy as sp

nelems = 10,10,10
domain, geom = mesh.rectilinear([np.linspace(0,1, n + 1) for n in nelems])
basis = domain.basis_bspline(degree = 2)

def metric_tensor(c):
    x = basis.vector(3).dot(c)
    x_g = x.grad(geom)
    a11 = (x_g**2)[:,0].sum(0)
    a12 = (x_g[:,0]*x_g[:,1]).sum(0)
    a13 = (x_g[:,0]*x_g[:,2]).sum(0)
    a22 = (x_g[:,1]*x_g[:,1]).sum(0)
    a23 = (x_g[:,1]*x_g[:,2]).sum(0)
    a33 = (x_g[:,2]*x_g[:,2]).sum(0)
    return {'g11': a22*a33 - a23**2, 'g12': a13*a23 - a12*a33, 'g13': a12*a23 - a13*a22, 'g22': a11*a33 - a13**2, 'g23': a13*a12 - a11*a23, 'g33': a11*a22 - a12**2}

def Elliptic(c):
    x = basis.vector(3).dot(c)
    t = metric_tensor(c)
    x_g_g = x[0].grad(geom).grad(geom)
    y_g_g = x[1].grad(geom).grad(geom)
    z_g_g = x[2].grad(geom).grad(geom)
    vec1, vec2, vec3 = [basis*(t['g11']*i[0,0] + 2*t['g12']*i[0,1] + 2*t['g13']*i[0,2] + t['g22']*i[1,1] + 2*t['g23']*i[1,2] + t['g33']*i[2,2]) for i in [x_g_g, y_g_g, z_g_g]]
    return -function.concatenate((vec1,vec2,vec3))

def solve(initial_guess, cons):
    target = function.DerivativeTarget([len(basis.vector(3))])
    res = model.Integral(Elliptic(target), domain=domain, geometry=geom, degree=6)
    lhs = model.newton(target, res, lhs0=initial_guess, freezedofs=cons.where).solve(1e-6)
    return lhs

geom_ = geom - np.array([0.5,0.5,0.5])
cons = domain.boundary.project(1/(function.sqrt((geom_**2).sum(0)))*geom_, geometry = geom, ischeme = 'gauss2', onto = basis.vector(3))
init = domain.project(geom_, geometry = geom, ischeme = 'gauss2', onto = basis.vector(3))

lhs = solve(cons|init, cons)

_map = basis.vector(3).dot(lhs)
det = function.determinant(_map.grad(geom))
points, det = domain.refine(1).elem_eval( [_map, det], ischeme='vtk', separate=True )
with plot.VTKFile('sphere') as vtu:
    vtu.unstructuredgrid( points, npars=2 )
    vtu.pointdataarray( 'det', det )
