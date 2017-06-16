import numpy as np
import scipy as sp
from nutils import *
from auxilliary_classes import *
import utilities as ut


def square_go(ndims,p):
    kv = np.prod([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,n+1)) for n in ndims])
    domain, geom = mesh.rectilinear(kv.knots)
    basis = domain.basis('bspline', degree = p, knotvalues = kv.knots).vector(2)
    cons = domain.boundary.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6')
    s = np.array(domain.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6', constraints = cons))
    return ut.tensor_grid_object.with_mapping(s,cons, knots = kv)


def circle_go(ndims,p):
    kv = np.prod([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,n+1)) for n in ndims])
    domain, geom = mesh.rectilinear(kv.knots)
    basis = domain.basis('bspline', degree = p, knotvalues = kv.knots).vector(2)
    x, y = 2*geom[0] - 1, 2*geom[1] - 1
    func = function.stack([1/2*x*function.sqrt(1 - y**2/2)+1/2*1, 1/2*y*function.sqrt(1 - x**2/2)+1/2*1])
    cons = domain.boundary.project(func, geometry = geom, onto = basis, ischeme = 'gauss6')
    s = np.array(domain.project(func, geometry = geom, onto = basis, ischeme = 'gauss6', constraints = cons))
    return ut.tensor_grid_object.with_mapping(s,cons, knots = kv)



def cube_go(ndims,p):
    kv = ut.tensor_kv(*[ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,n+1)) for n in ndims])
    domain, geom = mesh.rectilinear(kv.knots)
    basis = domain.basis('bspline', degree = p, knotvalues = kv.knots).vector(3)
    cons = domain.boundary.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6')
    s = np.array(domain.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6', constraints = cons))
    return ut.tensor_grid_object.with_mapping(s,cons, knots = kv)


def O_grid(ndims,p):
    kv = ut.tensor_kv([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,n+1), periodic = (n == ndims[1])) for n in ndims])
    go = ut.tensor_grid_object(knots = kv)
    def circle(R):
        return lambda g: R*function.stack([function.cos(2*np.pi*g[1]), function.sin(2*np.pi*g[1])])
    R1, R2 = 1,2
    func = (1 - go.geom[0])*circle(R1)(go.geom) + go.geom[0]*circle(R2)(go.geom)
    cons = util.NanVec(2*len(go.basis))
    for side in ['left', 'right']:
        cons |= go.domain.boundary[side].project(func, geometry = go.geom, onto = go.basis.vector(2), ischeme = 'gauss6')
    go.cons = cons
    go.s = go.project(func, onto = go.basis.vector(2))
    return go