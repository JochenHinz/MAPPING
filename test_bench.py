import numpy as np
import scipy as sp
from nutils import *
from auxilliary_classes import *
import utilities as ut


def square_go(ndims,p):
    kv = np.prod([ut.nonuniform_kv(np.linspace(0,1,n+1)) for n in ndims])
    domain, geom = mesh.rectilinear(kv.knots())
    basis = domain.basis('bspline', degree = p, knotvalues = kv.knots()).vector(2)
    cons = domain.boundary.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6')
    s = np.array(domain.project(geom, geometry = geom, onto = basis, ischeme = 'gauss6', constraints = cons))
    return ut.tensor_grid_object.with_mapping(s,cons,p, knots = kv)


def circle_go(ndims,p):
    kv = np.prod([ut.nonuniform_kv(np.linspace(0,1,n+1)) for n in ndims])
    domain, geom = mesh.rectilinear(kv.knots())
    basis = domain.basis('bspline', degree = p, knotvalues = kv.knots()).vector(2)
    x, y = 2*geom[0] - 1, 2*geom[1] - 1
    func = function.stack([1/2*x*function.sqrt(1 - y**2/2)+1/2*1, 1/2*y*function.sqrt(1 - x**2/2)+1/2*1])
    cons = domain.boundary.project(func, geometry = geom, onto = basis, ischeme = 'gauss6')
    s = np.array(domain.project(func, geometry = geom, onto = basis, ischeme = 'gauss6', constraints = cons))
    return ut.tensor_grid_object.with_mapping(s,cons,p, knots = kv)