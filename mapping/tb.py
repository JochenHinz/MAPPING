import numpy as np
import scipy as sp
from nutils import *
from .aux import *
from . import ut, pl


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


def O_grid(ndims,p, inner = 1, outer = 2):
    kv = ut.tensor_kv([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,ndims[i]+1), periodic = (i == 1)) for i in range(2)])
    go = ut.tensor_grid_object(knots = kv)
    def circle(R):
        return lambda g: R*function.stack([function.cos(2*np.pi*g[1]), function.sin(2*np.pi*g[1])])
    R1, R2 = inner, outer
    func = (1 - go.geom[0])*circle(R1)(go.geom) + go.geom[0]*circle(R2)(go.geom)
    cons = util.NanVec(2*len(go.basis))
    for side in ['left', 'right']:
        cons |= go.domain.boundary[side].project(func, geometry = go.geom, onto = go.basis.vector(2), ischeme = 'gauss6')
    go.cons = cons
    go.s = go.project(func, onto = go.basis.vector(2))
    return go

def nonconvex_go(ndims, p, mu = 0.3):
    kv = ut.tensor_kv([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,ndims[i]+1)) for i in range(2)])
    go = ut.tensor_grid_object(knots = kv)
    func = 2*(go.geom - np.array([1/2,1/2]))
    func = (1 - mu*(2 - (func**2).sum())*np.array([1,0]))*func
    cons = util.NanVec(2*len(go.basis))
    for side in ['left', 'right', 'bottom', 'top']:
        cons |= go.domain.boundary[side].project(func, geometry = go.geom, onto = go.basis.vector(2), ischeme = 'gauss6')
    go.cons = cons
    go.s = go.project(func, onto = go.basis.vector(2), constrain = cons)
    return go

def star_go(ndims, p, mu = 0.3, linear_spring = True):
    kv = ut.tensor_kv([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,ndims[i]+1)) for i in range(2)])
    go = ut.tensor_grid_object(knots = kv)
    func = 2*(go.geom - np.array([1/2,1/2]))
    func = (1 - mu*(2 - (func**2).sum()))*func
    cons = util.NanVec(2*len(go.basis))
    for side in ['left', 'right', 'bottom', 'top']:
        cons |= go.domain.boundary[side].project(func, geometry = go.geom, onto = go.basis.vector(2), ischeme = 'gauss6')
    go.cons = cons
    go.s = cons|0
    if linear_spring:
        go.set_initial_guess(method = linear_spring)
    else:
        go.s = go.project(func, onto = go.basis.vector(2), constrain = cons)
    return go

def dummy_go(ndims, p):
    go = circle_go(ndims,p)
    dgo = ut.dummy_go(**go.instantiation_lib)
    dgo.s = go.s
    dgo.cons = go.cons
    return dgo

def wedge(ndims, p):
    kv = ut.tensor_kv([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,ndims[i]+1)) for i in range(2)])
    go = ut.tensor_grid_object(knots = kv)
    go, goal_boundaries, corners = pl.wedge(go)
    go.set_cons(goal_boundaries, corners)
    go.set_initial_guess(goal_boundaries,corners)
    return go
    
