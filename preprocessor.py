import numpy as np
import scipy as sp
import scipy.interpolate
from nutils import *
import inspect
import collections
from problem_library import Pointset
import itertools
import utilities as ut
from auxilliary_classes import *


def vec_union(vec1, vec2):  ## return the union of two refine indices
    return np.asarray(sorted(list(set(list(vec1) + list(vec2)))))


class preproc_object(preproc_info):
    
    def __init__(self,cons_lib, dirichlet_lib, cons_lib_lib, error_lib):
        preproc_info.__init__(self)
        self.i =  -1
        
    def extend(self, *args):
        self.i += 1
        assert len(*args) == len(self)
        for i, thing in enumerate(*args):
            self[i].update({self.i: thing})
    
    def at(self,n):
        assert n < len(self)
        return [self[j][n] for j in range(len(self))]
            
            
def log_iter_sorted_dict_items(title, d):
    for k, v in sorted(d.items()):
        with log.context(title + ' ' + k):
            yield k, v
            
            
def std_goal_boundary_dependence(goal_boundaries_, go):
    goal_boundaries = goal_boundaries_.copy()
    for side in planar_sides:
        func = goal_boundaries[side]
        if isinstance(func, ut.tensor_grid_object):  ## take union
            temp = func + go 
            goal_boundaries[side] = temp
        elif isinstance(func, function.Evaluable):  ## nutils function - do nothing
            pass
        else: ## lambda function
            goal_boundaries[side] = func(go.geom)
    return goal_boundaries

            
            
def generate_cons(go, boundary_func_libray_, corners, btol = 1e-2):
    domain, geom, basis, degree, ischeme, basis_type, knots = go.domain, go.geom, go.basis.vector(2), go.degree, go.ischeme, go.basis_type, go.knots
    boundary_func_libray = boundary_func_libray_.copy()   
    cons = None
    for (i, j), v in log.iter('corners', corners.items()):  ## constrain the corners
        domain_ = (domain.levels[-1] if isinstance(domain, topology.HierarchicalTopology) else domain).boundary[{0: 'bottom', 1: 'top'}[j]].boundary[{0: 'left', 1: 'right'}[i]]
        cons = domain_.project(v, onto=basis, constrain=cons, geometry=geom, ischeme='vertex')
    # Project all boundaries onto `gbasis` and collect all elements where
    # the projection error is larger than `btol` in `refine_elems`.
    cons_library = {'left':0, 'right':0, 'top':0, 'bottom':0}
    #refine_elems = set()
    for side, goal in log_iter_sorted_dict_items('boundary', boundary_func_libray):
        dim = side_dict[side]
        if isinstance(goal, Pointset):
            domain_ = domain.boundary[side].locate(geom[dim], goal.verts)
            ischeme_ = 'vertex'
            goal = function.elemwise(
                dict(zip((elem.transform for elem in domain_), goal.geom)),
                [domain.ndims])
        elif isinstance(goal, ut.base_grid_object):
            temp = goal + go  ## take the union
            domain_ = temp[side].domain  ## restrict to boundary
            goal = temp.basis.vector(2).dot(temp.cons | 0)  ## create mapping
        else:
            domain_ = domain.boundary[side]
            ischeme_ = gauss(degree*2)
        cons_library[side] = domain_.refine(3).project(goal, onto=basis, geometry=geom, ischeme=ischeme_, constrain=cons)
        cons |= cons_library[side]
    return cons


def constrained_boundary_projection(go, goal_boundaries_, corners, btol = 1e-2, rep_functions = None, ref = 0):  #Needs some fixing
    degree, ischeme, basis_type = go.degree, go.ischeme, go.basis_type
    goal_boundaries = goal_boundaries_.copy()
    if rep_functions is None:  ## no reparam function: take standard dependency
        rep_functions = {'left': go.geom, 'right': go.geom, 'bottom': go.geom, 'top':go.geom}
    for side in planar_sides:  ## apply the reparam functions
        if all([not isinstance(goal_boundaries[side], Pointset), not isinstance(goal_boundaries[side], ut.base_grid_object)]):
            goal_boundaries[side] = goal_boundaries[side](rep_functions[side])
    if basis_type == 'bspline':  ## the basis type is b_spline = > we need to refine on knots
        assert go._knots is not None
    if go.cons is None:
        go.set_cons(goal_boundaries,corners)
    cons = go.cons
    refine_elems = set()
    error_dict = {'left':0, 'right':0, 'top':0, 'bottom':0}
    for side, goal in log_iter_sorted_dict_items('boundary', goal_boundaries):
        dim = side_dict[side]
        error = ((goal - go.bc())**2).sum(0)**0.5
        ## replace goal by goal.function() or something once it becomes an object
        go_ = go[side]
        if basis_type == 'spline':   ## basis is spline so operate on the elements
            error_ = go_.domain.project(error, ischeme= 2*degree)
            refine_elems.update(
                elem.transform.promote(domain.ndims)[0]
                for elem in go_.domain.supp(basis_, numpy.where(error_ > btol)[0]))
        elif basis_type == 'bspline':  ## basis is b_spline just compute error per element, refinement comes later
            basis0 = go_._basis(degree = 0)
            error_ = np.divide(*go_.integrate([basis0*error, basis0]))
            print(numpy.max(error_), side)
            error_dict[side] = error_
    if basis_type == 'spline':
        if len(refine_elems) == 0:  ## no refinement necessary = > finished
            return None
        else:   ## refinement necessary = > return new go with refined grid
            domain = domain.refined_by(refine_elems)
            new_go = go.update_from_domain(domain, ref_basis = True)   ## return hierarchically refined grid object
            new_go.set_cons(goal_boundaries, corners)
            return new_go
    else:
        union_dict = {0: ['bottom', 'top'], 1: ['left', 'right']}
        ## take union of refinement indices
        ref_indices = [vec_union(*[numpy.where(error_dict[i] > btol)[0] for i in union_dict[j]]) for j in range(2)]
        if len(ref_indices[0]) == 0 and len(ref_indices[1]) == 0:  ## no refinement
            return None
        else: ## return refined go
            ## refine according to union
            ## create new topology
            new_go = go.ref_by(ref_indices, prolong_mapping = False, prolong_constraint = False)
            cons_funcs = std_goal_boundary_dependence(goal_boundaries_, new_go)
            new_go.set_cons(cons_funcs, corners)
            return new_go



def boundary_projection(go, goal_boundaries, corners, btol = 1e-2, rep_functions = None):
    basis_type = go.basis_type
    go_list = [go]
    for bndrefine in log.count('boundary projection'):
        proj = constrained_boundary_projection(go_list[-1], goal_boundaries, corners, btol = btol, rep_functions = rep_functions)
        if proj is None:
            break
        else:
            go_list.append(proj)
    return ut.multigrid_object(go_list)