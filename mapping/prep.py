import numpy as np
import scipy as sp
import scipy.interpolate
from nutils import *
import inspect
import collections
import itertools
from . import ut
from .aux import *


def vec_union(vec1, vec2):  ## return the union of two refine indices
    return np.asarray(sorted(list(set(list(vec1) + list(vec2)))))

def unit_vec(length, i):
    ret = np.zeros(length)
    ret[i] = 1
    return ret

def tck(kv,i):
        ''' tck tuple for sp.interpolate.splev'''
        p = kv.degree
        knots = kv.extend_knots()
        #vec = np.zeros(kv.dim)
        #vec[i] = 1
        vec = unit_vec(kv.dim,i)
        return (knots, vec, p)

def l2_data(verts, points, kv):
    ''' Return l2 Mass matrix and rhs vector'''
    
    f = lambda x: np.array([scipy.interpolate.splev(x, tck(kv, i)) for i in range(kv.dim)])
    a = f(verts)
    a_ = sp.sparse.lil_matrix(a)
    rhs = np.concatenate([(a_.dot(sp.sparse.diags([points[i]],[0]))).sum(-1) for i in range(points.shape[0])])
    M = sp.sparse.csr_matrix((a[:,None,:]*a[None,:,:]).sum(-1))
    M = sp.sparse.csr_matrix(sp.sparse.block_diag([M]*points.shape[0]))
    return matrix.ScipyMatrix(M), np.array(rhs).flatten()
    
    
class preproc_dict:
    
    def __init__(self, dictionary, go):
        self._dict = dictionary
        self._go = go
        
    @property
    def go(self):
        return self._go
        
        
    def instantiate(self, rep_dict):
        if rep_dict is None:
            return self.from_geom()
        ret = self._dict.copy()
        for side in planar_sides:
            try:
                geom = rep_dict[side] ## key found, reparameterize
                if isinstance(geom, Pointset):
                    raise NotImplementedError
                elif isinstance(geom, ut.tensor_grid_object):
                    assert geom.repeat == 1
                    geom_ = geom + self._go  ## pull reparam function onto same grid
                    ret[side] = ret[side](function.stack([geom_.mapping()]*2))
                else:
                    ret[side] = ret[side](self._go.geom if rep_dict[side] is None else rep_dict[side])
            except:
                ## key not found, default instantiation
                if side in self._dict.keys():
                    ret[side] = ret[side](self._go.geom)
        return ret
        
        
    def from_geom(self):
        #rep_dict = {'left': self.go.geom, 'right': self.go.geom, 'bottom': self.go.geom, 'top': self.go.geom}
        goal_boundaries = self._dict.copy()
        rep_dict = dict(zip(goal_boundaries.keys(), [self.go.geom]*len(goal_boundaries.keys())))
        return self.instantiate(rep_dict)
    
    def items(self):
        return self._dict.items()
    
    def plot(self, go, name, ref = 1):
        d = self.from_geom(go.geom)
        for side in d.keys():
            points = go.domain.boundary[side].refine(ref).elem_eval( d[side], ischeme='vtk', separate=True)
            with plot.VTKFile(name+'_'+side) as vtu:
                vtu.unstructuredgrid( points, npars=2 )
            
            
def log_iter_sorted_dict_items(title, d):
    for k, v in sorted(d.items()):
        with log.context(title + ' ' + k):
            yield k, v

            
            
def generate_cons(go, boundary_funcs_, corners = None, l = None, stabilize = True):
    if l is None:
        l = 1/1000.0
    domain, geom, basis, degree, ischeme, basis_type, knots = go.domain, go.geom, go.basis.vector(2), go.degree, go.ischeme, go.basis_type, go.knots
    temp_go = go.empty_copy()
    cons_ = go.cons.copy()
    if not all(np.isnan(cons_)):
        log.warning('Warning, the grid object has nonempty constraints, some may be overwritten')
    cons = util.NanVec(len(cons_))
    boundary_funcs = boundary_funcs_.copy()
    ## constrain the corners
    if corners:
        for (i, j), v in log.iter('corners', corners.items()):
            domain_ = (domain.levels[-1] if isinstance(domain, topology.HierarchicalTopology) else domain).boundary[{0: 'bottom', 1: 'top'}[j]].boundary[{0: 'left', 1: 'right'}[i]]
            cons = domain_.project(v, onto=basis, constrain=cons, geometry=geom, ischeme='vertex')
    temp_go.cons = cons
    # Project all boundaries onto `gbasis` and collect all elements where
    # the projection error is larger than `btol` in `refine_elems`.
    cons_library = {'left':cons, 'right':cons, 'top':cons, 'bottom':cons}
    #refine_elems = set()
    for side, goal in log_iter_sorted_dict_items('boundary', boundary_funcs):
        dim = side_dict[side]
        if isinstance(goal, Pointset):
            assert dim not in go.periodic
            kv = go._knots[dim]
            #l = 1e-4
            go_ = ut.tensor_grid_object(knots = go._knots[dim], target_space = go.repeat)
            basis_ = go_.basis
            target = function.Argument('target', [len(basis_.vector(go_.repeat))])
            M, rhs = l2_data(goal.verts, goal.geom, go._knots[dim])
            func = basis_.vector(go_.repeat).dot(target)
            f = lambda i: np.array([scipy.interpolate.splev(goal.verts, tck(kv, i))])
            stab_indices = []
            for i in range(kv.dim):
                temp = f(i).flatten()
                if all(np.abs(temp) <= 1e-4):
                    stab_indices.append(i)
            if len(stab_indices) > 0 or stabilize:  ## we need to stabilize
                ## build stabilization matrix
                log.info('project > Stabilizing via least-distance penalty method')
                stab = go_.domain.integral((func.grad(go_.geom)**2).sum(), geometry = go_.geom, degree = \
                                           12).derivative('target').derivative('target')
                log.info('project > Building stabilization matrix')
                stab = solver.Integral.multieval(stab, arguments=collections.ChainMap({}, \
                                                                                      {'target': numpy.zeros(len(target))}))[0]
                mat = M + l*stab
            else:  ## don't stabilize
                mat = M
            side_cons = mat.solve(rhs,constrain = temp_go[side].cons)
            _go = go.empty_copy()
            _go.set_side(side, cons = side_cons, s = side_cons)
            cons_library[side] |= _go.cons
        elif isinstance(goal, ut.base_grid_object):
            _go = go[side]
            assert goal <= _go
            temp = goal + _go  ## take the union
            go_ = go.copy()
            go_.set_side(side, cons = temp.s)
            #domain_ = temp.domain  ## restrict to boundary
            #goal = temp.mapping()  ## create mapping
            cons_library[side] |= go_.cons
        else:  ## differentiable curve
            domain_ = domain.boundary[side]
            ischeme_ = gauss(ischeme*2)
            cons_library[side] = domain_.refine(2).project(goal, onto=basis, geometry=geom, ischeme=ischeme_, constrain=cons)
        cons |= cons_library[side]
    cons = cons | cons_  ##nonempty inner constraints are kept
    return cons


def constrained_boundary_projection(go, goal_boundaries_, corners, btol = 1e-2, rep_dict = None, ref = 0, **cons_args):  #Needs some fixing
    if isinstance(btol, (int, float)):  ## uniform btol => make them all equal
        btol = dict(zip(go._sides, [btol]*len(go._sides)))
    else:
        all(side in btol.keys() for side in go._sides)
    degree, ischeme, basis_type = go.degree, go.ischeme, go.basis_type
    prep_dict = preproc_dict(goal_boundaries_, go)
    goal_boundaries = prep_dict.instantiate(rep_dict)
    error_dict = {'left':0, 'right':0, 'top':0, 'bottom':0}
    ref_indices = {'left':[], 'right':[], 'top':[], 'bottom':[]}
    for side, goal in log_iter_sorted_dict_items('boundary', goal_boundaries):
        dim = side_dict[side]
        goal_ = goal if not isinstance(goal, ut.base_grid_object) else go.bc()  
        ## for now, if goal is a grid_object assume that go already satisfies the b.c.. Change this in the long run !
        ## replace goal by goal.function() or something once it becomes an object
        go_ = go[side]
        go_.s = go_.cons
        if basis_type == 'bspline':
            if not isinstance(goal, Pointset):  ## continuous function => integral error
                basis0 = go_.domain.basis_bspline(degree = 0, periodic = tuple(go.periodic))
                error = ((goal_ - go.bc())**2).sum(0)**0.5
                error_ = np.divide(*go_.integrate([basis0*error, basis0]))
                ref_indices[side] = numpy.where(error_ > btol[side])[0]
            else:  ## pointset => l2-residual
                error_ = np.sqrt(((goal.geom - go_.toscipy(goal.verts))**2).sum(0))
                verts = goal.verts[error_ > btol[side]]
                kv = go_._knots[0].knots
                ref_indices[side] = \
                list(set([i for i in range(len(kv) - 1) for s in verts if all([kv[i] <= s, s <= kv[i+1]])]))
            print(numpy.max(error_), side)
            error_dict[side] = error_
        else:
            raise NotImplementedError
    if False:  ## hierarchical grids forthcoming
        pass
    else:
        union_dict = {0: ['bottom', 'top'], 1: ['left', 'right']}
        ## take union of refinement indices
        ref_indices = [vec_union(*[ref_indices[side] for side in union_dict[j]]) for j in range(2)]
        print(ref_indices, 'ref_indices')
        if len(ref_indices[0]) == 0 and len(ref_indices[1]) == 0:  ## no refinement
            return None
        else: ## return refined go
            ## refine according to union
            ## create new topology
            new_go = go.ref_by(ref_indices).empty_copy()
            #cons_funcs = goal_boundaries_.from_geom(new_go.geom) if not rep_dict else goal_boundaries_.instantiate(rep_dict)
            new_go.set_cons(goal_boundaries_, corners, rep_dict = rep_dict, **cons_args)
            return new_go



def boundary_projection(go, goal_boundaries, corners = None, btol = 1e-2, rep_dict = None, maxref = 10, plot = True, **cons_args):
    basis_type = go.basis_type
    go.set_cons(goal_boundaries,corners, rep_dict = rep_dict,**cons_args)
    #go.quick_plot_boundary()
    go_list = [go]
    for bndrefine in log.count('boundary projection'):
        proj = constrained_boundary_projection(go_list[-1], goal_boundaries, corners, btol = btol, rep_dict = rep_dict,**cons_args)
        if proj is None or bndrefine == maxref:
            break
        else:
            go_list.append(proj)
            if plot:
                go_list[-1].quick_plot_boundary()
    return ut.multigrid_object(go_list)
