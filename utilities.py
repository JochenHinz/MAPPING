import numpy as np
import scipy as sp
import scipy.interpolate
from nutils import *
import inspect, collections, itertools, copy
from matplotlib import pyplot
from problem_library import Pointset
from auxilliary_classes import *
import preprocessor as prep
from scipy.linalg import block_diag


class interpolated_univariate_spline(function.Array): ## Need to figure out what this exactly does

    def __init__(self, vertices, values, position):
        assert function.isarray(position)
        assert values.shape[:1] == vertices.shape
        function.Array.__init__(self, args=[position], shape=position.shape+values.shape[1:], dtype=function._jointdtype(vertices.dtype, float))
        self._values_shape = values.shape[1:]
        self._splines = tuple(scipy.interpolate.InterpolatedUnivariateSpline(vertices, v) for v in values.reshape(values.shape[0], -1).T)

    def evalf(self, position):
        assert position.ndim == self.ndim
        shape = position.shape + self._values_shape
        position = position.ravel()
        return numpy.stack([spline(position) for spline in self._splines], axis=1).reshape(shape)
    

    
#########################################

## grid object and its auxilliary classes

#########################################


def open_kv_multiplicities(length, degree):
    return [degree+1] + [1]*(length - 2) + [degree+1]


    
class knot_object:  #a: beginning, b: end, n: steps  ## NEEDS FIXING, NEEDS TO BE INITIALIZED WITH KNOTS

    def __init__(self, *args, **kwargs):   ## either args = a,b,n or kwargs: knots = [....]
        if len(args) != 0:
            assert all([len(args) == 3, args[0] < args[1] and args[2] > 1])
            self.a, self.b, self.n = args[0], args[1], args[2] - 1
            self._knots = np.linspace(*args)
        else:
            assert 'knots' in kwargs
            _knots = kwargs['knots']
            assert all(np.diff(_knots) > 0) and len(_knots > 2), 'The knots-sequence needs to be strictly increasing'
            ## change this assertion to >= once I allow for knot-repetitions
            self._knots = _knots
            self.n = len(self._knots) - 1  ## amount of elements
            self.a, self.b = self._knots[0], self._knots[-1]
            
    def knots(self):
        return [self._knots]  ## Possibly get rid of the surrounding []
    
    
    def extend_knots(self, p):
        ret = [self.a]*p + list(self.knots()[0]) + [self.b]*p
        return np.asarray(ret)
    
    def __le__(self, other):  ## see if one is subset of other
        return set(self.knots()[0]) <= set(other.knots()[0])
    
    def __ge__(self, other):  ## the converse of __le__
        return other <= self
    
    
    
class uniform_kv(knot_object):
    
    def __init__(self, *args):
        knot_object.__init__(self,*args)
        
        
    def ref(self,r = 1):
        N = self.n
        for i in range(r):
            N += N - 1
        return knot_object(self.a, self.b, N)
    
    
class nonuniform_kv(knot_object):
    
    def __init__(self, knots):
        knot_object.__init__(self,knots = np.round(knots,10))   ### THIS IS FOR TESTING PURPOSES, I DON'T ACTUALLY WANNA BE ROUNDING HERE
        
        
    def ref_by(self,indices):
        if len(indices) == 0:
            return self
        print(len(indices), np.max(indices), self.n )
        assert all([len(indices) <= self.n, np.max(indices) < self.n])  ## amount of indices is of course smaller than the amount of elements
        new_knots = self._knots
        add = (np.asarray([new_knots[i+1] for i in indices]) + np.asarray([new_knots[i] for i in indices]))/2.0
        new_knots = numpy.insert(new_knots, [i + 1 for i in indices], add )
        return nonuniform_kv(new_knots)       
        
    def ref(self,ref = 1):
        if ref == 0:
            return self
        ret = copy.deepcopy(self)
        for i in range(ref):
            ret = ret.ref_by(range(len(ret.knots) - 1))
        return ret.ref_by(range(len(ret.knots) - 1))
    
    def __add__(self, other):  ## take the union, ## FIX: WE NEED TO ROUND OFF OR WE'LL GET DUPLICATES
        assert isinstance(other, type(self))
        ret = np.asarray(sorted(set( numpy.round(list(self.knots()[0]) + list(other.knots()[0]),10))))
        return nonuniform_kv(ret)
    
    def __mul__(self, other):
        assert isinstance(other, type(self))
        return tensor_kv(self,other)
    
    
class tensor_kv:  ## several knot_vectors
    
    ###################################################################
    
    ## Overall, in the long run, I'd like to allow for knot-repetitions
    
    ###################################################################
    
    
    _kvs = []
    index = -1
    
    def __init__(self,*args):  ## *args = kv1, kv2, ...
        self.ndims = len(args)
        self._kvs = list(args)
        
    def __getitem__(self,n):  ## in __getitem__ we do not return _kvs[n] but a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < self.ndims
        return tensor_kv(self._kvs[n])
    
    def __len__(self):
        return self.ndims
    
    def knots(self, ref = 0):  ## return 
        return [k.knots()[0] for k in self._kvs]
    
    def __add__(self, other):  ## take the elementwise union
        assert len(self._kvs) == len(other._kvs)
        return tensor_kv(*[self._kvs[i] + other._kvs[i] for i in range(len(self))])
    
    def ref_by(self, indices):  ## element-wise, indices = [[...], [...], ...]
        assert len(indices) == len(self)
        return tensor_kv(*[self._kvs[i].ref_by(indices[i]) for i in range(len(self))])
    
    def extend_knots(self, p):
        return [k.extend_knots(p) for k in self._kvs]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.ndims - 1:
            self.index = -1
            raise StopIteration
        self.index = self.index + 1
        return self[self.index]
    
    def __le__(self, other):
        if len(self) != len(other):  ## dimensionality does not match: return False
            return False
        elif len(self) == 1:
            return self._kvs[0] <= other._kvs[0]  ## if len(_kvs) == 1, we access _kvs[0] directly and compare
        else:
            ## if len(self) != 1, call __le__ len(self) times with len(item) == 1 tensor_kv's
            return all([self[i] <= other[i] for i in range(len(self))])
        
    def __ge__(self, other):
        return other <= self
    
    def __mul__(self,other):
        kvs = self._kvs
        kvs.extend(other._kvs)
        return tensor_kv(*kvs)
    
    

def grid_object(name, *args, **kwargs):
    if name == 'spline':
        return hierarchical_grid_object(*args, **kwargs)
    elif name == 'bspline':
        return tensor_grid_object(*args, **kwargs)
    else:
        raise ValueError('Unknown grid type ' + name)
        

class base_grid_object(object):
    s = None
    cons = None
    _knots = None
    domain = None
    geom = None
    degree = None
    sides = None
    ndims = None
    
    def __init__(self, grid_type_, *args, basis = None, ischeme = 6, knots = None, **kwargs):
        self.basis, self.ischeme, self._knots = basis, ischeme, knots
        if grid_type_ == 'spline': ## domain, geom, degree, for hierarchical
            assert len(args) == 3
            self.domain, self.geom, self.degree = args
        elif grid_type_ == 'bspline': ## degree and knots for tensor grid
            assert len(args) == 1 and self._knots is not None
            assert isinstance(knots, tensor_kv)
            self.degree, self.ischeme = *args, ischeme
            self.domain, self.geom = mesh.rectilinear(self.knots())
        else:
            raise ValueError('Unknown grid type ' + grid_type_)
        if not basis:  ## if the basis has not been stated specifically, we'll set it to the canonical choice of basis
                self.set_basis()
        
    def update_from_domain(self, domain_, new_knots = None, ref_basis = False):
        go_ = copy.deepcopy(self)
        go_.domain = domain_
        if new_knots is not None:
            go_.knots = new_knots
        if ref_basis:
            go_.set_basis()
        return go_
    
    def __len__(self):
        assert self._knots is not None
        return len(self._knots)
    
    
    def knots(self):
        return self._knots.knots()

    
    def _update(self,**kwargs):
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]
            
    def refine_knots(self,ref):
        assert self.knots is not None
        ret = [self.knots[i].ref(ref) for i in range(len(self.knots))]
        return ret
        
    def get_side(self,side):
        if not side in planar_sides:
            raise ValueError('side-keyword is invalid')
        else:
            go_ = copy.deepcopy(self)
            go_.domain, go_.side = go_.domain.boundary[side], side
            if go_._knots:
                go_._knots = go_._knots[1 if side in ['left', 'right'] else 0]
            return go_
        
    def make_cons(self, *args, **kwargs):
        assert self.basis is not None
        cons = prep.generate_cons(*args, **kwargs)
        return cons
    
    def set_cons(self, *args, **kwargs):
        self.cons = self.make_cons(self, *args, **kwargs)            
    
    
    def mapping(self):
        if self.s is None:
            return 0
        else:
            l = len(self.s) // len(self.basis)
            return (self.basis if l ==1 else self.basis.vector(l)).dot(self.s)
        
        
    #########################################################################
    
    ## IN THE LONG RUN THESE SHOULD BECOME METHODS OF THE CHILD CLASSES (so no assert necessary)
    
    
    @staticmethod
    def grid_union(leader,follower, prolong = True):  ##tensor_grid_objects
    ## take the union of two tensor grids, s and cons of leader will be prolonged
        assert all([isinstance(g, tensor_grid_object) for g in [leader, follower]] + [leader.degree == follower.degree]), 'Not yet implemented'
        ## make second assert statement compatible with len(args) > 2
        kv = leader._knots + follower._knots  ## take union of kvs
        ret = make_go(leader.basis_type, leader.degree, knots = kv)  ## initialize
        if prolong:
            ret.s, ret.cons = leader.prolong_func([leader.s, leader.cons], ret)  ## prolong first grid to unified grid
        return ret
    
    @staticmethod
    def mg_prolongation(fine, coarse, method = 'replace'):  ## multigrid_prolongation
    ## take the union of the grids but keep the bc of of fine while prolonging coarse.s
        assert all([isinstance(g, tensor_grid_object) for g in [fine, coarse]] + [fine.degree == coarse.degree]), 'Not yet implemented'
        ret = base_grid_object.grid_union(fine, coarse, prolong = False)  ## take grid union without prolongation
        ret.s = coarse.prolong_func([coarse.s], ret)   ## prolong coarse mapping to new grid (temporarily)
        ret.cons = fine.prolong_func([fine.cons], ret)   ## prolong fine constraints to new grid
        if method == 'project':  ## ret.s => constrained L2
            ret.s = np.asarray(ret.project(ret.mapping(), onto = ret.basis.vector(2), constrain = ret.cons))
        elif method == 'replace':  ## ret.s => combination of cons and s
            ret.s = np.asarray(ret.cons | ret.s) if len(ret) > 1 else ret.s
        return ret
    
    @staticmethod
    def grid_embedding(receiver, transmitter, prolong_cons = True, constrain_corners = True):  
    ## prolong / restrict s and possibly cons from transmitter to the grid of receiver (keep receiver.domain)
        assert all([isinstance(g, tensor_grid_object) for g in [receiver, transmitter]] + [receiver.degree == transmitter.degree]), 'Not yet implemented'
        ret = copy.deepcopy(receiver)  ## I ain't liking this
        ret.s, ret.cons = transmitter.prolong_func([transmitter.s, transmitter.cons if prolong_cons else None], ret)  
        ## prolong / restrict
        if ret.cons is None:  ## if ret.cons is None => prolong_cons is False, we take old constraints and combine with s
            ret.cons = receiver.cons
            ret.s = np.asarray(ret.cons | ret.s)
        if constrain_corners:  ## we make sure that the resulting geometry still satisfies s(0,0) = p0, s(1,0) = p1, ...
            assert len(ret.cons) == len(ret.s), 'The constraints and the mapping vector need to possess equal length'
            repeat, l  = [len(ret.s) // np.prod(ret.ndims), np.prod(ret.ndims)]
            ci = corner_indices(receiver.ndims)  ## extract the global index of the corners (0,0), (1,0), ...
            for i in range(repeat):  ## repeat repeat times
                for coord in ci.keys():
                    ret.cons[ci[coord] + i*l], ret.s[ci[coord]  + i*l ] = [receiver.s[ci[coord] + i*l]]*2          
        return ret
    
    
    #########################################################################
        
    
    def integrate(self,func, ref = 0, ischeme = None):
        if not ischeme:
            ischeme = self.ischeme
        return self.domain.refine(ref).integrate(func, geometry = self.geom, ischeme = gauss(ischeme))
    
    def project(self, func, ref = 0, ischeme = None, onto = None, constrain = None):
        ischeme = self.ischeme if ischeme is None else ischeme
        onto = self.basis if onto is None else onto
        return self.domain.refine(ref).project(func, geometry = self.geom, onto = onto, ischeme = gauss(ischeme), constrain = constrain)
    
    
    def set_basis(self):
        self.basis = self._basis()
        
        
    def dot(self,vec):
        assert self.basis is not None
        l = len(vec) // len(self.basis)
        return (self.basis if l == 1 else self.basis.vector(l)).dot(vec)
        
        
    def plot(self, name, ref = 0):
        assert self.s is not None, 'The grid object needs to have its weights specified in order to be plotted'
        _map = self.mapping()
        det = function.determinant(_map.grad(self.geom)) if len(self) > 1 else function.sqrt(_map.grad(self.geom).sum(-2))
        points, det = self.domain.refine(ref).elem_eval( [_map, det], ischeme='vtk', separate=True )
        with plot.VTKFile(name) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
            vtu.pointdataarray( 'det', det )
            
            
    def plot_boundary(self, name, ref = 0):
        assert self.s is not None and len(self) > 1
        points = self.domain.boundary.refine(ref).elem_eval( self.mapping(), ischeme='vtk', separate=True )
        with plot.VTKFile(name) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
            
    def plot_grid(self, name, ref = 0):
        points = self.domain.refine(ref).elem_eval( self.geom, ischeme='vtk', separate=True )
        with plot.VTKFile(name) as vtu:
            vtu.unstructuredgrid( points, npars=2 )
            
            
    def quick_plot(self, ref = 0):
        points = self.domain.refine(ref).elem_eval(self.mapping(), ischeme='bezier5', separate=True)
        plt = plot.PyPlot('I am a dummy')
        if len(self) >= 2:
            plt.mesh(points)
        else:
            plt.segments(np.array(points))
            plt.aspect('equal')
            plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
            
            
    def bc(self):
        assert self.cons is not None
        assert len(self) > 1, 'Not yet implemented.'
        return self.dot(self.cons | 0)
        
        
        
        
class hierarchical_grid_object(base_grid_object):
    
    basis_type = 'spline'
    
    def __init__(self, *args, **kwargs):
        base_grid_object.__init__(self, self.basis_type, *args, **kwargs)
        
        
    def _basis(self, degree = None, vector = None):
        if degree is None:
            degree = self.degree
        if vector is None:
            return self.domain.basis(self.basis_type, degree = degree)  ## make case distinction nicer
        else:
            return self.domain.basis(self.basis_type, degree = degree).vector(vector)
        
        
    def refine(self, ref, ref_basis = False):  # return a refined grid object with new coarse knots, refine basis is optional
        if ref == 0:
            return self
        else:
            go_ = copy.deepcopy(self)
            go_.knots, go_.domain = self.refine_knots(ref), self.domain.refine(ref)
            if ref_basis:
                go_.basis = self.domain.basis('spline', degree = go_.degree)
            return go_
        
    def pull_to_finest(self, ref_basis = False):
        if not isinstance(self.domain, topology.HierarchicalTopology):
            return self
        else:
            ref = len(self.domain.levels) - 1
            print(ref, 'ref')
            go_ = copy.deepcopy(self)
            go_.domain, go_.knots = go_.domain.levels[-1], go_.refine_knots(ref)
            if ref_basis:
                go_.basis = self.domain.basis('spline', degree = self.degree)
            return go_
        

class tensor_grid_object(base_grid_object):
    
    basis_type = 'bspline'
    
    def __init__(self, *args, **kwargs):
        base_grid_object.__init__(self, self.basis_type, *args, **kwargs)
        self.ndims = [len(k.knots()[0]) + self.degree - 1 for k in self._knots]
        self.sides = planar_sides
        if len(self.ndims) > 2:
            raise NotImplementedError
        
        
    def greville_abs(self):
        return ret_greville_abs(self.knots, self.degree)
    
    def _basis(self, degree = None, vector = None):
        if degree is None:
            degree = self.degree
        if vector is None:
            return self.domain.basis('bspline', degree = degree, knotvalues = self.knots())  ## make case distinction nicer
        else:
            return self.domain.basis('bspline', degree = degree, knotvalues = self.knots()).vector(vector)
        
        
    def ref_by(self, args, prolong_mapping = True, prolong_constraint = True):  ## args = [ref_index_1, ref_index2]
        assert len(args) == len(self.knots())
        new_knots = self._knots.ref_by(args)  ## refine the knot_vectors
        new_go = make_go(self.basis_type, self.degree, ischeme = self.ischeme, knots = new_knots) ## new grid with new kvs
        new_go.set_basis() 
        arg = [self.s if prolong_mapping else None, self.cons if prolong_constraint else None]  ## prolong or set to None
        new_go.s, new_go.cons = self.prolong_func(arg, new_go)
        return new_go
    
    def prolong_func(self, funcs, new_go, method = 'T'):  ## ugly, make prettier
        assert_params = [tensor_grid_object.are_nested(self,new_go)] + [self.degree <= new_go.degree]
        assert all(assert_params), 'the grid objects are not nested'
        if method == 'greville':  ## funcs = [basis.dot(vec), ...], this is for functions as opposed to vectors
            ## currently not in use
            assert all([isinstance(func, function.Evaluable) for func in funcs])
            return prolong_tensor_mapping_gos(funcs, new_go.basis.vector(2), self, new_go)
        elif method == 'T':  ## funcs = [vec1, vec2, ...]
            ## only handle vecs, bcs and None
            assert all([isinstance(func, np.ndarray or util.NanVec) or func is None for func in funcs]) 
            if all([func is None for func in funcs]):  ## only Nones: make no prolongation matrix but return Nones
                return funcs[0] if len(funcs) == 1 else funcs
            Ts = [prolongation_matrix(self.degree, *[new_go._knots[i], self._knots[i]]) for i in range(len(self._knots))]  ## make T_n, T_m, ....
            T = np.kron(*Ts) if len(Ts) != 1 else Ts[0]
            l = len(self.basis)
            f = lambda c: prolong_bc(c, *Ts) if isinstance(c, util.NanVec) else block_diag(*[T]*(len(c) // l)).dot(c)
            ## return prolongation of vec if not None else None
            ret = [f(func) if func is not None else None for func in funcs]
            return ret[0] if len(ret) == 1 else ret
    
    
    def __getitem__(self,side):
        assert side in planar_sides
        go_, dim = self.get_side(side), side_dict[side]
        go_.ndims = [self.ndims[dim]]
        go_.sides = dim_boundaries[dim]
        args = go_.s, go_.cons
        go_.set_basis()
        if args == [None]*len(args):  ## both s and cons None, do nothing
            pass
        else:  ## restrict to side
            go_.s = extract_sides(args[0], *self.ndims)[side] if args[0] is not None else None
            go_.cons = None
            if go_.s is not None:
                repeat = len(go_.s) // len(go_.basis)
                for side_ in dim_boundaries[dim]:  ## make this nicer, possibly manual
                    go_.cons = go_.domain.boundary[side_].project(go_.mapping(), geometry = go_.geom, onto = go_.basis.vector(repeat), constrain = go_.cons, ischeme = 'vertex')
        return go_
    
    
    ###################################################################################
    
    ## IMPLEMENT THESE FOR HIERARCHICAL GRIDS AND MAKE THEM PART OF THE BASE GRID CLASS (virtual functions or so)
    
    ##  Operator overloading
    
    
    def __add__(self, other):   ## self.cons and self.s are prolonged to unified grid
        if self >= other:  ## grids are nested
            return self
        else:
            return base_grid_object.grid_union(self, other)
        
    
    def __or__(self, other):  ## self.cons is kept and other.s is prolonged to unified grid
        return base_grid_object.mg_prolongation(self, other)
    
    
    def __sub__(self, other):  
        ## prolong / restrict everything from other to self while keeping self.domain, constrain the corners
        if not tensor_grid_object.are_nested(self,other):  ## grids are not nested => take grid union first
            fromgrid = other + self  ## other on the left because we need to keep other.cons and other.s
        else:
            fromgrid = other  ## grids are nested => simply take other
        return base_grid_object.grid_embedding(self, fromgrid)
            
    
    def __mod__(self, other): 
        ## self.cons is kept and other.s is prolonged / restricted into self.grid
        if tensor_grid_object.are_nested(self,other):  ## no grid union necessary
            fromgrid = other
        else:  ## grids are not nested => take union with other on the left because we need to retain other.s
            fromgrid = other + self
        if self >= fromgrid:  ## self is superset of other, just take self | other         
            return self | fromgrid
        elif self <= fromgrid:  ## self is subset of other, restrict other to self, while keeping self.cons
            ## We do not have to set constrain_corners to true assuming that self.cons satisfies s(0,0) = p0, ...
            return base_grid_object.grid_embedding(self, fromgrid, prolong_cons = False)
        
        
    ## go and go_[side] operations
    
    
    def extend(self,other):  ## exact 
        ## extend other to go[side] using a grid union in the side-direction replacing cons and s there, prolong the rest
        assert all([len(self) == 2, len(other) == 1]), 'Dimension mismatch'
        assert hasattr(other, 'side')
        ## Forthcoming
        return None
        
        
    def replace(self,other):  ## exact w.r.t. to other.side, possibly inexact w.r.t. self[oppositeside]
        ## replace self[side] by other go[oppositeside] is restricted / prolonged to kv in corresponding direction
        ## Forthcoming
        return None
    
    
    def inject(self,other):  ## possibly inexact
        ## coarsen other to self[side], keeping everythig else intact
        ## Forthcoming
        return None
    
    
    
    ## go[side], go_[otherisde] operations
    
    
    
    def __mul__(self,other):  ## axuilliary overload in order to make a grid with dimension self.ndims[0] * other.ndims[0]
        assert all([len(self.ndims) == 1,  len(other.ndims) == 1, self.side != other.side]), 'Not yet implemented'
        ret = make_go(self.basis_type, self.degree, knots = self._knots * other._knots)
        sides = [self.side, other.side]
        ## ret.s and ret.cons forthcoming
        return ret
        
        
        
        
        
    ####################################################################################   
    
    
    
    ## Logical operations
    
        
    @staticmethod
    def are_nested(leader,follower):  ## returns True when the the go's are nested else false
        return any([leader <= follower, follower <= leader])
    
    
    def __le__(self,other):
        if len(self.ndims) != len(other.ndims):
            return False
        else:
            k, k_n = self._knots, other._knots
            ## see if knot-vectors are all subsets of one another
            return all([set(k[i].knots()[0]) <= set(k_n[i].knots()[0]) for i in range(len(k))])
        
        
    def __pow__(self, other):  ## see if grids are nested
        return tensor_grid_object.are_nested(self,other)
            

def make_go(grid_type, *args, **kwargs):
    if grid_type == 'spline':
        return hierarchical_grid_object(*args, **kwargs)
    elif grid_type == 'bspline':
        return tensor_grid_object(*args, **kwargs)
    else:
        raise ValueError('unknown type ' + grid_type)
        
        
class multigrid_object(object):
    
    _gos = []
    
    def __init__(self, go_list):
        self._gos = go_list
        
    def __len__(self):
        return len(self._gos)
    
    def __getitem__(self,n):
        assert n < len(self)
        return self._gos[n]
    
    def __setitem__(self, key, value):
        assert key < len(self)
        self._gos[key] = value
        
    def plot(self, name, ref = 0):
        for i in range(len(self)):
            self[i].plot(name +'_%i' %i, ref = ref)
            

def nutils_function(func, derivative = np.polynomial.polynomial.Polynomial([0])):
    return lambda arg: function.pointwise([arg], func, nutils_function(lambda arg_: derivative(arg_)))

def zero_func():
    return nutils_function(lambda x:0)
  
        
def smart_plot(domain, funcs, name, names = [], ref = 0):
    cont = domain.refine(ref).elem_eval( funcs, ischeme='vtk', separate=True )
    with plot.VTKFile(name) as vtu:
        if len(funcs) > 1:
            vtu.unstructuredgrid( cont[0], npars=2 )
            for i in range(1, len(funcs)):
                vtu.pointdataarray( names[i - 1], cont[i] )
        else:
            vtu.unstructuredgrid( cont, npars=2 )
                
                
                
def prolongation_mat(go, frm, onto, mass = None, incidence = None, int_mat = None):  
    # slow & dirty implementataion of a prolongation matrix from basis to onto 
    # by assumption w_i = sum_j W_j only for supp W_j subset supp w_i
    domain, geom, ischeme = go.domain, go.geom, go.ischeme
    
    if incidence is None:
        incidence = domain.integrate(
                                     function.outer(onto**2, basis**2),
                                     geometry = geom, ischeme = 'gauss1'
                                    ).toscipy().tocsc()
        
    if mass is None:
        mass = domain.integrate(function.outer(onto), geometry = geom, ischeme = ischeme).toscipy().tocsr()
        
    if int_mat is None:
        int_mat = domain.integrate(function.outer(onto,frm), geometry = geom, ischeme = ischeme).toscipy().tocsr()
        
    Sigma = [numpy.nonzero(incidence[:,i])[0] for i in range(incidence.shape[1])]
    res = []
    
    for i in range(incidence.shape[1]):
        mat = mass[Sigma[i], :].tocsc()[:, Sigma[i]].todense()
        vec = int_mat[Sigma[i],i].todense()
        result = numpy.linalg.solve(mat,vec)
        result = numpy.asarray([result[i] if numpy.abs(result[i]) > 1e-12 else 0 for i in range(len(result))])
        ## MAKE ME MORE EFFICIENT !!
        retvec = numpy.zeros(incidence.shape[0])
        
        for j in range(len(Sigma[i])):
            retvec[Sigma[i][j]] = result[j]
        ##
        res.append(list(retvec))
        print(i/float(incidence.shape[1]),'% finished')
        
    return scipy.sparse.csr_matrix(res).transpose()

def project_unrelated_topologies(fromfun, *, onto, fromtopo, totopo, fromgeom, togeom, points=None, ischeme=None):
    '''project ``fromfun`` on ``fromtopo`` onto basis ``onto`` on ``totopo``

    Project a function ``fromfun``, evaluable on topology ``fromtopo`` with
    geometry ``fromgeom``, onto basis ``onto``, evaluable on ``totopo`` with
    geometry ``togeom``, using either the set of integration points ``points``
    or an integration scheme defined by ``ischeme``.  Contrary to
    ``totopo.project`` this function can be used when the target ``fromfun`` is
    not evaluable on topology ``totopo``.
    '''

    if points is not None:
        assert ischeme is None
    else:
        assert ischeme is not None
        points = totopo.elem_eval(togeom, ischeme=ischeme)
    if not isinstance(fromfun, list):
        fromfun = [fromfun]
    fromptopo = fromtopo.locate(fromgeom, points, eps = 1e-7)
    toptopo = totopo.locate(togeom, points, eps = 1e-7)
    values = [fromptopo.elem_eval(fun, ischeme='vertex') for fun in fromfun]
    topfun = [function.elemwise({e.transform: v for e, v in zip(toptopo, value)}, shape=value.shape[1:]) for value in values]
    ret = [toptopo.project(fun, onto=onto, geometry=togeom, ischeme='vertex') for fun in topfun]
    return ret if len(ret) > 1 else ret[0]

def projection_unrelated_topologies(fromfun, *, onto, **kwargs):
    return onto.dot(project_unrelated_topologies(fromfun, onto=onto, **kwargs))


def prolong_tensor_mapping_gos(fromfun, onto, *args): 
    ## args = [old_tensor_grid_object, new_tensor_grid_object] 
    ## this is greville l_2 via two tensor_grids, very slow
    ## make more efficient
    assert all([len(args) == 2] + [isinstance(args[i] , tensor_grid_object) for i in range(2)])
    g_o, g_n = args
    return project_unrelated_topologies(fromfun, onto = onto, fromtopo = g_o.domain, totopo = g_n.domain,                 fromgeom = g_o.geom, togeom = g_n.geom, points = g_n.greville_abs())
    
    
    


    
########################

## DEFECT DETECTION (MOSTLY)

########################
    


def ret_greville_abs(knots,p, ref = 0):  # knots = [knot_object_x, knot_object_y, ...]
    grevs = []
    for kv in knots:
        if isinstance(kv, knot_object):
            kv = kv.ref(ref)
            kvk = kv.extend_knots(p)
            grevs.append([1.0/p*np.sum(kvk[i+1:i+p+1]) for i in range(len(kvk) - p - 1)])
        else:
            assert ref == 0
            kvk = kv
            grevs.append([1.0/p*np.sum(kvk[i+1:i+p+1]) for i in range(len(kvk) - p - 1)])
            # for knots of the form returned by make_jac_basis, the greville absciassae are of the form [0,0.2,0.4, ...]
            # write something that simplifies construction
    if len (grevs) > 1:
        return np.asarray(list(itertools.product(*grevs)))
    else:
        return np.asarray(grevs[0])


def make_jac_basis(go,ref = 0):  ## make a B-spline basis of order 2p - 1 with p + 1 internal knot repetitions 
    assert go.knots is not None
    domain, geom, knots, p  = go.domain, go.geom, go.knots, go.degree
    assert isinstance(domain, topology.StructuredTopology)
    km = [[2*p] + [p + 1]*(len(knots[i].knots) - 2) + [2*p] for i in range(len(knots))]
    knots_jac = [list(itertools.chain.from_iterable([[knots[i].knots[j]]*km[i][j] for j in range(len(km[i]))])) for i in range(len(km))] # make the knot_vector with repeated knots
    return domain.basis_bspline(2*p - 1, knotmultiplicities = km), knots_jac


def collocate_greville(go, func, onto, onto_p, onto_knots = None, ref = 0, ret_domain = False):  # collocate func onto onto. the grid_object must have verts specified for the greville abscissae
    if onto_knots is None:    # if onto_knots is None we'll take those that belong to the grid_object
        assert go.knots is not None
        knots = go.knots
    else:
        knots = onto_knots
    domain, geom, p  = go.domain, go.geom, go.degree
    assert isinstance(domain, topology.StructuredTopology)
    if ref > 0:
        if isinstance(knots[0], knot_object):
            knots = [knots[i].ref(ref) for i in range(len(knots))]
        domain = domain.refine(ref)
    verts = ret_greville_abs(knots, onto_p, ref = ref if ref > 0 else 0)
    domain_ = domain.locate(geom, verts, eps = 1e-12)
    weights = domain_.project(func, onto=onto, geometry=geom, ischeme='vertex')
    if not ret_domain:
        return weights
    else:
        return weights, domain_
    
    
def get_basis_indices(basis,elements):  ## returns the indices of the basis that are nonzero on the elements
    indices = set()
    indfunc = function.Tuple([ ind[0] for ind, f in basis.blocks ])
    for elem in elements:
        indices.update(numpy.concatenate( indfunc.eval(elem), axis=1 )[0])
    return list(indices)
    
def check_elem(jac, element, check_ischeme):  ## returns True if an element is defective according to an ischeme
    c = jac.eval(elem = element, ischeme = check_ischeme)
    return numpy.min(c) < 0
        


def defect_check(go, jac, method = 'greville', ref = 0, check_ischeme = 'bezier5'):
### returns the indices of basis functions that are nonzero on defective elements
### detected via, 'greville': pull domain to finest level and do a tensor-product greville interpolation,
### 'discont': project determinant on a discont basis
    go_ = go.refine(ref)
    if method == 'greville':
        go_ = go.pull_to_finest().refine(ref)
        jac_basis, jac_knots = make_jac_basis(go_, ref = 0)
        d = collocate_greville(go_, jac, jac_basis, 2*go_.degree - 1, onto_knots = jac_knots)
    elif method == 'discont':     ## does not work that well yet
        jac_basis = go_.domain.basis('discont', 2*go_.degree - 1)
        d = go_.domain.project(jac, onto = jac_basis, geometry = go_.geom, ischeme = gauss(go_.ischeme))
    elif method == 'discrete':   ## in the discrete case, we skip the projection alltogether and check all elements afterwards
        jac_basis = go_.basis
        d = -1*numpy.ones(len(jac_basis))
    else:
        raise ValueError('Unknown method ' + method)
    elems = go_.domain.supp(jac_basis, numpy.where(d <0)[0])
    if len(elems) == 0:
        print('no defective elements found')
        return []
    else:
        elems = [elem for elem in elems if check_elem(jac, elem, check_ischeme)]
        return get_basis_indices(go.basis, elems)       
    
