import numpy as np
import scipy as sp
import scipy.interpolate, scipy.sparse
from nutils import *
import inspect, collections, itertools, copy, functools, abc, pickle
from matplotlib import pyplot
from auxilliary_classes import *
import preprocessor as prep
from scipy.linalg import block_diag
import Solver
import elasticity
    
def rotation_matrix(angle):
    return np.array([[np.cos(angle), - np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    
    
def interpolated_univariate_spline(vertices, values, position, center = None, k = 1):
    values = values.T  ## get rid of this somehow
    assert function.isarray(position)
    assert values.shape[:1] == vertices.shape
    splines = tuple(scipy.interpolate.InterpolatedUnivariateSpline(vertices, v, k = k) for v in values.reshape(values.shape[0], -1).T)
    return InterpolatedUnivariateSpline(splines, position, values.shape[1:], 0, center = center)


class InterpolatedUnivariateSpline(function.Array):

    def __init__(self, splines, position, values_shape, nderivs, center = None):
        self._splines = splines
        self._position = position
        self._values_shape = values_shape
        self._nderivs = nderivs
        self._angle = 0
        self._offset = 0
        self._mat = rotation_matrix(self._angle)
        if center is None:
            self._center = np.array([0.0,0.0])
        else:
            self._center = center
        super().__init__(args=[position], shape=position.shape+values_shape, dtype=float)

    def evalf(self, position):
        assert position.ndim == self.ndim
        shape = position.shape + self._values_shape
        position = position.ravel()
        offset = self._offset*np.ones(position.shape)  ## xi^prime = xi + offset (mod 1)
        ret = numpy.stack([spline((position + offset)%1, nu=self._nderivs) for spline in self._splines], axis=1).reshape(shape)
        return self._mat.dot(ret.T).T + (np.eye(2) - self._mat).dot(self._center.T).T

    def _derivative(self, var, axes, seen):
        return \
            InterpolatedUnivariateSpline(self._splines, self._position, self._values_shape, self._nderivs+1)[(...,)+(None,)*len(axes)] \
            * function.derivative(self._position[(...,)+(None,)*len(self._values_shape)], var, axes, seen)

    def _edit(self, op):
        return InterpolatedUnivariateSpline(self._splines, function.edit(self._position, op), self._values_shape, self._nderivs)
    
    def _rotate(self, angle):
        self._angle += angle
        self._mat = rotation_matrix(self._angle)  
        
        
    def _reparam(self, initial_guess = None):  ## introduce xi^prime such that evalf(0)[1] = 0, i.e. recompute self._offset
        if initial_guess is None:
            initial_guess = 0
        func = lambda t: self.evalf(np.array([t%1]))[0][1]
        res = scipy.optimize.newton(func, x0 = initial_guess)
        self._offset += res
        
    def update_geom(self, geom):
        super().__init__(args=[geom], shape=geom.shape+self._values_shape, dtype=float)
        return self
        
    def __call__(self, geom):
        return self.update_geom(geom)
        

    
#########################################

## grid object and its auxilliary classes

#########################################


def open_kv_multiplicities(length, degree):
    return [degree+1] + [1]*(length - 2) + [degree+1]


## Make this a vertiual class
@functools.total_ordering
class knot_object:
            
    def __init__(self, degree, knotvalues = None, knotmultiplicities = None, periodic = False):
        assert knotvalues is not None, 'The knot-sequence has to be provided'
        knotvalues = np.round(knotvalues,10)
        ### THIS IS FOR TESTING PURPOSES, I DON'T ACTUALLY WANNA BE ROUNDING HERE
        assert knotvalues[0] == 0 and knotvalues[-1] == 1, 'The knots-sequence needs to start on 0 and end on 1'
        assert all(np.diff(knotvalues) > 0) and len(knotvalues) >= 2, 'The knots-sequence needs to be strictly increasing'
        self._knots, self._knotmultiplicities, self._degree, self._periodic = knotvalues, knotmultiplicities, degree, periodic
        self.n = len(self._knots) - 1  ## amount of elements
        self.a, self.b = self._knots[0], self._knots[-1]
        if knotmultiplicities is not None:
            assert len(knotmultiplicities) == self.n + 1 and all([i <= degree + 1 for i in knotmultiplicities])
            if periodic:
                assert all([knotmultiplicities[0] == knotmultiplicities[-1], knotmultiplicities[-1] <= degree])
        else:
            knotmultiplicities = [1 if periodic else (degree + 1)] + [1]*(self.n - 1) + [1 if periodic else (degree + 1)]
        self._knotmultiplicities = np.array(knotmultiplicities, dtype = int)
            
    @property       
    def knots(self):
        return self._knots
    
    @property
    def periodic(self):
        return self._periodic
    
    @property
    def dim(self):  ## amount of resulting basis functions, possibly change this to amount of elements
        return np.sum(self._knotmultiplicities[:-1]) if self.periodic else len(self.extend_knots()) - self._degree - 1
    
    @property
    def knotmultiplicities(self):
        return self._knotmultiplicities
    
    @property
    def degree(self):
        return self._degree    
    
    def extend_knots(self):
        knots, km = self._knots, self._knotmultiplicities
        ret = np.array(list(itertools.chain.from_iterable([[knots[j]]*km[j] for j in range(len(km))])))
        if self.periodic:  ## this seems to work
            ret = ret[km[0] - 1:]
            p = self._degree
            ret = np.concatenate([ret[-p-1:-1] - 1, ret, ret[1:p+1] +1])
        return ret
    
    def to_c(self,n):
        assert n >= -1
        kv, km = self._knots.copy(), self._knotmultiplicities.copy()
        if self.periodic:
            km = np.array([np.min([i, self.degree - n]) for i in km])
        else:
            km = np.concatenate([[km[0]],[np.min([i, self.degree - n]) for i in km[1:-1]], [km[-1]]])
        return self.__class__(self.degree, knotvalues = kv, knotmultiplicities = km, periodic = self.periodic)
    
    def c0_indices(self):
        knots, km = self._knots, self._knotmultiplicities
        extknots = np.array(list(itertools.chain.from_iterable([[knots[j]]*km[j] for j in range(len(km))])))
        if self.periodic:
            extknots = extknots[:-km[-1]]
        #return [i for i in range(len(kv) - self._degree - 1) if np.all([extknots[i:i+
    
    @staticmethod
    def unify_kv(kv1, kv2):
        assert kv1.periodic == kv2.periodic, 'Cannot unify knot-vectors of periodic and non-periodic type'
        assert kv1._degree == kv2._degree
        dict1, dict2 = [dict(zip(*item)) for item in [[kv._knots,kv._knotmultiplicities] for kv in [kv1, kv2]]]
        union = np.array(sorted(set(list(kv1._knots) + list(kv2._knots))))
        km = [max(dict1[i] if i in dict1 else 1, dict2[i] if i in dict2 else 1) for i in union]
        return union, km
    
    def __le__(self, other):  ## see if one is subset of other
        if not (set(self.knots) <= set(other.knots) and self._degree <= other._degree and self.periodic == other.periodic): 
            ## knots no subset or self.p > other.p => return False
            return False
        else:  # check if knotmultiplicities are smaller or equal
            dict1, dict2 = [dict(zip(*item)) for item in [[kv._knots,kv._knotmultiplicities] for kv in [self, other]]]
            #{knotvalues: knotmultiplicity, ... }
            return all([dict1[i] <= dict2[i] for i in self._knots])
        
    def __ge__(self, other):  ## see if one is subset of other
        return other <= self
    
    
## Currently unused, reactivate when hierarchical grids get enabled again  
#class uniform_kv(knot_object):
#    
#    def __init__(self, *args):
#        knot_object.__init__(self,*args)
#        
#        
#    def ref(self,r = 1):
#        N = self.n
#        for i in range(r):
#            N += N - 1
#        return knot_object(self.a, self.b, N)
    
    
class nonuniform_kv(knot_object):
    
    def __init__(self, degree, **kwargs):
        super().__init__(degree, **kwargs) 
        
    def ref_by(self,indices):
        if len(indices) == 0:
            return self
        assert all([len(indices) <= self.n, np.max(indices) < self.n])
        ## amount of indices is of course smaller than the amount of elements
        new_knots, new_km = self._knots, self._knotmultiplicities
        add = (np.asarray([new_knots[i+1] for i in indices]) + np.asarray([new_knots[i] for i in indices]))/2.0
        new_knots = numpy.insert(new_knots, [i + 1 for i in indices], add )
        new_km = numpy.insert(new_km, [i + 1 for i in indices], [1]*len(indices) )
        return nonuniform_kv(self._degree, knotvalues = new_knots, knotmultiplicities = new_km, periodic = self.periodic) 
        
    def ref(self,ref = 1):
        assert ref >= 0
        if ref == 0:
            return self
        ret = copy.deepcopy(self)
        for i in range(ref):
            ret = ret.ref_by(range(len(ret.knots) - 1))
        return ret
    
    #def __add__(self, other):  ## take the union, ## FIX: WE NEED TO ROUND OFF OR WE'LL GET DUPLICATES
    #    assert isinstance(other, type(self))
    #    ret = np.asarray(sorted(set( numpy.round(list(self.knots[0]) + list(other.knots[0]),10))))
    #    return nonuniform_kv(ret)
    
    def raise_multiplicities(self, amount, indices = [], knotvalues = []):
        if all([len(indices) == 0, len(knotvalues) == 0]) or amount == 0:
            return self
        assert any([len(indices) != 0 and len(knotvalues) == 0, len(indices) == 0 and len(knotvalues) != 0])
        knots, new_km = self._knots, self._knotmultiplicities.copy()
        if not indices:
            indices = [i for i in range(len(knots)) for j in knotvalues if np.abs(knots[i] - j)<0.000001]
        if indices == []:
            raise ValueError('Either invalid knotvalues or indices supplied')
        for i in indices:
            new_km[i] = new_km[i] + amount if (new_km[i] + amount <= self._degree + 1) else self._degree + 1
        return nonuniform_kv(self._degree, knotvalues = knots, knotmultiplicities = new_km, periodic = self.periodic)
    
    
    def add_c0(self,knotvalues):
        kv = self.add_knots(knotvalues)
        kv = kv.raise_multiplicities(self._degree, knotvalues = knotvalues).to_c(0)
        return kv
        
    def __add__(self, other):
        p = max(self._degree, other._degree)
        kv, km = knot_object.unify_kv(self,other)
        return nonuniform_kv(p, knotvalues = kv, knotmultiplicities = km, periodic = self.periodic)
    
    def add_knots(self, knotvalues):
        if len(knotvalues) == 0:
            return self
        assert all([ i <= 1 and i >= 0 for i in knotvalues]) #and (0 not in knotvalues and 1 not in knotvalues)
        knotvalues = list(sorted(set([0] + list(knotvalues) + [1])))
        dummy = nonuniform_kv(self._degree, knotvalues = knotvalues, periodic = self.periodic) 
        return self + dummy
    
    def __mul__(self, other):
        assert isinstance(other, type(self))
        return tensor_kv([self,other])
    

###################################################################################

@functools.total_ordering
class tensor_kv(numpy.ndarray):
    def __new__(cls, data):
        return numpy.array(data).view(cls)
    
    def _vectorize(name, return_type = None):
        def wrapper(*args, **kwargs):
            self, *args = args
            assert all(len(a) <= len(self) for a in itertools.chain(args, kwargs.values()))
            assert len(set(type(a) for a in itertools.chain(args, kwargs.values()))) <= 1  ## all of equal type
            return (self.__class__ if return_type is None else return_type)([getattr(e, name)(*(a[i] for a in args), **{k: v[i] for k, v in kwargs.items()}) for i, e in enumerate(self)])
        return wrapper
    
    def _prop_wrapper(name, return_type = list):
        @property
        def wrapper(self):
            return return_type([getattr(e, name) for e in self])
        return wrapper
    
    extend_knots = _vectorize('extend_knots',list)
    knots = _prop_wrapper('knots')
    degree = _prop_wrapper('degree')
    #periodic = _prop_wrapper('periodic',tuple)
    ndims = _prop_wrapper('dim')
    knotmultiplicities = _prop_wrapper('knotmultiplicities')
    ref_by = _vectorize('ref_by')
    ref = _vectorize('ref')
    add_knots = _vectorize('add_knots')
    raise_multiplicities = _vectorize('raise_multiplicities')
    to_c = _vectorize('to_c')
    add_c0 = _vectorize('add_c0')
    
    @property
    def periodic(self):
        return tuple([i for i in self if i.periodic])
    
    def at(self,n):  ## in __getitem__[n] we return the n-th knot_object, here a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < len(self) and n >= -len(self)
        return tensor_kv([self[n if n >=0 else n%len(self)]])
        
    def __mul__(self,other):
        if isinstance(other, knot_object):
            l = [i for i in self] + [other]
            return tensor_kv(l)
        else:
            raise NotImplementedError
    
    del _vectorize
    del _prop_wrapper
    
    
@functools.total_ordering     
class _tensor_kv( numpy.ndarray ):

    def __new__( cls, *data ):
        return np.asarray(data).view(cls)
    
    @property    
    def knots(self, ref = 0):  ## return 
        return [k.knots[0] for k in self]
    
    @property
    def knotmultiplicities(self):
        return [k._knotmultiplicities for k in self]
    
    @property
    def periodic(self):
        return tuple([i for i in range(len(self)) if self[i].periodic])
    
    @property
    def ndims(self):
        return tuple([i.dim for i in self])
    
    def ref_by(self, indices):  ## element-wise, indices = [[...], [...], ...]
        assert len(indices) == len(self)
        return tensor_kv(*[self[i].ref_by(indices[i]) for i in range(len(self))])
    
    def extend_knots(self):
        return [k.extend_knots() for k in self]
    
    def at(self,n):  ## in __getitem__[n] we return the n-th knot_object, here a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < len(self)
        return tensor_kv(self[n])
    
    
    def add_knots(self, knots):
        assert len(indices) <= len(self)
        return tensor_kv(*[self[i].add_knots(knots[i]) for i in range(len(knots))])
    
    def raise_multiplicity(self, amount, indices = None, knotvalues = None):
        if indices:
            assert knotvalues is None
            return tensor_kv(*[self[i].raise_multiplicity(amount, indices = indices[i]) for i in range(len(indices))])
        else:
            assert knotvalues is not None
            return tensor_kv(*[self[i].raise_multiplicity(amount, knotvalues = knotvalues[i]) for i in range(len(knotvalues))])            
    
    def __le__(self, other):
        if len(self) != len(other):  ## dimensionality does not match: return False
            return False
        elif len(self) == 1:
            return self[0] <= other[0]  ## if len(_kvs) == 1, we access _kvs[0] directly and compare
        else:
            ## if len(self) != 1, call __le__ len(self) times with len(item) == 1 tensor_kv's
            return all([self[i] <= other[i] for i in range(len(self))])
        
    def __mul__(self,other):
        if isinstance(other, knot_object):
            l = [i for i in self] + [other]
            return tensor_kv(*l)
        else:
            raise NotImplementedError
    

def grid_object(name, *args, **kwargs):
    if name == 'spline':
        return hierarchical_grid_object(*args, **kwargs)
    elif name == 'bspline':
        return tensor_grid_object(*args, **kwargs)
    else:
        raise ValueError('Unknown grid type ' + name)

class base_grid_object(metaclass=abc.ABCMeta):   ## IMPLEMENT ABSTRACT METHODS
    _s = None
    _cons = None
    _knots = None
    domain = None
    geom = None
    degree = None
    _sides = None
        
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
    
    ##########################################################
    
    ## Handling of s (mapping) and cons (constraints) virtual
    
    @abc.abstractmethod
    def gets(self):
        pass
    
    @abc.abstractmethod
    def sets(self, value):
        pass
        
    @abc.abstractmethod
    def getcons(self):
        pass
    
    @abc.abstractmethod
    def setcons(self, value):
        pass
    
    @abc.abstractmethod
    def s(self):
        pass
    
    @abc.abstractmethod
    def cons(self):
        pass
    
    
    ##########################################################
        
    
    
    @property
    def knots(self):
        return self._knots.knots
    
    @property
    def knotmultiplicities(self):
        return self._knots.knotmultiplicities
    
    @abc.abstractmethod
    def degree(self):
        pass
    
    @property
    def sides(self):
        return self._sides

    
    def _update(self,**kwargs):
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]
            
    def refine_knots(self,ref):
        assert self.knots is not None
        ret = [self.knots[i].ref(ref) for i in range(len(self.knots))]
        return ret
    
    @abc.abstractmethod        
    def get_side(self,side):
        pass

        
    def make_cons(self, goal_boundaries_, corners, rep_dict = None, **kwargs):
        assert self.basis is not None
        goal_boundaries = prep.preproc_dict(goal_boundaries_, self)
        funcs = goal_boundaries.instantiate(rep_dict) if rep_dict is not None else goal_boundaries.from_geom()
        cons = prep.generate_cons(self, funcs, corners, **kwargs)
        return cons
    
    def set_cons(self, *args, **kwargs):
        self.cons = self.make_cons(*args, **kwargs)            
    
    
    def mapping(self):
        if self.s is None:
            return 0
        else:
            return self.dot(self.s)
    
    @property
    def determinant(self):
        _map = self.mapping()
        return function.determinant(_map.grad(self.geom)) if len(self) > 1 else function.sqrt(_map.grad(self.geom).sum(-2))
        
        
    @property
    def repeat(self):  ## this one gives the ratio len(s) // len(basis), make this adaptive to nD
        return self._target_space
        
        
    #########################################################################
    
    ## NECESSARY INGREDIENTS FOR + - // | %
    
    
    @staticmethod
    @abc.abstractmethod  
    def grid_union(*args, **kwargs):  ##tensor_grid_objects
        pass
    
    @staticmethod
    @abc.abstractmethod  
    def mg_prolongation(*args, **kwargs):  ## multigrid_prolongation
        pass
    
    @staticmethod
    @abc.abstractmethod  
    def grid_embedding(*args, **kwargs):  
        pass
    
    
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
        l = self.repeat
        return (self.basis if l == 1 else self.basis.vector(l)).dot(vec)
    
    def bc(self):
        assert len(self) > 1, 'Not yet implemented.'
        return self.dot(self.cons | 0)
    
    def quick_solve(self, **kwargs):
        kwargs.setdefault('method', 'Elliptic')
        kwargs.setdefault('solutionmethod', 'Newton')
        solver = Solver.Solver(self, self.cons)   
        self.s = solver.solve(self.s, **kwargs)
        
    @abc.abstractmethod
    def detect_defects(self):
        pass
    
    
    #########################################################################
    
    ## PLOTTING
    
    ## MAKE LESS REPETETIVE with decorator
        
        
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
            
    def quick_plot_grid(self, ref = 0):
        points = self.domain.refine(ref).elem_eval(self.geom, ischeme='bezier5', separate=True)
        plt = plot.PyPlot('I am a dummy')
        if len(self) >= 2:
            plt.mesh(points)
        else:
            plt.segments(np.array(points))
            plt.aspect('equal')
            plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
            
            
    def quick_plot(self, *args):
        points = (self.domain.refine(args[0]) if len(args) != 0 else self.domain).elem_eval(self.mapping(), ischeme='bezier5', separate=True)
        plt = plot.PyPlot('I am a dummy')
        if len(self) >= 2:
            plt.mesh(points)
        else:
            plt.segments(np.array(points))
            plt.aspect('equal')
            plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
        
        
    def quick_plot_boundary(self, ref = 0):
        points = self.domain.boundary.refine(ref).elem_eval(self.bc(), ischeme='bezier5', separate=True)
        plt = plot.PyPlot('I am a dummy')
        plt.segments(np.array(points))
        plt.aspect('equal')
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
    
    
    
##########################################################

## HIERARCHICAL GRID OBJECT, AIN'T WORK YET       
        
        
class hierarchical_grid_object(base_grid_object):
    
    basis_type = 'spline'
    
    def __init__(self, *args, basis = None, ischeme = 6, knots = None, **kwargs):
        self.basis, self.ischeme, self._knots = basis, ischeme, knots
        assert len(args) == 3
        self.domain, self.geom, self.degree = args
        if not basis:  ## if the basis has not been stated specifically, we'll set it to the canonical choice of basis
            self.set_basis()
        
        
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
        
        
###################################################################

## TENSOR GRID OBJECT, WORKS 70 %, missing: extension to nD; n <= 3
## injection from lower order tensor_gos to parent gos
        
class tensor_grid_object(base_grid_object):
    
    basis_type = 'bspline' 
    _p = None
    _side = None
    
    #####################################
    
    ## VARIOUS CALLS TO __init__
    
    @classmethod
    def with_mapping(cls, s, cons, *args, **kwargs):
        ret = cls(*args, **kwargs)
        ret.s, ret.cons = s, cons
        return ret
    
    @classmethod
    def from_parent(cls, parent, side):
        entries = parent.get_side(side)
        init_lib = parent.instantiation_lib
        init_lib['knots'] = parent._knots.at(side_dict[side])
        ret = cls.with_mapping(entries[0], entries[1], parent.domain.boundary[side], parent.geom, **init_lib)
        ret._p = parent
        return ret
    
    
    ##Forthcoming
    @classmethod
    def from_domain(cls):
        return None
    
    @property
    def _init_kwargs(self):
        pass
        #return dict(ischeme=self.ischeme, knots=self.knots, ...)
    
    def _update(self, **update):
        kwargs = self._init_kwargs
        kwargs.update(update)
        return type(self)(**kwargs)
    
    
    ######################################################################################
    
    ### INITIALIZATION, MAKE SHORTER
    
                    
    def __init__(self, *args, ischeme = 6, knots = None, side = None, target_space = None):
        assert knots is not None, 'Keyword-argument \'knots\' needs to be provided'
        self.ischeme, self._knots = ischeme, knots.copy()
        if len(args) == 2: ## instantiation via domain, geom
            assert args[0].ndims == 1
            self.domain, self.geom = args
        elif len(args) == 0:  ## canonical instantiation via knots
            assert isinstance(knots, tensor_kv)
            self.domain, self.geom = mesh.rectilinear(self.knots, periodic = self.periodic)
        else:
            raise ValueError('Invalid amount of arguments supplied')
           
        ## _ndims according to basis, might change - making it adaptable to order elevation
        ## If target_space is not specified assume it equals the dimension of the domain
        self._target_space = len(self.ndims) if not target_space else target_space 
        if len(self.ndims) > 2:  ## don't allow for 3D yet
            raise NotImplementedError
        self._side = side  ## set self._side, this is gonna be handy when instantiating from a parent
        self.set_sides()  ## initialize boundary sides, ugly, find better solution
        self.set_basis()
        self.sets()
        self.setcons()
        self._indices = tensor_index.from_go(self)
        
    @property   
    def instantiation_lib(self):  ## for making a (altered) copy
        if self._side is not None:
            print('Warning self.sides information gets lost if not instantiated from parent!!')
        return { 'ischeme': self.ischeme, 'knots': self._knots, 'side': self._side, 'target_space': self.repeat}
    
    
    #######################################################################################
    
    ## Inheritance of self._knots
    
    
    def _kv_wrapper(name):
        def wrapper(*args, **kwargs):
            self, *args = args
            _kwargs = self.instantiation_lib
            _kwargs['knots'] = getattr(self._knots, name)(*args, **kwargs)
            new_go = tensor_grid_object(**_kwargs)
            new_mapping = self.prolong_weights(new_go, s = True, c = True)
            return tensor_grid_object.with_mapping(*new_mapping, **_kwargs)
        return wrapper
    
    def _prop_wrapper(name):
        @property
        def wrapper(self):
            return getattr(self._knots, name)
        return wrapper
    
    def extend_knots(self):
        return self._knots.extend_knots()
    
    knots = _prop_wrapper('knots')
    periodic = _prop_wrapper('periodic')
    ndims = _prop_wrapper('ndims')
    knotmultiplicities = _prop_wrapper('knotmultiplicities')
    #ref_by = _kv_wrapper('ref_by')
    add_knots = _kv_wrapper('add_knots')
    raise_multiplicities = _kv_wrapper('raise_multiplicities')
    to_c = _kv_wrapper('to_c')
    add_c0 = _kv_wrapper('add_c0')
    
    del _kv_wrapper
    del _prop_wrapper
        
        
    ###########################################################################################
        
    def set_sides(self):
        if self.periodic == (0,):
            raise ValueError('Periodicity in the xi-direction is prohibited, use eta instead')
        self._sides = self.domain._bnames
            
            
    @property
    def p(self):
        return self._p if self._p is not None else self
    
    ## I only from 2D to 1D, 0D not allowed yet, which is why we return self when len(self) == 1
    def c(self, side):
        return tensor_grid_object.from_parent(self,side) if len(self) > 1 else self     
    
    @property
    def degree(self):
        return [k._degree for k in self._knots]
    
    @property
    def internal_dofs(self):
        a = self.cons.where
        return np.array([i for i in range(len(go.s)) if not a[i]])
            
            
            
    #############################################################################################
    
    ## sides, indices & stuff

    
    ## Fugly, too repetetive, try fewer lines of code    
    
    def gets(self):
        return self._s
    
    def sets(self, value = None):
        if value is None:
            self._s = np.zeros(self.repeat*len(self.basis))
        else:
            assert len(value) == len(self.basis)*self.repeat
            self._s = value  
            
    def getcons(self):
        return self._cons
    
    def setcons(self, value = None):
        if value is None:
            self._cons = util.NanVec(len(self.s))
        else:
            assert len(value) == len(self.basis)*self.repeat
            self._cons = value
        
    s = property(gets, sets)
    cons = property(getcons, setcons)
    
    def get_side(self, side):  ## get self.s, self.cons constrained to side
        assert side in self._sides
        ind = self._indices[side].indices
        return self._s[ind], self._cons[ind]
    
    def set_side(self, side, s = None, cons = None):
        assert side in self._sides
        ind = self._indices[side].indices
        if s is not None:
            self._s[ind] = s  ## s_ is None => nothing happens
        if cons is not None:
            self._cons[ind] = cons  ## cons_ is None => nothing happens
            
    def get_corner_indices(self):
        assert len(self) == 2, 'Corners are only implemented for 2D'
        ret = []
        for side1 in ['left', 'right']:
            for side2 in ['bottom', 'top']:
                ret = ret + list(self._indices[side1][side2].indices)
        return np.array(ret, dtype = np.int)
    
    
    def quick_project(self, funcs):
        assert len(self) == 2, 'Not yet implemented'
        basis = self.basis
        try:
            l = len(funcs)
        except:
            l = False
        integr = basis*funcs if not l else [basis*i for i in funcs]
        rhs = self.integrate(integr)
        mass = sp.sparse.kron(*[self[side].integrate(function.outer(self[side].basis)).toscipy() for side in ['bottom', 'left']])
        return sp.sparse.linalg.cg(mass, rhs)[0] if not l else [sp.sparse.linalg.cg(mass, i)[0] for i in rhs]
        
    
    def quick_projection(self, funcs):
        try:
            l = len(funcs)
        except:
            l = False
        rhs = self.quick_project(funcs)
        return self.basis.dot(rhs) if not l else [self.basis.dot(rhs[i]) for i in range(l)]
        
        
        
        
    ###################################################
    
    ## REQUIRED FOR GRID OPERATIONS
    
    ## MAKE ADAPTIVE TO 3D
    
    
    @staticmethod
    def grid_union(leader,follower, prolong = True):  ##tensor_grid_objects
    ## take the union of two tensor grids, s and cons of leader will be prolonged
        assert leader.degree == follower.degree, 'Not yet implemented'
        ## make second assert statement compatible with len(args) > 2
        new_knots = leader._knots + follower._knots  ## take union of kvs
        ret = tensor_grid_object(knots = new_knots, side = leader._side, target_space = leader._target_space)
        if prolong:
            ret.s, ret.cons = leader.prolong_weights(ret)  ## prolong first grid to unified grid
        return ret
    
    @staticmethod
    def mg_prolongation(fine, coarse, method = 'replace'):  ## multigrid_prolongation
    ## take the union of the grids but keep the bc of of fine while prolonging coarse.s
        assert fine.degree == coarse.degree, 'Not yet implemented'
        ret = tensor_grid_object.grid_union(fine, coarse, prolong = False)  ## take grid union without prolongation
        ret.s = coarse.prolong_weights(ret, c = False)[0]   ## prolong coarse mapping to new grid (temporarily)
        ret.cons = fine.prolong_weights(ret, s = False)[1]   ## prolong fine constraints to new grid
        if method == 'project':  ## ret.s => constrained L2
            ret.s = np.asarray(ret.project(ret.mapping(), onto = ret.basis.vector(ret.repeat), constrain = ret.cons))
        elif method == 'replace':  ## ret.s => combination of cons and s
            ret.s = np.asarray(ret.cons | ret.s) #if len(ret) > 1 else ret.s
        return ret
    
    
    ################### IMPLEMENT CONSTRAIN CORNERS FOR nD !!!! #####################
    @staticmethod
    def grid_embedding(receiver, transmitter, prolong_constraints = True, constrain_corners = True):  
    ## prolong / restrict s and possibly cons from transmitter to the grid of receiver (keep receiver.domain)
        assert receiver.degree == transmitter.degree, 'Not yet implemented'
        ret = copy.deepcopy(receiver)  ## I ain't liking this
        if prolong_constraints:
            ret.s, ret.cons = transmitter.prolong_weights(receiver)  
        ## prolong / restrict
        else:
            ## if prolong_cons is False, we take old constraints and combine with s
            ret.cons = receiver.cons
            ret.s = np.asarray(ret.cons | transmitter.prolong_weights(ret, c= False)[0])
        if constrain_corners: ## we make sure that the resulting geometry still satisfies s(0,0) = p0, s(1,0) = p1, ...
            if len(ret) == 1:  ## for 1D not yet implemented
                for side in transmitter.sides:
                    temp = receiver.get_side(side)
                    ret.set_side(side, s = temp[0], cons = temp[1])
            elif len(ret) == 2:
                toindex = ret.get_corner_indices()
                fromindex = receiver.get_corner_indices()
                ret._s[toindex], ret._cons[toindex] = [receiver._cons[fromindex]]*2
            else:
                raise NotImplementedError
        return ret
    
    @staticmethod
    def grid_interpolation(verts, grid_objects):  ## return a function that generates an interpolation of previous grids
    ## verts = [t_0, t_1, t_2, ...], grid_objects = [go(t_0), go(t_1), go(t_2), ...]
        assert len(verts) == len(grid_objects), 'The amount of verts must match the amount of grid objects'
        assert len(verts) <= 6, 'Interpolation order is bounded from above by 5'
        if len(verts) == 1:  ## zeroth order interpolation
            return lambda x: grid_objects[0].s
        go = np.sum(grid_objects)
        gos = [i + go for i in grid_objects]
        s = lambda x:np.array([sp.interpolate.InterpolatedUnivariateSpline(verts, [i.s[j] for i in gos], k = len(verts) - 1)(x) for j in range(len(go.s))])
        return s
    
    
    #########################################################################
    
    @property
    def ndims(self):
        return self._knots.ndims
    
    @property
    def periodic(self):
        return [i for i in range(len(self)) if self._knots[i].periodic]
        
    #def greville_abs(self):
    #    return ret_greville_abs(self.knots, self.degree)
    
    def _basis(self, degree = None, vector = None):
        if degree is None:
            degree = self.degree
        basis = self.domain.basis('bspline', degree = degree, knotvalues = self.knots, knotmultiplicities = self.knotmultiplicities)  ## make case distinction nicer
        if vector is not None:
            basis = basis.vector(vector)
        return basis
         
    def ref_by(self, args, prolong_mapping = True, prolong_constraint = True):  ## args = [ref_index_1, ref_index2]
        assert len(args) == len(self.knots)
        new_knots = self._knots.ref_by(args)  ## refine the knot_vectors
        ## dummy go for prolong
        new_go = tensor_grid_object(knots = new_knots, side = self._side, target_space = self._target_space)
        ## prolong or set to None
        new_mapping = self.prolong_weights(new_go, s = prolong_mapping, c = prolong_constraint)
        return tensor_grid_object.with_mapping(*new_mapping, knots = new_knots, side = self._side, target_space = self._target_space)
    
    def ref(self,ref):
        assert ref >= 0
        if ref == 0:
            return self
        ret = copy.deepcopy(self)
        for i in range(ref):
            ret = ret.ref_by([range(len(k) - 1) for k in ret.knots])
        return ret            
        
    
    def prolong_weights(self, new_go, method = 'T', s = True, c = True):  ## ugly, make prettier
        assert_params = [tensor_grid_object.are_nested(self,new_go)] #+ [self.degree <= new_go.degree]
        assert all(assert_params), 'the grid objects are not nested'
        if method == 'T':  ## funcs = [vec1, vec2, ...]
            ## make T_n, T_m, ....
            Ts = [prolongation_matrix(*[new_go._knots[i], self._knots[i]]) for i in range(len(self._knots))]
            if len(Ts) == 1:
                T = Ts[0]
            else:
                ## allow for sparse later
                #T = (sp.sparse if all([sp.sparse.issparse(t) for t in Ts]) else np).kron(*Ts)
                T = np.kron(*Ts)
            l = self.repeat
            ret = [block_diag(*[T]*l).dot(self.s) if s else None, prolong_bc_go(self, new_go, *Ts) if c else None]
            return ret
        elif method == 'greville':
            raise ValueError('Yet to be implemented')
            #assert all([isinstance(func, function.Evaluable) for func in funcs])
            #return prolong_tensor_mapping_gos(funcs, new_go.basis.vector(2), self, new_go)
        else:
            raise NotImplementedError
            
    def prolong_function(self):
        ## Forthcoming
        raise NotImplementedError
    
    
    def __getitem__(self,side):
        assert side in self._sides
        return self.c(side)
    
    
    def detect_defects(self):
        jgo = jac_go(self)
        jgo.s = jgo.project(self.determinant)
        return jgo
    
    
    
    def requires_dependence(*requirements, operator = all):  ## decorator to ensure that self and other satisfy certain requirements 
        def decorator(fn):                                       
            def decorated(*args):                              
                if operator([req(*args) for req in requirements]):                  
                    return fn(*args)                         
                raise Exception("cannot perform requested operation with given arguments")                  
            return decorated                                          
        return decorator
    
    ###################################################################################
    
    ## for use in requires_dependence(...)
    
    sameclass = lambda x,y:  type(x) == type(y)
    subclass = lambda x,y: issubclass(type(y), type(x))
    superclass = lambda x,y: issubclass(type(x), type(y))
    samedim = lambda x,y: len(x) == len(y)
    subdim = lambda x,y: len(y) == len(x) - 1
    same_degree = lambda x,y: x.degree == y.degree
    has_side = lambda x,y: hasattr(y, '_side')
    not_periodic = lambda x,y: all([len(x.periodic) == 0, len(y.periodic) == 0])
    
    
    ###################################################################################
    
    ## IMPLEMENT THESE FOR HIERARCHICAL GRIDS AND MAKE THEM PART OF THE BASE GRID CLASS (virtual functions or so)
    
    ##  Operator overloading
    
    @requires_dependence(samedim)
    def __add__(self, other):   ## self.cons and self.s are prolonged to unified grid
        if self >= other:  ## grids are nested
            return self
        else:
            return tensor_grid_object.grid_union(self, other)
        
    
    @requires_dependence(samedim, same_degree)
    def __or__(self, other):   ## self.cons is kept and other.s is prolonged to unified grid
        return tensor_grid_object.mg_prolongation(self, other)
    
    @requires_dependence(samedim, same_degree)
    def __sub__(self, other):  
        ## prolong / restrict everything from other to self while keeping self.domain, constrain the corners
        if not tensor_grid_object.are_nested(self,other):  ## grids are not nested => take grid union first
            fromgrid = other + self  ## other on the left because we need to keep other.cons and other.s
        else:
            fromgrid = other  ## grids are nested => simply take other
        return tensor_grid_object.grid_embedding(self, fromgrid)
    
    @requires_dependence(samedim, same_degree)
    def __floordiv__(self, other):
        ## same as self - other but without constraining the corners
        if not tensor_grid_object.are_nested(self,other):  ## grids are not nested => take grid union first
            fromgrid = other + self  ## other on the left because we need to keep other.cons and other.s
        else:
            fromgrid = other  ## grids are nested => simply take other
        return tensor_grid_object.grid_embedding(self, fromgrid, constrain_corners = False)
        
            
    @requires_dependence(samedim, same_degree)
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
            return tensor_grid_object.grid_embedding(self, fromgrid, prolong_constraints = False)
        
    @requires_dependence(samedim, same_degree)
    def elast(self, other, alpha = 1):
        return elasticity.main(self,other, alpha = alpha)
        
        
    
    #######################################################################################        
        
    ## go and go_[side] operations
    
    @requires_dependence(subdim, has_side)
    def extend(self,other):  ## exact 
        ## extend other to go[side] using a grid union in the side-direction replacing cons and s there, prolong the rest
        dim = side_dict[other._side]
        ## prolong 1D go
        other_ = copy.deepcopy(other) + tensor_grid_object(knots = other._knots + self._knots.at(dim), side = other._side)
        new_knots = copy.deepcopy(self._knots)
        new_knots[dim] = other_._knots[0]  ## EXTEND knots in corresponding direction
        new_go = copy.deepcopy(self) + tensor_grid_object(knots = new_knots)
        new_go.set_side(other._side, s = other_.s, cons = other_.s)
        return new_go
        
    @requires_dependence(subdim, has_side)    
    def replace(self,other):  ## exact w.r.t. to other.side, possibly inexact w.r.t. self[oppositeside]
        ## replace self[side] by other, go[oppositeside] is restricted / prolonged to kv in corresponding direction
        dim = side_dict[other._side]
        new_knots = copy.deepcopy(self._knots)
        new_knots[dim] = other._knots[0]  ## REPLACE knots in corresponding direction
        new_go = tensor_grid_object(knots = new_knots, side = self._side, target_space = self._target_space)
        new_go = new_go // copy.deepcopy(self)
        new_go.set_side(other._side, s = other.s, cons = other.s)
        return new_go
    
    @requires_dependence(subdim, has_side)
    def inject(self,other):  ## possibly inexact
        ## coarsen other to self[side], keeping everythig else intact
        temp = self[other._side] - copy.deepcopy(other)
        return self.extend(temp)
    
    
    
    ## go[side], go_[otherisde] operations
    
    
    
    def __mul__(self,other):  ## axuilliary overload in order to make a grid with dimension self.ndims[0] * other.ndims[0]
        assert all([len(self) == 1,  len(other) == 1, self.side != other.side]), 'Not yet implemented'
        ret = tensor_grid_object(knots = self._knots * other._knots)
        sides = [self.side, other.side]
        ## ret.s and ret.cons forthcoming
        return ret
        
        
        
        
        
    ####################################################################################   
    
    
    
    ## Logical operations
    
        
    @staticmethod
    def are_nested(leader,follower):  ## returns True when the the go's are nested else false
        return any([leader <= follower, follower <= leader])
    
    @requires_dependence(subclass, superclass, operator = any)
    def __le__(self,other):
        if len(self.ndims) != len(other.ndims):
            return False
        else:
            k, k_n = self._knots, other._knots
            ## see if knot-vectors are all subsets of one another
            #return all([set(k[i].knots[0]) <= set(k_n[i].knots[0]) for i in range(len(k))])
            return all([k[i] <= k_n[i] for i in range(len(k))])
        
        
    def __pow__(self, other):  ## see if grids are nested
        return tensor_grid_object.are_nested(self,other)
    
    
class dummy_go(tensor_grid_object):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _kv_wrapper(name):
        def wrapper(*args, **kwargs):
            self, *args = args
            _kwargs = self.instantiation_lib
            _kwargs['knots'] = getattr(self._knots, name)(*args, **kwargs)
            new_go = dummy_go(**_kwargs)
            new_mapping = self.prolong_weights(new_go, s = True, c = True)
            return dummy_go.with_mapping(*new_mapping, **_kwargs)
        return wrapper
    
    def _prop_wrapper(name):
        @property
        def wrapper(self):
            return getattr(self._knots, name)
        return wrapper
    
    def extend_knots(self):
        return self._knots.extend_knots()
    knots = _prop_wrapper('knots')
    periodic = _prop_wrapper('periodic')
    ndims = _prop_wrapper('ndims')
    knotmultiplicities = _prop_wrapper('knotmultiplicities')
    ref_by = _kv_wrapper('ref_by')
    add_knots = _kv_wrapper('add_knots')
    raise_multiplicities = _kv_wrapper('raise_multiplicities')
    
    del _kv_wrapper
    del _prop_wrapper


def make_go(grid_type, *args, **kwargs):
    if grid_type == 'spline':
        return hierarchical_grid_object(*args, **kwargs)
    elif grid_type == 'bspline':
        return tensor_grid_object(*args, **kwargs)
    else:
        raise ValueError('unknown type ' + grid_type)
        
        
class multigrid_object:
    
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
            
    @staticmethod
    def from_file(name):
        pkl_file = open(name + '.pkl', 'rb')
        mgo = pickle.load(pkl_file)
        pkl_file.close()
        return multigrid_object(mgo._gos)
        
            

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
            kvk = kv.extend_knots()
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


def jac_go(go,ref = 0):  ## make a B-spline basis of order 2p - 1 with p + 1 internal knot repetitions 
    assert go.knots is not None
    domain, geom, knots, p, periodic  = go.domain, go.geom, go.knots, np.array(go.degree), go.periodic
    _p = 2*p - 1
    assert isinstance(domain, topology.StructuredTopology)
    km = [[p[i] + 1 if i in periodic else 2*p[i]] + [p[i] + 1]*(len(knots[i]) - 2) + [p[i] + 1 if i in periodic else 2*p[i]] for i in range(len(knots))]
    #knots_jac = [list(itertools.chain.from_iterable([[knots[i][j]]*km[i][j] for j in range(len(km[i]))])) for i in range(len(km))] # make the knot_vector with repeated knots
    _knots = np.prod([nonuniform_kv(_p[i], knotvalues = go._knots[i].knots, periodic = go._knots[i].periodic, knotmultiplicities = km[i]) for i in range(len(km))])
    return tensor_grid_object(knots = _knots, target_space = 1)


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
    
