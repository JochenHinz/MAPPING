import numpy as np
import scipy as sp
import scipy.interpolate
from nutils import *
import inspect, collections, itertools, copy, functools, abc, pickle
from matplotlib import pyplot
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


@functools.total_ordering
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
        knot_object.__init__(self,knots = np.round(knots,10)) 
        ### THIS IS FOR TESTING PURPOSES, I DON'T ACTUALLY WANNA BE ROUNDING HERE
        
        
    def ref_by(self,indices):
        if len(indices) == 0:
            return self
        assert all([len(indices) <= self.n, np.max(indices) < self.n])
        ## amount of indices is of course smaller than the amount of elements
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
    

###########################################################################################

## old version of tensor_kv, replaced by a version that inherits from np.ndarray
    
    
@functools.total_ordering  
class tensor_kv_:  ## several knot_vectors
    
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
    
    def __setitem__(self,n, value):  ## in __getitem__ we do not return _kvs[n] but a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < self.ndims
        new_kvs = self._kvs.copy()
        new_kvs[n] = value
        return tensor_kv(new_kvs)
    
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
    
    def __mul__(self,other):
        kvs = self._kvs
        kvs.extend(other._kvs)
        return tensor_kv(*kvs)
    
    def delete(self,index):
        kvs = self._kvs.copy()
        del kvs[index]
        return tensor_kv[kvs]
    
    
###################################################################################
    
    
@functools.total_ordering     
class tensor_kv( numpy.ndarray ):

    def __new__( cls, *args ):
        obj = np.asarray(args).view(cls)
        return obj
    
    def knots(self, ref = 0):  ## return 
        return [k.knots()[0] for k in self]
    
    def ref_by(self, indices):  ## element-wise, indices = [[...], [...], ...]
        assert len(indices) == len(self)
        return tensor_kv(*[self[i].ref_by(indices[i]) for i in range(len(self))])
    
    def extend_knots(self, p):
        return [k.extend_knots(p) for k in self]  
    
    def at(self,n):  ## in __getitem__ we do not return _kvs[n] but a new tensor_kv with new._kvs = [self._kvs[n]]
        assert n < len(self)
        return tensor_kv(self[n])
    
    def __le__(self, other):
        if len(self) != len(other):  ## dimensionality does not match: return False
            return False
        elif len(self) == 1:
            return self._kvs[0] <= other._kvs[0]  ## if len(_kvs) == 1, we access _kvs[0] directly and compare
        else:
            ## if len(self) != 1, call __le__ len(self) times with len(item) == 1 tensor_kv's
            return all([self[i] <= other[i] for i in range(len(self))])
        
    def __mul__(self,other):
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
    sides = None
    ndims = None
        
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
        return self._knots.knots()

    
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

        
    def make_cons(self, goal_boundaries, corners, rep_dict = None, **kwargs):
        assert self.basis is not None
        funcs = goal_boundaries.instantiate(rep_dict) if rep_dict is not None else goal_boundaries.from_geom(self.geom)
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
        return cls(*args, s = s, cons = cons, **kwargs)    
    
    @classmethod
    def from_parent(cls, parent, side):
        knots = tensor_kv(parent._knots[side_dict[side]])
        entries = parent.get_side(side)
        ret = cls.with_mapping(entries[0], entries[1], parent.degree, parent.domain.boundary[side], parent.geom, side = side, target_space = parent._target_space, knots = knots)
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
    
                    
    def __init__(self, p, *args, ischeme = 6, knots = None, s = None, cons = None, side = None, target_space = None):
        assert knots is not None, 'Keyword-argument \'knots\' needs to be provided'
        self.degree, self.ischeme, self._knots = p, ischeme, knots.copy()
        if len(args) == 2: ## instantiation via domain, geom
            assert args[0].ndims == 1
            self.domain, self.geom = args
        elif len(args) == 0:  ## canonical instantiation via knots
            assert isinstance(knots, tensor_kv)
            self.domain, self.geom = mesh.rectilinear(self.knots)
        else:
            raise ValueError('Invalid amount of arguments supplied')
           
        ## _ndims according to basis, might change - making it adaptable to order elevation
        self._ndims = [len(k.knots()[0]) + self.degree - 1 for k in self._knots]
        ## If target_space is not specified assume it equals the dimension of the domain
        self._target_space = len(self._ndims) if not target_space else target_space 
        #if len(self.ndims) > 2:  ## don't allow for 3D yet
        #    raise NotImplementedError
        self._side = side  ## set self._side, this is gonna be handy when instantiating from a parent
        self.set_sides()  ## initialize boundary sides, ugly, find better solution
        self._s = np.zeros(self.repeat*np.prod(self.ndims)) if s is None else s
        self._cons = util.NanVec(len(self._s)) if cons is None else cons
        self.set_basis()
        self._indices = tensor_index.from_go(self, side = side)
        
        
    ###########################################################################################
     
    def set_sides(self):
        if len(self) == 2:
            self._sides = ['bottom', 'right', 'top', 'left']
        elif len(self) == 1:
            if self._side is None:
                self._sides = ['left', 'right']
            else:
                self._sides = ['bottom', 'top'] if self._side in ['left', 'right'] else ['left', 'right']
        else:
            self._sides = ['left', 'right', 'bottom', 'top', 'front', 'back']
            
            
    @property
    def p(self):
        return self._p if self._p is not None else self
    
    ## I only from 2D to 1D, 0D not allowed yet, which is why we return self when len(self) == 1
    def c(self, side):
        return tensor_grid_object.from_parent(self,side) if len(self) > 1 else self            
            
            
            
    #############################################################################################
    
    ## sides, indices & stuff

    
    ## Fugly, too repetetive, try fewer lines of code    
    
    def gets(self):
        return self._s
    def sets(self, value):
        self._s = value  
    def getcons(self):
        return self._cons
    def setcons(self, value):
        self._cons = value
        
    s = property(gets, sets)
    cons = property(getcons, setcons)
    
    def get_side(self, side):  ## get self.s, self.cons constrained to side
        ind = self._indices[side].indices
        return self._s[ind], self._cons[ind]
    
    def set_side(self, side, s = None, cons = None):
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
        
        
    ###################################################
    
    ## REQUIRED FOR GRID OPERATIONS
    
    ## MAKE ADAPTIVE TO 3D
    
    
    @staticmethod
    def grid_union(leader,follower, prolong = True):  ##tensor_grid_objects
    ## take the union of two tensor grids, s and cons of leader will be prolonged
        assert leader.degree == follower.degree, 'Not yet implemented'
        ## make second assert statement compatible with len(args) > 2
        new_knots = leader._knots + follower._knots  ## take union of kvs
        ret = tensor_grid_object(leader.degree, knots = new_knots, side = leader._side, target_space = leader._target_space)
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
                pass
            elif len(ret) == 2:
                toindex = ret.get_corner_indices()
                fromindex = receiver.get_corner_indices()
                ret._s[toindex], ret._cons[toindex] = [receiver._cons[fromindex]]*2
            else:
                raise NotImplementedError
        return ret
    
    
    #########################################################################
    
    @property
    def ndims(self):
        return self._ndims
        
        
    def greville_abs(self):
        return ret_greville_abs(self.knots, self.degree)
    
    def _basis(self, degree = None, vector = None):
        if degree is None:
            degree = self.degree
        if vector is None:
            return self.domain.basis('bspline', degree = degree, knotvalues = self.knots)  ## make case distinction nicer
        else:
            return self.domain.basis('bspline', degree = degree, knotvalues = self.knots).vector(vector)
         
    def ref_by(self, args, prolong_mapping = True, prolong_constraint = True):  ## args = [ref_index_1, ref_index2]
        assert len(args) == len(self.knots)
        new_knots = self._knots.ref_by(args)  ## refine the knot_vectors
        ## dummy go for prolong
        new_go = tensor_grid_object(self.degree, knots = new_knots, side = self._side, target_space = self._target_space)
        ## prolong or set to None
        new_mapping = self.prolong_weights(new_go, s = prolong_mapping, c = prolong_constraint)
        return tensor_grid_object.with_mapping(*new_mapping, self.degree, knots = new_knots, side = self._side, target_space = self._target_space)
    
    def prolong_weights(self, new_go, method = 'T', s = True, c = True):  ## ugly, make prettier
        assert_params = [tensor_grid_object.are_nested(self,new_go)] + [self.degree <= new_go.degree]
        assert all(assert_params), 'the grid objects are not nested'
        if method == 'T':  ## funcs = [vec1, vec2, ...]
            ## make T_n, T_m, ....
            Ts = [prolongation_matrix(self.degree, *[new_go._knots[i], self._knots[i]]) for i in range(len(self._knots))]  
            T = np.kron(*Ts) if len(Ts) != 1 else Ts[0]
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
        pass
    
    
    def __getitem__(self,side):
        return self.c(side)
    
    
    
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
        
        
    
    #######################################################################################        
        
    ## go and go_[side] operations
    
    @requires_dependence(subdim, has_side)
    def extend(self,other):  ## exact 
        ## extend other to go[side] using a grid union in the side-direction replacing cons and s there, prolong the rest
        dim = side_dict[other._side]
        ## prolong 1D go
        other_ = copy.deepcopy(other) + tensor_grid_object(other.degree, knots = other._knots + tensor_kv(self._knots[dim]), side = other._side)
        new_knots = copy.deepcopy(self._knots)
        new_knots[dim] = other_._knots[0]  ## EXTEND knots in corresponding direction
        new_go = copy.deepcopy(self) + tensor_grid_object(self.degree, knots = new_knots)
        new_go.set_side(other._side, s = other_.s, cons = other_.s)
        return new_go
        
    @requires_dependence(subdim, has_side)    
    def replace(self,other):  ## exact w.r.t. to other.side, possibly inexact w.r.t. self[oppositeside]
        ## replace self[side] by other, go[oppositeside] is restricted / prolonged to kv in corresponding direction
        dim = side_dict[other._side]
        new_knots = copy.deepcopy(self._knots)
        new_knots[dim] = other._knots[0]  ## REPLACE knots in corresponding direction
        new_go = tensor_grid_object(self.degree, knots = new_knots, side = self._side, target_space = self._target_space)
        new_go = new_go - copy.deepcopy(self)
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
        ret = tensor_grid_object(self.degree, knots = self._knots * other._knots)
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
    
