import collections, functools
import numpy as np
from nutils import *
from scipy.linalg import block_diag


########################################

## MAKE ALL THIS STUFF ADAPTIVE TO nD

########################################


Pointset = collections.namedtuple('Pointset', ['verts', 'geom'])
gauss = 'gauss{:d}'.format
planar_sides = ['left', 'right', 'bottom', 'top']
side_dict = {'left':1, 'right':1, 'bottom':0, 'top':0}
opposite_side = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom'}
dim_boundaries = {0: ['left', 'right'], 1: ['bottom', 'top']}
corners = lambda n,m: {'left':[0, m-1], 'bottom':[0, n-1], 'top':[0, n-1], 'right':[0, m-1]}

bnames = ('left', 'right'), ('bottom', 'top'), ('back', 'front')
map_bnames_dim = {name: dim for dim, names in enumerate(bnames) for name in names}
slices = {name: (slice(None),)*dim+(side,) for dim, names in enumerate(bnames) for name, side in zip(names, ([0],[-1]))}
    
    
class tensor_index:  ## for now only planar, make more efficient
    'returns indices of sides and corners'
    
    _p = None 
    _side = None  
    _l = 1  ## godfather length
    _n = 0 ## amount of dims
    
    @classmethod
    def from_go(cls, go, *args, **kwargs):
        ret = cls(go._ndims, repeat = go.repeat, side = go._side)
        ret._n, ret._l = len(go.ndims), np.prod(go.ndims)
        ret._indices = np.arange(np.array(go.ndims).prod()).reshape(go.ndims)  ## instantiate with range(N)
        ret._bnames = bnames[:len(go._ndims)]
        return ret
    
    @classmethod
    def from_parent(cls,parent,side):
        dim = map_bnames_dim[side]
        ndims = [parent._ndims[i] if i!= dim else 1 for i in range(len(parent._ndims))]
        ## select dimension corresponding to side
        ret = cls(ndims, repeat = parent._repeat, side = side, fromside = parent._side)  ## instantiate
        ret._p = parent  ## set parent
        ret._l, ret._n = parent._l, parent._n - 1
        ret._indices = parent._indices[slices[side]]
        ret._bnames = parent._bnames
        return ret
    
    def __init__(self, ndims, repeat = 1, side = None, fromside = None):  ## adapt to dims of any size
        assert len(ndims) < 4, 'Not yet implemented'
        self._ndims, self._repeat = ndims.copy(), repeat
        self._side = side
        
        
    def __len__(self):
        return len(self._ndims)
    
    
    @property
    def sides(self):
        return [element for tupl in self._bnames for element in tupl]
    
    
    def getslice(self,index,sl):
        index = (slice(None),)*index+(sl,)+(slice(None),)*(len(self) - index - 1)
        print(index)
        return self._indices[index].flatten()
        
        
    @property
    def p(self):
        assert self._p is not None
        return self._p
    
    def c(self, side):
        assert self._n > 0
        return tensor_index.from_parent(self,side)
    
    def __getitem__(self,side_):
        if side_ in self.sides and self._ndims[map_bnames_dim[side_]] != 1:
            return self.c(side_)
        else:
            return self._indices[side_]
    
    @property
    def indices(self):
        return np.concatenate([self._indices.flatten() + i*self._l*np.ones(np.prod(self._ndims), dtype=np.int) for i in range(self._repeat)])


###############################################################


## Prolongation / restriction matrix

## Make prolongation object-oriented maybe
        
        
def prolongation_matrix(p, *args):  ## MAKE THIS SPARSE
    ## args = [kv_new, kv_old] is [tensor_kv]*2, if len(kv_new) < len(kv_old) return restriction
    #assert all([len(args) == 2] + [len(k) == 1 for k in args])
    ## make sure we've got 2 tensor_kvs with dimension 1
    assert_params = [args[0] <= args[1], args[1] <= args[0]]
    assert any(assert_params), 'The kvs must be nested'  ## check for nestedness
    kv_new, kv_old = [k.extend_knots(p) for k in args] ## repeat first and last knots
    if assert_params[0]:  ## kv_new <= kv_old, reverse order 
        kv_new, kv_old = list(reversed([kv_new, kv_old]))
    n = len(kv_new) - 1
    m = len(kv_old) - 1
    T = numpy.zeros([n, m])
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if kv_new[i] >= kv_old[j] and kv_new[i] < kv_old[j+1]:
                T[i,j] = 1
    for q in range(p):
        q = q+1
        T_new = numpy.zeros([n - q, m - q])
        for i in range(T_new.shape[0]):
            for j in range(T_new.shape[1]):
                fac1 = (kv_new[i + q] - kv_old[j])/(kv_old[j+q] - kv_old[j]) if kv_old[j+q] != kv_old[j] else 0
                fac2 = (kv_old[j + 1 + q] - kv_new[i + q])/(kv_old[j + q + 1] - kv_old[j + 1]) if kv_old[j + q + 1] != kv_old[j + 1] else 0
                T_new[i,j] = fac1*T[i,j] + fac2*T[i,j + 1]
        T = T_new
    ## return T if kv_new >= kv_old else the restriction
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T)


### go.cons prolongation / restriction

def prolong_bc_go(fromgo, togo, *args, return_type = 'nan'):  ## args = [T_n, T_m , ...]
    to_shape = np.prod(togo._ndims)
    repeat = togo.repeat
    if return_type == 'nan':
        ret = util.NanVec(repeat*to_shape)  ## create empty NanVec of appropriate size
    else:
        ret = np.zeros(repeat*to_shape)
    if len(args) == 2:  ## more than one dimension
        for side in togo._sides:
            T = block_diag(*[args[side_dict[side]]]*repeat)
            vecs = fromgo.get_side(side)
            ret[togo._indices[side].indices] = T.dot(vecs[1])
        return ret
    elif len(args) == 1:
        for side in togo._sides:
            vecs = fromgo.get_side(side)
            ret[togo._indices[side].indices] = vecs[1]
        return ret
    else:
        raise NotImplementedError