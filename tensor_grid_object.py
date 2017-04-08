cube_sides_ = ['left', 'right', 'bottom', 'top', 'back', 'front']
planar_sides_ = ['left', 'right', 'bottom', 'top']
opposite_side = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom', 'front': 'back', 'back': 'front'}
dims = {'left':0, 'right':0, 'bottom':1, 'top':1, 'back':2, 'front': 2}

class side_base(metaclass=abc.ABCMeta):
    
    _n = 0    
    
    def __init__(self, ndims , target_dim, side = None):  ## ndims = [n,m,...], target_dims => R^target_dims
        #assert len(ndims) == self._n
        assert side in cube_sides_ + [None]
        self._side = side
        self._ndims = ndims
        self._target_dim = target_dim
        
    @property
    def side(self):
        return self._side
        
    @property
    def p(self):
        return self._p if self._p else self
    
    @abc.abstractmethod
    def c(self,side_):  ## children
        pass
    
    def __getitem__(self,side_):
        return self.c(side_)
    
    
class cube(side_base): 
    
    _n = 3
      
    def __init__(self, *args):
        side_base.__init__(self, *args)
    
    @property
    def p(self):
        return self
    
    def c(self, side_):
        assert side_ in ['left', 'right', 'bottom', 'top', 'back', 'front']
        return plane.from_parent(self, side_)
    
    
class plane(cube):
    
    _p = None
    _n = 2
    
    @classmethod
    def from_parent(cls, parent, side_, *args, **kwargs):
        assert issubclass(cls, type(parent))
        #n,m,o = parent._ndims
        ## get this right
        #ndims = {'left':[n,o], 'front':[m,o], 'back':[m,o], 'right':[n,o], 'bottom':[m,n], 'top':[m,n]}[side_]
        ndims = parent._ndims.copy()
        ndims[dims[side_]] = 0
        ret = cls(ndims, parent._target_dim, *args, side = side_)
        ret._p = parent
        return ret 
    
    def __init__(self, *args, side = None):
        side_base.__init__(self, *args, side = side)
        
    @property
    def p(self):
        return self._p if self._p else self
    
    def c(self, side_):
        print(side_, self._side)
        #assert side_ in list(set(cube_sides_) - set([self._side, opposite_side[self._side]]))
        return line.from_parent(self, side_)
        
        
class line(plane):
    
    _p = None
    _n = 1
    
    #@classmethod
    #def from_parent(cls, parent, side_, *args, **kwargs):
    #    assert issubclass(cls, type(parent))
    #    n,m = parent._ndims
    #    ndims = {'left': [m], 'right': [m], 'bottom': [n], 'top': [n]}[side_]
    #    ret = cls(ndims, parent._target_dim, *args, side = side_)
    #    ret._p = parent
    #    return ret 
    
    def __init__(self, *args, side = None):
        side_base.__init__(self, *args, side = side)
    
    def c(self, side_):
        #if self._side in ['left', 'right']:
        #    assert side_ in ['bottom', 'top']
        #elif self._side in ['bottom', 'top']:
        #    assert side_ in ['left', 'right']
        #else:
        #    raise NotImplementedError
        return dot.from_parent(self, side_)
    
    
class dot(line):
    
    _p = None  
    _n = 0
    
    #@classmethod
    #def from_parent(cls, parent, side_, *args, **kwargs):
    #    assert issubclass(cls, type(parent))
    #    ndims = [0]
    #    ret = cls(ndims, parent._target_dim, *args, side = side_)
    #    ret._p = parent
    #    return ret 
    
    def __init__(self, *args, side = None):
        side_base.__init__(self,side = side)
      
    def c(self, *args, **kwargs):
        return self
