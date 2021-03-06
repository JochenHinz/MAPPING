import collections
import numpy as np
from nutils import *
from scipy.linalg import block_diag
import utilities as ut


########################################

## MAKE ALL THIS STUFF ADAPTIVE TO nD


Pointset = collections.namedtuple('Pointset', ['verts', 'geom'])
gauss = 'gauss{:d}'.format
preproc_info_list = ['cons_lib', 'dirichlet_lib', 'cons_lib_lib', 'error_lib']
preproc_info = collections.namedtuple('preproc_info', preproc_info_list, verbose = True)
planar_sides = ['left', 'right', 'bottom', 'top']
side_dict = {'left':1, 'right':1, 'bottom':0, 'top':0}
opposite_side = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom'}
dim_boundaries = {0: ['left', 'right'], 1: ['bottom', 'top']}

def side_indices(n, m, repeat = 1):  ## args = [n,m], ## soon [n,m, ...]
    ret_ = {'bottom': [i*m for i in range(n)], 'top': [i*m + m - 1 for i in range(n)], 'left': list(range(m)), 'right': list(range((n - 1)*m, n*m))}
    l = n*m
    ret = ret_.copy()
    for i in range(1,repeat):
        for key in planar_sides:
            ret[key] += [j + i*l for j in ret_[key]]
    return ret
            
        
        

def corner_indices(ndims):  ## 2D => 0D
    if len(ndims) == 1:
        return {0: 0, 1: ndims[0] - 1}
    elif len(ndims) == 2:
        n,m = ndims
        return {(0,0): 0,  (1,0): m - 1, (0,1): (n-1)*m, (1,1): n*m - 1}
    else:
        raise NotImplementedError

def unit_vector(length, i):
    x = np.zeros(length)
    x[i] = 1
    return x
        
        
def prolongation_matrix(p, *args):  ## MAKE THIS SPARSE
    ## args = [kv_new, kv_old] is [tensor_kv]*2, if len(kv_new) < len(kv_old) return restriction
    assert all([len(args) == 2] + [isinstance(k, ut.tensor_kv) and len(k) == 1 for k in args])
    ## make sure we've got 2 tensor_kvs with dimension 1
    assert_params = [args[0] <= args[1], args[1] <= args[0]]
    assert any(assert_params), 'The kvs must be nested'  ## check for nestedness
    kv_new, kv_old = [k.extend_knots(p)[0] for k in args] ## repeat first and last knots
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
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T) ## return T if kv_new >= kv_old else the restriction

def extract_sides(*args):  ## for now only in 2D, args = [s, dim1, dim2, ...]
    if len(args[0]) == np.prod([i for i in args[1:]]):
        return extract_sides_single(*args)
    else:
        return extract_sides_multi(*args)

def extract_sides_single(s,*args):  ## s refers to a tensor-product weight-vector with univariate bases of length n and m
    assert len(s) == np.prod(args)
    d = side_indices(*args)
    ret = dict([(side, s[d[side]]) for side in planar_sides])
    return ret

def extract_sides_multi(s,*args):
    l = np.prod(args)
    assert l > 1
    repeat = len(s) // np.prod(args)
    ret = extract_sides_single(s[0:l], *args)
    for i in range(1,repeat):
        temp = extract_sides_single(s[i*l:(i+1)*l], *args)
        for side in ret.keys():
            ret[side] = np.concatenate(tuple([ret[side], temp[side]]))
    return ret
    

def prolong_bc(s, *args, return_type = 'nan'):  ## *args = T_n, T_m,  case distinction ugly !! temporary solution !!
    assert isinstance(s, util.NanVec or np.ndarray)
    if len(args) > 1:
        return prolong_bc_nD(s, *args, return_type = return_type)
    else:
        return prolong_bc_1D(s, *args, return_type = return_type)
    
    
def prolong_bc_1D(s, T, return_type = 'nan'):
    from_shape = T.shape[1]
    to_shape = T.shape[0]
    repeat = len(s) // from_shape  ## determine dimension of vector 
    if return_type == 'nan':
        ret = util.NanVec(repeat*to_shape)  ## create empty NanVec of appropriate size
    else:
        ret = np.zeros(repeat*to_shape)
    for i in range(repeat):
        ret[[j + i*to_shape for j in corner_indices([to_shape])]] = s[[j + i*from_shape for j in corner_indices([from_shape])]]
    return ret
    
    
def prolong_bc_nD(s, *args, return_type = 'nan'):
    from_shape = np.prod([T.shape[1] for T in args])
    to_shape = np.prod([T.shape[0] for T in args])
    repeat = len(s) // from_shape  ## determine dimension of vector 
    if return_type == 'nan':
        ret = util.NanVec(repeat*to_shape)  ## create empty NanVec of appropriate size
    else:
        ret = np.zeros(repeat*to_shape)
    if len(args) != 1:  ## more than one dimension
        goal_sides = side_indices(*[a.shape[0] for a in args])  ## extract indices w.r.t. to the sides of tensor-product vec
        for i in range(repeat):  ## make this more elegant (block_diag & stuff)
            d = extract_sides(s[range(i*from_shape, (i+1)*from_shape)], *[a.shape[1] for a in args])
            for side in planar_sides:
                rhs = args[side_dict[side]].dot(d[side])
                ret[[j + i*to_shape for j in goal_sides[side]]] = rhs
    else:  ## one dimension: simply prolong with prolongation matrix (times l)
        ret[:] = block_diag(*[args[1]]*l).dot(s) 
    return ret