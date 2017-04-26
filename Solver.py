from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
from problem_library import Pointset
from auxilliary_classes import *
from preprocessor import preproc_dict

class Solver_info(object):
    def __init__(self, geom, domain, basis, ischeme, hom_index = None):
        if hom_index is None:
            bnd_int = domain.boundary.integrate(basis, geometry = geom, ischeme = ischeme)
            hom_index = np.where(np.abs(bnd_int) < 0.0001)[0]
        self.geom, self.domain, self.basis, self.ischeme, self.hom_index = geom, domain, basis, ischeme, hom_index
    

def boundary_projection(proj_info, curves, ref = 0):    #For now slow and not versatile enough
    sides = ['bottom', 'right', 'top', 'left']
    cons_x, cons_y = [n.util.NanVec(len(proj_info.basis))]*2
    for i in self.sides:
        cons_x |= domain.refine(ref).boundary[i].project(
                curves[i][0], onto = proj_info.basis, geometry = proj_info.geom, ischeme = proj_info.ischeme ) 
        cons_y |= domain.refine(ref).boundary[i].project(
                curves[i][1], onto = proj_info.basis, geometry = proj_info.geom, ischeme = proj_info.ischeme )
        vec_x, vec_y = [np.zeros(len(basis))]*2
    for i in list(set(range(len(basis))) - set(proj_info.hom_index)):
        vec_x[i] = cons_x[i]
        vec_y[i] = cons_y[i]
    return [vec_x, vec_y]


def BC_Transfinite_Interpolation(BC_info, input_curves, pols, P):
    sides = ['bottom', 'right', 'top', 'left']
    topo_dict = {'bottom':[1,0], 'right': [0,1], 'top':[1,0], 'left':[0,1]}
    for i in self.sides:
        arg = pols[i](BC_info.geom.dot(topo_dict[i])) if i in pols else BC_info.geom.dot(topo_dict[i])
        curves[i] = n.function.stack([ut.nutils_function(input_curves[i][j])(arg) for j in range(2)])
    vec_x, vec_y = boundary_projection(BC_info, input_curves)
    


class Solver(object):
    
    
    
    sides = ['bottom', 'right', 'top', 'left']
    topo_dict = {'bottom':[1,0], 'right': [0,1], 'top':[1,0], 'left':[0,1]}
    dual_basis = 0
    line_search_always = False
    prev_dirichlet_ext_x = 0
    prev_dirichlet_ext_y = 0
    prev_x = [0,0]
    jac_basis = 0
    mass = None
    mass_hom = None
    
    
  
    def vec_maker(self,c):
        l = len(self.basis_hom)
        a = self.dirichlet_vec[0]
        a[self.hom_index] = c[0:l]
        b = self.dirichlet_vec[1]
        b[self.hom_index] = c[l:2*l]
        return numpy.concatenate((a,b))
    
    
        
    def std_int(self, func, ref = 0):
        go = self.go
        return go.domain.refine(ref).integrate(func, geometry = go.geom, ischeme = gauss(go.ischeme))
    
    
    
    def make_basis_hom(self, basis, domain = None, ret_index_set = True):
        if domain is None:
            domain = self.domain
        index_set = np.where(np.abs(self.std_int(basis, domain = domain.boundary) < 0.0001)[0]) 
        ret = basis[index_set]
        if not ret_index_set:
            return ret
        else:
            return [ret, index_set]
        
        
  
    def make_gradients(self, basis):
        grad = basis.grad(self.geom,ndims = 2)
        return [grad, grad.grad(self.geom, ndims = 2)]
    
    
    def projection(self, func, mass = None):
        if mass is None:
            if not self.mass is None:
                mass = self.mass
            else:
                mass, self.mass = [self.std_int(n.function.outer(self.basis)).toscipy()]*2
        rhs = self.std_int(self.basis*func)
        return mass.solve(rhs)[0]
    
    
    
    def constrained_projection(self, func, constraint = 0, mass = None):
        if mass is None:
            if not self.mass_hom is None:
                mass = self.mass_hom
            else:
                mass, self.mass_hom = [self.std_int(n.function.outer(self.basis_hom)).toscipy()]*2
        rhs = self.std_int(self.basis_hom*(func - constraint))
        return mass.solve(rhs)[0]
        
    
    
    
    def set_basis_hom(self, basis, ret = False):
        basis_hom_0, self.hom_index = self.make_basis_hom(basis = basis)
        basis_lst = [basis_hom_0].extend(self.make_gradients(basis_hom_0))
        self.basis_hom = basis_lst
        if ret:
            return basis_lst
        
    
        
        
  
    def __init__(   
                    self, 
                    grid_object,             # [geom, domain, basis, ischeme,...]
                    corners,                       # [P0, P1, P2, P3]
                    cons,           # must be compatible with domain
                    dirichlet_ext_vec = None,# If None, dirichlet_ext is projected onto basis
                    initial_guess = None,    # initial guess such that the initial geometry = \
                                             # basis_hom.vector(2).dot(initial_guess) + dirichlet_ext
                    bad_initial_guess = False,
                    maxiter = 12,
                    mass_hom = None,
                    mass = None,
                    curves = None
                                                 ):
        
        
        #self.domain, self.geom, self.basis, self.ischeme, self.degree = grid_object.domain, grid_object.geom, grid_object.basis, grid_object.ischeme, grid_object.degree
        self.go = grid_object
        self.corners = corners
        self.cons = cons     
                
        if mass_hom is not None:
            self.mass_hom = mass_hom
            
            
            
        if mass is not None:
            self.mass = mass
            
            
    def update_all(self, additional_rhs = None):
        go = self.go
        rhs = [go.basis.vector(2).dot(self.cons | 0)]
        if additional_rhs is not None:
            rhs.append(additional_rhs)
        print(rhs, 'rhs')
        self.go = self.go.update_from_domain(self.go.domain, ref_basis = True)
        rhs = [rhs[i]*go.basis for i in range(len(rhs))]
        rhs.append(function.outer(go.basis))        
        rhs = self.std_int(rhs)
        print(rhs, 'rhs')
        mass = rhs[-1]
        rhs = rhs[:-1]
        lhs = mass.solve(rhs)
        self.cons = lhs[0]
        return lhs[1:] if len(lhs) > 0 else 0
            
            
    def one_d_laplace(self, ltol = 1e-7):
        go = self.go
        gbasis = go.basis.vector(2)
        target = function.DerivativeTarget([len(go.basis.vector(2))])
        res = model.Integral(gbasis['ik,1']*gbasis.dot(target)['k,1'], domain=go.domain, geometry=go.geom, degree=go.degree*3)
        lhs = model.newton(target, res, lhs0=self.cons | 0, freezedofs=self.cons.where).solve(ltol)
        return lhs
    
    
    def transfinite_interpolation(self, curves_library_, corners, rep_dict = None):    ## NEEDS FIXING
        go = self.go
        geom = go.geom
        curves_library = preproc_dict(curves_library_, go).instantiate(rep_dict)
        for item in curves_library:
            if isinstance(curves_library[item], Pointset):
                pnts = curves_library[item]
                curves_library[item] = ut.interpolated_univariate_spline(pnts.verts, pnts.geom, {'left': geom[1], 'right': geom[1], 'bottom': geom[0], 'top':geom[0]}[item])
        basis = go.basis.vector(2)
        expression = (1 - geom[1])*curves_library['bottom'] + geom[1]*curves_library['top']
        expression += (1 - geom[0])*curves_library['left'] + geom[0]*curves_library['right']
        expression += -(1 - geom[0])*(1 - geom[1])*np.array(corners[(0,0)]) - geom[0]*geom[1]*np.array(corners[(1,1)])
        expression += -geom[0]*(1 - geom[1])*np.array(corners[(1,0)]) - (1 - geom[0])*geom[1]*np.array(corners[(0,1)])
        return go.domain.project(expression, onto=basis, geometry=geom, ischeme=gauss(go.ischeme), constraints = go.cons)
            
              
    def main_function(self,c, russian = True):
        go = self.go
        s = go.basis.vector(2).dot(c)
        x_g, y_g = [s[i].grad(go.geom,ndims = 2) for i in range(2)]
        x_xi, x_eta = [x_g[i] for i in range(2)]
        y_xi, y_eta = [y_g[i] for i in range(2)]
        g11, g12, g22 = x_xi**2 + y_xi**2, x_xi*x_eta + y_xi*y_eta, x_eta**2 + y_eta**2
        vec1 = go.basis*(g22*x_xi.grad(go.geom,ndims = 2)[0] - 2*g12*x_xi.grad(go.geom,ndims = 2)[1] + g11*x_eta.grad(go.geom,ndims = 2)[1])
        vec2 = go.basis*(g22*y_xi.grad(go.geom,ndims = 2)[0] - 2*g12*y_xi.grad(go.geom,ndims = 2)[1] + g11*y_eta.grad(go.geom,ndims = 2)[1])
        
        if russian:
            return function.concatenate((vec1,vec2))/(2*g11 + 2*g22)
        else:
            return function.concatenate((vec1,vec2))
        
        
    def g12(self, c):
        go = self.go
        s = go.basis.vector(2).dot(c)
        x_g, y_g = [s[i].grad(go.geom,ndims = 2) for i in range(2)]
        x_xi, x_eta = [x_g[i] for i in range(2)]
        y_xi, y_eta = [y_g[i] for i in range(2)]
        return x_xi*x_eta + y_xi*y_eta
        
    
    
    def solve(self, init = None, ltol = 1e-7, method = 'Newton', t0 = None, cons = None, bnd = False):
        go = self.go
        basis = go.basis
        if cons is None:
            cons = self.cons
        if init is None:
            init = self.cons|0
        target = function.DerivativeTarget([len(go.basis.vector(2))])
        res = model.Integral(self.main_function(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3)
        if bnd:
            res += model.Integral(2*1e9*go.basis.vector(2).sum(-1)*self.g12(target)**2, domain = go.domain.boundary, geometry=go.geom, degree=go.ischeme*3)
        if method == 'Newton':
            lhs = model.newton(target, res, lhs0=init, freezedofs=cons.where).solve(ltol)
        elif method == 'Pseudotime':
            if t0 is None:
                t0 = 0.01
            term = basis.vector(2).dot(target)
            inert = -model.Integral(function.concatenate((basis*term[0], basis*term[1])), domain=go.domain, geometry=go.geom, degree=go.ischeme*3)
            lhs = model.pseudotime(target, res, inert, t0, lhs0=init, freezedofs=cons.where).solve(ltol)
        else:
            raise ValueError('unknown method: ' + method)
        return lhs
    
    
    def solve_with_repair(self, s):  ## repair defects using 'strategy' while keeping the dirichlet condition fixed
        go = self.go
        if not isinstance(s, np.ndarray):
            raise ValueError('Parameter s needs to be an np.ndarray')
        s = self.solve(init = s, cons = self.cons)
        jac = function.determinant(go.basis.vector(2).dot(s).grad(go.geom))
        basis_indices = ut.defect_check(self.go, jac)
        for irefine in log.count('Defect correction '):
            self.go.domain = self.go.domain.refined_by(elem for elem in self.go.domain.supp(self.go.basis, basis_indices))
            s = self.update_all(go.basis.vector(2).dot(s))
            s = self.solve(init = s, cons = self.cons)
            jac = function.determinant(go.basis.vector(2).dot(s).grad(go.geom))
            basis_indices = ut.defect_check(self.go, jac)
            if len(basis_indices) == 0:
                break
        return s
            
            