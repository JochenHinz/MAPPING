from nutils import *
import numpy as np
import scipy as sp
from . import ut
from .pl import Pointset
from .aux import *
from .prep import preproc_dict

class Solver(object):
    
    
    
    sides = ['bottom', 'right', 'top', 'left']
    topo_dict = {'bottom':[1,0], 'right': [0,1], 'top':[1,0], 'left':[0,1]}
    dual_basis = 0
    mass = None
    mass_hom = None 
        
  
    def __init__(   
                    self, 
                    grid_object,             # [geom, domain, basis, ischeme,...]
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
        
        self.go = grid_object
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
            
            
    def one_d_laplace(self, direction, ltol = 1e-7):  ## 0:xi, 1:eta
        go = self.go
        gbasis = go.basis.vector(2)
        target = function.Argument('target', [len(go.basis.vector(2))])
        res = go.domain.integral(gbasis['ik,' + str(direction)]*gbasis.dot(target)['k,' + str(direction)], geometry=go.geom, degree=go.degree*3)
        lhs = solver.newton('target', res, lhs0=self.cons | 0, freezedofs=self.cons.where).solve(ltol)
        return lhs
    
    
    def linear_spring(self):
        go = self.go
        mat = sp.sparse.csr_matrix(sp.sparse.block_diag([sp.sparse.diags([-1, -1, 4, -1, -1], [-go.ndims[1], -1, 0, 1, go.ndims[1]], shape=[np.prod(go.ndims)]*2)]*2))
        mat = matrix.ScipyMatrix(mat)
        return mat.solve(constrain = go.cons)
    
    
    def transfinite_interpolation(self, curves_library_, corners = None, rep_dict = None):    ## NEEDS FIXING
        go = self.go
        geom = go.geom
        curves_library = preproc_dict(curves_library_, go).instantiate(rep_dict)
        for item in curves_library:
            if isinstance(curves_library[item], Pointset):
                pnts = curves_library[item]
                curves_library[item] = ut.interpolated_univariate_spline(pnts.verts, pnts.geom, {'left': geom[1], 'right': geom[1], 'bottom': geom[0], 'top':geom[0]}[item])
        basis = go.basis.vector(2)
        expression = 0
        expression += (1 - geom[1])*curves_library['bottom'] + geom[1]*curves_library['top'] if 'top' in curves_library else 0
        expression += (1 - geom[0])*curves_library['left'] + geom[0]*curves_library['right'] if 'left' in curves_library else 0
        if corners is not None:
            expression += -(1 - geom[0])*(1 - geom[1])*np.array(corners[(0,0)]) - geom[0]*geom[1]*np.array(corners[(1,1)])
            expression += -geom[0]*(1 - geom[1])*np.array(corners[(1,0)]) - (1 - geom[0])*geom[1]*np.array(corners[(0,1)])
        return go.domain.project(expression, onto=basis, geometry=geom, ischeme=gauss(go.ischeme), constrain = go.cons)
            
              
    def Elliptic(self,c, russian = True):
        go = self.go
        g11, g12, g22 = self.fff(c)
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        vec1 = go.basis*(g22*x_xi.grad(go.geom,ndims = 2)[0] - 2*g12*x_xi.grad(go.geom,ndims = 2)[1] + g11*x_eta.grad(go.geom,ndims = 2)[1])
        vec2 = go.basis*(g22*y_xi.grad(go.geom,ndims = 2)[0] - 2*g12*y_xi.grad(go.geom,ndims = 2)[1] + g11*y_eta.grad(go.geom,ndims = 2)[1])
        if russian:
            return -function.concatenate((vec1,vec2))/(2.0*g11 + 2.0*g22)
        else:
            return -function.concatenate((vec1,vec2))
        
        
    def Elliptic_DG(self,c):
        go = self.go
        g11, g12, g22 = self.fff(c)
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        b = go.basis
        b_xi, b_eta = [b.grad(go.geom)[:,i] for i in range(2)]
        x = go.basis.vector(go.repeat).dot(c)
        #beta = function.repeat(x[_], go.ndims, axis = 0)
        J = function.determinant(x.grad(go.geom))
        vec1 = x_xi*((b*g22).grad(go.geom)[:,0] - (b*g12).grad(go.geom)[:,1]) + x_eta*(-(b*g12).grad(go.geom)[:,0] + (b*g11).grad(go.geom)[:,1])
        vec2 = y_xi*((b*g22).grad(go.geom)[:,0] - (b*g12).grad(go.geom)[:,1]) + y_eta*(-(b*g12).grad(go.geom)[:,0] + (b*g11).grad(go.geom)[:,1])
        A = function.stack([(x_xi*function.stack([g22, -g12]) + x_eta*function.stack([-g12, g11])).dotnorm(go.geom), (y_xi*function.stack([g22, -g12]) + y_eta*function.stack([-g12, g11])).dotnorm(go.geom)])
        #A = function.stack([(function.stack([x_xi*g22, x_eta*g11])).dotnorm(go.geom), (function.stack([y_xi*g22, y_eta*g11])).dotnorm(go.geom)]) ## set g_12 to 0
        alpha = 1000000
        vec_d_1 = b*(function.mean(A[0])) + alpha*b*function.jump(x_xi) + alpha*b*function.jump(x_eta)
        vec_d_2 = b*(function.mean(A[1])) + alpha*b*function.jump(y_xi) + alpha*b*function.jump(y_eta)
        #vec_d_1 -= alpha*b*function.jump((x.grad(go.geom)[:,0]).dotnorm(go.geom))
        #vec_d_2 -= alpha*b*function.jump((x.grad(go.geom)[:,1]).dotnorm(go.geom))
        return -function.concatenate((vec1,vec2)), function.concatenate((vec_d_1,vec_d_2))
    
    def Elliptic_partial_bnd_orth(self,c):
        pass
        
        
    def Liao(self,c):
        g11, g12, g22 = self.fff(c)
        return g11**2 + g22**2 + 2*g12**2
    
    def AO(self,c):
        g11, g12, g22 = self.fff(c)
        return g11*g22
    
    def Winslow(self,c):
        go = self.go
        g11, g12, g22 = self.fff(c)
        det =function.determinant(go.basis.vector(go.repeat).dot(c).grad(go.geom))
        return (g11+g22)/det
    
    def func_derivs(self,c):
        go = self.go
        s = go.basis.vector(2).dot(c)
        x_g, y_g = [s[i].grad(go.geom,ndims = 2) for i in range(2)]
        x_xi, x_eta = [x_g[i] for i in range(2)]
        y_xi, y_eta = [y_g[i] for i in range(2)]
        return x_xi, x_eta, y_xi, y_eta
    
    def elliptic_conformal(self,c, alpha_1 = 25, alpha_2 = 2):
        g11, g12, g22 = self.fff(c)
        return alpha_1*(g11- g22)**2 + alpha_2*g12**2
    
    def conformal(self,c):
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        return (x_xi - y_eta)**2 + (y_xi + x_eta)**2
        
        
        
    def fff(self, c):  ## first fundamental form
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        g11, g12, g22 = x_xi**2 + y_xi**2, x_xi*x_eta + y_xi*y_eta, x_eta**2 + y_eta**2
        return g11, g12, g22
    
    
    def solve(self, init = None, method = 'Elliptic', solutionmethod = 'Newton',\
              t0 = None, cons = None, ltol = None, bnd_AO = None, maxiter = np.inf, **solveargs):
        go = self.go
        assert len(go) == 2
        basis = go.basis
        if ltol is None:
            ltol = 1e-1/float(len(basis))
        if cons is None:
            cons = self.cons
        if init is None:
            init = self.cons|0
        target = function.Argument('target', [len(go.basis.vector(2))])
        if method == 'Elliptic':
            res = go.domain.integral(self.Elliptic(target), geometry=go.geom, degree=go.ischeme*3)
        elif method == 'Laplace':
            f = (go.basis.vector(2).dot(target).grad(go.geom)**2).sum()
            res = go.domain.integral(f, geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Elliptic_norus':
            res = go.domain.integral(self.Elliptic(target, russian = False), geometry=go.geom, degree=go.ischeme*3)
        elif method == 'Liao':
            res = go.domain.integral(self.Liao(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'AO':
            res = go.domain.integral(self.AO(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Winslow':
            res = go.domain.integral(self.Winslow(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Elliptic_conformal':
            res = go.domain.integral(self.elliptic_conformal(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Elliptic_partial':     
            #res = go.domain.integral(go.basis.vector(go.repeat)['ik,l']*go.geom['k,l'], geometry=go.basis.vector(go.repeat).dot(target), degree=go.ischeme*3)
            g11, g12, g22 = self.fff(target)
            b = basis.grad(go.geom,ndims = 2)
            res = go.domain.integral(function.concatenate([b[:,0]*g22 - b[:,1]*g12, b[:,1]*g11 - b[:,0]*g12]), geometry=go.geom, degree=go.ischeme*3)
        elif method == 'conformal':
            res = go.domain.integral(self.conformal(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Elliptic_optimize':
            g11, g12, g22 = self.fff(target)
            x_xi, x_eta, y_xi, y_eta = self.func_derivs(target)
            term1 = g22*x_xi.grad(go.geom,ndims = 2)[0] - 2*g12*x_xi.grad(go.geom,ndims = 2)[1] + g11*x_eta.grad(go.geom,ndims = 2)[1]
            term2 = g22*y_xi.grad(go.geom,ndims = 2)[0] - 2*g12*y_xi.grad(go.geom,ndims = 2)[1] + g11*y_eta.grad(go.geom,ndims = 2)[1]
            opt = term1**2 + term2**2
            res = go.domain.integral(opt, geometry=go.geom, degree=go.ischeme*3).derivative('target')
        elif method == 'Elliptic_DG':
            G, DG = self.Elliptic_DG(target)
            res = go.domain.integral(G, geometry=go.geom, degree=go.ischeme*3)
            res += go.domain.interfaces.integral(DG, geometry=go.geom, degree=go.ischeme*3)
        else:
            raise ValueError('unknown method: ' + method)
        if bnd_AO is not None:
            assert len(go.periodic) == 0
            pen, power = bnd_AO['pen'], bnd_AO['power']
            assert pen > 0 and power >= 0
            #scale_ = lambda x: np.piecewise(x, [0 <= x <= eps, eps < x < 1 - eps, 1 - eps <= x <= 1],\
            #                                    [lambda x: 1 - x/eps, 0, lambda x: 1 - 1/eps + x/eps])
            scale_ = lambda x: (1 - function.Exp(-1*(x - 0.5)**power))/(function.Exp(-1*(-0.5)**power))
            scale = scale_(go.geom[0])*(1 - scale_(go.geom[1])) + scale_(go.geom[1])*(1 - scale_(go.geom[0]))
            #if not all([eps in go.knots[0], 1-eps in go.knots[0]]) and all([eps in go.knots[1], 1-eps in go.knots[1]]):
            #    log.warning('Warning, the boundary orthogonality scaling function does not align with the knot-vector(s)')
            res += pen*go.domain.integral(scale*self.AO(target), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        if solutionmethod == 'Newton':
            lhs = solver.newton('target', res, lhs0=init, constrain=cons.where, **solveargs).solve(ltol,maxiter = maxiter)
        elif solutionmethod == 'Pseudotime':
            if t0 is None:
                t0 = 0.01
            term = basis.vector(2).dot(target)
            inert = go.domain.integral(function.concatenate((basis*term[0], basis*term[1])), geometry=go.geom, degree=go.ischeme*3)
            lhs = solver.pseudotime('target', res, inert, t0, lhs0=init, constrain=cons.where).solve(ltol, maxiter = maxiter)
        else:
            raise ValueError('unknown solution method: ' + solutionmethod)
        return lhs
    
    def optimize(self,init = None, weights = [1,0,0], solutionmethod = 'Newton', t0 = None, cons = None, ltol = None):
        go = self.go
        basis = go.basis
        if ltol is None:
            ltol = 1e-1/float(len(basis))
        if cons is None:
            cons = self.cons
        if init is None:
            init = self.cons|0
        target = function.Argument('target', [len(go.basis.vector(2))])
        g11, g12, g22 = self.fff(target)
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(target)
        term1 = g22*x_xi.grad(go.geom,ndims = 2)[0] - 2*g12*x_xi.grad(go.geom,ndims = 2)[1] + g11*x_eta.grad(go.geom,ndims = 2)[1]
        term2 = g22*y_xi.grad(go.geom,ndims = 2)[0] - 2*g12*y_xi.grad(go.geom,ndims = 2)[1] + g11*y_eta.grad(go.geom,ndims = 2)[1]
        elliptic_opt = term1**2 + term2**2
        uniform = np.sum([deriv.grad(go.geom,ndims = 2)[0]**2 for deriv in [x_xi,y_xi]])
        uniform += np.sum([deriv.grad(go.geom,ndims = 2)[1]**2 for deriv in [x_eta,y_eta]])
        uniform += 2*np.sum([deriv.grad(go.geom,ndims = 2)[1]**2 for deriv in [x_xi,y_xi]])
        orth = x_xi**2 + y_xi**2 + x_eta**2 + y_eta**2
        funcs = [elliptic_opt,uniform,orth]
        res = np.sum([funcs[i]*weights[i] for i in range(len(funcs))])
        res = go.domain.integral(res, geometry=go.geom, degree=go.ischeme*3).derivative('target')
        if solutionmethod == 'Newton':
            lhs = solver.newton('target', res, lhs0=init, constrain=cons.where).solve(ltol)
        elif solutionmethod == 'Pseudotime':
            if t0 is None:
                t0 = 0.01
            term = basis.vector(2).dot(target)
            inert = go.domain.integral(function.concatenate((basis*term[0], basis*term[1])), geometry=go.geom, degree=go.ischeme*3)
            lhs = solver.pseudotime('target', res, inert, t0, lhs0=init, constrain=cons.where).solve(ltol)
        else:
            raise ValueError('unknown solution method: ' + solutionmethod)
        return lhs
    
    def teichmuller(self, init = None, k0 = 0.5, solutionmethod = 'Newton', t0 = None, cons = None, ltol = None):
        go = self.go
        basis = go.basis
        if ltol is None:
            ltol = 1e-1/float(len(basis))
        if cons is None:
            cons = np.stack(self.cons)
        a = util.NanVec(len(cons) + 1)
        a[:-1] = cons
        cons = a
        if init is None:
            init = np.concatenate([self.cons|0,[k0]])
        else:
            init = np.concatenate([init,[k0]])
        target = function.Argument('target', [len(go.basis.vector(2)) + 1])
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(target[:-1])
        f_zc = (x_xi - y_eta)**2 + (y_xi + x_eta)**2
        f_z = (x_xi + y_eta)**2 + (y_xi - x_eta)**2
        res = (f_zc/f_z - target[-1]**2)**2
        res = go.domain.integral(res, geometry=go.geom, degree=go.ischeme*3).derivative('target')
        if solutionmethod == 'Newton':
            lhs = solver.newton('target', res, lhs0=init, constrain=cons.where).solve(ltol)
        elif solutionmethod == 'Pseudotime':
            if t0 is None:
                t0 = 0.01
            term = basis.vector(2).dot(target[:-1])
            inert = go.domain.integral(function.concatenate((basis*term[0], basis*term[1], target[-1])), geometry=go.geom, degree=go.ischeme*3)
            lhs = solver.pseudotime('target', res, inert, t0, lhs0=init, constrain=cons.where).solve(ltol)
        else:
            raise ValueError('unknown solution method: ' + solutionmethod)
        return lhs
    
    def teichmuller_(self,init = None, solutionmethod = 'Newton', t0 = None, cons = None, ltol = None):
        go = self.go
        basis = go.basis
        if ltol is None:
            ltol = 1e-1/float(len(basis))
        if cons is None:
            cons = self.cons
        if init is None:
            init = self.cons|0
        target = function.Argument('target', [len(go.basis.vector(2))])
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(target)
        f_zc = (x_xi - y_eta)**2 + (y_xi + x_eta)**2
        f_z = (x_xi + y_eta)**2 + (y_xi - x_eta)**2
        res = f_zc/f_z
        res = go.domain.integral((res.grad(go.geom,ndims = 2)**2).sum(), geometry=go.geom, degree=go.ischeme*3).derivative('target')
        if solutionmethod == 'Newton':
            lhs = solver.newton('target', res, lhs0=init, constrain=cons.where).solve(ltol)
        elif solutionmethod == 'Pseudotime':
            if t0 is None:
                t0 = 0.01
            term = basis.vector(2).dot(target)
            inert = go.domain.integral(function.concatenate((basis*term[0], basis*term[1])), geometry=go.geom, degree=go.ischeme*3)
            lhs = solver.pseudotime('target', res, inert, t0, lhs0=init, constrain=cons.where).solve(ltol)
        else:
            raise ValueError('unknown solution method: ' + solutionmethod)
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
            
            
