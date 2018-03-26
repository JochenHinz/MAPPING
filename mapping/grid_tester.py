from nutils import *
from . import ut
import numpy as np
import scipy as sp
from . import Solver, prep
from .aux import *
numpy.set_printoptions(threshold=numpy.nan)


def grid_convergence(go, load = None):
    domain, geom, basis, s, p, ischeme = go.domain, go.geom, go.basis, go.s, go.degree, go.ischeme
    if not load:
        load = numpy.ones(len(basis))
    mapping = basis.vector(2).dot(s)
    funcs = []
    for i in range(3):
        if i != 0:
            domain = domain.refine(1)
            basis = domain.basis('bspline', degree = p)
        mat = domain.integrate(function.outer( basis.grad(mapping,ndims=2), basis.grad(mapping,ndims=2) ).sum(2), geometry = mapping, ischeme = gauss(ischeme))
        rhs = domain.integrate(basis,geometry = geom, ischeme = gauss(ischeme))
        cons = domain.boundary.project(0, onto = basis, geometry = geom, ischeme = gauss(ischeme))
        lhs = mat.solve(rhs, constrain=cons, symmetric = True)
        funcs.append(basis.dot(lhs))
    diff1, diff2 = np.sqrt(domain.integrate([(funcs[1] - funcs[0])**2, (funcs[2] - funcs[1])**2], geometry = geom, ischeme = gauss(ischeme)))
    return np.log(diff1/diff2)/np.log(2)


def flatten_weights_to_gismo(go):
    assert len(go) in [1,2], NotImplementedError
    s, knots, p = go.s, go.knots, go.degree
    l = len(s) // go.repeat
    ss = [vec.reshape(go.ndims).T.flatten()[:,None] for vec in chunks(s,l)]
    return str(np.hstack(ss)).replace('[','').replace(']','')


def export_to_gismo(go):
    if len(go) == 2:
        return export_to_gismo_bivariate(go)
    elif len(go) == 1:
        return export_to_gismo_univariate(go)
    raise NotImplementedError


def export_to_gismo_bivariate(go):
    p, knots = go.degree, go._knots
    knots_xi, knots_eta = [(str(kn.extend_knots())[1:-1]).replace(',','') for kn in knots]
    string = '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n' \
             ' <xml> \n' \
             '   <Geometry type=\"TensorBSpline2\" id=\"0\"> \n' \
             '   <Basis type=\"TensorBSplineBasis2\"> \n' \
            '     <Basis type=\"BSplineBasis\" index=\"0\"> \n' \
            '<KnotVector degree=\"' + str(p[0]) + '\"> ' + knots_xi +' </KnotVector> \n' \
            '     </Basis> \n' \
            '     <Basis type=\"BSplineBasis\" index=\"1\"> \n' \
            '<KnotVector degree=\"' + str(p[1]) + '\"> ' + knots_eta +' </KnotVector> \n' \
            '     </Basis> \n' \
            '    </Basis> \n' \
            '    <coefs geoDim=\"' + str(go.repeat) + '\"> \n'\
            +flatten_weights_to_gismo(go)+'\n' \
            '    </coefs> \n' \
            '   </Geometry> \n' \
            '</xml>'
    return string


def export_to_gismo_univariate(go):
    assert len(go) == 1
    p, knots = go.degree[0], go._knots
    knots_xi = (str(knots.extend_knots()[0])[1:-1]).replace(',','')
    string = '<?xml version="1.0" encoding="UTF-8"?> \n' \
             '    <xml> \n' \
             '    <Geometry type="BSpline"> \n' \
             '    <Basis type="BSplineBasis"> \n' \
            '    <KnotVector degree=\"' + str(p) + '\"> ' + knots_xi +' </KnotVector> \n' \
             '    </Basis> \n' \
            '   <coefs geoDim=\"' + str(go.repeat) + '\"> \n' \
            + flatten_weights_to_gismo(go) + '\n' \
            '</coefs> \n' \
            '</Geometry> \n' \
            '</xml>'
    return string
