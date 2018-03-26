#! /usr/bin/env python3

from nutils import *
from .aux import *
import numpy

def _tryall( obj, prefix, kwargs ):
    for name in dir( obj ):
        if name.startswith( prefix ):
            try:
                return getattr( obj, name )( **kwargs )
            except TypeError:
                pass
    raise Exception( 'not supported: ' + ', '.join( kwargs.keys() ) )

class Hooke:

    def __init__( self, **kwargs ):
    
        verify = kwargs.pop( 'verify', True )

        if len(kwargs)!=2:
            raise ValueError( 'exactly two arguments expected, found %d' % len(kwargs) )

        _tryall( self, '_set_from_', kwargs )

        if verify:
            for key, value in kwargs.items():
                numpy.testing.assert_approx_equal( value, getattr(self,key) )

    def _set_from_lame( self, lmbda, mu ):
        self.lmbda = float(lmbda)
        self.mu = float(mu)

    def _set_from_poisson_young( self, nu, E ):
        self.lmbda = (nu*E)/((1.+nu)*(1.-2.*nu))
        self.mu = E/(2.*(1.+nu))

    def __call__ ( self, epsilon ):
        ndims = epsilon.shape[-2]
        assert epsilon.shape[-1] == ndims
        return self.lmbda * function.trace( epsilon )[...,_,_] * function.eye(ndims) + 2 * self.mu * epsilon
    
    def __str__( self ):
        return 'Hooke(mu=%s,lmbda=%s)' % ( self.mu, self.lmbda )

    @property
    def E( self ):
        return self.mu * (3.*self.lmbda+2.*self.mu) / (self.lmbda+self.mu)

    @property
    def nu( self ):
        return self.lmbda / (2.*(self.lmbda+self.mu))


@log.title
def makeplots( domain, geom, stress ):

  points, colors = domain.elem_eval( [ geom, stress[0,1] ], ischeme='bezier3', separate=True )
  with plot.PyPlot( 'stress', ndigits=0 ) as plt:
    plt.mesh( points, colors, tight=False )
    plt.colorbar()


def main(from_go_, to_go_, lmbda = 1., mu = 1., alpha = 1):
    assert from_go_.periodic == to_go_.periodic
    assert all([len(from_go_.ndims) == 2, len(to_go_.ndims) == 2])
    
    ## prolong everything to unified grid

    from_go = from_go_ + to_go_
    
    to_go = to_go_ + from_go_
    
    final_go = from_go_ + to_go_
    
    #final_go.quick_plot_boundary()
    
    ##
    
    dbasis = to_go.basis.vector(2)
    domain = to_go.domain

  # construct matrix
    stress = Hooke( lmbda=lmbda, mu=mu )
    elasticity = function.outer( dbasis.grad(from_go.mapping()), stress(dbasis.symgrad(from_go.mapping())) ).sum([2,3])
    matrix = domain.integrate( elasticity, geometry=from_go.mapping(), ischeme = gauss(from_go.ischeme) )

  # construct dirichlet boundary constraints
    cons = alpha*(to_go.cons - from_go.cons)

  # solve system
    lhs = matrix.solve( constrain=cons, tol=1e-10, symmetric=True, precon='diag' )

  # construct solution function
    final_go.cons = final_go.cons + cons
    final_go.s = final_go.s + lhs

    return final_go
