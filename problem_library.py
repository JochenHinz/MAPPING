import numpy as np
import scipy as sp
from nutils import *
from auxilliary_classes import *
import utilities as ut
import reparam as rep
import collections
import os
import sys
import xml.etree.ElementTree as ET

pathdir = os.path.dirname(os.path.realpath(__file__))


###################################################

## Tidy up ! Use as much from utilities as possible


def arr(thing):
    return np.asarray(thing)

def read_xml(path):
    xml = ET.parse(path).getroot()
    kv = xml[0][0][0].text
    kv = numpy.asarray(sorted(set([float(i) for i in kv.split()])))  ## remove duplicate entries
    s = xml[0][1].text.split()
    s = np.asarray([float(i) for i in s])
    l = len(s) // 2
    s = np.asarray([s[2*i] for i in range(l)] + [s[2*i + 1] for i in range(l)])
    return ut.nonuniform_kv(kv), s

def middle(go, reparam = 'Joost', splinify = True): #Return either a spline or point-cloud representation of the screw-machine with casing problem
    pathdir = os.path.dirname(os.path.realpath(__file__))
    #ref_geom = go.geom
    
    top = numpy.stack([numpy.loadtxt(pathdir+'/xml/t2_row_{}'.format(i)) for i in range(1, 3)]).T
    top = top[numpy.concatenate([((top[1:]-top[:-1])**2).sum(1) > 1e-8, [True]])]
    top = top[65:230][::-1]

    bot = np.stack([np.loadtxt(pathdir+'/xml/t1_row_{}'.format(i)) for i in range(1, 3)]).T
    bot = bot[numpy.concatenate([((bot[1:]-bot[:-1])**2).sum(1) > 1.5*1e-4, [True]])]
    bot = bot[430:1390][::-1]
    print(type(bot), 'bot')
    if True: # fix
        if reparam == 'standard' or reparam == 'length':
            top_verts, bot_verts = [rep.reparam.reparam(reparam, 'discrete',[item]) for item in [top, bot]] ## for now only upper and lower
        elif reparam == 'Joost':
            top_verts, bot_verts = rep.reparam.reparam(reparam, 'discrete', [top, bot])
        else:
            raise ValueError(reparam +' not supported')
        corners = {(0,0): bot[0], (1,0): bot[-1], (0,1): top[0], (1,1): top[-1]}
        goal_boundaries = dict(
            left = lambda g: corners[0,0]*(1-g[1]) + corners[0,1]*g[1],
            right = lambda g: corners[1,0]*(1-g[1]) + corners[1,1]*g[1],
        )
        if splinify:
            goal_boundaries.update(
                top = lambda g: ut.interpolated_univariate_spline(top_verts, top, g[0]),
                bottom = lambda g: ut.interpolated_univariate_spline(bot_verts, bot, g[0]),
            )
        else:
            goal_boundaries.update(
                top = Pointset(top_verts[1:-1], top[1:-1]),
                bottom = Pointset(bot_verts[1:-1], bot[1:-1]),
            )
            
    return goal_boundaries, corners


def sines(go):
    corners = {(0,0): (0,0), (1,0): (2,0), (0,1): (0,1), (1,1): (2,1)}
    _g = lambda g: function.stack([g[0]*2, (1+0.5*function.sin(g[0]*numpy.pi*3))*g[1]+(-0.6*function.sin(g[0]*numpy.pi*4))*(1-g[1])])
    goal_boundaries = dict(left=_g, right=_g, top=_g, bottom=_g)
    return goal_boundaries, corners


def bottom(go):  ## make shorter, compacter, more elegant
    assert isinstance(go, ut.tensor_grid_object), 'This problem has to be instantiated with a tensor-product grid'
    print('knots in xi - direction will be overwritten')
    interior_corners = middle(go, reparam = 'standard', splinify = False)[1]
    p3, p4 = interior_corners[(1,0)], interior_corners[(0,0)]
    p1, p2 = [np.array([p_[0], 35]) for p_ in [p4,p3]]
    corners = {(0,0): p1, (1,0): p2, (0,1): p4, (1,1): p3}
    knots_xi, c = read_xml(pathdir + '/xml/motor_grid_bspline_3_278_13_bot_curve.xml')
    kv = knots_xi*go.knots[1]
    go = ut.make_go('bspline', go.degree, knots = kv)
    go.set_basis()
    go.cons = util.NanVec(2*len(go.basis))
    go.cons[side_indices(*go.ndims)['top']] |= c
    goal_boundaries = dict()
    goal_boundaries.update(bottom = lambda g: arr(p1)*(1 - g[0]) + arr(p2)*g[0], right = lambda g: arr(p2)*(1 - g[1]) + arr(p3)*g[1])
    print(len(go.domain.boundary['top'].basis('bspline', degree = go.degree, knotvalues = [go.knots.knots()[0]])), len(c))
    goal_boundaries.update(top = go, left = lambda g: arr(p1)*(1 - g[1]) + arr(p4)*g[1])
    return go, goal_boundaries, corners
    
