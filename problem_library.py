import numpy as np
import scipy as sp
from nutils import *
from auxilliary_classes import *
import utilities as ut
import reparam as rep
import collections, re, os, sys
import xml.etree.ElementTree as ET

pathdir = os.path.dirname(os.path.realpath(__file__))


###################################################

## Tidy up ! Use as much from utilities as possible


def rot(weights, angle_):
    assert weights.shape[0] == 2
    mat = np.array([[np.cos(angle_), -np.sin(angle_)], [np.sin(angle_), np.cos(angle_)]])
    return mat.dot(weights.T).T


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


def single_female_casing(go, angle = 0, radius = 38, splinify = True):
    xml = ET.parse(pathdir +'/xml/SRM4+6.xml').getroot()
    female = xml[0].text.split()
    female = np.asarray([float(i) for i in female])
    female = np.reshape(female,[2, len(female)//2])
    female = np.vstack([female.T, female[:,0]])
    #female = female[np.concatenate([((female[1:]-female[:-1])**2).sum(1) > 8.0*1e-5, [True]])]
    #female = np.delete(female,[1, 642, 643, 644, 1285, 1286, 1287, 1928, 1929, 1930],0)
    steps = female.shape[0]
    absc = np.linspace(0,2*np.pi, steps)
    casing = (radius*np.vstack([np.cos(absc), np.sin(absc)])).T
    corners = {(0,0): (female[0,0],female[0,1]), (1,0): (casing[0,0],casing[0,1]), (0,1): (female[0,0],female[0,1]), (1,1): (casing[-1,0],casing[-1,1])}
    leftverts, rightverts = [rep.reparam.reparam('length', 'discrete',[item]) for item in [female, casing]]
    goal_boundaries = dict(
            bottom = lambda g: corners[0,0]*(1-g[0]) + corners[1,0]*g[0],
            top = lambda g: corners[0,1]*(1-g[0]) + corners[1,1]*g[0],
        )
    if splinify:
            goal_boundaries.update(
                left = lambda g: ut.interpolated_univariate_spline(leftverts, female, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(rightverts, casing, g[1]),
            )
    else:
            goal_boundaries.update(
                left = Pointset(leftverts[1:-1], female[1:-1]),
                bottom = Pointset(rightverts[1:-1], casing[1:-1]),
            )
    return goal_boundaries, corners


def single_male_casing(go, angle = 0, radius = 38, splinify = True, dims = None):
    xml = ET.parse(pathdir +'/xml/SRM4+6.xml').getroot()
    male = xml[1].text.split()
    male = np.asarray([float(i) for i in male])
    male = np.reshape(male,[2, len(male)//2])
    delentries = [37,38,39, 40, 419, 420, 421, 422, 801, 802, 803, 804, 1183, 1184, 1185, 1186, 1565, 1566, 1567, 1568, 1947, 1948, 1949, 1950]
    male = np.delete(male,delentries,1)
    male = np.vstack([male.T, male[:,0]])[::-1]
    steps = male.shape[0]
    offset = - np.pi/10.1
    absc = np.linspace(offset,2*np.pi + offset, steps)
    casing = (radius*np.vstack([np.cos(absc), np.sin(absc)])).T + np.asarray([56.52, 0.])[None,:]
    casing = casing[::-1]
    if dims is not None:
        male, casing = male[dims[0]: dims[1], :], fecasing[dims[0]: dims[1], :]
    corners = {(1,0): (male[0,0],male[0,1]), (0,0): (casing[0,0],casing[0,1]), (1,1): (male[0,0],male[0,1]), (0,1): (casing[-1,0],casing[-1,1])}
    rightverts, leftverts = [rep.reparam.reparam('length', 'discrete',[item]) for item in [male, casing]]
    goal_boundaries = dict(
            bottom = lambda g: corners[0,0]*(1-g[0]) + corners[1,0]*g[0],
            top = lambda g: corners[0,1]*(1-g[0]) + corners[1,1]*g[0],
        )
    if splinify:
            goal_boundaries.update(
                left = lambda g: ut.interpolated_univariate_spline(leftverts, casing, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(rightverts, male, g[1], center = np.array([56.52, 0.0])),
            )
    else:
            goal_boundaries.update(
                left = Pointset(leftverts[1:-1], casing[1:-1]),
                bottom = Pointset(rightverts[1:-1], male[1:-1]),
            )
    return goal_boundaries, corners
    

def middle(go, reparam = 'Joost', splinify = True): #Return either a spline or point-cloud representation of the screw-machine with casing problem
    pathdir = os.path.dirname(os.path.realpath(__file__))
    #ref_geom = go.geom
    
    top = numpy.stack([numpy.loadtxt(pathdir+'/xml/t2_row_{}'.format(i)) for i in range(1, 3)]).T
    top = top[numpy.concatenate([((top[1:]-top[:-1])**2).sum(1) > 1e-8, [True]])]
    top = top[65:230][::-1]

    bot = np.stack([np.loadtxt(pathdir+'/xml/t1_row_{}'.format(i)) for i in range(1, 3)]).T
    bot = bot[numpy.concatenate([((bot[1:]-bot[:-1])**2).sum(1) > 1.5*1e-4, [True]])]
    bot = bot[430:1390][::-1]
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
    kv = knots_xi*go._knots[1]
    go = ut.tensor_grid_object(go.degree, knots = kv)
    go.set_side('top', cons = c)
    goal_boundaries = dict()
    goal_boundaries.update(bottom = lambda g: arr(p1)*(1 - g[0]) + arr(p2)*g[0], right = lambda g: arr(p2)*(1 - g[1]) + arr(p3)*g[1])
    goal_boundaries.update(top = go, left = lambda g: arr(p1)*(1 - g[1]) + arr(p4)*g[1])
    return go, goal_boundaries, corners

def nrw(go, stepsize = 1):
    with open (pathdir + '/nrw.txt', "r") as nrw:
        data=nrw.readlines()
    data = str(data)
    vec = re.findall(r'[+-]?[0-9.]+', data[0:])
    vec = np.array([float(i) for i in vec])[::stepsize]
    l = np.nonzero(vec)[0]
    vec = vec[l]
    vec_ = np.reshape(vec,[len(vec)//2, 2]).T
    stepsize = vec_.shape[1]//4
    vec_0, vec_1, vec_2, vec_3 = vec_[:,0:stepsize].T, vec_[:,stepsize:2*stepsize - 3].T, vec_[:,2*stepsize-3:3*stepsize-5].T[::-1,:], vec_[:,3*stepsize-5:].T[::-1,:]
    verts_0, verts_1, verts_2, verts_3 = [rep.reparam.reparam('length', 'discrete',[item]) for item in [vec_0, vec_1, vec_2, vec_3]]
    #verts_2, verts_3 = [np.array(list(reversed(verts))) for verts in [verts_2, verts_3]]
    goal_boundaries = dict()
    goal_boundaries.update(
                top = lambda g: ut.interpolated_univariate_spline(verts_1, vec_1, g[0]),
                bottom = lambda g: ut.interpolated_univariate_spline(verts_3, vec_3, g[0]),
                left = lambda g: ut.interpolated_univariate_spline(verts_0, vec_0, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(verts_2, vec_2, g[1]),
            )
    corners = {(0,0): (vec_0[0,0],vec_0[0,1]), (1,0): (vec_3[-1,0],vec_3[-1,1]), (0,1): (vec_1[0,0],vec_1[0,1]), (1,1): (vec_1[-1,0],vec_1[-1,1])}
    return goal_boundaries, corners
    
    
