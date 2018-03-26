import numpy as np
import scipy as sp
from nutils import *
from .aux import *
from . import ut, rep, sep
import collections, re, os, sys
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

pathdir = os.path.dirname(os.path.realpath(__file__))

__ = (slice(None), ) + (np.newaxis,)

###################################################

## Tidy up ! Use as much from utilities as possible


def theta(x_, center = np.array([0,0])):
    assert x_.shape[0] == 2
    x = x_.copy() - center[__]
    try:
        vec = np.array([0 if i[0] >= 0 else np.pi for i in x.T])
        return np.arctan(np.divide(*[x[i,:] for i in range(1,-1,-1)])) + vec
    except:
        return np.arctan(np.divide(*[x[i] for i in range(1,-1,-1)])) + (0 if x[0] >= 0 else np.pi)

def circle_point(r,angle, a = 1, b = 1):
    return r*np.vstack([a*np.cos(angle),b*np.sin(angle)])

def norm(x_, center = np.array([0,0])):
    assert x_.shape[0] == 2
    x = x_.copy() - center[__]
    return np.sqrt((x**2).sum(0))

def thin(x, tolerance = 1e-6):
    return x.T[np.concatenate([((x.T[1:]-x.T[:-1])**2).sum(1) > 1e-4, [True]])].T
    
def cusp(a,b,c):  ## x**2 + y**2 = a**2;   (x-c)**2 + y**2 = b**2
    x = (a**2 - b**2 + c**2)/(2*c)
    y = np.sqrt(a**2 - x**2)
    return np.array([x,y]), np.array([x,-y])

def single_casing(radius,steps= 1000):
    angles = np.linspace(0,2*np.pi,steps)
    return circle_point(radius,angles)

distances = rep.discrete_length

def c0_indices(c, thresh = 0.5):
    assert c.shape[0] == 2
    periodic = False
    x = c.copy()  ## don't mess with x passed by reference
    length = x.shape[1]
    if all(x[:,0] == x[:,-1]):  ## include first and last index in angle computation
        periodic = True
        x = np.concatenate([x[:,-2][__],x,x[:,2][__]], axis = 1)
    x_ = x[:,1:] - x[:,:-1]
    inner = (x_[:,1:]*x_[:,:-1]).sum(0)
    dist = np.sqrt((x_**2).sum(0))
    dist = dist[1:]*dist[:-1]
    ret = np.arccos(inner/dist)
    indices = [i if periodic else i+1 for i in range(len(ret)) if ret[i] > thresh]
    return indices

def snail(angle = 0):
    left = []
    with open(pathdir +  '/xml/screwLeft.txt') as fl:
        for l in fl:
            row = l.split()
            left.append(np.array([float(row[i]) for i in range(2)])[__])
    left = np.roll(np.concatenate(left, axis = 1),250, axis = 1)
    left = np.concatenate([left, left.T[0].T[__]],axis = 1)

    thet = theta(left, center = np.array([-13.1,0]))
    casing_left = circle_point(16,thet)

    right = []
    with open(pathdir +  '/xml/screwRight.txt') as fl:
        for l in fl:
            row = l.split()
            right.append(np.array([float(row[i]) for i in range(2)])[__])
    right = np.roll(np.concatenate(right, axis = 1),250, axis = 1)
    right = np.concatenate([right, right.T[0].T[__]],axis = 1)

    thet = theta(right, center = np.array([13.1,0]))
    casing_right = circle_point(16,thet)
    
    return left, casing_left, right, casing_right


class ndspline:
    
    @classmethod
    def from_splines(cls, splines):
        dummy = cls(np.array([1,2,3,4,5,6]), [np.array([1,2,3,4,5,6])])
        dummy._splines = splines
        return dummy
    
    def __init__(self, verts, points, k = 1):
        self._splines = tuple([sp.interpolate.InterpolatedUnivariateSpline(verts,i, k=k) for i in points])
    
    def __call__(self, x):
        return np.vstack([spl(x) for spl in self._splines])
    
    @property
    def derivative(self):
        return self._derivative()
    
    def _derivative(self):
        return ndspline.from_splines(tuple([spl.derivative() for spl in self._splines]))
    
    def normal(self, x):
        if isinstance(x, (int, float)):
            x = [x]
        deriv = self.derivative
        normal = np.array([np.linalg.norm(deriv(i))**(-1) for i in x])[None,:]
        return normal*np.array([[0, -1],[1,0]]).dot(deriv(x))


def rotate(x_,angle_, center = np.array([0,0])):
    assert x_.shape[0] == 2
    if len(center.shape) == 1:
        center = center[__]
    x = x_.copy()
    x -= center
    mat = np.array([[np.cos(angle_), -np.sin(angle_)], [np.sin(angle_), np.cos(angle_)]])
    x = mat.dot(x) + center
    return x


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

def twin_screw_(angle = 0):  ## angle corresponds to male
    xml = ET.parse(pathdir +  '/xml/SRM4_6_gap0.1mm.xml').getroot()
    male, female, casing = [xml[i].text.split() for i in [0,1,2]]
    male, female, casing = [np.asarray([float(i) for i in entry]).T for entry in [male, female, casing]]
    male, female, casing = [np.reshape(entry,[2, len(entry)//2]) for entry in [male, female, casing]]
    #female = np.roll(female,female.shape[1]//2, axis = 1)
    female = np.hstack([female, female[:,0][__]]).T[::-1].T - np.array([56.52,0])[__]
    male = np.roll(male,male.shape[1]//2, axis = 1)
    male = np.hstack([male, male[:,0][__]])
    return ut.rotation_matrix(angle).dot(male), ut.rotation_matrix(-2/3.0*angle).dot(female) + np.array([56.52,0])[__]

def twin_screw(angle = 0, amount = 4000): ## uniform point-distribution
    male, female = twin_screw_(angle = angle)
    male, female = [ndspline(rep.discrete_length_param(points), points) for points in [male,female]]
    male, female = [item(np.linspace(0,1,amount)) for item in [male,female]]
    return male, female


def wedge(go):
    corners = {(0,0): (0,0), (1,0): (1,0), (0,1): (0,1), (1,1): (1,2)}
    goal_boundaries = dict(left= lambda g: function.stack([0,g[1]]), right= lambda g: function.stack([1,2*g[1]]), top=lambda g: function.stack([g[0],1+g[0]**2]), bottom=lambda g: function.stack([g[0],0]))
    return go, goal_boundaries, corners

def V(go):
    assert len(go.periodic) == 0
    corners = {(0,0): (0,-1), (1,0): (1,1), (0,1): (-1,1), (1,1): (0,0)}
    goal_boundaries = dict(left = lambda g: (1-g[1])*np.array(corners[(0,0)]) + g[1]*np.array(corners[(0,1)]))
    goal_boundaries.update(right = lambda g: (1-g[1])*np.array(corners[(1,0)]) + g[1]*np.array(corners[(1,1)]))
    goal_boundaries.update(bottom = lambda g: (1-g[0])*np.array(corners[(0,0)]) + g[0]*np.array(corners[(1,0)]))
    goal_boundaries.update(top = lambda g: (1-g[0])*np.array(corners[(0,1)]) + g[0]*np.array(corners[(1,1)]))
    return go, goal_boundaries, corners

def single_left_snail(go, c0 = True):
    assert go.periodic == [1]
    left, casing_left = snail()[:2]
    verts = rep.discrete_length_param(left)
    if c0:
        indices = c0_indices(left, thresh = 0.3)
        go = go.add_knots([[],verts[indices]])
        go = go.raise_multiplicities([0,go.degree[1]-1], knotvalues = [[],verts[indices]])
    goal_boundaries = dict()
    casing_spline = lambda g: ut.interpolated_univariate_spline(verts, casing_left, g[1])
    goal_boundaries.update(
                right = lambda g: 16*casing_spline(g)/function.sqrt((casing_spline(g)**2).sum(0)) - np.array([13.1,0]),
                left = lambda g: (ut.interpolated_univariate_spline(verts, left, g[1])),
            )
    return go, goal_boundaries, None
    


def single_male_casing(go, radius = 36.1, c0 = True, reparam = True):
    assert go.periodic == [1]
    male = twin_screw()[0]
    angle = theta(male[:,0][__])
    absc = np.linspace(angle,angle+2*np.pi, male.shape[1])
    casing = circle_point(radius, np.linspace(-np.pi,np.pi, 4000))
    if reparam:
        verts_casing, verts_male = rep.constrained_arc_length(casing, male)
    else:
        verts_male, verts_casing = [rep.discrete_length_param(item) for item in [male, casing]]
    if c0:
        indices = c0_indices(male)
        print(verts_male[indices], go.knots)
        go = go.add_knots([[],verts_male[indices]])
        go = go.raise_multiplicities([0,go.degree[1]-1], knotvalues = [[],verts_male[indices]])
    goal_boundaries = dict()
    casing_spline = lambda g: ut.interpolated_univariate_spline(verts_casing, casing, g[1])
    goal_boundaries.update(
                right = lambda g: radius*casing_spline(g)/function.sqrt((casing_spline(g)**2).sum(0)),
                left = lambda g: ut.interpolated_univariate_spline(verts_male, male, g[1])
            )
    return go, goal_boundaries, None


def single_female_casing(go, radius = 36, reparam = True, repfac = 1.5):
    assert go.periodic == [1]
    center = np.array([56.52,0])[__]
    female = twin_screw()[1] - center
    goal_boundaries = dict()
    if reparam:
        theta0 = theta(female.T[0][__])
        casing = circle_point(radius,np.linspace(theta0, theta0 -2*np.pi,4000))
        verts_casing, verts_female = rep.constrained_arc_length(casing, female, smooth = True, fac = repfac)
        female = female + center
        goal_boundaries.update(
                left = lambda g: ut.interpolated_univariate_spline(verts_casing, casing+center, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(verts_female, female, g[1]),
            )
    else:       
        verts = rep.discrete_length_param(female)
        spl = ndspline(verts, female)
        thresh = np.max(norm(female))
        indices = [i for i in range(female.shape[1]) if norm(female[:,i][__]) > 0.97*thresh]
        casing = circle_point(radius,theta(female.T[indices].T))
        casing_spline = lambda g: ut.interpolated_univariate_spline(verts[indices], casing, g[1])
        female = twin_screw()[1]
        goal_boundaries.update(
                left = lambda g: radius*casing_spline(g)/function.sqrt((casing_spline(g)**2).sum(0)) + np.array([56.52,0]),
                right = lambda g: ut.interpolated_univariate_spline(verts, female, g[1]),
            )
    return go, goal_boundaries, None


def separator(go, angle_, reparam = True):
    bottom, top, left, right = sep.separator(angle_)
    #c_1, c_0 = pl.cusp(36.1,36, 56.52)
    if reparam:
        left_verts, right_verts = rep.constrained_arc_length(left, right, 6, smooth = True)
    else:
        left_verts, right_verts = [rep.discrete_length_param(j) for j in [left,right]]
    bottom_verts, top_verts = [rep.discrete_length_param(j) for j in [bottom,top]]
    goal_boundaries = dict(left = lambda g: ut.interpolated_univariate_spline(left_verts, left, g[1]))
    goal_boundaries.update(right = lambda g: ut.interpolated_univariate_spline(right_verts, right, g[1]))
    goal_boundaries.update(bottom = lambda g: ut.interpolated_univariate_spline(bottom_verts, bottom, g[0]))
    goal_boundaries.update(top = lambda g: ut.interpolated_univariate_spline(top_verts, top, g[0]))
                
    corners = {(0,0): bottom.T[0], (1,0): bottom.T[-1], (0,1): top.T[0], (1,1): top.T[-1]}
    return go, goal_boundaries, corners

def separator_pointset(go, angle_, reparam = True, npoints = 1001, return_verts = False):
    bottom, top, left, right = sep.separator(angle_, npoints = npoints)
    #c_1, c_0 = pl.cusp(36.1,36, 56.52)
    if reparam:
        left_verts, right_verts = rep.constrained_arc_length(left, right, 6, smooth = True)
    else:
        left_verts, right_verts = [rep.discrete_length_param(j) for j in [left,right]]
    bottom_verts, top_verts = [rep.discrete_length_param(j) for j in [bottom,top]]
    goal_boundaries = dict(left = lambda g: Pointset(left_verts, left))
    goal_boundaries.update(right = lambda g: Pointset(right_verts, right))
    goal_boundaries.update(bottom = lambda g: Pointset(bottom_verts, bottom))
    goal_boundaries.update(top = lambda g: Pointset(top_verts, top))
                
    corners = {(0,0): bottom.T[0], (1,0): bottom.T[-1], (0,1): top.T[0], (1,1): top.T[-1]}
    if not return_verts:
        return go, goal_boundaries, corners
    return go, goal_boundaries, corners, [left_verts, right_verts]
        



def isoline(go, radius_male = 36.1, radius_female = 36):
    center = np.array([56.52,0])
    x1, x2 = cusp(radius_male, radius_female, 56.52)
    male_angles, female_angles = [theta(i[__]) for i in [x1,x2]], [theta(i[__], center = center) for i in [x1,x2]]
    male, female = twin_screw(angle = np.pi/4.0)
    left_casing, right_casing = [single_casing(r) for r in [radius_male, radius_female]]
    male_indices = [i for i in range(male.shape[1]) if (theta(male[:,i][__]) >= male_angles[1] and theta(male[:,i][__]) <= male_angles[0]) ]
    female_indices = [i for i in range(female.shape[1]) if (theta(female[:,i][__], center = center) >= female_angles[0] and theta(female[:,i][__], center = center) <= female_angles[1] and norm(female[:,i][__]) <= radius_male)]
    left_casing_indices = [i for i in range(left_casing.shape[1]) if (theta(left_casing[:,i][__]) >= male_angles[1] and theta(left_casing[:,i][__]) <= male_angles[0])]
    right = np.hstack((left_casing.T[left_casing_indices][range(107,180)].T,female.T[female_indices].T, left_casing.T[left_casing_indices][range(30,107)].T))
    left = male.T[male_indices].T                                
    corners = {(0,0):left.T[0], (0,1): left.T[-1], (1,0): right.T[0], (1,1): right.T[-1]}
    #left_verts, right_verts = rep.discrete_reparam_joost(left, right)
    right_verts, left_verts = rep.constrained_arc_length(right, left, 5)
    goal_boundaries = dict(
            bottom = lambda g: corners[0,0]*(1-g[0]) + corners[1,0]*g[0],
            top = lambda g: corners[0,1]*(1-g[0]) + corners[1,1]*g[0],
        )
    goal_boundaries.update(
                left = lambda g: ut.interpolated_univariate_spline(left_verts, left, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(right_verts, right, g[1]),
            )
    return goal_boundaries, corners

#def CUSP_problem(go, radius_male = 36.1, radius_female = 36):
    
    

def middle(go, reparam = 'constrained_arc_length', splinify = True, c0 = True): #Return either a spline or point-cloud representation of the screw-machine with casing problem
    pathdir = os.path.dirname(os.path.realpath(__file__))
    #ref_geom = go.geom
    
    top = numpy.stack([numpy.loadtxt(pathdir+'/xml/t2_row_{}'.format(i)) for i in range(1, 3)]).T
    top = top[numpy.concatenate([((top[1:]-top[:-1])**2).sum(1) > 1e-4, [True]])]
    top = top[65:230][::-1].T

    bot = np.stack([np.loadtxt(pathdir+'/xml/t1_row_{}'.format(i)) for i in range(1, 3)]).T
    bot = bot[numpy.concatenate([((bot[1:]-bot[:-1])**2).sum(1) > 1.5*1e-4, [True]])]
    bot = bot[430:1390][::-1].T
    #bot = np.delete(bot.T, 568, 0).T
    
    indices = c0_indices(bot)
    
    if True: # fix
        if reparam == 'standard' or reparam == 'length':
            top_verts, bot_verts = [rep.discrete_length_param(item) for item in [top, bot]] ## for now only upper and lower
        elif reparam == 'Joost':
            top_verts, bot_verts = rep.discrete_reparam_joost(top, bot)
        elif reparam == 'constrained_arc_length':
            top_verts, bot_verts = rep.constrained_arc_length(top, bot, 2)
        else:
            raise ValueError(reparam +' not supported')
            
        if c0:
            #go = go.add_c0([go.knots[1],[]]).to_c([1,1])
            go = go.add_c0([bot_verts[indices],[]])
        
        corners = {(0,0): bot.T[0], (1,0): bot.T[-1], (0,1): top.T[0], (1,1): top.T[-1]}
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
                top = lambda x: Pointset(top_verts, top),
                bottom = lambda x: Pointset(bot_verts, bot),
            )
            
    return go, goal_boundaries, corners


def sines(go):
    corners = {(0,0): (0,0), (1,0): (2,0), (0,1): (0,1), (1,1): (2,1)}
    _g = lambda g: function.stack([g[0]*2, (1+0.2*function.sin(g[0]*numpy.pi*3))*g[1]+(-0.2*function.sin(g[0]*numpy.pi*3))*(1-g[1])])
    goal_boundaries = dict(left=_g, right=_g, top=_g, bottom=_g)
    return go, goal_boundaries, corners


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
    vec_0, vec_1, vec_2, vec_3 = vec_[:,0:stepsize], vec_[:,stepsize:2*stepsize - 3], vec_[:,2*stepsize-3:3*stepsize-5].T[::-1,:].T, vec_[:,3*stepsize-5:].T[::-1,:].T
    verts_0, verts_1, verts_2, verts_3 = [rep.discrete_length_param(item) for item in [vec_0, vec_1, vec_2, vec_3]]
    print(vec_1.shape)
    goal_boundaries = dict()
    goal_boundaries.update(
                top = lambda g: ut.interpolated_univariate_spline(verts_1, vec_1, g[0]),
                bottom = lambda g: ut.interpolated_univariate_spline(verts_3, vec_3, g[0]),
                left = lambda g: ut.interpolated_univariate_spline(verts_0, vec_0, g[1]),
                right = lambda g: ut.interpolated_univariate_spline(verts_2, vec_2, g[1]),
            )
    corners = {(0,0): (vec_0[0,0],vec_0[1,0]), (1,0): (vec_3[0,-1],vec_3[1,-1]), (0,1): (vec_1[0,0],vec_1[1,0]), (1,1): (vec_1[0,-1],vec_1[1,-1])}
    return goal_boundaries, corners
    
    
