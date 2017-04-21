#! /usr/bin/env python3

import collections
import matplotlib
import scipy.interpolate
from nutils import *
import reparam as rep


def log_iter_sorted_dict_items(title, d):
    for k, v in sorted(d.items()):
        with log.context(title + ' ' + k):
            yield k, v


Pointset = collections.namedtuple('Pointset', ['verts', 'geom'])


class interpolated_univariate_spline(function.Array):

    def __init__(self, vertices, values, position):
        assert function.isarray(position)
        assert values.shape[:1] == vertices.shape
        function.Array.__init__(self, args=[position], shape=position.shape+values.shape[1:], dtype=function._jointdtype(vertices.dtype, float))
        self._values_shape = values.shape[1:]
        self._splines = tuple(scipy.interpolate.InterpolatedUnivariateSpline(vertices, v) for v in values.reshape(values.shape[0], -1).T)

    def evalf(self, position):
        assert position.ndim == self.ndim
        shape = position.shape + self._values_shape
        position = position.ravel()
        return numpy.stack([spline(position) for spline in self._splines], axis=1).reshape(shape)


def main(
    nelems:'number of elements'=4,
    degree:'degree of basis functions'=3,
    btol:'tolerance for boundary refinement'=1e-2,
    ltol:'tolerance for solving laplace'=1e-7,
    plot_2d:'make 2D plots'=True,
    pathdir:'directory with top and bottom paths'=cli.Path('xml'),
    goal:'"curves" or "sines"'='curves',
    splinify:'convert a pointset to splines before processing'=False,
):
    nelems = nelems, nelems // 2
    domain, ref_geom = mesh.rectilinear([numpy.linspace(0, 1, n+1) for n in nelems])

    gauss = 'gauss{:d}'.format

    if goal == 'curves':

        top = numpy.stack([numpy.loadtxt(str(pathdir)+'/t2_row_{}'.format(i)) for i in range(1, 3)]).T
        top = top[numpy.concatenate([((top[1:]-top[:-1])**2).sum(1) > 1e-8, [True]])]
        top = top[65:230][::-1]
        top_distances = ((top[1:] - top[:-1])**2).sum(1)**0.5
        top_verts = numpy.concatenate([[0], top_distances]).cumsum()
        top_verts /= top_verts[-1]

        bot = numpy.stack([numpy.loadtxt(str(pathdir)+'/t1_row_{}'.format(i)) for i in range(1, 3)]).T
        bot = bot[numpy.concatenate([((bot[1:]-bot[:-1])**2).sum(1) > 1e-8, [True]])]
        bot = bot[430:1390][::-1]
        bot_distances = ((bot[1:] - bot[:-1])**2).sum(1)**0.5
        if True: # fix
            # Find `i` and `j` such that the distance between `top[i]` and `bottom[j]` is minized.
            i, j = numpy.unravel_index(numpy.argmin(((top[:,None,:]-bot[None,:,:])**2).sum(2)), [len(top), len(bot)])
            # Scale `bot_distances` such that `bot_verts[j]` is close to `top_verts[i]`.
            bot_distances[:j] *= top_distances[:i].sum() / bot_distances[:j].sum()
            bot_distances[j:] *= top_distances[i:].sum() / bot_distances[j:].sum()
        bot_verts = numpy.concatenate([[0], bot_distances]).cumsum()
        bot_verts /= bot_verts[-1]

        corners = {(0,0): bot[0], (1,0): bot[-1], (0,1): top[0], (1,1): top[-1]}
        goal_boundaries = dict(
            left=corners[0,0]*(1-ref_geom[1]) + corners[0,1]*ref_geom[1],
            right=corners[1,0]*(1-ref_geom[1]) + corners[1,1]*ref_geom[1],
        )
        if splinify:
            goal_boundaries.update(
                top=interpolated_univariate_spline(top_verts, top, ref_geom[0]),
                bottom=interpolated_univariate_spline(bot_verts, bot, ref_geom[0]),
            )
        else:
            goal_boundaries.update(
                top=Pointset(top_verts[1:-1], top[1:-1]),
                bottom=Pointset(bot_verts[1:-1], bot[1:-1]),
            )

    elif goal == 'sines':

        corners = {(0,0): (0,0), (1,0): (2,0), (0,1): (0,1), (1,1): (2,1)}
        _g = ref_geom
        _g = function.stack([_g[0]*2, (1+0.5*function.sin(_g[0]*numpy.pi*3))*_g[1]+(-0.6*function.sin(_g[0]*numpy.pi*4))*(1-_g[1])])
        goal_boundaries = dict(left=_g, right=_g, top=_g, bottom=_g)

    else:

        raise ValueError('unknown goal: {}'.format(goal))

    geom = None

    for irefine in log.count('refine'):

        if plot_2d:
            points = domain.elem_eval(ref_geom, ischeme='bezier5', separate=True)
            with plot.PyPlot('topo') as plt:
                plt.mesh(points)

        gbasis = domain.basis('spline', degree=degree).vector(2)

        # Constrain the corners.
        cons = None
        for (i, j), v in log.iter('corners', corners.items()):
            domain_ = (domain.levels[-1] if isinstance(domain, topology.HierarchicalTopology) else domain).boundary[{0: 'bottom', 1: 'top'}[j]].boundary[{0: 'left', 1: 'right'}[i]]
            cons = domain_.project(v, onto=gbasis, constrain=cons, geometry=ref_geom, ischeme='vertex')

        # Project all boundaries onto `gbasis` and collect all elements where
        # the projection error is larger than `btol` in `refine_elems`.
        refine_elems = set()
        if plot_2d:
            plot_points, plot_error = [], []
        for side, goal in log_iter_sorted_dict_items('boundary', goal_boundaries):
            if isinstance(goal, Pointset):
                dim = {'top': 0, 'bottom': 0, 'left': 1, 'right': 1}[side]
                domain_ = domain.boundary[side].locate(ref_geom[dim], goal.verts)
                print(type(goal.verts), 'verts')
                ischeme_ = 'vertex'
                goal = function.elemwise(
                    dict(zip((elem.transform for elem in domain_), goal.geom)),
                    [domain.ndims])
            else:
                domain_ = domain.boundary[side]
                ischeme_ = gauss(degree*2)
            cons = domain_.project(goal, onto=gbasis, geometry=ref_geom, ischeme=ischeme_, constrain=cons)

            basis = domain.boundary[side].basis('spline', degree=degree)
            error = ((goal - gbasis.dot(cons | 0))**2).sum(0)**0.5
            error = domain_.project(error, onto=basis, geometry=ref_geom, ischeme=gauss(degree*2))
            refine_elems.update(
                elem.transform.promote(domain.ndims)[0]
                for elem in domain_.supp(basis, numpy.where(error > btol)[0]))

            if plot_2d:
                p, e = domain.boundary[side].elem_eval([gbasis.dot(cons | 0), basis.dot(error)], ischeme='bezier5', separate=True)
                plot_points.extend(p)
                plot_error.extend(e)

        if plot_2d:
            with plot.PyPlot('boundary') as plt:
                plt.segments(numpy.array(plot_points), numpy.array(plot_error), norm=matplotlib.colors.LogNorm(vmin=btol*1e-3, vmax=1))
                plt.aspect('equal')
                plt.autoscale(enable=True, axis='both', tight=True)
                plt.colorbar()

        # Solve inner geometry.
        with log.context('inner'):

            target = function.DerivativeTarget([len(gbasis)])
            with log.context('initial state'):
                # Generate an initial guess for the solver of the laplace equation
                # below.
                if geom is None:
                    # Compute an initial guess by solving a laplace equation in
                    # second dimension only in reference coordinates.
                    res = model.Integral(gbasis['ik,1']*gbasis.dot(target)['k,1'], domain=domain, geometry=ref_geom, degree=degree*3)
                    lhs = model.newton(target, res, lhs0=cons | 0, freezedofs=cons.where).solve(ltol)
                else:
                    # Project the geometry found in the previous iteration onto the
                    # new `gbasis`.
                    lhs = domain.project(geom, onto=gbasis, geometry=ref_geom, ischeme=gauss(degree*2), constrain=cons)
                if plot_2d:
                    points = domain.elem_eval(gbasis.dot(lhs), ischeme='bezier5', separate=True)
                    with plot.PyPlot('initial_geom') as plt:
                        plt.mesh(points)

            with log.context('laplace'):
                res = model.Integral(gbasis['ik,l']*ref_geom['k,l'], domain=domain, geometry=gbasis.dot(target), degree=degree*3)
                lhs = model.newton(target, res, lhs0=lhs, freezedofs=cons.where).solve(ltol)

            geom = gbasis.dot(lhs)

            if plot_2d:
                points = domain.elem_eval(geom, ischeme='bezier5', separate=True)
                with plot.PyPlot('geom') as plt:
                    plt.mesh(points)

        if len(refine_elems) == 0:
            break
        elif len(refine_elems) == 1:
            log.info('refining 1 element')
        else:
            log.info('refining {} elements'.format(len(refine_elems)))
        domain = domain.refined_by(refine_elems)


if __name__ == '__main__':
    cli.run(main)
