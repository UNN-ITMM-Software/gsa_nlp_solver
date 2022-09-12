import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from os.path import isfile, join

matplotlib.style.use('classic')

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:orange')
linestyles = ['--', '-', '--', ':', '-.', '--', ':', '-', '--']

def plot_cmcs(curves, captures=None, show=True, filename=None, log=False):
    if captures != None:
        assert len(captures) == len(curves)

    plt.xlabel(r'$K$', fontsize=15)
    plt.ylabel(r'$P$', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    min_iters = sys.maxsize
    max_iters = 0
    for curve in curves:
        min_iters = min(curve[0][-1], min_iters)
        max_iters = max(curve[0][-1], max_iters)

    if not log:
        plt_fun = plt.plot
        plt.xlim([0., min(max_iters, min_iters*2)])
    else:
        plt_fun = plt.semilogx
        start = 10
        if min_iters > 10**3:
            start = 10**2
        elif min_iters > 10**4:
            start = 2*10**2
        plt.xlim([start, max_iters + 10**math.log(max_iters, 10)/4.])

    for i, curve in enumerate(curves):
        plt_fun(curve[0], curve[1], color = colors[i], \
            #linestyle = linestyles[i],
             label = captures[i], markersize=3, linewidth=2)

    plt.grid()
    plt.legend(loc = 'best', fontsize = 14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    if show:
        plt.show()

def plot_countours(functions, bounds, filename, show=False, points=None):
    lb, ub = bounds
    assert len(lb) == len(ub) == 2
    assert len(functions) > 0

    n_cells = 300
    x = np.linspace(lb[0], ub[0], n_cells, endpoint=True)
    y = np.linspace(lb[1], ub[1], n_cells, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z_obj = np.zeros((n_cells, n_cells), dtype=np.float)
    for i in range(len(X)):
        for j in range(len(Y)):
            Z_obj[i, j] = functions[-1]([X[i, j], Y[i, j]])

    plt.figure()
    if points:
        points = np.array(points)
        plt.scatter(points[:,0], points[:,1], c='r')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    CS = plt.contour(X, Y, Z_obj, colors='k', extent=[lb[0],ub[0],lb[1],ub[1]])

    n_constraints = len(functions) - 1
    if n_constraints > 0:
        Z_c = np.zeros((n_cells, n_cells), dtype=np.float)
        for i in range(len(X)):
            for j in range(len(Y)):
                val = -np.inf
                for f in functions[:-1]:
                    val = f([X[i, j], Y[i, j]])
                    if val > 0:
                        break
                Z_c[i, j] = val
        CS_c = plt.contourf(X, Y, Z_c, [-np.inf, 0], colors=['green'], \
                            alpha=.4, extent=[lb[0],ub[0],lb[1],ub[1]])

        for f in functions[:-1]:
            for i in range(len(X)):
                for j in range(len(Y)):
                    Z_c[i, j] = f([X[i, j], Y[i, j]])
            plt.contour(X, Y, Z_c, [0], colors='k', extent=[lb[0],ub[0],lb[1],ub[1]])

    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$x$', fontsize=15)
    plt.ylabel(r'$y$', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim([lb[0], ub[0]])
    plt.ylim([lb[1], ub[1]])
    plt.grid()

    if len(filename) > 0:
        plt.savefig(filename, dpi=300)

    if show:
        plt.show()
