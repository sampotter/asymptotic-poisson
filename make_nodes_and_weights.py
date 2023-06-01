import colorcet as cc
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import meshpy.triangle
import scipy.io
import time

from modepy.quadrature.vioreanu_rokhlin \
    import VioreanuRokhlinSimplexQuadrature as VRSQ

hmax = 0.025
quad_order = 2
use_quadratic_triangles_on_boundary = True

A = 0
N = 0
x = lambda t: (1 + A*np.cos(N*t))*np.cos(t)
y = lambda t: (1 + A*np.cos(N*t))*np.sin(t)

t0 = time.time()

dl = hmax
while True:
    T = np.linspace(0, 2*np.pi, int(np.ceil(2*np.pi/dl)))
    X, Y = x(T), y(T)
    dX, dY = np.diff(X), np.diff(Y)
    dL = np.sqrt(dX**2 + dY**2)
    if dL.max() <= hmax:
        break
    dl *= dl/dL.mean()

points = np.array([X[:-1], Y[:-1]]).T

n = points.shape[0]

facets = np.empty((n, 2), dtype=int)
facets[:, 0] = np.arange(n)
facets[:-1, 1] = np.arange(n - 1) + 1
facets[-1, 1] = 0

info = meshpy.triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets)

def should_refine(verts, area):
    x0, x1, x2 = np.array(verts)
    d01 = np.linalg.norm(x1 - x0)
    d12 = np.linalg.norm(x2 - x1)
    d20 = np.linalg.norm(x0 - x2)
    diam = max(d01, d12, d20)
    return diam > hmax

mesh = meshpy.triangle.build(info, refinement_func=should_refine)

points = np.array(mesh.points)
faces = np.array(mesh.elements)

quad = VRSQ(quad_order, points.shape[1])
ref_nodes = (quad.nodes.T + 1)/2
ref_weights = quad.weights/4

nodes, weights = [], []
for face in faces:
    num_bd = (face < n).sum()
    assert num_bd <= 2
    if not use_quadratic_triangles_on_boundary or num_bd < 2:
        x0, x1, x2 = points[face]
        dx1, dx2 = x1 - x0, x2 - x0
        jac = abs(dx1[0]*dx2[1] - dx1[1]*dx2[0])
        for s1, s2 in ref_nodes:
            nodes.append(x0 + s1*dx1 + s2*dx2)
        for w in ref_weights:
            weights.append(w*jac)
    else:
        i0 = face[face >= n][0] # find the lone interior vertex
        i1, i2 = np.setdiff1d(face, i0) # the others are the boundary verts
        x0, x1, x2 = points[i0], points[i1], points[i2]
        dx1, dx2 = x1 - x0, x2 - x0
        if abs(i2 - i1) == 1:
            tm = (T[i1] + T[i2])/2
        else:
            assert (i1 == 0 and i2 == n - 1) or (i1 == n - 1 and i2 == 0)
            tm = (T[n - 1] + T[n])/2
        xm = np.array([x(tm), y(tm)])
        dxm = xm - (x1 + x2)/2

        # NOTE: this snippet plots the current quadratic
        # triangle---maybe useful for debugging...

        # plt.figure()
        # for _ in [x0, x1, x2, (x0+x1)/2, (x0+x2)/2]:
        #     print(_)
        #     plt.scatter(*_, s=15, c='k', zorder=1)
        # plt.scatter(*xm, s=20, facecolors='r', edgecolors='k', zorder=1)
        # for s2 in np.linspace(0, 1, 11):
        #     s1 = np.linspace(0, 1 - s2)
        #     s2 = s2*np.ones_like(s1)
        #     X = x0 + np.outer(s1, dx1) + np.outer(s2, dx2) + np.outer(4*s1*s2, xm - (x1 + x2)/2)
        #     plt.plot(*X.T, linewidth=1, c='k', zorder=0)
        # for s1 in np.linspace(0, 1, 11):
        #     s2 = np.linspace(0, 1 - s1)
        #     s1 = s1*np.ones_like(s2)
        #     X = x0 + np.outer(s1, dx1) + np.outer(s2, dx2) + np.outer(4*s1*s2, xm - (x1 + x2)/2)
        #     plt.plot(*X.T, linewidth=1, c='k', zorder=0)
        # for s0 in np.linspace(0, 1, 11):
        #     s1 = np.linspace(0, s0)
        #     s2 = np.linspace(s0, 0)
        #     X = x0 + np.outer(s1, dx1) + np.outer(s2, dx2) + np.outer(4*s1*s2, xm - (x1 + x2)/2)
        #     plt.plot(*X.T, linewidth=1, c='k', zorder=0)
        # plt.show()

        for s1, s2 in ref_nodes:
            nodes.append(x0 + s1*dx1 + s2*dx2 + 4*s1*s2*dxm)
        for w in ref_weights:
            a11 = dx1[0] + 4*s2*dxm[0]
            a12 = dx2[0] + 4*s1*dxm[0]
            a21 = dx1[1] + 4*s2*dxm[1]
            a22 = dx2[1] + 4*s1*dxm[1]
            jac = abs(a11*a22 - a21*a12)
            weights.append(w*jac)

nodes = np.array(nodes)
weights = np.array(weights)

t1 = time.time() # finished!

print(f'built {nodes.size} node quadrature in {t1 - t0:.1f} s ({nodes.size/(t1 - t0):.1f} pps)')

if A == 0 and N == 0:
    disk_area_rel_error = abs(np.pi - weights.sum())/np.pi
    print(f'A == 0 && N == 0: rel error for area of disk is {disk_area_rel_error:.1E}')

Tplot = np.linspace(0, 2*np.pi, 201)

smin = 3
smax = 6
s = (smax - smin)*(weights - weights.min())/weights.ptp() + smin

plt.figure(figsize=(12, 10))
# plt.triplot(*points.T, faces, linewidth=1, c='k', zorder=0)
plt.plot(x(Tplot), y(Tplot), c='k', linewidth=1)
# plt.scatter(x(T), y(T), s=10, c='k')
plt.scatter(*nodes.T, s=s, c=weights, cmap=cc.cm.rainbow)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

scipy.io.savemat('nodes_and_weights.mat', {'nodes': nodes, 'weights': weights})
