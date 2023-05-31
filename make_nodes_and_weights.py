import colorcet as cc
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import meshpy.triangle
import scipy.io

from modepy.quadrature.vioreanu_rokhlin \
    import VioreanuRokhlinSimplexQuadrature as VRSQ

hmax = 0.1
quad_order = 2

A = 0.3
N = 5
x = lambda t: (1 + A*np.cos(N*t))*np.cos(t)
y = lambda t: (1 + A*np.cos(N*t))*np.sin(t)

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
    x0, x1, x2 = points[face]
    dx1, dx2 = x1 - x0, x2 - x0
    jac = abs(dx1[0]*dx2[1] - dx1[1]*dx2[0])
    for s1, s2 in ref_nodes:
        nodes.append(x0 + s1*dx1 + s2*dx2)
    for w in ref_weights:
        weights.append(w*jac)
nodes = np.array(nodes)
weights = np.array(weights)

Tplot = np.linspace(0, 2*np.pi, 201)

smin = 3
smax = 6
s = (smax - smin)*(weights - weights.min())/weights.ptp() + smin

plt.figure(figsize=(12, 10))
plt.triplot(*points.T, faces, linewidth=1, c='k', zorder=0)
plt.plot(x(Tplot), y(Tplot), c='k', linewidth=1)
# plt.scatter(x(T), y(T), s=10, c='k')
plt.scatter(*nodes.T, s=s, c=weights, cmap=cc.cm.gouldian)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

scipy.io.savemat('nodes_and_weights.mat', {'nodes': nodes, 'weights': weights})
