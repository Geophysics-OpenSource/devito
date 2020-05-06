import numpy as np
from devito import (Grid, Constant, Function, SpaceDimension, configuration)
from examples.seismic.skew_self_adjoint import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline
mpl.rc('font', size=14)
plt.rcParams['figure.facecolor'] = 'white'
configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

n = 601
shape = (n, n)
dtype = np.float32
npad = 50
qmin = 0.1
qmax = 100.0
fpeak = 0.010
w = 2.0 * np.pi * fpeak
d = 10.0
origin = tuple([0.0 - d * npad for s in shape])
spacing = (d, d)
extent = tuple([d * (s - 1) for o, s in zip(origin, shape)])
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
grid = Grid(extent=extent, shape=shape, origin=origin, dimensions=(x, z), dtype=dtype)

print("shape;      ", shape)
print("origin;     ", origin)
print("spacing;    ", spacing)
print("extent;     ", extent)

wOverQ_a = Function(name='wOverQ', grid=grid)
wOverQ_b = Function(name='wOverQ', grid=grid)

sigma = 0
setup_wOverQ_numpy(wOverQ_a, w, qmin, qmax, npad, sigma=sigma)
setup_wOverQ(wOverQ_b, w, qmin, qmax, npad, sigma=sigma)

# Plot
qa = w / wOverQ_a.data
qb = w / wOverQ_b.data
qc = qa - qb
print("min,max Q value difference; ", np.min(qc), np.max(qc))

plt_extent = [origin[0], origin[0] + spacing[0]*(shape[0]-1),
              origin[1] + spacing[1]*(shape[1]-1), origin[1]]

b1 = 0.0
b2 = d * (n - 2 * npad)

plt.figure(figsize=(21, 10))

plt.subplot(1, 3, 1)
plt.imshow(np.transpose(qa.data), cmap=cm.jet_r,
           vmin=qmin, vmax=qmax, extent=plt_extent)
plt.colorbar(orientation='horizontal', label='Q(x,z)')
plt.plot([b1, b1, b2, b2, b1],
         [b1, b2, b2, b1, b1],
         'white', linewidth=4, linestyle=':', label="Absorbing Boundary")
plt.xlabel("X Coordinate (m)", labelpad=10)
plt.ylabel("Z Coordinate (m)", labelpad=10)
plt.title("Q for numpy")

plt.subplot(1, 3, 2)
plt.imshow(np.transpose(qb.data), cmap=cm.jet_r,
           vmin=qmin, vmax=qmax, extent=plt_extent)
plt.colorbar(orientation='horizontal', label='Q(x,z)')
plt.plot([b1, b1, b2, b2, b1],
         [b1, b2, b2, b1, b1],
         'white', linewidth=4, linestyle=':', label="Absorbing Boundary")
plt.xlabel("X Coordinate (m)", labelpad=10)
plt.ylabel("Z Coordinate (m)", labelpad=10)
plt.title("Q for devito Eq")

plt.subplot(1, 3, 3)
amin, amax = -1.e-5, +1.e-5
plt.imshow(np.transpose(qc.data), cmap="seismic",
           vmin=amin, vmax=amax, extent=plt_extent)
plt.colorbar(orientation='horizontal', label='Q difference')
plt.plot([b1, b1, b2, b2, b1],
         [b1, b2, b2, b1, b1],
         'black', linewidth=4, linestyle=':', label="Absorbing Boundary")
plt.xlabel("X Coordinate (m)", labelpad=10)
plt.ylabel("Z Coordinate (m)", labelpad=10)
plt.title("Difference in Q values")

plt.tight_layout()
# plt.savefig("qdiff.png")
plt.show()
