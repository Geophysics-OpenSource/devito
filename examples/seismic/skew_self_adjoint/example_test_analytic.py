import numpy as np
from devito import configuration, Grid, Function, SpaceDimension, Constant
from examples.seismic import RickerSource, Receiver, TimeAxis
from examples.seismic.skew_self_adjoint import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline
mpl.rc('font', size=14)
plt.rcParams['figure.facecolor'] = 'white'
configuration['language'] = 'openmp'
# configuration['log-level'] = 'DEBUG'

# Time / frequency
nt = 1001
dt = 0.1
tmin = 0.0
tmax = dt * (nt - 1)
fpeak = 0.090
omega = 2.0 * np.pi * fpeak
time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)
time = np.linspace(tmin, tmax, nt)

# Model
space_order = 8
npad = 100
dx, dz = 1.0, 1.0
nx, nz = 401, 401
shape = (nx, nz)
spacing = (dx, dz)
extent = (dx * (nx - 1), dz * (nz - 1))
origin = (0.0, 0.0)
dtype = np.float64
qmin = 0.1
qmax = 1.0e6
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
grid = Grid(extent=extent, shape=shape, origin=origin,
            dimensions=(x, z), dtype=dtype)
b = Function(name='b', grid=grid, space_order=space_order)
v = Function(name='v', grid=grid, space_order=space_order)
wOverQ = Function(name='wOverQ', grid=v.grid, space_order=v.space_order)
b.data[:] = 1.0 / 1000.0
v.data[:] = 1.5
setup_wOverQ(wOverQ, omega, qmin, qmax, npad)

# Source and reciver coordinates 
src_coords = np.empty((1, 2), dtype=dtype)
rec_coords = np.empty((1, 2), dtype=dtype)
src_coords[:, 0] = origin[0] + extent[0] / 2
src_coords[:, 1] = origin[1] + extent[1] / 2
rec_coords[:, 0] = origin[0] + extent[0] / 2 + 60 / dx
rec_coords[:, 1] = origin[1] + extent[1] / 2 + 60 / dz

print(time_axis)
print("origin;     %12.4f %12.4f" % (origin[0], origin[1]))
print("extent;     %12.4f %12.4f" % (extent[0], extent[1]))
print("spacing;    %12.4f %12.4f" % (spacing[0], spacing[1]))
print("src_coords; %12.4f %12.4f" % (src_coords[0,0], src_coords[0,1]))
print("rec_coords; %12.4f %12.4f" % (rec_coords[0,0], rec_coords[0,1]))

# Solver setup
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src_coords, rec_coords, time_axis,
                                    space_order=space_order)

# Source function 
src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1, 
                   time_range=time_axis)
src.coordinates.data[:] = src_coords[:]

# Modeling
recNum, uNum, _ = solver.forward(src)

# Plot
x1 = origin[0]
x2 = origin[0] + extent[0]
z1 = origin[1]
z2 = origin[1] + extent[1]

xABC1 = origin[0] + dx * npad
xABC2 = origin[0] + extent[0] - dx * npad
zABC1 = origin[1] + dx * npad
zABC2 = origin[1] + extent[1] - dx * npad

plt_extent = [x1, x2, z2, z1]
abc_pairsX = [x1, x1, x2, x2, x1] 
abc_pairsZ = [z1, z2, z2, z1, z1] 

plt.figure(figsize=(12.5,12.5))

amax = np.max(np.abs(uNum.data[1,:,:]))

# Plot wavefield
plt.subplot(2,2,1)
plt.imshow(uNum.data[1,:,:], vmin=-amax, vmax=+amax, cmap="seismic",
           aspect="auto", extent=plt_extent)
plt.plot(src_coords[0, 0], src_coords[0, 1], 'r*', markersize=11, label='Source') 
plt.plot(rec_coords[0, 0], rec_coords[0, 1], 'k^', markersize=11, label='Receiver') 
plt.plot(abc_pairsX, abc_pairsZ, 'black', linewidth=4, linestyle=':', 
         label="ABC")
plt.legend()
plt.xlabel('x position (m)')
plt.ylabel('z position (m)')
plt.title('Wavefield of numerical solution')
plt.tight_layout()
# plt.savefig('wavefieldperf.pdf')

# Plot trace
plt.subplot(2,2,3)
plt.plot(time, recNum.data[:, 0], '-b', label='Numerical solution')
# plt.plot(time, U_t[:], '--r', label='Analytic solution')
# plt.xlim([0,150])
# plt.ylim([1.15*np.min(U_t[:]), 1.15*np.max(U_t[:])])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
# plt.subplot(2,1,2)
# plt.plot(time, 100 *(ref_rec.data[:, 0] - U_t[:]), '-b', label='difference x100')
# plt.xlim([0,150])
# plt.ylim([1.15*np.min(U_t[:]), 1.15*np.max(U_t[:])])
# plt.xlabel('time (ms)')
# plt.ylabel('amplitude x100')
# plt.legend()
plt.show()