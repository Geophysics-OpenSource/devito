import numpy as np
from scipy.special import hankel2
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
configuration['log-level'] = 'DEBUG'

# Time / frequency
nt = 1251
dt = 0.1
tmin = 0.0
tmax = dt * (nt - 1)
fpeak = 0.090
t0w = 1.0 / fpeak
omega = 2.0 * np.pi * fpeak
time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)
time = np.linspace(tmin, tmax, nt)

# Model
space_order = 8
npad = 50
dx, dz = 0.5, 0.5
nx, nz = 801 + 2 * npad, 801 + 2 * npad
shape = (nx, nz)
spacing = (dx, dz)
extent = (dx * (nx - 1), dz * (nz - 1))
origin = (0.0 - dx * npad, 0.0 - dz * npad)
dtype = np.float64
qmin = 0.1
qmax = 100000
v0 = 1.5
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
grid = Grid(extent=extent, shape=shape, origin=origin,
            dimensions=(x, z), dtype=dtype)
b = Function(name='b', grid=grid, space_order=space_order)
v = Function(name='v', grid=grid, space_order=space_order)
b.data[:] = 1.0
v.data[:] = v0

# Source and reciver coordinates 
src_coords = np.empty((1, 2), dtype=dtype)
rec_coords = np.empty((1, 2), dtype=dtype)
src_coords[:, 0] = origin[0] + extent[0] / 2
src_coords[:, 1] = origin[1] + extent[1] / 2
rec_coords[:, 0] = origin[0] + extent[0] / 2 + 60
rec_coords[:, 1] = origin[1] + extent[1] / 2 + 60

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
                   time_range=time_axis, t0w=t0w)
src.coordinates.data[:] = src_coords[:]

# Numerical solution
recNum, uNum, _ = solver.forward(src)

# Analytic response
def analytic_response():
    """
    Computes analytic solution of 2D acoustic wave-equation with Ricker wavelet 
    peak frequency fpeak, temporal padding 20x per the accuracy notebook:
    examples/seismic/acoustic/accuracy.ipynb
        u(r,t) = 1/(2 pi) sum[ -i pi H_0^2(k,r) q(w) e^{i w t} dw 
        where:
            r = sqrt{(x_s - x_r)^2 + (z_s - z_r)^2}
            w = 2 pi f 
            q(w) = Fourier transform of Ricker source wavelet
            H_0^2(k,r) Hankel function of the second kind
            k = w/v (wavenumber) 
    """
    sx, sz = src_coords[0, :]
    rx, rz = rec_coords[0, :]
    print("fpeak; ", fpeak)
    print("sx,sz; ", sx, sz)
    print("rx,rz; ", rx, rz)
    ntpad = 20 * (nt - 1) + 1
    tmaxpad = dt * (ntpad - 1)
    time_axis_pad = TimeAxis(start=tmin, stop=tmaxpad, step=dt)
    timepad = np.linspace(tmin, tmaxpad, ntpad)
    print(time_axis_pad)
    srcpad = RickerSource(name='srcpad', grid=v.grid, f0=fpeak, npoint=1, 
                          time_range=time_axis_pad, t0w=t0w)
    nf = int(ntpad / 2 + 1)
    fnyq = 1.0 / (2 * dt)
    df = 1.0 / tmaxpad
    faxis = df * np.arange(nf)

    # Take the Fourier transform of the source time-function
    R = np.fft.fft(srcpad.wavelet[:])
    R = R[0:nf]
    nf = len(R)

    # Compute the Hankel function and multiply by the source spectrum
    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf - 1):
        w = 2 * np.pi * faxis[a] 
        r = np.sqrt((rx - sx)**2 + (rz - sz)**2)
        U_a[a] = -1j * np.pi * hankel2(0.0,  w * r / v0) * R[a]

    # Do inverse fft on 0:dt:T and you have analytical solution
    U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], ntpad))
    
    # Note that the analytic solution is scaled by dx^2 to convert to pressure
    return (np.real(U_t) * (dx**2)), srcpad

uAnaPad, srcpad = analytic_response()
uAna = uAnaPad[0:nt]

nmin, nmax = np.min(recNum.data), np.max(recNum.data)
amin, amax = np.min(uAna), np.max(uAna)

print("")
print("Numerical min/max; %+12.6e %+12.6e" % (nmin, nmax))
print("Analytic  min/max; %+12.6e %+12.6e" % (amin, amax))

# Plot
x1 = origin[0]
x2 = origin[0] + extent[0]
z1 = origin[1]
z2 = origin[1] + extent[1]

xABC1 = origin[0] + dx * npad
xABC2 = origin[0] + extent[0] - dx * npad
zABC1 = origin[1] + dz * npad
zABC2 = origin[1] + extent[1] - dz * npad

plt_extent = [x1, x2, z2, z1]
abc_pairsX = [xABC1, xABC1, xABC2, xABC2, xABC1] 
abc_pairsZ = [zABC1, zABC2, zABC2, zABC1, zABC1] 

plt.figure(figsize=(12.5,12))

# Plot wavefield
plt.subplot(2,2,1)
amax = 1.1 * np.max(np.abs(recNum.data[:]))
plt.imshow(uNum.data[1,:,:], vmin=-amax, vmax=+amax, cmap="seismic",
           aspect="auto", extent=plt_extent)
plt.plot(src_coords[0, 0], src_coords[0, 1], 'r*', markersize=15, label='Source') 
plt.plot(rec_coords[0, 0], rec_coords[0, 1], 'k^', markersize=11, label='Receiver') 
plt.plot(abc_pairsX, abc_pairsZ, 'black', linewidth=4, linestyle=':', 
         label="ABC")
plt.legend(loc="center", bbox_to_anchor=(0.325, 0.625, 0.35, .1))
plt.xlabel('x position (m)')
plt.ylabel('z position (m)')
plt.title('Wavefield of numerical solution')
plt.tight_layout()

# Plot trace
plt.subplot(2,2,3)
plt.plot(time, recNum.data[:, 0], '-b', label='Numerical solution')
plt.plot(time, uAna[:], '--r', label='Analytic solution')
plt.xlim([50,90])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Trace comparison of solutions')
plt.legend(loc="lower right")
plt.ylim([-amax, +amax])

plt.subplot(2,2,4)
plt.plot(time, 100 * (recNum.data[:, 0] -  uAna[:]), '-k', label='Difference x100')
plt.xlim([50,90])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Difference of solutions (x100)')
plt.legend(loc="lower right")
plt.ylim([-amax, +amax])

plt.tight_layout()
plt.savefig("accuracy.png")
plt.show()