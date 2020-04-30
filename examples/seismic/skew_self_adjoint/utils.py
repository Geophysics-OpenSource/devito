from timeit import default_timer as timer
from sympy import exp, Min
import numpy as np
from devito import (Grid, Constant, Function, SpaceDimension, Eq, Operator)
from examples.seismic import RickerSource, Receiver, TimeAxis
from devito.builtins import gaussian_smooth

__all__ = ['critical_dt', 'setup_wOverQ', 'setup_wOverQ_numpy', 'defaultSetupIso']


def critical_dt(v):
    """
    Determine the temporal sampling to satisfy CFL stability.
    This method replicates the functionality in the Model class.

    Note we add a safety factor, reducing dt by a factor 0.9.

    Parameters
    ----------
    v : Function
        velocity
    """
    coeff = 0.38 if len(v.grid.shape) == 3 else 0.42
    dt = 0.9 * v.dtype(coeff * np.min(v.grid.spacing) / (np.max(v.data)))
    return v.dtype("%.5e" % dt)


def setup_wOverQ(wOverQ, w, qmin, qmax, npad, sigma=0):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model. Uses Devito Operator.

    Parameters
    ----------
    wOverQ : Function, required
        The omega over Q field used to implement attenuation in the model,
        and the absorbing boundary condition for outgoing waves.
    w : float32, required
        center angular frequency, e.g. peak frequency of Ricker source wavelet
        used for modeling.
    qmin : float32, required
        Q value at the edge of the model. Typically set to 0.1 to strongly
        attenuate outgoing waves.
    qmax : float32, required
        Q value in the interior of the model. Typically set to 100 as a
        reasonable and physically meaningful Q value.
    npad : int, required
        Number of points in the absorbing boundary region. Note that we expect
        this to be the same on all sides of the model.
    sigma : float32, optional, defaults to None
        sigma value for call to scipy gaussian smoother, default 5.
    """
    # sanity checks
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)
    for n in wOverQ.grid.shape:
        if n - 2*npad < 1:
            raise ValueError("2 * npad must not exceed dimension size!")

    t1 = timer()
    lqmin = np.log(qmin)
    lqmax = np.log(qmax)

    # 1. Get distance to closest boundary in all dimensions
    # 2. Logarithmic variation between qmin, qmax across the absorbing boundary
    pos = Min(1, Min(*[Min(d - d.symbolic_min, d.symbolic_max - d)
                       for d in wOverQ.dimensions]) / npad)
    val = exp(lqmin + pos * (lqmax - lqmin))
    eqn = Eq(wOverQ, val)

    Operator([eqn], name='Initialize_wOverQ_Operator')()

    # If we apply the smoother, we must renormalize output to [qmin,qmax]
    if sigma > 0:
        print("sigma=", sigma)
        smooth = gaussian_smooth(wOverQ.data, sigma=sigma)
        smin, smax = np.min(smooth), np.max(smooth)
        smooth[:] = qmin + (qmax - qmin) * (smooth - smin) / (smax - smin)
        wOverQ.data[:] = smooth

    wOverQ.data[:] = w / wOverQ.data[:]

    # report min/max output Q value
    q1, q2 = np.min(wOverQ.data), np.max(wOverQ.data)
    t2 = timer()
    print("setup_wOverQ ran in %.4f seconds -- min/max w/Q values; %.6e %.6e"
          % (t2-t1, q1, q2))


def setup_wOverQ_numpy(wOverQ, w, qmin, qmax, npad, sigma=0):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model.

    Uses an outer product via numpy.ogrid[:n1, :n2] to speed up loop traversal
    for 2d and 3d. TODO: stop wasting so much memory with 9 tmp arrays ...
    Note results in 9 temporary numpy arrays for 3D.

    Parameters
    ----------
    wOverQ : Function, required
        The omega over Q field used to implement attenuation in the model,
        and the absorbing boundary condition for outgoing waves.
    w : float32, required
        center angular frequency, e.g. peak frequency of Ricker source wavelet
        used for modeling.
    qmin : float32, required
        Q value at the edge of the model. Typically set to 0.1 to strongly
        attenuate outgoing waves.
    qmax : float32, required
        Q value in the interior of the model. Typically set to 100 as a
        reasonable and physically meaningful Q value.
    npad : int, required
        Number of points in the absorbing boundary region. Note that we expect
        this to be the same on all sides of the model.
    sigma : float32, optional, defaults to None
        sigma value for call to scipy gaussian smoother, default 5.
    """
    # sanity checks
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)
    for n in wOverQ.grid.shape:
        if n - 2*npad < 1:
            raise ValueError("2 * npad must not exceed dimension size!")

    t1 = timer()
    lqmin = np.log(qmin)
    lqmax = np.log(qmax)

    if len(wOverQ.grid.shape) == 2:
        # 2d operations
        nx, nz = wOverQ.grid.shape
        kxMin, kzMin = np.ogrid[:nx, :nz]
        kxArr, kzArr = np.minimum(kxMin, nx-1-kxMin), np.minimum(kzMin, nz-1-kzMin)
        nval1 = np.minimum(kxArr, kzArr)
        nval2 = np.minimum(1, nval1/(npad))
        nval3 = np.exp(lqmin+nval2*(lqmax-lqmin))
        wOverQ.data[:, :] = w / nval3

    else:
        # 3d operations
        nx, ny, nz = wOverQ.grid.shape
        kxMin, kyMin, kzMin = np.ogrid[:nx, :ny, :nz]
        kxArr = np.minimum(kxMin, nx-1-kxMin)
        kyArr = np.minimum(kyMin, ny-1-kyMin)
        kzArr = np.minimum(kzMin, nz-1-kzMin)
        nval1 = np.minimum(kxArr, np.minimum(kyArr, kzArr))
        nval2 = np.minimum(1, nval1/(npad))
        nval3 = np.exp(lqmin+nval2*(lqmax-lqmin))
        wOverQ.data[:, :, :] = w / nval3

    # Note if we apply the gaussian smoother, renormalize output to [qmin,qmax]
    if sigma > 0:
        print("sigma=", sigma)
        nval2[:] = gaussian_smooth(nval3, sigma=sigma)
        nmin2, nmax2 = np.min(nval2), np.max(nval2)
        nval3[:] = qmin + (qmax - qmin) * (nval2 - nmin2) / (nmax2 - nmin2)

    wOverQ.data[:] = w / nval3

    # report min/max output Q value
    q1, q2 = np.min(wOverQ.data), np.max(wOverQ.data)
    t2 = timer()
    print("setup_wOverQ_numpy ran in %.4f seconds -- min/max w/Q values; %.6e %.6e"
          % (t2-t1, q1, q2))


def defaultSetupIso(npad, shape, dtype,
                    sigma=0, fpeak=0.010, qmin=0.1, qmax=100.0, tmin=0.0, tmax=2000.0,
                    bvalue=1.0/1000.0, vvalue=1.5, space_order=8, info=False):
    """
    For isotropic propagator build default model with 10m spacing,
        and 1.5 m/msec velocity

    Return:
        dictionary of velocity, buoyancy, and wOverQ
        TimeAxis defining temporal sampling
        SparseTimeFunction for source located at [center x, center y, top z]
        SparseTimeFunction for receiver line located at [line x, center y, center z)
    """
    d = 10.0
    origin = tuple([0.0 - d * npad for s in shape])
    extent = tuple([d * (s - 1) for s in shape])
    
    # Define dimensions
    if len(shape) == 2:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin,
                    dimensions=(x, z), dtype=dtype)
    else:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin,
                    dimensions=(x, y, z), dtype=dtype)

    b = Function(name='b', grid=grid, space_order=space_order)
    v = Function(name='v', grid=grid, space_order=space_order)
    b.data[:] = bvalue
    v.data[:] = vvalue

    dt = dtype("%.6f" % (0.8 * critical_dt(v)))
    time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)

    nr = shape[0] - 2 * npad
    src = RickerSource(name='src', grid=grid, f0=fpeak, npoint=1, time_range=time_axis)
    rec = Receiver(name='rec', grid=grid, npoint=nr, time_range=time_axis)

    if len(shape) == 2:
        src.coordinates.data[:, 0] = origin[0] + d * (shape[0] - 2 * npad) / 2
        src.coordinates.data[:, 1] = d

        rec.coordinates.data[:, 0] = np.linspace(0.0, d * (nr - 1), nr)
        rec.coordinates.data[:, 1] = origin[1] + d * (shape[1] - 2 * npad) / 2

        if info:
            print("src X min/max; %+12.6f %+12.6f" %
                (np.min(src.coordinates.data[:,0]), np.max(src.coordinates.data[:,0])))
            print("src Z min/max; %+12.6f %+12.6f" %
                (np.min(src.coordinates.data[:,1]), np.max(src.coordinates.data[:,1])))
            print("rec X min/max; %+12.6f %+12.6f" %
                (np.min(rec.coordinates.data[:,0]), np.max(rec.coordinates.data[:,0])))
            print("rec Z min/max; %+12.6f %+12.6f" %
                (np.min(rec.coordinates.data[:,1]), np.max(rec.coordinates.data[:,1])))
    else:
        xcenter = origin[0] + extent[0] / 2
        ycenter = origin[1] + extent[1] / 2
        zcenter = origin[2] + extent[2] / 2

        src.coordinates.data[:, 0] = origin[0] + d * (shape[0] - 2 * npad) / 2
        src.coordinates.data[:, 1] = origin[1] + d * (shape[1] - 2 * npad) / 2
        src.coordinates.data[:, 2] = d

        rec.coordinates.data[:, 0] = np.linspace(0.0, d * (nr - 1), nr)
        rec.coordinates.data[:, 1] = origin[1] + d * (shape[1] - 2 * npad) / 2
        rec.coordinates.data[:, 2] = origin[2] + d * (shape[2] - 2 * npad) / 2

        if info:
            print("src X min/max; %+12.6f %+12.6f" %
                (np.min(src.coordinates.data[:,0]), np.max(src.coordinates.data[:,0])))
            print("src Y min/max; %+12.6f %+12.6f" %
                (np.min(src.coordinates.data[:,1]), np.max(src.coordinates.data[:,1])))
            print("src Z min/max; %+12.6f %+12.6f" %
                (np.min(src.coordinates.data[:,2]), np.max(src.coordinates.data[:,2])))
            print("rec X min/max; %+12.6f %+12.6f" %
                (np.min(rec.coordinates.data[:,0]), np.max(rec.coordinates.data[:,0])))
            print("rec Y min/max; %+12.6f %+12.6f" %
                (np.min(rec.coordinates.data[:,1]), np.max(rec.coordinates.data[:,1])))
            print("rec Z min/max; %+12.6f %+12.6f" %
                (np.min(rec.coordinates.data[:,2]), np.max(rec.coordinates.data[:,2])))

    return b, v, time_axis, src, rec
