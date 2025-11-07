import itertools
import tqdm
import numpy as np
import scipy.interpolate as interp

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

from juwavelet.fft import fft


_C_DELTA = {
    1: {
        4: 0.6457893510648375,
        4.5: 0.5896218724987083,
        5: 0.5215756951098879,
        5.5: 0.47218128638000867,
        6: 0.4300073911120658,
        2 * np.pi: 0.4095070406736934,
        6.5: 0.39511772750623536,
        7: 0.36553908182140044,
        7.5: 0.34017621107087914,
        8: 0.3181670622089918,
        8.5: 0.2988644127864009,
        9: 0.28180282794462663,
        3 * np.pi: 0.2688117661517729,
        9.5: 0.2666382478787306,
        10: 0.25302545068504456,
        10.5: 0.24068078416456265,
        11: 0.2295267271821073,
        11.5: 0.21949881506527225,
        12: 0.21028605373689963,
        12.5: 0.20159862394076225,
        4 * np.pi: 0.20048720344035054,
        13: 0.19356501856723043,
        13.5: 0.18645502571059624,
        14: 0.1800762793761847,
        14.5: 0.17385783666835034,
        15: 0.16758555844563683,
        15.5: 0.1617139146986407,
        5 * np.pi: 0.15953351911290717,
        16: 0.1567716728512244,
        16.5: 0.1526346385049266,
        17: 0.1485770986616525,
        17.5: 0.1440309804264572,
        18: 0.13918829414859932,
        18.5: 0.13479802149801484,
        6 * np.pi: 0.1323226767564661,
    },
    2: {
        4: 0.38386267534279117,
        4.5: 0.34754280195085274,
        5: 0.2658384779339945,
        5.5: 0.22194729662461601,
        6: 0.18350289526638397,
        2 * np.pi: 0.16653934961216302,
        6.5: 0.15513900903701414,
        7: 0.13269903857596307,
        7.5: 0.1148834933565721,
        8: 0.10046252792957325,
        8.5: 0.08862196016506302,
        9: 0.07877445901309464,
        3 * np.pi: 0.07165549155660157,
        9.5: 0.07049707847389897,
        10: 0.06346963046301672,
        10.5: 0.057442764587270725,
        11: 0.05223697923625438,
        11.5: 0.04772288488587469,
        12: 0.04377915029009549,
        12.5: 0.040291574982466105,
        4 * np.pi: 0.03985855074602544,
        13: 0.03719089351967183,
        13.5: 0.03444988900656289,
        14: 0.03202944940037944,
        14.5: 0.029855569782654218,
        15: 0.027863240714281055,
        15.5: 0.026040788656900304,
        5 * np.pi: 0.02533827988162018,
        16: 0.02441019237750478,
        16.5: 0.02296983642467311,
        17: 0.021671216296693448,
        17.5: 0.020453106633691737,
        18: 0.019290504619499525,
        18.5: 0.018205607961328966,
        6 * np.pi: 0.017513039976598387,
    },
    3: {
        4: 0.4283931029686469,
        4.5: 0.42047569245486466,
        5: 0.2688474549244627,
        5.5: 0.2134134364511464,
        6: 0.1599145855107853,
        2 * np.pi: 0.13872442531321855,
        6.5: 0.12447207644291042,
        7: 0.09894500779683672,
        7.5: 0.07925792092712564,
        8: 0.06517957522086339,
        8.5: 0.053679758210994834,
        9: 0.045247285069208915,
        3 * np.pi: 0.03925077298053942,
        9.5: 0.03807758748726513,
        10: 0.03271602584682166,
        10.5: 0.028004791810484153,
        11: 0.024431416853378008,
        11.5: 0.021204770011526614,
        12: 0.01873433872973038,
        12.5: 0.016446730090162932,
        4 * np.pi: 0.016279024130202914,
        13: 0.01459496452755837,
        13.5: 0.013012634630626288,
        14: 0.011653945264912313,
        14.5: 0.010478861696135688,
        15: 0.009454560802338252,
        15.5: 0.00855767369066144,
        5 * np.pi: 0.008218156287244356,
        16: 0.007771817142929901,
        16.5: 0.007082360545665025,
        17: 0.006473864624532983,
        17.5: 0.005931881898818549,
        18: 0.005445969426091458,
        18.5: 0.005010221976479837,
        6 * np.pi: 0.004733375817052859,
    },
}

PSI_0 = np.pi ** -0.25


def compute_c_delta(dims, param, force=False):
    """
    Numerically computes the c_delta normalization factor needed for reconstruction.
    Very time consuming for higher dimensions.

    Parameters
    ----------
    dims : int
        number of dimensions (1, 2, or 3)
    param : float
        Morlet parameter
    force: bool
        Force recomputation in contrast to using cached values

    Returns
    -------
    [float]
        reconstruction factor c_delta
    """
    assert dims in (1, 2, 3)

    # these parameters work well and are not too costly
    nx = 2 ** 11
    dx = 1
    s0 = 0.25
    dj = 0.25
    js = 70
    scales = s0 * 2 ** (np.arange(js) * dj)

    if dims == 1 and (force or not (min(_C_DELTA[dims]) <= param <= max(_C_DELTA[dims]))):
        c_delta = 0
        for scale in scales:
            daughter = morlet1d_fourier(
                nx, dx, scale, param, normed=False)
            c_delta += np.mean(daughter)
        _C_DELTA[1][param] = c_delta * dj * np.log(2)
    elif dims == 2 and (force or not (min(_C_DELTA[dims]) <= param <= max(_C_DELTA[dims]))):
        ts = 30
        thetas = np.linspace(0, np.pi / 4, ts, endpoint=False)
        c_delta = 0
        for scale, theta in tqdm.tqdm(list(itertools.product(scales, thetas))):
            daughter = morlet2d_fourier(
                nx, nx, dx, dx, scale, theta, param, normed=False)
            c_delta += np.mean(daughter)
        c_delta *= 4  # due to integration only to pi/4
        _C_DELTA[2][param] = c_delta * dj * np.log(2) * ((np.pi / 4) / ts)
    elif dims == 3 and (force or not (min(_C_DELTA[dims]) <= param <= max(_C_DELTA[dims]))):
        nx = 2 ** 8
        dj = 0.5
        js = 25
        ts = 25
        thetas = np.linspace(0, np.pi / 4, ts, endpoint=False)
        ps = 25
        phis = np.linspace(0, np.pi / 2, ps, endpoint=False)
        c_delta = 0
        for scale, theta, phi in tqdm.tqdm(list(itertools.product(scales, thetas, phis))):
            daughter = morlet3d_fourier(
                nx, nx, nx, dx, dx, dx, scale, theta, phi, param, normed=False)
            c_delta += np.mean(daughter) * np.cos(phi)
        c_delta *= 4  # due to integration of theta to only pi/4
        c_delta *= 2  # due to integration of phi over only positive numbers
        _C_DELTA[3][param] = c_delta * dj * np.log(2) * ((np.pi / 2) / ps) * ((np.pi / 4) / ts)
    if param not in _C_DELTA[dims]:
        xs = sorted(_C_DELTA[dims])
        ys = [_C_DELTA[dims][_x] for _x in xs]
        _C_DELTA[dims][param] = float(interp.interp1d(xs, ys, kind="cubic")(param))
    return _C_DELTA[dims][param]


def morlet1d_fourier(nx, dx, scale, param=6, normed=False):
    """
    Computes the spectral Morlet wavelt

    Parameters
    ----------
    nx : int
        length of vector
    dx : float
        horizontal sampling
    scale : float
        scale parameter
    param : float
        Morlet parameter
    normed : bool
        if true, the wavelet is normed to 1

    Returns
    -------
    [ndarray]
        spectral Morlet wavelet
    """
    norm = 2
    if normed:
        norm *= np.sqrt(scale) / (np.sqrt(2) * PSI_0)
        norm /= dx
    expnt = -0.5 * np.square(2 * np.pi * scale * np.fft.fftfreq(nx, d=dx) - param)
    daughter = norm * np.exp(expnt)
    return daughter


def _morlet1d(nx, dx, scale, param, normed=False):
    xs = np.arange(-nx // 2, nx // 2) * dx
    gauss = np.exp(-(xs / scale) ** 2 / 2)
    if normed:
        norm = np.sqrt(1 / scale) * PSI_0
    else:
        norm = dx * np.sqrt(2 / np.pi) / scale
    return norm * np.exp(1j * param * (xs / scale)) * gauss


def morlet1d(nx, dx, scale, param, normed=False):
    """
    Computes the spatial Morlet wavelt

    Parameters
    ----------
    nx : int
        length of vector
    dx : float
        horizontal sampling
    scale : float
        scale parameter
    param : float
        Morlet parameter
    normed : bool
        if true, the wavelet is normed to 1

    Returns
    -------
    [ndarray]
        spatial Morlet wavelet
    """

    daughter = morlet1d_fourier(nx, dx, scale, param, normed=normed)
    return np.fft.fftshift(fft.ifft(daughter), 0)


def morlet2d_fourier(nx, ny, dx, dy, scale, theta, param=6, normed=False):
    """
    Provides the Morlet 2-D wavelet in realspace, mostly
    for plotting and debugging purposes.

    Parameters
    ----------
    nx : int
        length of x-axis
    ny : int
        length of y-axis
    dx : float
        delta of x-axis
    dy : float
        delta of y-axis
    scale : float
        scaling parameter of wavelet
    theta : float
        angle of Morlet wavelet
    param : float
        Morlet oscillation parameter
    normed : bool
        whether to use normed wavelets

    Returns
    -------
    [ndarray]
       2-D Morlet wavelet in frequency space
    """

    k0 = param
    # Wavenumber
    k01 = k0 * np.cos(theta)
    k02 = k0 * np.sin(theta)
    expnt = -0.5 * (
        ((scale * 2 * np.pi) * np.fft.fftfreq(nx, d=dx).reshape(-1, 1) - k01) ** 2 +
        ((scale * 2 * np.pi) * np.fft.fftfreq(ny, d=dy).reshape(1, -1) - k02) ** 2)
    # 2 instead of sqrt(2) as in Chen to get rid of the otherwise necessary scaling
    # of Wf with sqrt(2). Thus C_delta has been computed such that the mean
    # power corresponds to A^2, not 0.5A^2.
    norm = 2
    if normed:
        norm *= scale * np.sqrt(np.pi) / (dx * dy)
    daughter = norm * np.exp(expnt)
    return daughter


def _morlet2d(nx, ny, dx, dy, scale, theta, param, normed=False):
    xs = np.arange(-nx // 2, nx // 2) * dx
    ys = np.arange(-ny // 2, ny // 2) * dy
    ys, xs = np.meshgrid(ys, xs)
    gauss = np.exp(-((xs / scale) ** 2 + (ys / scale) ** 2) / 2)
    if normed:
        norm = np.sqrt(1 / np.pi) / scale
    else:
        norm = dx * dy / ((scale ** 2) * np.pi)
    return np.exp(1j * param * (np.cos(theta) * xs + np.sin(theta) * ys) / scale) * gauss * norm


def morlet2d(nx, ny, dx, dy, scale, theta, param, normed=False):
    """
    Provides the Morlet 2-D wavelet in fourier space

    Parameters
    ----------
    nx : int
        length of x-axis
    ny : int
        length of y-axis
    dx : float
        delta of x-axis
    dy : float
        delta of y-axis
    scale : float
        scaling parameter of wavelet
    theta : float
        angle of Morlet wavelet
    param : float
        Morlet oscillation parameter
    normed : bool
        whether to use normed wavelets

    Returns
    -------
    [ndarray]
       2-D Morlet wavelet in real space
    """

    daughter = morlet2d_fourier(nx, ny, dx, dy, scale, theta, param=param, normed=normed)
    return np.fft.fftshift(np.fft.fftshift(fft.ifft2(daughter), 0), 1)


@njit
def myfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.empty(shape=n, dtype=np.int64)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=np.int64)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=np.int64)
    results[N:] = p2
    return results * val


@njit(parallel=True)
def morlet3d_fourier(
        nx, ny, nz,
        dx, dy, dz,
        scale, theta, phi, param=6, normed=False):
    """
    Provides the Morlet 3-D wavelet in fourier space

    Parameters
    ----------
    nx : int
        length of x-axis
    ny : int
        length of y-axis
    nz : int
        length of z-axis
    dx : float
        delta of x-axis
    dy : float
        delta of y-axis
    dz : float
        delta of z-axis
    scale : float
        scaling parameter of wavelet
    theta : float
        first angle of Morlet wavelet
    phi : float
        second angle of Morlet wavelet
    param : float
        Morlet oscillation parameter
    normed : bool
        whether to use normed wavelets

    Returns
    -------
    [ndarray]
       3-D Morlet wavelet in frequency space
    """
    k0 = param
    # Wavenumber
    k01 = k0 * np.cos(phi) * np.cos(theta)
    k02 = k0 * np.cos(phi) * np.sin(theta)
    k03 = k0 * np.sin(phi)

    expnt = -0.5 * (
        ((scale * 2 * np.pi) * myfftfreq(nx, d=dx).reshape(-1, 1, 1) - k01) ** 2 +
        ((scale * 2 * np.pi) * myfftfreq(ny, d=dy).reshape(1, -1, 1) - k02) ** 2 +
        ((scale * 2 * np.pi) * myfftfreq(nz, d=dz).reshape(1, 1, -1) - k03) ** 2)
    norm = 2
    if normed:
        norm *= np.sqrt(2) * np.pi * PSI_0 * (scale ** 1.5) / (dx * dy * dz)
    daughter = norm * np.exp(expnt)
    return daughter


def _morlet3d(nx, ny, nz, dx, dy, dz, scale, theta, phi, param, normed=False):
    xs = np.arange(-nx // 2, nx // 2) * dx
    ys = np.arange(-ny // 2, ny // 2) * dy
    zs = np.arange(-nz // 2, nz // 2) * dz
    ys, xs, zs = np.meshgrid(ys, xs, zs)
    gauss = np.exp(-((xs / scale) ** 2 + (ys / scale) ** 2 + (zs / scale) ** 2) / 2)
    if normed:
        norm = PSI_0 * np.sqrt(1 / np.pi) / (scale ** 1.5)
    else:
        norm = (dx * dy * dz) / ((scale ** 3) * np.pi * np.sqrt(2 * np.pi))
    return np.exp(1j * param * (
        np.cos(phi) * np.cos(theta) * xs +
        np.cos(phi) * np.sin(theta) * ys +
        np.sin(phi) * zs) / scale) * gauss * norm


def morlet3d(nx, ny, nz, dx, dy, dz, scale, theta, phi, param, normed=False):
    """
    Provides the Morlet 3-D wavelet in realspace, mostly
    for plotting and debugging purposes.

    Parameters
    ----------
    nx : int
        length of x-axis
    ny : int
        length of y-axis
    nz : int
        length of z-axis
    dx : float
        delta of x-axis
    dy : float
        delta of y-axis
    dz : float
        delta of z-axis
    aspect : float
        aspect ratio of x to y-axis
    scale : float
        scaling parameter of wavelet
    theta : float
        angle of Morlet wavelet
    phi : float
        angle of Morlet wavelet
    param : float
        Morlet oscillation parameter
    normed : bool
        whether to use normed wavelets

    Returns
    -------
    [ndarray]
       3-D Morlet wavelet in real space
    """

    daughter = morlet3d_fourier(
        nx, ny, nz, dx, dy, dz,
        scale, theta, phi, param, normed=normed)
    return np.fft.fftshift(np.fft.fftshift(np.fft.fftshift(
        fft.ifftn(daughter), 0), 1), 2)
