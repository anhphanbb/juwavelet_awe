import itertools
import tqdm
import numpy as np

from juwavelet.fft import fft
from juwavelet.morlet import (
    compute_c_delta, PSI_0, morlet1d_fourier, morlet2d_fourier, morlet3d_fourier)


def _get_padding(n, npad):
    if npad is not None:
        return npad
    return int(2 ** round(np.log2(n) + 1.4999))


def decompose1d(data, dx, s0, dj, js,
                nxpad=None, opts=None, filt=None,
                mode="stockwell", dtype=np.complex128):
    """
    Computes the 1-D time-frequency decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of scales
    opts : dict [optional]
        additional parameters
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x and max_wavelength_x
        are the possible entries in the dictionary, each
        having a float value. Filtered coefficients are assigned NaN.
    nxpad : int [optional]
        zero-pad to this
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.

    Returns
    -------
    [dict]
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"]:
        opts.setdefault("window", 100)
    else:
        opts.setdefault("param", 2 * np.pi)
    if filt is None:
        filt = {}
    nx = len(data)
    nxpad = _get_padding(nx, nxpad)

    data_pad = np.zeros(nxpad)
    data_pad[:nx] = data

    f_hat = fft.fft(data_pad)

    scales = s0 * (2 ** (np.arange(js) * dj))
    periods = scales.copy()
    if mode in ["cwt", "scaled", "stockwell"]:
        periods *= 2 * np.pi / opts["param"]
    phase_corr = (2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2)

    coeff = np.full((js, nx), np.nan, dtype=dtype)
    for j in range(js):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(periods[j])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(periods[j]))):
            continue
        if mode == "gabor":
            daughter = morlet1d_fourier(
                nxpad, dx, opts["window"],
                opts["window"] * 2 * np.pi / scales[j], normed=False)
        else:
            daughter = morlet1d_fourier(
                nxpad, dx, scales[j], opts["param"], normed=(mode == "cwt"))

        coeff[j, :] = fft.ifft(daughter * f_hat)[:nx]

        if mode in ["stockwell", "gabor"]:
            coeff[j, :] /= np.exp(phase_corr / periods[j])
    if mode in ["cwt"]:
        coeff *= dx

    return {"decomposition": coeff,
            "dx": dx,
            "dj": dj,
            "js": js,
            "scale": scales,
            "period": periods,
            "mode": mode,
            "opts": opts}


def decompose1d_dominant(
        data, dx, s0=None, dj=None, js=None,
        scales=None, previous=None,
        nxpad=None, opts=None, filt=None,
        mode="stockwell", dtype=np.complex128):
    """
    Computes the 2-D Morlet wavelet decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling distance (second axis of data)
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of spatial scales
    aspect : float [optional]
        stretching factor for y-dimension
    nxpad : int [optional]
        padding in x-direction
    opts : dict [optional]
        additional options.
        * "param" selects the Morlet parameter, default is 2pi
        * "window" selects the window length for "gabor" mode
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x, max_wavelength_x, min_wavelength_y, max_wavelength_y,
        and min_coefficient are the possible entries in the dictionary, each
        having a float value.
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.
    dtype : numpy complex type
        type to store coefficients in

    Returns
    -------
    [dict]
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"]:
        opts.setdefault("window", 100)
    else:
        opts.setdefault("param", 2 * np.pi)
    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)
    nx = len(data)
    nxpad = _get_padding(nx, nxpad)

    data_pad = np.zeros(nxpad)
    data_pad[:nx] = data

    f_hat = fft.fft(data_pad)

    # ....construct SCALE array & empty PERIOD & WAVE arrays
    if s0 is not None:
        assert scales is None
        scales = s0 * (2. ** (np.arange(js) * dj))
    else:
        assert js is None and dj is None
    periods = scales.copy()
    if mode in ["cwt", "scaled", "stockwell"]:
        periods *= 2 * np.pi / opts["param"]

    phase_corr = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))

    result = {
        _name: np.full_like(data, np.nan) for _name in [
            "energy", "coefficient", "phase", "scale", "theta", "period"]}

    if previous is not None:
        for key in result:
            result[key][:] = previous[key]

    for j in range(js):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(periods[j])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(periods[j]))):
            continue

        if mode == "gabor":
            daughter = morlet1d_fourier(
                nxpad, dx, opts["window"],
                opts["window"] * 2 * np.pi / scales[j], normed=False)
        else:
            daughter = morlet1d_fourier(
                nxpad, dx, scales[j],
                opts["param"], normed=(mode == "cwt"))

        coeff = fft.ifft(f_hat * daughter)[:nx]

        if mode in ["stockwell", "gabor"]:
            coeff /= np.exp(phase_corr / periods[j])
        if mode in ["cwt"]:
            coeff *= dx

        sel = abs(coeff) > filt["min_coefficient"]
        sel &= (scales[j] * coeff > result["energy"]) | (~np.isfinite(result["energy"]))
        result["coefficient"][sel] = coeff[sel]
        result["energy"][sel] = scales[j] * coeff[sel]
        result["phase"][sel] = np.angle(coeff[sel])
        result["scale"][sel] = scales[j]
        result["period"][sel] = periods[j]

    return result


def reconstruct1d(decomposition):
    """
    Reconstructs data from a wavelet decomposition

    Parameters
    ----------
    decomposition : dict
        decomposition from decompose1d

    Returns
    -------
    [ndarray]
    """
    coeff, scales, periods, dx, dj, js, mode = [
        decomposition[_x] for _x in [
            "decomposition", "scale", "period", "dx", "dj", "js", "mode"]]

    rec_fac = dj * np.log(2) / compute_c_delta(1, decomposition["opts"]["param"])
    if mode == "cwt":
        # add back the terms missing from normed wavelets
        coeff = coeff / np.sqrt(scales[:, np.newaxis])
        rec_fac *= np.sqrt(2) * PSI_0
    elif mode == "scaled":
        pass
    elif mode == "gabor":
        raise NotImplementedError
    elif mode == "stockwell":
        phase_corr = (2 * np.pi * 1j * dx) * np.arange(-coeff.shape[1] // 2, coeff.shape[1] // 2)
        coeff = (coeff * np.exp(phase_corr[np.newaxis, :] / periods[:, np.newaxis])).real
    else:
        raise NotImplementedError
    return rec_fac * coeff.real.sum(0)


def decompose2d(
        data, dx, dy, s0, dj, js, jt, aspect=1,
        nxpad=None, nypad=None, opts=None, filt=None,
        mode="stockwell", dtype=np.complex128):
    """
    Computes the 2-D Morlet wavelet decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling distance (second axis of data)
    dy : float
        vertical sampling distance (first axis of data)
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of spatial scales
    jt : int
        number of angular scales
    aspect : float [optional]
        stretching factor for y-dimension
    nxpad : int [optional]
        padding in x-direction
    nypad : int [optional]
        padding in y-direction
    opts : dict [optional]
        additional options.
        * "param" selects the Morlet parameter, default is 2pi
        * "window" selects the window length for "gabor" mode
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x, max_wavelength_x, min_wavelength_y, max_wavelength_y,
        min_theta, and max_theta
        are the possible entries in the dictionary, each
        having a float value. Filtered coefficients are assigned NaN.
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.
    dtype : numpy complex type
        type to store coefficients in

    Returns
    -------
    [dict]
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"]:
        opts.setdefault("window", 100)
    else:
        opts.setdefault("param", 2 * np.pi)
    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)
    nx, ny = data.shape
    nxpad, nypad = [_get_padding(_n, _pad) for _n, _pad in zip((nx, ny), (nxpad, nypad))]
    dy_p = aspect * dy

    data_pad = np.zeros((nxpad, nypad))
    data_pad[:nx, :ny] = data

    # ....compute FFT of the (padded) time series
    f_hat = fft.fft2(data_pad)    # [Eqn(3)]

    # ....construct SCALE array & empty PERIOD & WAVE arrays
    scales = s0 * (2. ** (np.arange(js) * dj))
    thetas = np.linspace(0, np.pi, jt, endpoint=False)
    periods = scales.copy()
    if mode in ["cwt", "scaled", "stockwell"]:
        periods *= (2 * np.pi / opts["param"])
    wxs = periods[:, np.newaxis] / (np.cos(thetas[np.newaxis, :]))
    wys = periods[:, np.newaxis] / (aspect * np.sin(thetas[np.newaxis, :]))

    phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[:, np.newaxis]
    phase_corr_y = ((2 * np.pi * 1j * dy_p) * np.arange(-ny // 2, ny // 2))[np.newaxis, :]

    decomposition = np.full((js, jt, nx, ny), np.nan, dtype=dtype)  # define the wavelet array
    # loop through all scales and compute transform
    for a1, a2 in itertools.product(range(js), range(jt)):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(wxs[a1, a2])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(wxs[a1, a2])) or
                ("min_wavelength_y" in filt and filt["min_wavelength_y"] > abs(wys[a1, a2])) or
                ("max_wavelength_y" in filt and filt["max_wavelength_y"] < abs(wys[a1, a2])) or
                ("min_theta" in filt and filt["min_theta"] > thetas[a2]) or
                ("max_theta" in filt and filt["max_theta"] < thetas[a2])):
            continue
        if mode == "gabor":
            daughter = morlet2d_fourier(
                nxpad, nypad, dx, dy_p, opts["window"],
                thetas[a2], opts["window"] * 2 * np.pi / periods[a1], normed=False)
        else:
            daughter = morlet2d_fourier(
                nxpad, nypad, dx, dy_p, scales[a1],
                thetas[a2], opts["param"], normed=(mode == "cwt"))

        decomposition[a1, a2, :, :] = fft.ifft2(f_hat * daughter)[:nx, :ny]

        if mode in ["stockwell", "gabor"]:
            decomposition[a1, a2, :, :] /= np.exp(
                (phase_corr_x * np.cos(thetas[a2]) +
                 phase_corr_y * np.sin(thetas[a2])) / periods[a1])
    if mode in ["cwt"]:
        decomposition *= dx * dy * aspect

    return {"decomposition": decomposition,
            "dx": dx,
            "dy": dy,
            "dj": dj,
            "js": js,
            "jt": jt,
            "scale": scales,
            "theta": thetas,
            "period": periods,
            "wavelength_x": periods[:, np.newaxis] / np.cos(thetas[np.newaxis, :]),
            "wavelength_y": periods[:, np.newaxis] / (aspect * np.sin(thetas[np.newaxis, :])),
            "aspect": aspect,
            "mode": mode,
            "opts": opts}


def decompose2d_dominant(
        data, dx, dy, s0=None, dj=None, js=None, jt=None,
        scales=None, thetas=None, previous=None, aspect=1,
        nxpad=None, nypad=None, opts=None, filt=None,
        mode="stockwell", dtype=np.complex128):
    """
    Computes the 2-D Morlet wavelet decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling distance (second axis of data)
    dy : float
        vertical sampling distance (first axis of data)
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of spatial scales
    jt : int
        number of angular scales
    aspect : float [optional]
        stretching factor for y-dimension
    nxpad : int [optional]
        padding in x-direction
    nypad : int [optional]
        padding in y-direction
    opts : dict [optional]
        additional options.
        * "param" selects the Morlet parameter, default is 2pi
        * "window" selects the window length for "gabor" mode
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x, max_wavelength_x, min_wavelength_y, max_wavelength_y,
        min_theta, max_theta, and min_coefficient
        are the possible entries in the dictionary, each
        having a float value
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.
    dtype : numpy complex type
        type to store coefficients in

    Returns
    -------
    [dict]
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"]:
        opts.setdefault("window", 100)
    else:
        opts.setdefault("param", 2 * np.pi)
    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)
    nx, ny = data.shape
    nxpad, nypad = [_get_padding(_n, _pad) for _n, _pad in zip((nx, ny), (nxpad, nypad))]

    dy_p = aspect * dy

    data_pad = np.zeros((nxpad, nypad))
    data_pad[:nx, :ny] = data

    # ....compute FFT of the (padded) time series
    f_hat = fft.fft2(data_pad)    # [Eqn(3)]

    # ....construct SCALE array & empty PERIOD & WAVE arrays
    if s0 is not None:
        assert scales is None and thetas is None
        scales = s0 * (2. ** (np.arange(js) * dj))
        thetas = np.linspace(0, np.pi, jt, endpoint=False)
    else:
        assert js is None and jt is None and dj is None
    periods = scales
    if mode in ["cwt", "scaled", "stockwell"]:
        periods = (2 * np.pi / opts["param"]) * scales
    wxs = periods[:, np.newaxis] / (np.cos(thetas[np.newaxis, :]))
    wys = periods[:, np.newaxis] / (aspect * np.sin(thetas[np.newaxis, :]))

    phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[:, np.newaxis]
    phase_corr_y = ((2 * np.pi * 1j * dy_p) * np.arange(-ny // 2, ny // 2))[np.newaxis, :]

    result = {
        _name: np.full_like(data, np.nan) for _name in [
            "energy", "coefficient", "wavelength_x", "wavelength_y", "phase", "scale", "theta", "period"]}

    if previous is not None:
        for key in result:
            result[key][:] = previous[key]

    for a1, a2 in itertools.product(range(len(scales)), range(len(thetas))):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(wxs[a1, a2])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(wxs[a1, a2])) or
                ("min_wavelength_y" in filt and filt["min_wavelength_y"] > abs(wys[a1, a2])) or
                ("max_wavelength_y" in filt and filt["max_wavelength_y"] < abs(wys[a1, a2])) or
                ("min_theta" in filt and filt["min_theta"] > thetas[a2]) or
                ("max_theta" in filt and filt["max_theta"] < thetas[a2])):
            continue

        if mode == "gabor":
            daughter = morlet2d_fourier(
                nxpad, nypad, dx, dy_p, opts["window"],
                thetas[a2], opts["window"] * 2 * np.pi / periods[a1], normed=False)
        else:
            daughter = morlet2d_fourier(
                nxpad, nypad, dx, dy_p, scales[a1],
                thetas[a2], opts["param"], normed=(mode == "cwt"))

        coeff = fft.ifft2(f_hat * daughter)[:nx, :ny]

        if mode in ["stockwell", "gabor"]:
            coeff /= np.exp(
                (phase_corr_x * np.cos(thetas[a2]) +
                 phase_corr_y * np.sin(thetas[a2])) / periods[a1])
        if mode in ["cwt"]:
            coeff *= dx * dy * aspect

        sel = abs(coeff) > filt["min_coefficient"]
        sel &= (scales[a1] * coeff > result["energy"]) | (~np.isfinite(result["energy"]))
        result["coefficient"][sel] = coeff[sel]
        result["energy"][sel] = scales[a1] * coeff[sel]
        result["phase"][sel] = np.angle(coeff[sel])
        result["scale"][sel] = scales[a2]
        result["theta"][sel] = thetas[a1]
        result["period"][sel] = periods[a2]
        result["wavelength_x"][sel] = wxs[a1, a2]
        result["wavelength_y"][sel] = wys[a1, a2]

    return result


def reconstruct2d(decomposition):
    """
    Reconstructs from a 2-D wavelet decomposition

    Parameters
    ----------
    decomposition : dict
        decomposition

    Returns
    -------
    [ndarray]
    """
    coeff, scales, periods, thetas, dx, dy, dj, js, jt, aspect, mode = [
        decomposition[_x] for _x in [
            "decomposition", "scale", "period", "theta",
            "dx", "dy", "dj", "js", "jt", "aspect", "mode"]]
    dy_p = dy * aspect
    rec_fac = np.log(2) * (np.pi / jt) * dj / compute_c_delta(2, decomposition["opts"]["param"])
    if mode == "cwt":
        rec_fac /= np.sqrt(np.pi)
        coeff = coeff.real / scales[:, np.newaxis, np.newaxis, np.newaxis]
    elif mode == "scaled":
        pass
    elif mode == "gabor":
        raise NotImplementedError
    elif mode == "stockwell":
        nx, ny = coeff.shape[2:]
        phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[np.newaxis, np.newaxis, :, np.newaxis]
        phase_corr_y = ((2 * np.pi * 1j * dy_p) * np.arange(-ny // 2, ny // 2))[np.newaxis, np.newaxis, np.newaxis, :]
        theta_bc = thetas[np.newaxis, :, np.newaxis, np.newaxis]
        period_bc = periods[:, np.newaxis, np.newaxis, np.newaxis]
        coeff = (coeff * np.exp(
            (phase_corr_x * np.cos(theta_bc) +
             phase_corr_y * np.sin(theta_bc)) / period_bc)).real
    else:
        raise NotImplementedError
    return rec_fac * coeff.real.sum(0).sum(0)


def decompose3d(
        data, dx, dy, dz, s0, dj, js, jt, jp, aspect=1,
        nxpad=None, nypad=None, nzpad=None, opts=None, filt=None,
        mode="stockwell", dtype=np.complex64):
    """
    Computes the 2-D Morlet wavelet decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling distance
    dy : float
        vertical sampling distance
    dz : float
        vertical sampling distance
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of spatial scales
    jt : int
        number of theta scales
    jp : int
        number of phi scales
    aspect : float [optional]
        stretching factor for y-dimension
    nxpad : int [optional]
        padding in x-direction
    nypad : int [optional]
        padding in y-direction
    nzpad : int [optional]
        padding in z-direction
    opts : dict [optional]
        additional options.
        * "param" selects the Morlet parameter, default is 2pi
        * "window" selects the window length for "gabor" mode
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x, max_wavelength_x, min_wavelength_y, max_wavelength_y,
        min_wavelength_z, max_wavelength_z, min_theta, max_theta, min_phi, and
        max_phi are the possible entries in the
        dictionary, each having a float value. Coefficients for filtered
        components are set to NaN.
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.
    dtype : numpy complex type [optional]
        type to store coefficients in

    Returns
    -------
    [dict]
        3-D wavelet decomposition
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"] and "window" not in opts:
        opts["window"] = 100
    elif "param" not in opts:
        opts["param"] = 2 * np.pi
    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)
    nx, ny, nz = data.shape
    nxpad, nypad, nzpad = [_get_padding(_n, _pad) for _n, _pad in zip((nx, ny, nz), (nxpad, nypad, nzpad))]
    dz_p = dz * aspect

    data_pad = np.zeros((nxpad, nypad, nzpad))
    data_pad[:nx, :ny, :nz] = data

    # ....compute FFT of the (padded) time series
    f_hat = fft.fftn(data_pad)    # [Eqn(3)]

    # ....construct SCALE array & empty PERIOD & WAVE arrays
    scales = s0 * (2. ** (np.arange(js) * dj))
    thetas = np.linspace(0, np.pi, jt, endpoint=False)
    phis = np.linspace(-np.pi / 2, np.pi / 2, jp, endpoint=False) + (np.pi / jp) / 2
    periods = scales.copy()
    if mode in ["cwt", "scaled", "stockwell"]:
        periods *= (2 * np.pi / opts["param"])
    wxs = periods[:, np.newaxis, np.newaxis] / (
        np.cos(phis[np.newaxis, np.newaxis, :]) * np.cos(thetas[np.newaxis, :, np.newaxis]))
    wys = periods[:, np.newaxis, np.newaxis] / (
        np.sin(phis[np.newaxis, np.newaxis, :]) * np.sin(thetas[np.newaxis, :, np.newaxis]))
    wzs = np.tile(periods[:, np.newaxis, np.newaxis] /
                  (aspect * np.sin(phis[np.newaxis, np.newaxis])), [1, len(thetas), 1])

    if mode in ["stockwell", "gabor"]:
        phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[:, np.newaxis, np.newaxis]
        phase_corr_y = ((2 * np.pi * 1j * dy) * np.arange(-ny // 2, ny // 2))[np.newaxis, :, np.newaxis]
        phase_corr_z = ((2 * np.pi * 1j * dz_p) * np.arange(-nz // 2, nz // 2))[np.newaxis, np.newaxis, :]

    decomposition = np.full((js, jt, jp, nx, ny, nz), np.nan, dtype=dtype)  # define the wavelet array
    # loop through all scales and compute transform
    for a3, a2, a1 in tqdm.tqdm(list(itertools.product(range(jp), range(jt), range(js)))):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(wxs[a1, a2, a3])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(wxs[a1, a2, a3])) or
                ("min_wavelength_y" in filt and filt["min_wavelength_y"] > abs(wys[a1, a2, a3])) or
                ("max_wavelength_y" in filt and filt["max_wavelength_y"] < abs(wys[a1, a2, a3])) or
                ("min_wavelength_z" in filt and filt["min_wavelength_z"] > abs(wzs[a1, a2, a3])) or
                ("max_wavelength_z" in filt and filt["max_wavelength_z"] < abs(wzs[a1, a2, a3])) or
                ("min_theta" in filt and filt["min_theta"] > thetas[a2]) or
                ("max_theta" in filt and filt["max_theta"] < thetas[a2]) or
                ("min_phi" in filt and filt["min_phi"] > phis[a3]) or
                ("max_phi" in filt and filt["max_phi"] < phis[a3])):
            continue
        if mode == "gabor":
            raise NotImplementedError
        else:
            daughter = morlet3d_fourier(
                nxpad, nypad, nzpad, dx, dy, dz_p,
                scales[a1], thetas[a2], phis[a3], opts["param"], normed=(mode == "cwt"))
        decomposition[a1, a2, a3, :, :, :] = fft.ifftn(f_hat * daughter)[:nx, :ny, :nz]

        if mode in ["stockwell", "gabor"]:
            decomposition[a1, a2, a3, ...] /= np.exp(
                (phase_corr_x * np.cos(phis[a3]) * np.cos(thetas[a2]) +
                 phase_corr_y * np.cos(phis[a3]) * np.sin(thetas[a2]) +
                 phase_corr_z * np.sin(phis[a3])) / periods[a1])
    if mode in ["cwt"]:
        decomposition *= dx * dy * dz_p

    return {"decomposition": decomposition,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dj": dj,
            "js": js,
            "jt": jt,
            "jp": jp,
            "scale": scales,
            "theta": thetas,
            "phi": phis,
            "period": periods,
            "wavelength_x": wxs,
            "wavelength_y": wys,
            "wavelength_z": wzs,
            "aspect": aspect,
            "mode": mode,
            "opts": opts}


def decompose3d_dominant(
        data, dx, dy, dz, s0=None, dj=None, js=None, jt=None, jp=None,
        scales=None, thetas=None, phis=None, previous=None, aspect=1,
        nxpad=None, nypad=None, nzpad=None, opts=None, filt=None,
        mode="stockwell", dtype=np.complex64):
    """
    Computes the 2-D Morlet wavelet decomposition

    Parameters
    ----------
    data : ndarray
        data to be analysed
    dx : float
        horizontal sampling distance
    dy : float
        vertical sampling distance
    dz : float
        vertical sampling distance
    s0 : float
        smallest scale to take into account
    dj : float
        subscale exponent. Use 0.25 in doubt
    js : int
        number of spatial scales
    jt : int
        number of theta scales
    jp : int
        number of phi scales
    aspect : float [optional]
        stretching factor for y-dimension
    nxpad : int [optional]
        padding in x-direction
    nypad : int [optional]
        padding in y-direction
    nzpad : int [optional]
        padding in z-direction
    opts : dict [optional]
        additional options.
        * "param" selects the Morlet parameter, default is 2pi
        * "window" selects the window length for "gabor" mode
    filt : dict [optional]
        A dictionary with filter options.
        min_wavelength_x, max_wavelength_x, min_wavelength_y, max_wavelength_y,
        min_wavelength_z, max_wavelength_z, min_theta, max_theta, min_phi,
        max_phi, and min_coefficient are the possible entries in the
        dictionary, each having a float value. Coefficients for filtered
        components are set to NaN.
    mode : str [optional]
        choose between "cwt", "scaled", "gabor", and "stockwell" modes.
    dtype : numpy complex type [optional]
        type to store coefficients in

    Returns
    -------
    [dict]
        3-D wavelet decomposition
    """
    assert mode in ["cwt", "scaled", "gabor", "stockwell"]
    if opts is None:
        opts = {}
    if mode in ["gabor"] and "window" not in opts:
        opts["window"] = 100
    elif "param" not in opts:
        opts["param"] = 2 * np.pi
    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)
    nx, ny, nz = data.shape
    nxpad, nypad, nzpad = [_get_padding(_n, _pad) for _n, _pad in zip((nx, ny, nz), (nxpad, nypad, nzpad))]
    dz_p = dz * aspect

    data_pad = np.zeros((nxpad, nypad, nzpad))
    data_pad[:nx, :ny, :nz] = data

    # ....compute FFT of the (padded) time series
    f_hat = fft.fftn(data_pad)    # [Eqn(3)]

    # ....construct SCALE array & empty PERIOD & WAVE arrays
    if s0 is not None:
        scales = s0 * (2. ** (np.arange(js) * dj))
        thetas = np.linspace(0, np.pi, jt, endpoint=False)
        phis = np.linspace(-np.pi / 2, np.pi / 2, jp, endpoint=False)
    else:
        assert js is None and jt is None and jp is None and dj is None
    periods = scales.copy()
    if mode in ["cwt", "scaled", "stockwell"]:
        periods *= (2 * np.pi / opts["param"])
    wxs = periods[:, np.newaxis, np.newaxis] / (
        np.cos(phis[np.newaxis, np.newaxis, :]) * np.cos(thetas[np.newaxis, :, np.newaxis]))
    wys = periods[:, np.newaxis, np.newaxis] / (
        np.sin(phis[np.newaxis, np.newaxis, :]) * np.sin(thetas[np.newaxis, :, np.newaxis]))
    wzs = np.tile(periods[:, np.newaxis, np.newaxis] /
                  (aspect * np.sin(phis[np.newaxis, np.newaxis])), [1, len(thetas), 1])

    if mode in ["stockwell", "gabor"]:
        phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[:, np.newaxis, np.newaxis]
        phase_corr_y = ((2 * np.pi * 1j * dy) * np.arange(-ny // 2, ny // 2))[np.newaxis, :, np.newaxis]
        phase_corr_z = ((2 * np.pi * 1j * dz_p) * np.arange(-nz // 2, nz // 2))[np.newaxis, np.newaxis, :]

    result = {
        _name: np.full_like(data, np.nan) for _name in [
            "energy", "coefficient", "wavelength_x", "wavelength_y", "wavelength_z",
            "phase", "scale", "theta", "period"]}

    # loop through all scales and compute transform
    for a3, a2, a1 in tqdm.tqdm(list(itertools.product(range(jp), range(jt), range(js)))):
        if (("min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(wxs[a1, a2, a3])) or
                ("max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(wxs[a1, a2, a3])) or
                ("min_wavelength_y" in filt and filt["min_wavelength_y"] > abs(wys[a1, a2, a3])) or
                ("max_wavelength_y" in filt and filt["max_wavelength_y"] < abs(wys[a1, a2, a3])) or
                ("min_wavelength_z" in filt and filt["min_wavelength_z"] > abs(wzs[a1, a2, a3])) or
                ("max_wavelength_z" in filt and filt["max_wavelength_z"] < abs(wzs[a1, a2, a3]))):
            continue

        if mode == "gabor":
            raise NotImplementedError
        else:
            daughter = morlet3d_fourier(
                nxpad, nypad, nzpad, dx, dy, dz_p,
                scales[a1], thetas[a2], phis[a3], opts["param"], normed=(mode == "cwt"))

        coeff = fft.ifftn(f_hat * daughter)[:nx, :ny, :nz]

        if mode in ["stockwell", "gabor"]:
            coeff /= np.exp(
                (phase_corr_x * np.cos(phis[a3]) * np.cos(thetas[a2]) +
                 phase_corr_y * np.cos(phis[a3]) * np.sin(thetas[a2]) +
                 phase_corr_z * np.sin(phis[a3])) / periods[a1])
        elif mode in ["cwt"]:
            coeff *= dx * dy * dz_p

        sel = abs(coeff) > filt["min_coefficient"]
        sel &= (scales[a1] * coeff > result["energy"]) | (~np.isfinite(result["energy"]))
        result["coefficient"][sel] = coeff[sel]
        result["energy"][sel] = scales[a1] * coeff[sel]
        result["phase"][sel] = np.angle(coeff[sel])
        result["scale"][sel] = scales[a2]
        result["theta"][sel] = thetas[a1]
        result["period"][sel] = periods[a2]
        result["wavelength_x"][sel] = wxs[a1, a2]
        result["wavelength_y"][sel] = wys[a1, a2]
        result["wavelength_z"][sel] = wzs[a1, a2]

    return result


def reconstruct3d(decomposition):
    """
    Reconstructs from a 3-D wavelet decomposition

    Parameters
    ----------
    decomposition : dict
        decomposition

    Returns
    -------
    The reconstructed field
    """

    coeff, scales, periods, thetas, phis, dx, dy, dz, dj, js, jt, jp, aspect, mode = [
        decomposition[_x] for _x in [
            "decomposition", "scale", "period", "theta", "phi",
            "dx", "dy", "dz", "dj", "js", "jt", "jp", "aspect", "mode"]]
    # discritized integration and c_delta
    rec_fac = 2 * np.log(2) * (np.pi / jp) * (np.pi / jt) * dj / compute_c_delta(3, decomposition["opts"]["param"])
    if mode == "cwt":
        rec_fac /= PSI_0 * np.sqrt(2) * np.pi
        coeff = coeff.real / (scales[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis] ** 1.5)
    elif mode == "scaled":
        pass
    elif mode == "gabor":
        raise NotImplementedError
    elif mode == "stockwell":
        nx, ny, nz = coeff.shape[3:]
        dz_p = dz * aspect
        phase_corr_x = ((2 * np.pi * 1j * dx) * np.arange(-nx // 2, nx // 2))[:, np.newaxis, np.newaxis]
        phase_corr_y = ((2 * np.pi * 1j * dy) * np.arange(-ny // 2, ny // 2))[np.newaxis, :, np.newaxis]
        phase_corr_z = ((2 * np.pi * 1j * dz_p) * np.arange(-nz // 2, nz // 2))[np.newaxis, np.newaxis, :]
        phi_bc = phis[np.newaxis, np.newaxis, :]
        theta_bc = thetas[np.newaxis, :, np.newaxis]
        periods_bc = periods[:, np.newaxis, np.newaxis]
        coeff = coeff * np.exp(
            (phase_corr_x[np.newaxis, np.newaxis, np.newaxis, :, :, :] *
                (np.cos(phi_bc) * np.cos(theta_bc))[:, :, :, np.newaxis, np.newaxis, np.newaxis] +
             phase_corr_y[np.newaxis, np.newaxis, np.newaxis, :, :, :] *
                (np.cos(phi_bc) * np.sin(theta_bc))[:, :, :, np.newaxis, np.newaxis, np.newaxis] +
             phase_corr_z[np.newaxis, np.newaxis, np.newaxis, :, :, :] *
                np.sin(phi_bc)[:, :, :, np.newaxis, np.newaxis, np.newaxis]) /
            periods_bc[:, :, :, np.newaxis, np.newaxis, np.newaxis])
    else:
        raise NotImplementedError
    result = coeff.real.sum(0).sum(0)
    result *= np.cos(phis)[:, np.newaxis, np.newaxis, np.newaxis]
    return rec_fac * result.sum(0)
