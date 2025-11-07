import itertools
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid


def cone_of_influence_1d(dec):
    nx = dec["decomposition"].shape[1]
    return dec["dx"] * (2 * np.pi / (dec["opts"]["param"] * np.sqrt(2))) * np.min(
        np.vstack([np.arange(nx), np.arange(nx)[::-1]]), axis=0)


def plot_decomposition2d(
        decomposition,
        redux_s=1, redux_t=1, norm=None, cmap=None, vmax=None,
        func=abs, sel_s=None, sel_t=None):
    """
    plots the 2-D Morlet wavelet decomposition of a regularliy sampled 2-D array

    Parameters
    ----------
    decomposition : dictionary
    redux_s : int
        how many scales to skip in plot
    redux_t : int
        how many angles to skip in plot
    sel_s : list of int
        ignore redux_s, provide an explicit list of scale indexes to plot
    sel_t : list of int
        ignore redux_t, provide an explicit list of angle indexes to plot
    cmap : colormap (optional)
        matplotlib colormap
    vmax : float (optional)
        maximum value for imshow plots

    Returns
    -------
    fig
        figure for saving
    """

    coeff, scale, theta, period, aspect = [
        decomposition[_x] for _x in [
            "decomposition", "scale", "theta",
            "period", "aspect"]]

    if sel_t is None:
        sel_t = list(range(0, coeff.shape[1], redux_t))
    if sel_s is None:
        sel_s = list(range(0, coeff.shape[0], redux_s))
    shape = (len(sel_s), len(sel_t))
    fig = plt.figure(figsize=(shape[1], shape[0]))
    grid = ImageGrid(fig, 111, nrows_ncols=shape,
                     share_all=False, aspect=False, axes_pad=0.02,
                     cbar_mode="single", cbar_size='2%')
    axs = np.asarray(grid.axes_all).reshape(shape)
    for i in range(axs.shape[0]):
        axs[i, 0].set_ylabel(f"{sel_s[i]}\n{period[sel_s[i]]:.0f}")
    for j in range(axs.shape[1]):
        hl = 1. / np.sin(theta[sel_t[j]])
        vl = 1. / np.cos(theta[sel_t[j]])
        if abs(vl) > 1000:
            vl = np.inf
        if abs(hl) > 1000:
            hl = np.inf
        axs[-1, j].set_xlabel(
            f"{sel_t[j]}\n"
            f"{np.rad2deg(theta[sel_t[j]]):.0f}°\n"
            f"{hl:.2f}\n"
            f"{1 / (aspect * np.cos(theta[sel_t[j]])):.3f}")
    n = 0
    if vmax is None:
        vmax = abs(func(coeff)).max()
    if cmap is None:
        cmap = matplotlib.cm.get_cmap("plasma")
    if norm is None:
        norm = matplotlib.colors.BoundaryNorm(np.linspace(0, vmax, 11), cmap.N)
    for i in tqdm.tqdm(range(0, shape[0], 1)):
        for j in range(0, shape[1], 1):
            pm = axs[i, j].imshow(
                func(coeff[sel_s[i], sel_t[j]]),
                rasterized=True, origin="lower", norm=norm, cmap=cmap)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            n += 1
    grid.cbar_axes[0].colorbar(pm)
    fig.tight_layout()
    return fig, grid


def identify_cluster2d(decomposition, min_amp=2, thr=0.5):
    """
    Identify clusters of high magnitude

    Parameters
    ----------
    decomposition : dictionary
    min_amp : float
        smallest amplitude to consider a new cluster
    thr : float
        smallest amplitude to gather up into cluster

    Returns
    -------
    list
        maximal amplitude of cluster
    ndarray
        array with number of cluster at its position in the
        wavelet decomposition
    """

    adec = abs(decomposition["decomposition"])
    shape = adec.shape
    iwave = np.full_like(adec, -1, dtype=np.int16)
    amps = []
    idxs = []

    sel = np.zeros(shape, dtype=bool)

    sel1 = np.zeros((shape[0], shape[1]), dtype=bool)
    new1 = np.zeros((shape[0], shape[1]), dtype=bool)

    sel2 = np.zeros((shape[2], shape[3]), dtype=bool)
    new2 = np.zeros((shape[2], shape[3]), dtype=bool)

    for j in itertools.count():
        wmax = adec.max()
        if wmax < min_amp:
            break
        amps.append(wmax)
        udx = np.unravel_index(adec.argmax(), adec.shape)
        idxs.append(udx)
        print("------------")
        print("cluster=", j)
        print("amplitude=", wmax, "idx=", udx)
        sel[:] = False
        sel[udx] = True
        last_max = wmax

        awave2 = adec[:, :, udx[2], udx[3]]
        min_sel = awave2 > thr / 2
        sel1[:] = False
        sel1[udx[0], udx[1]] = True
        for i in itertools.count():
            new1[:] = False
            new1[1:, :] |= sel1[:-1, :]
            new1[:-1, :] |= sel1[1:, :]
            new1[:, 1:] |= sel1[:, :-1]
            new1[:, :-1] |= sel1[:, 1:]
            new1 &= min_sel
            new1 &= ~sel1
            new1 &= awave2 <= last_max
            if new1.sum() == 0:
                break
            new_last_max = awave2[new1].max()
            assert new_last_max <= last_max
            last_max = new_last_max
            if new1.sum() == 0:
                break
            sel1 |= new1

        for ix, iy in zip(*np.where(sel1)):
            sel[ix, iy, udx[2], udx[3]] = True
        for ix, iy in zip(*np.where(sel1)):
            sel2[:] = sel[ix, iy]
            awave2 = adec[ix, iy]
            min_sel = awave2 > thr
            last_max = awave2.max()
            for i in itertools.count():
                new2[:] = sel2
                new2[1:, :] |= sel2[:-1, :]
                new2[:-1, :] |= sel2[1:, :]
                new2[:, 1:] |= sel2[:, :-1]
                new2[:, :-1] |= sel2[:, 1:]
                new2 &= min_sel
                last_max = wmax
                new2 &= awave2 <= last_max
                new2 &= ~sel2
                if new2.sum() == 0:
                    break
                new_last_max = awave2[new2].max()
                assert new_last_max <= last_max
                last_max = new_last_max
                sel2 |= new2
            sel[ix, iy, :, :] = sel2

        print("selected elements:", sel.sum())
        adec[sel] = 0
        iwave[sel] = j

    return amps, idxs, iwave


def smooth_edges(data, width, window="blackman", out=None):
    """
    Tapering of edges of data to reduce boundary effects. Works in n-dimensions.


    Parameters
    ----------
    data : numpy array
        data to be tapered
    width: int
        number of points on the borders to user for tapering
    window: str
        taper-window to be used. Options are blackman, kaiser, linear, and cos.
    out: None or numpy array
        array to use for storing the result. Supply data again to modify the data in-place.
        Must be of same size as data.

    Returns
    -------
    array
         tapered data

    """
    if window == "blackman":
        tapering = np.blackman(2 * width)[:width]
    elif window == "kaiser":
        tapering = np.kaiser(2 * width, 14)[:width]
    elif window == "linear":
        tapering = np.linspace(0, 1, width + 1)[:-1]
    elif window == "cos":
        tapering = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, width + 1)[:-1])
    else:
        raise RuntimeError
    if out is None:
        result = data.copy()
    else:
        result = out
        result[:] = data
    for dim in range(len(data.shape)):
        temp = result.swapaxes(dim, -1)
        temp[..., :width] *= tapering
        temp[..., -width:] *= tapering[::-1]

    return result


def analyse2d_dominant(dec, filt=None, energy=False):
    """Analyses 2-D wavelet decomposition for dominant waves

    Return set of 2-D fields containing parameters for the wavelet
    identified with the largest energy. Mostly this should simplify
    visualising the many 2-D fields by collapsing relevant information
    to single 2-D arrays.

    Args:
        dec (dict): 2-D wavelet decomposition
        filt (dict): several filter options. keys are
            "min_coefficient": minimum abs coefficient to consider
            "min_wavelength_x": minimal wavelength in x-direction to consider
            "max_wavelength_x": maximum wavelength in x-direction to consider
            "min_wavelength_y": minimal wavelength in y-direction to consider
            "max_wavelength_y": maximum wavelength in y-direction to consider
        energy: bool
            whether to use energy instead of coefficient magnitude for selection

    Returns:
        dict: dictionary with 2-D fields of attributes associated with the
            non-filtered wavelet with the maximum energy.
    """
    coeff = np.abs(dec["decomposition"])
    aspect = dec["aspect"]

    if filt is None:
        filt = {}
    filt.setdefault("min_coefficient", 0)

    result = {_name: np.full((coeff.shape[2], coeff.shape[3]), np.nan) for _name in [
        "energy", "coefficient", "wavelength_x", "wavelength_y", "phase", "scale", "theta", "period"]}
    for ip, period in enumerate(dec["period"]):
        for it, theta in enumerate(dec["theta"]):
            wx = period / (np.cos(theta))
            wy = period / (aspect * np.sin(theta))
            if "min_wavelength_y" in filt and filt["min_wavelength_y"] > abs(wy):
                continue
            if "max_wavelength_y" in filt and filt["max_wavelength_y"] < abs(wy):
                continue
            if "min_wavelength_x" in filt and filt["min_wavelength_x"] > abs(wx):
                continue
            if "max_wavelength_x" in filt and filt["max_wavelength_x"] < abs(wx):
                continue
            sel = abs(coeff[ip, it]) >= filt["min_coefficient"]
            if energy:
                sel &= (dec["scale"][ip] * coeff[ip, it] > result["energy"]) | ~np.isfinite(result["energy"])
            else:
                sel &= (coeff[ip, it] > result["coefficient"]) | ~np.isfinite(result["coefficient"])

            result["coefficient"][sel] = coeff[ip, it][sel]
            result["energy"][sel] = dec["scale"][ip] * coeff[ip, it][sel]
            result["phase"][sel] = np.angle(coeff[ip, it][sel])
            result["scale"][sel] = dec["scale"][ip]
            result["theta"][sel] = theta
            result["period"][sel] = period
            result["wavelength_x"][sel] = wx
            result["wavelength_y"][sel] = wy

    return result
