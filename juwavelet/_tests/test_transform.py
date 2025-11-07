# import matplotlib.pyplot as plt
import numpy as np
import pytest

import juwavelet.transform as transform
import juwavelet.morlet as morlet
import juwavelet.utils as utils


_C_PSI = {
    1: {
        4: 1.6254360986894005,
        4.5: 1.4336797751044845,
        5: 1.2834546315099138,
        5.5: 1.162304236537642,
        6: 1.0623947623886103,
        2 * np.pi: 1.0131799006578437,
        6.5: 0.978515872310787,
        7: 0.9070528788610637,
        7.5: 0.8454127825082485,
        8: 0.7916838457102651,
        8.5: 0.7444239007489395,
        9: 0.7025235363512534,
        3 * np.pi: 0.670484530974965,
        9.5: 0.6651147828322892,
        10: 0.631508468556733,
        10.5: 0.6011502143114966,
        11: 0.5735888722948547,
        11.5: 0.5484534718496332,
        12: 0.5254361019732275,
        12.5: 0.5042790143336111,
        4 * np.pi: 0.5015984250609288,
        13: 0.4847647768719809,
        13.5: 0.4667086656665705,
        14: 0.44995272155090643,
        14.5: 0.4343610604106778,
        15: 0.4198161384073333,
        15.5: 0.4062157522369974,
        5 * np.pi: 0.4008155477898055,
        16: 0.3934706106753533,
        16.5: 0.3815023541432945,
        17: 0.370241928569532,
        17.5: 0.35962824162114615,
        18: 0.34960704561814704,
        18.5: 0.3401300036727136,
        6 * np.pi: 0.33380440728082655,
    },
    2: {
    },
    3: {
    }
}


SST = np.asarray([
    -0.15, -0.30, -0.14, -0.41, -0.46, -0.66, -0.50, -0.80, -0.95,
    -0.72, -0.31, -0.71, -1.04, -0.77, -0.86, -0.84, -0.41, -0.49,
    -0.48, -0.72, -1.21, -0.80, 0.16, 0.46, 0.40, 1.00, 2.17,
    2.50, 2.34, 0.80, 0.14, -0.06, -0.34, -0.71, -0.34, -0.73,
    -0.48, -0.11, 0.22, 0.51, 0.51, 0.25, -0.10, -0.33, -0.42,
    -0.23, -0.53, -0.44, -0.30, 0.15, 0.09, 0.19, -0.06, 0.25,
    0.30, 0.81, 0.26, 0.10, 0.34, 1.01, -0.31, -0.90, -0.73,
    -0.92, -0.73, -0.31, -0.03, 0.12, 0.37, 0.82, 1.22, 1.83,
    1.60, 0.34, -0.72, -0.87, -0.85, -0.40, -0.39, -0.65, 0.07,
    0.67, 0.39, 0.03, -0.17, -0.76, -0.87, -1.36, -1.10, -0.99,
    -0.78, -0.93, -0.87, -0.44, -0.34, -0.50, -0.39, -0.04, 0.42,
    0.62, 0.17, 0.23, 1.03, 1.54, 1.09, 0.01, 0.12, -0.27,
    -0.47, -0.41, -0.37, -0.36, -0.39, 0.43, 1.05, 1.58, 1.25,
    0.86, 0.60, 0.21, 0.19, -0.23, -0.29, 0.18, 0.12, 0.71,
    1.42, 1.59, 0.93, -0.25, -0.66, -0.95, -0.47, 0.06, 0.70,
    0.81, 0.78, 1.43, 1.22, 1.05, 0.44, -0.35, -0.67, -0.84,
    -0.66, -0.45, -0.12, -0.20, -0.16, -0.47, -0.52, -0.79, -0.80,
    -0.62, -0.86, -1.29, -1.04, -1.05, -0.75, -0.81, -0.90, -0.25,
    0.62, 1.22, 0.96, 0.21, -0.11, -0.25, -0.24, -0.43, 0.23,
    0.67, 0.78, 0.41, 0.98, 1.28, 1.45, 1.02, 0.03, -0.59,
    -1.34, -0.99, -1.49, -1.74, -1.33, -0.55, -0.51, -0.36, -0.99,
    0.32, 1.04, 1.41, 0.99, 0.66, 0.50, 0.22, 0.71, -0.16,
    0.38, 0.00, -1.11, -1.04, 0.05, -0.64, -0.34, -0.50, -1.85,
    -0.94, -0.78, 0.29, 0.27, 0.69, -0.06, -0.83, -0.80, -1.02,
    -0.96, -0.09, 0.62, 0.87, 1.03, 0.70, -0.10, -0.31, 0.04,
    -0.46, 0.04, 0.24, -0.08, -0.28, 0.06, 0.05, -0.31, 0.11,
    0.27, 0.26, 0.04, 0.12, 1.11, 1.53, 1.23, 0.17, -0.18,
    -0.56, 0.05, 0.41, 0.22, 0.04, -0.19, -0.46, -0.65, -1.06,
    -0.54, 0.14, 0.25, -0.21, -0.73, -0.43, 0.48, 0.26, 0.05,
    0.11, -0.27, -0.08, -0.10, 0.29, -0.15, -0.28, -0.55, -0.44,
    -1.40, -0.55, -0.69, 0.58, 0.37, 0.42, 1.83, 1.23, 0.65,
    0.41, 1.03, 0.64, -0.07, 0.98, 0.36, -0.30, -1.33, -1.39,
    -0.94, 0.34, -0.00, -0.15, 0.06, 0.39, 0.36, -0.49, -0.53,
    0.35, 0.07, -0.24, 0.20, -0.22, -0.68, -0.44, 0.02, -0.22,
    -0.30, -0.59, 0.10, -0.02, -0.27, -0.60, -0.48, -0.37, -0.53,
    -1.35, -1.22, -0.99, -0.34, -0.79, -0.24, 0.02, 0.69, 0.78,
    0.17, -0.17, -0.29, -0.27, 0.31, 0.44, 0.38, 0.24, -0.13,
    -0.89, -0.76, -0.71, -0.37, -0.59, -0.63, -1.47, -0.40, -0.18,
    -0.37, -0.43, -0.06, 0.61, 1.33, 1.19, 1.13, 0.31, 0.14,
    0.03, 0.21, 0.15, -0.22, -0.02, 0.03, -0.17, 0.12, -0.35,
    -0.06, 0.38, -0.45, -0.32, -0.33, -0.49, -0.14, -0.56, -0.18,
    0.46, 1.09, 1.04, 0.23, -0.99, -0.59, -0.92, -0.28, 0.52,
    1.31, 1.45, 0.61, -0.11, -0.18, -0.39, -0.39, -0.36, -0.50,
    -0.81, -1.10, -0.29, 0.57, 0.68, 0.78, 0.78, 0.63, 0.98,
    0.49, -0.42, -1.34, -1.20, -1.18, -0.65, -0.42, -0.97, -0.28,
    0.77, 1.77, 2.22, 1.05, -0.67, -0.99, -1.52, -1.17, -0.22,
    -0.04, -0.45, -0.46, -0.75, -0.70, -1.38, -1.15, -0.01, 0.97,
    1.10, 0.68, -0.02, -0.04, 0.47, 0.30, -0.55, -0.51, -0.09,
    -0.01, 0.34, 0.61, 0.58, 0.33, 0.38, 0.10, 0.18, -0.30,
    -0.06, -0.28, 0.12, 0.58, 0.89, 0.93, 2.39, 2.44, 1.92,
    0.64, -0.24, 0.27, -0.13, -0.16, -0.54, -0.13, -0.37, -0.78,
    -0.22, 0.03, 0.25, 0.31, 1.03, 1.10, 1.05, 1.11, 1.28,
    0.57, -0.55, -1.16, -0.99, -0.38, 0.01, -0.29, 0.09, 0.46,
    0.57, 0.24, 0.39, 0.49, 0.86, 0.51, 0.95, 1.25, 1.33,
    -0.00, 0.34, 0.66, 1.11, 0.34, 0.48, 0.56, 0.39, -0.17,
    1.04, 0.77, 0.12, -0.35, -0.22, 0.08, -0.08, -0.18, -0.06])


def _compute_c_psi(dims, param):
    if param in _C_PSI[dims]:
        return _C_PSI[dims][param]
    if dims == 1:
        oms = np.arange(1, 4000) / 1000
        ddx = 0.005
        mr = np.fft.fftshift(np.fft.ifft(morlet.morlet1d_fourier(2 * 4096, ddx, 1, param, normed=True)), 0)
        xs = (np.arange(2 * 4096) - 2 * 2048) * ddx
        c_psi = []
        for om in oms:
            c_psi.append(np.abs(ddx * (mr * np.exp(-2 * np.pi * 1j * om * xs)).sum())**2 / om)
        c_psi = sum(c_psi)
        c_psi *= (oms[1] - oms[0])
    else:
        assert False
    _C_PSI[dims][param] = c_psi
    return c_psi


def _reconstruct1d(decomposition):
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

    param = decomposition["opts"]["param"]
    rec_fac = 2 * dj * np.log(2) / _compute_c_psi(1, param)
    assert mode == "cwt"

    rec1 = np.zeros(coeff.shape[1])
    for i, s in enumerate(scales):
        mr = np.fft.fftshift(np.fft.ifft(morlet.morlet1d_fourier(coeff.shape[1], dx, s, param, normed=True)), 0)
        rec1 += dx * np.convolve(coeff[i] / scales[i], mr, mode="same").real

    return rec_fac * rec1


def test_wavelet1d_sst():
    dx = 0.25
    s0 = dx
    dj = 0.25
    jtot = int(11 / dj)

    dec = transform.decompose1d(SST, dx, s0, dj, jtot, mode="cwt")
    rec = transform.reconstruct1d(dec)

    assert abs(rec - SST).max() < 0.02


@pytest.mark.parametrize("amp", [5, 10])
@pytest.mark.parametrize("wl", [67.4, 116.8])
@pytest.mark.parametrize("dx", [1, 4])
@pytest.mark.parametrize("param", [5, 6, 2 * np.pi, 8, 12, 18])
@pytest.mark.parametrize("mode", ["cwt", "scaled", "gabor", "stockwell"])
def test_decompose1d(amp, wl, dx, param, mode):
    print(amp, wl, dx, param, mode)
    n = 4096
    s0 = 2 * dx
    dj = 0.1
    js = int(13 / dj)
    xs = np.arange(n) * dx
    ys = amp * np.sin(2 * np.pi * xs / wl)

    dec = transform.decompose1d(
        ys, dx, s0, dj=dj, js=js, mode=mode, opts={"param": param})

    if mode in ["scaled", "gabor", "stockwell"]:
        i0 = abs(dec["period"] - wl).argmin()
        assert abs(dec["decomposition"][i0]).max() >= abs(dec["decomposition"]).max()
        assert abs((abs(dec["decomposition"]).max() / amp) - 1) < 0.05

    if mode == "gabor":
        pytest.skip("not implemented yet")

    rec = transform.reconstruct1d(dec)

    if mode in ["cwt"]:
        c_psi = _compute_c_psi(1, param)
        rec1 = _reconstruct1d(dec)

        # plt.plot(ys, label="orig")
        # plt.plot(rec, label="dut1")
        # plt.plot(rec1, label="dut2")
        # plt.legend()
        # plt.show()

        # test power conservation
        power_r = np.sum(ys ** 2) * dx
        power_cwt = (2 / c_psi) * dx * dj * np.log(2) * \
            np.sum(np.abs(dec["decomposition"]) ** 2 / dec["scale"][:, None])
        fac = ys[500:-500].max() / rec1[500:-500].max()
        print(f"REC {param:6.2f} ", c_psi, fac, 1 / fac)
        print(f"POW {param:6.2f} ", power_cwt / power_r)

        assert abs(1 - fac) < 0.05
        assert abs(1 - (power_cwt / power_r)) < 0.05

    # import matplotlib.pyplot as plt
    # plt.plot(dec["decomposition"][50], label="orig")
    # plt.plot(rec, label="dut1")
    # plt.plot(rec2, label="dut2")
    # plt.legend()
    # plt.show()

    sel = (ys != 0)
    assert abs(np.median(rec[sel] / ys[sel]) - 1) < 0.05
    assert abs(rec - ys)[200:-200].max() < 0.05 * amp


@pytest.mark.parametrize("aspect", (1, 10))
@pytest.mark.parametrize("wave_param", ((5, 2, 2), (1, 3, 2), (1, 1, 3)))
@pytest.mark.parametrize("mode", ["scaled", "cwt", "stockwell"])
@pytest.mark.parametrize("k", [5, 2 * np.pi, 3 * np.pi])
@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi / 8])
def test_decompose2d(aspect, wave_param, mode, k, angle):
    print("aspect", aspect)
    print("wave_param", wave_param)
    print("mode", mode)
    print("k", k)
    print("angle", angle)

    amp, dx, dy = wave_param
    dy = dy / aspect
    xs = np.arange(0, 220., dx)
    ys = np.arange(0, 200 / aspect, dy)
    YS, XS = np.meshgrid(ys, xs)
    param = k
    wl = 40 * (2 * np.pi) / param
    kx = (2 * np.pi / wl) * np.cos(angle)
    ky = (2 * np.pi / wl) * np.sin(angle)
    print(xs, ys)
    print("wl", wl)
    data = amp * np.sin((YS * ky * aspect + XS * kx))

    js, jt, dj = 16, 16, 0.25
    dec = transform.decompose2d(
        data, dx=dx, dy=dy, s0=10, js=js, jt=jt, dj=dj,
        aspect=aspect, mode=mode, opts={"param": param})

    i0 = abs(dec["period"] - wl).argmin()
    i1 = abs(dec["theta"] - angle).argmin()
    print("i0/i1", i0, i1)
    print(wl, dec["period"][i0])
    print(angle, dec["theta"][i1])

    rec = transform.reconstruct2d(dec)
    print("wama", abs(data).max())
    print("rema", abs(rec).max())
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 4)
    # pm = axs[0].pcolormesh(ys, xs, data, vmin=-amp, vmax=amp, shading="auto")
    # plt.colorbar(pm, ax=axs[0])
    # pm = axs[1].pcolormesh(ys, xs, rec, vmin=-amp, vmax=amp, shading="auto")
    # plt.colorbar(pm, ax=axs[1])
    # pm = axs[2].pcolormesh(ys, xs, rec - data, cmap="RdBu", shading="auto")
    # plt.colorbar(pm, ax=axs[2])
    # nx = len(xs)
    # ny = len(ys)
    # scale = 40
    # mor = morlet.morlet2d(nx, ny, dx, dy, scale, angle, k, normed=False).real
    # print(data.shape, mor.shape)
    # pm = axs[3].pcolormesh(ys, xs, mor, cmap="RdBu")
    # plt.colorbar(pm, ax=axs[3])
    # plt.show()
    print(np.median(rec / data))
    print("Period", dec["period"])
    print("Angles", dec["theta"])
    print("max/rec/amp", abs(dec["decomposition"]).max(), abs(rec).max(), amp)
    # for ii0 in range(len(dec["period"])):
    #     for ii1 in range(len(dec["theta"])):
    #         print(ii0, ii1, abs(dec["decomposition"][ii0, ii1]).max())
    assert abs(dec["decomposition"][i0, i1]).max() >= abs(dec["decomposition"]).max()
    if mode in ["scaled", "gabor", "stockwell"]:
        assert abs((abs(dec["decomposition"]).max() / amp) - 1) < 0.05
    sel = (data != 0)
    print("abs diff", (rec - data).max())
    print("rel diff", np.median(rec[sel] / data[sel]))
    assert abs(np.median(rec[sel] / data[sel]) - 1) < 0.1
    assert np.linalg.norm((rec - data)[10:-10, 10:-10], 2) < \
        0.1 * np.linalg.norm(data[10:-10, 10:-10], 2)


@pytest.mark.parametrize("amp", [5])
@pytest.mark.parametrize("dx", [1])
@pytest.mark.parametrize("dy", [1])
@pytest.mark.parametrize("dz", [1])
@pytest.mark.parametrize("aspect", [1])
@pytest.mark.parametrize("param", [2 * np.pi])
@pytest.mark.parametrize("mode", ["cwt", "stockwell"])
@pytest.mark.parametrize("theta", [0])
@pytest.mark.parametrize("phi", [0])
def test_decompose3d(amp, dx, dy, dz, aspect, param, mode, theta, phi):
    # pytest.skip("too expensive")
    print(amp, dx, dy, dz, aspect, param, mode, theta, phi)
    xs = np.arange(0., 64., dx)
    ys = np.arange(0., 64., dy)
    zs = np.arange(0., 64., dz)
    YS, XS, ZS = np.meshgrid(ys, xs, zs)
    wl = 8 * (2 * np.pi) / param
    kx = (2 * np.pi / wl) * np.cos(phi) * np.cos(theta)
    ky = (2 * np.pi / wl) * np.cos(phi) * np.sin(theta)
    kz = (2 * np.pi / wl) * np.sin(phi)
    print("wl", wl)
    data = amp * np.sin((ZS * kz * aspect + YS * ky + XS * kx))
    # s0, js, jt, jp, dj = 2, 16, 16, 16, 0.25
    s0, js, jt, jp, dj = 2 * np.sqrt(2), 12, 16, 16, 0.25
    # s0, js, jt, jp, dj = 4, 8, 16, 16, 0.25
    # s0, js, jt, jp, dj = 8, 1, 1, 2, 0.25
    print("s0, js, jt, jp, dj", s0, js, jt, jp, dj)
    dec = transform.decompose3d(
        data, dx=dx, dy=dy, dz=dz, s0=s0, js=js, jt=jt, jp=jp, dj=dj, aspect=aspect, mode=mode, opts={"param": param})
    print("period", dec["period"])
    print("theta", dec["theta"])
    print("phi", dec["phi"])

    i0 = abs(dec["period"] - wl).argmin()
    i1 = abs(dec["theta"] - theta).argmin()
    i2 = abs(dec["phi"] - phi).argmin()

    print(i0, i1, i2)
    print(abs(dec["decomposition"]).shape)
    print(abs(dec["decomposition"]).argmax())
    print(dec["wavelength_x"].shape)
    print(dec["wavelength_y"].shape)
    print(dec["wavelength_z"].shape)
    print(dec["wavelength_x"])
    print(dec["wavelength_y"])
    print(dec["wavelength_z"])
    rec = transform.reconstruct3d(dec)
    # plt.figure()
    # plt.pcolormesh(np.angle(dec["decomposition"][i0,i1,i2,:,:,32]))
    # plt.colorbar()
    # plt.show()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3)
    axs = axs.T.reshape(-1)
    data_p = rec
    pm = axs[0].pcolormesh(zs, ys, data[data.shape[0] // 2].real, vmin=-amp, vmax=amp, shading="auto")
    plt.colorbar(pm, ax=axs[0])
    pm = axs[1].pcolormesh(zs, xs, data[:, data.shape[1] // 2].real, vmin=-amp, vmax=amp, shading="auto")
    plt.colorbar(pm, ax=axs[1])
    pm = axs[2].pcolormesh(ys, xs, data[:, :, data.shape[2] // 2].real, vmin=-amp, vmax=amp, shading="auto")
    plt.colorbar(pm, ax=axs[2])
    pm = axs[3].pcolormesh(zs, ys, data_p[data.shape[0] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[3])
    pm = axs[4].pcolormesh(zs, xs, data_p[:, data.shape[1] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[4])
    pm = axs[5].pcolormesh(ys, xs, data_p[:, :, data.shape[2] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[5])
    pm = axs[6].pcolormesh(zs, ys, (data - data_p)[data.shape[0] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[6])

    pm = axs[7].pcolormesh(zs, xs, (data - data_p)[:, data.shape[1] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[7])

    pm = axs[8].pcolormesh(ys, xs, (data - data_p)[:, :, data.shape[2] // 2].real, shading="auto")
    plt.colorbar(pm, ax=axs[8])
    plt.savefig(f"{amp}_{dx}_{dy}_{dz}_{aspect}_{param}_{mode}_{theta}_{phi}.png")
    plt.close()
    print("max,amp", abs(dec["decomposition"]).max(), amp)
    sel = (data != 0)
    sel[:10] = False
    sel[-10:] = False
    sel[:, :10] = False
    sel[:, -10:] = False
    sel[:, :, :10] = False
    sel[:, :, -10:] = False
    print("abs diff", (rec - data)[10:-10, 10:-10, 10:-10].max())
    print("rel diff", np.median((rec[sel] / data[sel])))
    print("max rec", rec.max())
    print("max rec2", rec[10:-10, 10:-10, 10:-10].max())
    print("max rec3", rec[20:-20, 20:-20, 20:-20].max())
    assert abs(dec["decomposition"][i0, i1, i2]).max() >= abs(dec["decomposition"]).max()
    if mode in ["scaled", "stockwell"]:
        assert abs((abs(dec["decomposition"]).max() / amp) - 1) < 0.2
    assert abs(np.median(rec[sel] / data[sel]) - 1) < 0.1
    assert np.linalg.norm((rec - data)[10:-10, 10:-10, 10:-10].reshape(-1), 2) < \
        0.1 * np.linalg.norm(data[10:-10, 10:-10, 10:-10].reshape(-1), 2)


@pytest.mark.parametrize("shape", [100, (100, 100)])
@pytest.mark.parametrize("mode", ["blackman", "kaiser", "cos", "linear"])
def test_tapering(mode, shape):
    data = np.ones(shape)
    result = utils.smooth_edges(data, 20, window=mode)
    assert data.reshape(-1)[0] == 1
    assert result.reshape(-1)[0] < 1
    assert data.reshape(-1)[-1] == 1
    assert result.reshape(-1)[-1] < 1
    utils.smooth_edges(data, 20, window=mode, out=data)
    assert data.reshape(-1)[0] < 1
    assert data.reshape(-1)[-1] < 1
    assert np.allclose(data, result)


def test_ana2d():
    amp, dx, dy, aspect, angle, mode = 5, 2, 2, 10, np.pi / 4, "stockwell"
    xs = np.arange(0, 200., dx)
    ys = np.arange(0, 200 / aspect, dy)
    XS, YS = np.meshgrid(xs, ys)
    param = np.pi
    wl = 40 * (2 * np.pi) / param
    kx = (1 / wl) * np.cos(angle)
    ky = (1 / wl) * np.sin(angle)
    wx = (1 / kx)
    wy = (1 / ky) / aspect
    if (kx > 1e-6) and (ky > 1e-6):
        data = amp * np.sin(2 * np.pi * (YS / wy + XS / wx))
    elif (kx > 1e-6):
        data = amp * np.sin(2 * np.pi * (XS / wx))
    elif (ky > 1e-6):
        data = amp * np.sin(2 * np.pi * (YS / wy))

    print("shape", data.shape, len(xs), len(ys))
    js, jt, dj = 16, 16, 0.25
    dec = transform.decompose2d(
        data, dx=dx, dy=dy, s0=10, js=js, jt=jt, dj=dj,
        aspect=aspect, mode=mode, opts={"param": param})

    utils.analyse2d_dominant(dec)
    transform.decompose2d_dominant(
        data, dx=dx, dy=dy, s0=10, js=js, jt=jt, dj=dj,
        aspect=aspect, mode=mode, opts={"param": param})
