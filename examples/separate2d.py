import copy

from juwavelet import transform, utils

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': [2 * 6.94, 2 * 4.29],
})

titlebox = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
letterbox = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

label_x1, label_y1 = 0.08, 0.13


def sphere_wave(X, Y, x0, y0, lh):
    """
    Computes a spherical wave

    Parameters
    ----------
    X, Y : arrays of x- and y-coordinates
    x0, y0 : ints of x- and y-offests
    lh : float defining the wavelength
    """

    k = 2 * np.pi / lh
    signal = np.cos(k * np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)) * \
        (np.exp(-(X - x0)**2 / (4 * lh**2) - (Y - y0)**2 / (4 * lh**2)) ** 0.5)

    return signal


def wavepaket(X, Y, x0, y0, sx, sy, lh, theta):
    """
    Computes a Gaussian wave paket

    Parameters
    ----------
    X, Y : arrays of x- and y-coordinates
    x0, y0 : ints of x- and y-offests
    sx, sy : floats defining the width of the Gaussian
    lh : float defining the wavelength
    theta: float defining the orientation (clockwise from 12o'clock)
    """

    kx = 2 * np.pi / lh * np.sin(theta)
    ky = 2 * np.pi / lh * np.cos(theta)
    oscillation = np.cos(kx * (X - x0) + ky * (Y - y0))
    envelope = np.exp(-(X - x0)**2 / (2 * sx**2) - (Y - y0)**2 / (2 * sy**2))

    return oscillation * envelope


def extract_max_real_component(decomposition):
    """
    Extracts the real part of the max-energy complex value across (scale_dim, theta_dim) for each (x_dim, y_dim) position
    in the 'decomposition' array.

    Parameters:
        decomposition (dict): Dictionary containing a 'decomposition' key with a 4D complex-valued ndarray.

    Returns:
        np.ndarray: 2D array of shape (x_dim, y_dim) with the real part of the selected complex entries.
    """

    decomp_copy = copy.deepcopy(decomposition)
    scale_dim, theta_dim, x_dim, y_dim = decomp_copy['decomposition'].shape
    abs_decomp = np.abs(decomp_copy['decomposition'])
    abs_decomp_flat = abs_decomp.reshape(scale_dim * theta_dim, x_dim, y_dim)
    idx_decomp_flat = abs_decomp_flat.argmax(axis=0)
    i_idx, j_idx = np.unravel_index(idx_decomp_flat, (scale_dim, theta_dim))
    k_idx, l_idx = np.indices((x_dim, y_dim))
    dominant_modes = decomp_copy['decomposition'][i_idx, j_idx, k_idx, l_idx]

    return np.real(dominant_modes)


x = np.linspace(0, 100, 201)
y = np.linspace(0, 100, 201)
dx, dy = 0.5, 0.5
nx, ny = 201, 201

X, Y = np.meshgrid(x, y)

x0 = [50, 60, 30, 70, 80, 25, 65, 22]
y0 = [50, 70, 60, 40, 80, 80, 20, 22]
lh0 = [6, 3, 8, 14, 13, 20, 25, 5]
th0 = [60, 0, 110, 10, 110, 160, 20, np.nan]

wp_1 = wavepaket(X, Y, 50, 50, 7, 10, 6, np.deg2rad(60))
wp_2 = wavepaket(X, Y, 60, 70, 4, 4, 3, np.deg2rad(0))
wp_3 = wavepaket(X, Y, 30, 60, 20, 5, 8, np.deg2rad(110))
wp_4 = wavepaket(X, Y, 70, 40, 10, 10, 14, np.deg2rad(10))
wp_5 = wavepaket(X, Y, 80, 80, 15, 15, 13, np.deg2rad(110))
wp_6 = wavepaket(X, Y, 25, 80, 20, 5, 20, np.deg2rad(160))
wp_7 = wavepaket(X, Y, 65, 20, 20, 5, 25, np.deg2rad(20))
wp_8 = 2 * sphere_wave(X, Y, 22, 22, 5)

wavefield = wp_1 + wp_2 + wp_3 + wp_4 + wp_5 + wp_6 + wp_7 + wp_8
s0 = 2 * dx
dj = 1 / 16
js = int(1 / dj * np.log2(ny / s0))
jt = 18

cwt_result = transform.decompose2d(wavefield, dx, dy, s0, dj, js, jt, aspect=1,
                                   nxpad=None, nypad=None, opts={'param': 2 * np.pi},
                                   mode="scaled", dtype=np.complex128)

dominant = utils.analyse2d_dominant(cwt_result)
lh, ori, amp = [dominant[_x] for _x in ["period", "theta", "coefficient"]]

peak_lh, peak_ori = [], []
for wp in range(8):
    peak_lh = np.append(peak_lh, lh[2 * y0[wp] - 1, 2 * x0[wp] - 1])
    peak_ori = np.append(peak_ori, ori[2 * y0[wp] - 1, 2 * x0[wp] - 1])

recon = extract_max_real_component(cwt_result)

letter = ['a', 'b', 'c', 'd', 'e', 'f']

num_levels = 11

fig, axes = plt.subplots(2, 3, sharey=False, sharex=False)
plot_field = axes[0, 0].pcolormesh(X, Y, wavefield, cmap="RdBu_r", rasterized=True, vmin=-1, vmax=1)
cbar_field = fig.colorbar(plot_field, ax=axes[0, 0], location="right", ticks=[-1, -0.5, 0, 0.5, 1],
                          label="Perturbation")

plot_recon = axes[0, 1].pcolormesh(X, Y, recon, cmap="RdBu_r", rasterized=True, vmin=-1, vmax=1)
cbar_recon = fig.colorbar(plot_recon, ax=axes[0, 1], location="right", ticks=[-1, -0.5, 0, 0.5, 1],
                          label="Perturbation")

plot_amp = axes[0, 2].contourf(X, Y, amp, np.linspace(0, 1, num_levels), rasterized=True, cmap="plasma_r")
cbar_amp = fig.colorbar(plot_amp, ax=axes[0, 2], location="right", ticks=[0, 0.25, 0.5, 0.75, 1],
                        label="Amplitude")

plot_lh = axes[1, 0].contourf(X, Y, lh, levels=np.linspace(0, 25, num_levels), cmap="turbo", extend="neither")
cbar_lh = fig.colorbar(plot_lh, ax=axes[1, 0], location="right", ticks=[0, 5, 10, 15, 20, 25],
                       extend="both", label="Wavelength")

plot_dir = axes[1, 1].contourf(X, Y, ori, levels=np.linspace(0, np.pi, num_levels), cmap="twilight", extend="neither")
cbar_dir = fig.colorbar(plot_dir, ax=axes[1, 1], location="right", ticks=[0, np.pi / 2, np.pi],
                        extend="both", label="Orientation")
cbar_dir.set_ticks(ticks=[0, np.pi / 2, np.pi], labels=(['0°', '90°', '180°']))
axes[1, 1].quiver(X[0::10, 0::10], Y[0::10, 0::10], np.sin(ori[0::10, 0::10]), np.cos(ori[0::10, 0::10]),
                  angles='xy', scale_units='xy', scale=0.3)


axes[1, 2].scatter(lh0, peak_lh, s=5)
axes[1, 2].plot([0, 30], [0, 30], c='k')
axes[1, 2].set_xlabel(r'$\lambda_{in}$')
axes[1, 2].set_ylabel(r'$\lambda_{out}$')
axes[1, 2].set_xlim([0, 30])
axes[1, 2].set_ylim([0, 30])
axes[1, 2].grid(which='both')
for k in range(8):
    axes[1, 2].text(lh0[k] + 0.2, peak_lh[k] + 0.2, str(k + 1), c='tab:blue')
axes[1, 2].xaxis.label.set_color('tab:blue')
axes[1, 2].tick_params(axis='x', colors='tab:blue')
axes[1, 2].yaxis.label.set_color('tab:blue')
axes[1, 2].tick_params(axis='y', colors='tab:blue')


theta_ax = axes[1, 2].twinx().twiny()
theta_ax.scatter(th0, np.rad2deg(peak_ori), color='k', s=5)
for k in range(8):
    theta_ax.text(th0[k] + 3, np.rad2deg(peak_ori[k]) + 3, str(k + 1), c='k')
theta_ax.set_xlim([0, 180])
theta_ax.set_ylim([0, 180])
theta_ax.set_xlabel(r'$\theta_{in}$')
theta_ax.set_ylabel(r'$\theta_{out}$')

axes[0, 0].set_ylabel('y')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_xlabel('x')
axes[1, 1].set_xlabel('x')

for i in range(2):
    for j in range(3):
        axes[i, j].grid(which='both')
        axes[i, j].text(label_x1, label_y1, letter[3 * i + j], transform=axes[i][j].transAxes,
                        bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")
        for k in range(8):
            if 3 * i + j > 4:
                continue
            else:
                axes[i, j].text(x0[k], y0[k], str(k + 1), bbox=titlebox,
                                verticalalignment="top", horizontalalignment="right")

plt.tight_layout()
plt.savefig('example_separate2d.png', dpi=120, facecolor="w", edgecolor="w", bbox_inches="tight")
plt.savefig('example_separate2d.pdf', dpi=120, facecolor="w", edgecolor="w", bbox_inches="tight")
plt.show()
