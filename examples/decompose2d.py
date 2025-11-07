import os
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from juwavelet import transform, utils


matplotlib.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': [2 * 6.94, 2 * 4.29],
})


def example_decompose2d():
    storage = np.loadtxt(os.path.join(os.path.dirname(__file__), "alima.txt"))
    xs, ys, wave = storage[0, 1:], storage[1:, 0], storage[1:, 1:].T
    xs -= xs[0]

    dx = np.diff(xs).mean()
    dy = np.diff(ys).mean()
    cwt = transform.decompose2d(
        wave, dx=dx, dy=dy,
        s0=20, dj=0.25, js=20, jt=18, aspect=40)

    amps, idxs, iwave = utils.identify_cluster2d(
        cwt, min_amp=2.0, thr=1.0)

    decomposition, period, theta = [
        cwt[_x] for _x in ["decomposition", "period", "theta"]]

    orig = decomposition.copy()

    cmap = matplotlib.cm.turbo
    norm = matplotlib.colors.BoundaryNorm(
        np.exp(np.linspace(np.log(0.25), np.log(5.5), 10)), cmap.N)
    fig1, _ = utils.plot_decomposition2d(
        cwt, redux_s=2, redux_t=2, cmap=cmap, norm=norm)

    fig2, axs = plt.subplots(2, 3)
    axs = axs.T
    opts = {"cmap": "RdBu_r", "vmin": -5, "vmax": 5, "rasterized": True}
    axs[0, 0].set_title("original")
    axs[0, 0].pcolormesh(xs, ys, wave.T, **opts)

    decomposition[:] = orig
    decomposition[:, (np.pi / 2 < theta)] = 0
    rec = transform.reconstruct2d(cwt)
    axs[1, 0].set_title("left slanted")
    axs[1, 0].pcolormesh(xs, ys, rec.T, **opts)

    decomposition[:] = orig
    decomposition[:, (theta <= np.pi / 2)] = 0
    rec = transform.reconstruct2d(cwt)
    axs[2, 0].set_title("right slanted")
    axs[2, 0].pcolormesh(xs, ys, rec.T, **opts)

    decomposition[:] = orig
    decomposition[period < 100] = 0
    rec = transform.reconstruct2d(cwt)
    axs[0, 1].set_title("low pass")
    axs[0, 1].pcolormesh(xs, ys, rec.T, **opts)

    for idx, ax in ((1, axs[1, 1]), (7, axs[2, 1])):
        decomposition[:] = orig
        decomposition[iwave != idx] = 0
        udx = idxs[idx]
        rec = transform.reconstruct2d(cwt)
        ax.set_title(
            f"$\\lambda_x$={cwt['wavelength_x'][udx[0], udx[1]]:3.0f}km "
            f"$\\lambda_z$={cwt['wavelength_y'][udx[0], udx[1]]:3.0f}km")
        ax.pcolormesh(xs, ys, rec.T, **opts)
        decomposition[:] = abs(decomposition)
        max_idx = np.unravel_index(decomposition[udx[0], udx[1]].argmax(), decomposition[0, 0].shape)
        ax.plot(xs[max_idx[0]], ys[max_idx[1]], "o", color="w", mec="k", ms=10)
        print(idx, decomposition[udx[0], udx[1]].max(), decomposition[udx[0], udx[1], max_idx[0], max_idx[1]], decomposition.max())

    for ax in axs[:, 1]:
        ax.set_xlabel("distance (km)")
    for ax in axs[0, :]:
        ax.set_ylabel("altitude (km)")

    letterbox = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

    for ax, letter in zip(axs.T.reshape(-1), string.ascii_lowercase):
        ax.text(0.12, 0.15, letter, transform=ax.transAxes,
                bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")
    fig2.tight_layout()

    fig1.savefig("example_decompose2d_a.png")
    fig1.savefig("example_decompose2d_a.pdf")
    fig2.savefig("example_decompose2d_b.png")
    fig2.savefig("example_decompose2d_b.pdf")
    # plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    example_decompose2d()
