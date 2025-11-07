import string
import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import juwavelet.transform as transform


matplotlib.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': [2 * 6.94, 1.5 * 4.29],
})


def example_decompose1d():
    title = {"cwt": "CWT", "gabor": "Gabor", "stockwell": "Stockwell"}
    amp2, amp3 = 10, 5
    wl1 = 16
    wl2 = 256
    dx = 1
    s0 = 5 * dx
    dj = 0.125
    js = int(7 / dj)
    xs = np.arange(2048)
    ys1 = np.linspace(10, 0, len(xs)) * np.sin(2 * np.pi * xs / wl1)
    ys2 = np.zeros_like(ys1)
    ys2[len(ys2) // 2:] += amp2 * np.cos(2 * np.pi * xs[len(xs) // 2:] / wl2)
    ys3 = amp3 * np.sin(2 * np.pi * xs * np.linspace(1 / 1000, 1 / 40, len(xs)))
    ys = ys1 + ys2 + ys3

    opts = {"window": 100}
    decs = {}
    modes = ["cwt", "gabor", "stockwell"]
    for mode in modes:
        decs[mode] = transform.decompose1d(
            ys, dx=dx, s0=s0, dj=dj, js=js, opts=opts, mode=mode)

    fig, axs = plt.subplots(2, 1 + len(modes))
    axs[0, 0].plot(xs, ys)
    axs[1, 0].plot(xs, ys1)
    axs[1, 0].plot(xs, ys2)
    axs[1, 0].plot(xs, ys3)
    axs[0, 0].set_yticks(np.arange(-20, 21, 5))
    axs[1, 0].set_yticks(np.arange(-10, 11, 5))

    cbs = []
    for j, mode in enumerate(modes):
        i = j + 1
        axs[0, i].set_title(f"{title[mode]}\nMagnitude")
        vmax = 200
        if i > 1:
            vmax = 10
        pm = axs[0, i].pcolormesh(xs, decs[mode]["period"], abs(decs[mode]["decomposition"]),
                                  cmap="turbo", vmin=0, vmax=vmax, rasterized=True)
        axs[0, i].set_ylabel("period")
        plt.colorbar(pm, ax=axs[0, i])
        axs[1, i].set_title("Phase")
        pm = axs[1, i].pcolormesh(xs, decs[mode]["period"], np.angle(decs[mode]["decomposition"]),
                                  vmin=-np.pi, vmax=np.pi, cmap=cmocean.cm.phase, rasterized=True)
        axs[1, i].set_ylabel("period")
        cbs.append(plt.colorbar(pm, ax=axs[1, i]))

    for ax in axs[:, 1:].reshape(-1):
        ax.set_yscale('log')
    for ax in axs.reshape(-1):
        ax.set_xlim(0, 2048)

    for cb in cbs:
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(["$-\pi$", "0", "$\pi$"])

    letterbox = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

    for ax, letter in zip(axs.reshape(-1), string.ascii_lowercase):
        ax.text(0.15, 0.15, letter, transform=ax.transAxes,
                bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")

    fig.tight_layout()
    fig.savefig("example_decompose1d.png")
    fig.savefig("example_decompose1d.pdf")
    plt.close(fig)


if __name__ == "__main__":
    example_decompose1d()
