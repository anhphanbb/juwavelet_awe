import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import juwavelet.morlet as wavelet


matplotlib.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': [2 * 6.94, 4.29],
})


def example_plot_wavelet2d():
    dx = 0.1
    xs = np.arange(-30, 30, dx)
    period = 10
    param = 2 * np.pi

    fig, axs = plt.subplots(1, 3)
    for ax in axs:
        ax.set_aspect(1)
    period, theta = 10, 0
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=dx,
        scale=period, theta=theta, param=param)
    wave /= wave.max()
    axs[0].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[0].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)
    period, theta = 10, np.pi / 4
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=dx,
        scale=period, theta=theta, param=param)
    wave /= wave.max()
    axs[1].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[1].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)
    period, theta = 5, np.pi / 2
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=dx,
        scale=period, theta=np.pi / 2, param=param)
    wave /= wave.max()
    axs[2].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[2].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)
    letterbox = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

    for ax, letter in zip(axs, ["a", "b", "c"]):
        ax.text(0.12, 0.15, letter, transform=ax.transAxes,
                bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")

    fig.tight_layout()
    fig.savefig("example_2dmorlet_a.png")
    fig.savefig("example_2dmorlet_a.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3)
    aspect = 10
    for ax in axs:
        ax.set_aspect(1)
    period, theta = 10, 0
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=aspect * dx,
        scale=period, theta=theta, param=param)
    wave /= wave.max()
    axs[0].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[0].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)
    period, theta = 10, np.pi / 4
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=aspect * dx,
        scale=period, theta=theta, param=param)
    wave /= wave.max()
    axs[1].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[1].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)
    period, theta = 5, np.pi / 2
    wave = wavelet.morlet2d_real(
        len(xs), len(xs), dx=dx, dy=aspect * dx,
        scale=period, theta=np.pi / 2, param=param)
    wave /= wave.max()
    axs[2].set_title(f"period={period}, theta={np.rad2deg(theta):4.1f}")
    axs[2].pcolormesh(xs, xs, wave, vmin=-1, vmax=1, cmap="RdBu", rasterized=True)

    for ax, letter in zip(axs, ["a", "b", "c"]):
        ax.text(0.12, 0.15, letter, transform=ax.transAxes,
                bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")

    fig.tight_layout()
    fig.savefig("example_2dmorlet_b.png")
    fig.savefig("example_2dmorlet_b.pdf")
    plt.close(fig)


if __name__ == "__main__":
    example_plot_wavelet2d()
