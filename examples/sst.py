import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from juwavelet import transform, utils
PSI_0 = np.pi ** -0.25


matplotlib.rcParams.update({
    'axes.labelsize': 16,
    'font.size': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': [6.94, 2 * 4.29],
})


def main():
    sst = np.loadtxt(os.path.join(os.path.dirname(__file__), "sst_nino3.txt"))
    dx = 0.25
    s0 = 2 * dx
    dj = 0.25
    jtot = int(7 / dj) + 1
    xs = np.arange(len(sst)) * dx + 1871

    cwt = transform.decompose1d(sst, dx, s0, dj, jtot, opts={"param": 6}, mode="cwt")
    stw = transform.decompose1d(sst, dx, s0, dj, jtot, opts={"param": 6}, mode="stockwell")
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(xs, sst)
    levels = [0, 0.125, 0.25, 0.5, 1]
    pm = axs[1].contourf(xs, cwt["period"], abs(cwt["decomposition"]) ** 2, levels, cmap="plasma_r", extend="max")
    plt.colorbar(pm, ax=pm.axes, label="Power (K$^2$)")
    levels = [0, 0.1, 0.25, 0.5, 1]
    pm = axs[2].contourf(xs, cwt["period"], abs(stw["decomposition"]), levels, cmap="plasma_r", extend="max")
    plt.colorbar(pm, ax=pm.axes, label="Amplitude (K)")
    coi = utils.cone_of_influence_1d(cwt)
    for ax in axs[1:]:
        ax.fill_between(
            xs, coi, 9999,
            facecolor="none", hatch="x", edgecolor="white")
    plt.colorbar(pm, ax=axs[0])  # this colorbar must be manually removed
    axs[0].set_ylabel("temperature (K)")
    for ax in axs:
        ax.set_xlim(1870, 2000)
    for ax in axs[1:]:
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.5, 64)
        ax.invert_yaxis()
        ax.set_yticks([2 ** x for x in range(7)],
                      [str(2 ** x) for x in range(7)])
        ax.set_ylabel("Period (years)")
    axs[2].set_xlabel("Time (year)")
    letterbox = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

    for ax, letter in zip(axs, ["a", "b", "c"]):
        ax.text(0.08, 0.15, letter, transform=ax.transAxes,
                bbox=letterbox, verticalalignment="top", horizontalalignment="right", weight="bold")
    plt.tight_layout()
    plt.savefig("example_sst.pdf")
    plt.savefig("example_sst.png")
    # plt.show()


if __name__ == "__main__":
    main()
