import os
import matplotlib.pyplot as plt
import numpy as np

from juwavelet import transform, utils


def example_identify_clusters():
    storage = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "alima.txt"))
    wave = storage[1:, 1:].T
    xs = storage[0, 1:]
    ys = storage[1:, 0]

    dx = np.diff(xs).mean()
    dy = np.diff(ys).mean()
    dec = transform.decompose2d(
        wave, dx=dx, dy=dy,
        s0=20, dj=0.25, js=20, jt=18, aspect=40)

    decomposition, period, theta = [
        dec[_x] for _x in ["decomposition", "period", "theta"]]

    amps, idxs, iwave = utils.identify_cluster2d(
        dec, min_amp=2.0, thr=1.0)

    wave2 = dec["decomposition"].copy()
    ma = 5
    for j, (amp, udx) in enumerate(zip(amps, idxs)):
        wave2[:] = dec["decomposition"]
        sel = (iwave == j)
        # filter small clusters
        if sel.sum() < 1000:
            continue
        print(j, sel.sum())
        wave2[~sel] = 0
        dec2 = dec.copy()
        dec2["decomposition"] = wave2
        rec = transform.reconstruct2d(dec2)
        fig, axs = plt.subplots()
        axs.set_title(
            f"amplitude={amp:4.1f}K\n"
            f"period={period[udx[0]]:5.1f}km theta={np.rad2deg(theta[udx[1]]):4.1f}\n"
            f"hor_wl={dec['wavelength_x'][udx[0], udx[1]]:5.1f}km "
            f"ver_wl={dec['wavelength_y'][udx[0], udx[1]]:4.1f}km")
        pm = axs.pcolormesh(
            xs, ys, rec.T, vmin=-ma, vmax=ma, cmap="RdBu", rasterized=True)
        plt.colorbar(pm, label="temperature (K)")
        fig.tight_layout()
        fig.savefig(f"example_cluster_{j:02}.png")
        plt.close(fig)


if __name__ == "__main__":
    example_identify_clusters()
