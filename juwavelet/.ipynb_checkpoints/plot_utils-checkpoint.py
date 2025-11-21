import numpy as np
import matplotlib.pyplot as plt

def plot_fov_wavelet_spectrum(cwt, stat: str = "mean"):
    """
    Make a 0–180° half-circle polar plot of wave energy vs orientation and wavelength.
    Radial axis uses the juwavelet wavelengths (km), log scaled.
    Major tick labels: 8, 16, 32, 64, 128, ...
    """

    # 1. Amplitude statistic
    amp = np.abs(cwt["decomposition"])  # (ns, nt, nx, ny)

    if stat == "mean":
        spec = np.nanmean(amp, axis=(2, 3))  # (ns, nt)
    elif stat == "max":
        spec = np.nanmax(amp, axis=(2, 3))   # (ns, nt)
    else:
        raise ValueError("stat must be 'mean' or 'max'")

    ns, nt = spec.shape

    # 2. Wavelengths
    if "wavelength" in cwt:
        lam = np.asarray(cwt["wavelength"])
    else:
        lam = np.asarray(cwt["period"])

    lam1d = lam[:, 0] if lam.ndim == 2 else lam  # (ns,)

    # 3. Angles: convert to radians, then fold to orientation in [0, π)
    theta_arr = np.asarray(cwt["theta"])         # shape (ns, nt) or (nt,)
    theta1d = theta_arr[0, :] if theta_arr.ndim == 2 else theta_arr  # (nt,)

    th_max = np.nanmax(np.abs(theta1d))
    if th_max <= 2 * np.pi + 1e-3:
        theta_rad = theta1d
    else:
        theta_rad = np.radians(theta1d)

    # orientation is 180 degree periodic
    ori = np.mod(theta_rad, np.pi)  # [0, π)

    # Decide number of orientation bins
    # If coverage is more than π, we probably had 0..2π, so fold to nt//2 bins
    if th_max > np.pi + 0.1:
        n_bins = nt // 2
    else:
        n_bins = nt

    # Uniform binning in [0, π)
    th_edges = np.linspace(0.0, np.pi, n_bins + 1)      # edges in radians
    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])   # centers in radians

    # Map each original orientation to a bin
    bin_idx = np.digitize(ori, th_edges) - 1  # indices 0..n_bins-1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # 4. Fold spectrum: average amplitude in each orientation bin
    folded_spec = np.full((ns, n_bins), np.nan)

    for i in range(ns):
        row = spec[i, :]
        # mask NaN before accumulating
        valid = ~np.isnan(row)
        vals = row[valid]
        idx = bin_idx[valid]

        sums = np.zeros(n_bins, dtype=float)
        counts = np.zeros(n_bins, dtype=float)

        np.add.at(sums, idx, vals)
        np.add.at(counts, idx, 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            folded_spec[i, :] = np.where(counts > 0, sums / counts, np.nan)

    # 5. Build radial edges for pcolormesh
    def edges(x):
        mid = 0.5 * (x[:-1] + x[1:])
        first = x[0] - (mid[0] - x[0])
        last  = x[-1] + (x[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    lam_edges = edges(lam1d)  # radial edges

    TH, R = np.meshgrid(th_edges, lam_edges)  # TH in radians

    # 6. Plot
    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": "polar"}
    )

    # Half circle in degrees
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.set_theta_zero_location("N")  # 0° at the top
    ax.set_theta_direction(-1)       # clockwise

    # Radial (wavelength) axis
    ax.set_yscale("log")
    ax.set_ylim(lam1d.min(), lam1d.max())

    major_ticks = []
    for v in [8, 16, 32, 64, 128, 256, 512]:
        if lam1d.min() <= v <= lam1d.max():
            major_ticks.append(v)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f"{int(v)} km" for v in major_ticks])

    im = ax.pcolormesh(TH, R, folded_spec, shading="auto", cmap="viridis")

    ax.set_title("FOV Wavelet Spectrum", pad=24)

    cbar = fig.colorbar(im, pad=0.12)
    cbar.set_label("mean |wavelet coeff|" if stat == "mean" else "max |wavelet coeff|")

    plt.tight_layout()
    plt.show()

    

def quicklook_background_decomp(
    x1d: np.ndarray,
    y1d: np.ndarray,
    original: np.ndarray,
    background: np.ndarray,
    remaining: np.ndarray,
    vmin: float,
    vmax: float,
) -> None:
    """
    Show three panels:
      (a) original field (median-subtracted)
      (b) background = 2D plane + first Fourier component in x and y (median-subtracted)
      (c) remaining = original - background  (no extra median subtraction)
    Uses provided vmin/vmax so colors match other plots.
    """
    med = np.nanmedian(original)

    # Median-center original and background
    orig_c = original - med
    bg_c = background - med

    # Remaining is exactly original - background
    rem_c = original - background

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    im0 = axs[0].pcolormesh(
        x1d, y1d, orig_c,
        shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    axs[0].set_title("original\n(median-subtracted)")

    axs[1].pcolormesh(
        x1d, y1d, bg_c,
        shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    axs[1].set_title("background\n(plane + first Fourier)")

    axs[2].pcolormesh(
        x1d, y1d, rem_c,
        shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    axs[2].set_title("remaining\n(original - background)")

    for ax in axs:
        ax.set_xlabel("x (pixels)")
        ax.set_aspect("equal", adjustable="box")
    axs[0].set_ylabel("y (pixels)")

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("Intensity (a.u.)")

    plt.show()
