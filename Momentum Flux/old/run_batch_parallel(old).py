#!/usr/bin/env python3
from __future__ import annotations

"""
Parallel batch runner (per file parallelism).

Example:
  python run_batch_parallel.py --input l3 --glob "*.nc" --workers 4

Notes:
- Parallelizes across files (recommended). Slices remain sequential inside a file.
- Uses matplotlib Agg backend so no GUI windows.

Option A implemented:
- Do not trim the whole orbit
- For each slice:
  1) slice from raw 2D frame (still has NaNs)
  2) trim all-NaN edge columns in that slice window
  3) flatten swath on the trimmed window
  4) run preprocessing and CWT on the trimmed+flattened window
  5) pad core outputs back so every slice core has the same width
"""

import os
import gc
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from concurrent.futures import ProcessPoolExecutor, as_completed

from juwavelet import transform, utils, plot_utils

try:
    from scipy import stats
except Exception:
    stats = None


# ==========================
# DEFAULT PARAMS
# ==========================
DEFAULT_INPUT_FOLDER = Path("l3_Dominique")
DEFAULT_FILE_GLOB = "*.nc"

DEFAULT_OUTPUT_ROOT = Path("outputs_decompose2d_slices_single_noise_filtering_AUTO_BATCH")
DEFAULT_SAVE_DPI = 200
DEFAULT_OUTPUT_L6_FOLDER = Path("l6")
DEFAULT_STITCHED_FOLDERNAME = "_STITCHED"   # inside OUTPUT_ROOT

DEFAULT_FRAME_IDX = 0
DEFAULT_X_CHUNK = 600
DEFAULT_Y_SLICE = slice(None, None)
DEFAULT_SLICE_PAD = 50
DEFAULT_X_OFFSET = 0

DEFAULT_REMOVE_TREND = True
DEFAULT_STANDARDIZE = False
DEFAULT_GAUSS_BLUR_SIGMA = 0.0

DEFAULT_APPLY_BINNING = True
DEFAULT_BIN_FACTOR = 2

DEFAULT_APPLY_TAPER = True
DEFAULT_TAPER_EDGE_PIXELS = 10

DEFAULT_S0 = 24
DEFAULT_DJ = 1 / 8
DEFAULT_JS = 28     # Max wavelength around 272 km
DEFAULT_JT = 18
DEFAULT_ASPECT = 1

DEFAULT_DX_KM_BASE = 2.0
DEFAULT_DY_KM_BASE = 2.0

DEFAULT_N_SHOW = 6
DEFAULT_AMP_MIN_FRACTION_OF_CLUSTER_MAX = 0.10

DEFAULT_QUIVER_STEP_X = 20
DEFAULT_QUIVER_STEP_Y = 20
DEFAULT_QUIVER_SCALE = 1

DEFAULT_APPLY_RED_NOISE_FILTER = True

DEFAULT_SAVE_REDNOISE_DIAGNOSTICS = False
DEFAULT_SAVE_SLICE_PLOTS = False
DEFAULT_SAVE_STITCHED_PLOT = True


# --------------------------
# Plot style
# --------------------------
matplotlib.rcParams.update(
    {
        "axes.labelsize": 16,
        "font.size": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.figsize": [2 * 6.94, 2 * 4.29],
    }
)


# --------------------------
# Small helpers
# --------------------------
def save_fig(fig: plt.Figure, outpath: Path, *, enable: bool, save_dpi: int) -> None:
    try:
        if enable:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath, dpi=save_dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def cleanup_memory() -> None:
    try:
        plt.close("all")
    except Exception:
        pass
    gc.collect()


def diverging_limits(*arrays: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> Tuple[float, float]:
    vals = []
    for A in arrays:
        if A is None:
            continue
        a = np.asarray(A)
        good = np.isfinite(a)
        if np.any(good):
            vals.append(a[good].ravel())
    if not vals:
        return (-1.0, 1.0)
    stacked = np.concatenate(vals)
    lo, hi = np.nanpercentile(stacked, [p_lo, p_hi])
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return (-vmax, vmax)


def trim_all_nan_edge_columns(Z: np.ndarray) -> Tuple[np.ndarray, int, int]:
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D array, got {Z.shape}")

    col_has_data = np.any(np.isfinite(Z), axis=0)
    if not np.any(col_has_data):
        return Z, 0, Z.shape[1]

    x0 = int(np.argmax(col_has_data))
    x1 = int(len(col_has_data) - np.argmax(col_has_data[::-1]))
    return Z[:, x0:x1], x0, x1


def trim_window_all_nan_edges(win: np.ndarray) -> Tuple[np.ndarray, int, int]:
    win_trim, x0, x1 = trim_all_nan_edge_columns(win)
    trim_left = int(x0)
    trim_right = int(win.shape[1] - x1)
    return win_trim, trim_left, trim_right


def crop_pad_core_from_window(
    A: np.ndarray,
    *,
    rel0_binned: int,
    core_w_binned_target: int,
    pad_left_binned: int,
    pad_right_binned: int,
) -> np.ndarray:
    nyA, nxA = A.shape

    inner_w = int(core_w_binned_target - pad_left_binned - pad_right_binned)
    if inner_w < 0:
        inner_w = 0

    left = max(0, int(rel0_binned))
    right = min(nxA, left + inner_w)
    core = A[:, left:right]

    out = np.full((nyA, core_w_binned_target), np.nan, dtype=A.dtype)

    x_ins0 = int(pad_left_binned)
    x_ins1 = min(core_w_binned_target, x_ins0 + core.shape[1])
    if x_ins0 < core_w_binned_target and x_ins1 > x_ins0:
        out[:, x_ins0:x_ins1] = core[:, : (x_ins1 - x_ins0)]
    return out


# --------------------------
# Product detection + RUNS
# --------------------------
def detect_product_type(file_path: Path) -> str:
    name = file_path.name.lower()
    if "_l3a_" in name:
        return "l3a"
    if "_l3c_" in name:
        return "l3c"
    if "_l5c_" in name:
        return "l5c"
    raise ValueError(f"Cannot detect product type from file name: {file_path.name}")


def get_runs_for_product(product_type: str) -> List[Dict[str, Any]]:
    if product_type == "l3a":
        return [
            dict(
                name="base",
                MIN_AMP=1.25,
                THR=0.6,
                WHITE_NOISE_THRESHOLD=0.1,
                WPS_NOISE_THRESHOLD=8.0,
            ),
        ]
    return [
        dict(
            name="base",
            MIN_AMP=0.2,
            THR=0.15,
            WHITE_NOISE_THRESHOLD=0.02,
            WPS_NOISE_THRESHOLD=8.0,
        ),
    ]


# --------------------------
# Data helpers
# --------------------------
def read_frame(path: str | Path, frame_idx: int, product_type: str) -> np.ndarray:
    with Dataset(str(path), "r") as nc:
        varname = "Temperature" if product_type == "l3a" else "Radiance"
        data = nc.variables[varname][:]  # (t, y, x)

    if data.ndim != 3:
        raise ValueError(f"{varname} expected 3D (t,y,x); got {data.shape}")

    if not (0 <= frame_idx < data.shape[0]):
        raise IndexError(f"FRAME_IDX={frame_idx} out of range 0..{data.shape[0]-1}")

    return np.asarray(data[frame_idx, :, :], dtype=float)


def flatten_swath(Z: np.ndarray, target_ny: int = 300) -> np.ndarray:
    ny, nx = Z.shape
    out = np.full_like(Z, np.nan)

    bottoms = np.full(nx, np.nan, dtype=float)
    for j in range(nx):
        col = Z[:, j]
        valid_idx = np.where(np.isfinite(col))[0]
        if valid_idx.size:
            bottoms[j] = valid_idx.max()

    valid_bottoms = bottoms[np.isfinite(bottoms)]
    if valid_bottoms.size == 0:
        med = float(np.nanmedian(Z)) if np.isfinite(np.nanmedian(Z)) else 0.0
        out = np.nan_to_num(Z, nan=med, posinf=med, neginf=med)
        if target_ny is not None and ny > target_ny:
            out = out[-target_ny:, :]
        return out

    max_bottom = int(valid_bottoms.max())

    for j in range(nx):
        col = Z[:, j]
        valid_idx = np.where(np.isfinite(col))[0]
        if valid_idx.size == 0:
            continue
        bottom = int(valid_idx.max())
        shift = max_bottom - bottom
        if shift == 0:
            out[:, j] = col
        elif shift > 0:
            out[shift:, j] = col[: ny - shift]
        else:
            shift_up = -shift
            out[: ny - shift_up, j] = col[shift_up:]

    med = float(np.nanmedian(out)) if np.isfinite(np.nanmedian(out)) else 0.0
    out = np.nan_to_num(out, nan=med, posinf=med, neginf=med)

    if target_ny is not None and ny > target_ny:
        start = max_bottom + 1 - target_ny
        start = max(start, 0)
        end = start + target_ny
        out = out[start:end, :]

    return out


def bin2d_block_mean(Z: np.ndarray, factor: int) -> np.ndarray:
    if factor is None or int(factor) <= 1:
        return Z
    f = int(factor)
    ny, nx = Z.shape
    ny2 = (ny // f) * f
    nx2 = (nx // f) * f
    if ny2 == 0 or nx2 == 0:
        raise ValueError(f"Binning factor {f} too large for array {Z.shape}")
    Z2 = Z[:ny2, :nx2]
    return Z2.reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))


def fit_plane(Z: np.ndarray) -> np.ndarray:
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    G = np.stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)], axis=1)
    coeff, *_ = np.linalg.lstsq(G, Z.ravel(), rcond=None)
    return coeff[0] * xx + coeff[1] * yy + coeff[2]


def extract_first_fourier_bg(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = Z.shape
    F = np.fft.fft2(Z)
    F_bg = np.zeros_like(F, dtype=complex)

    if nx > 1:
        F_bg[0, 1] = F[0, 1]
        F_bg[0, -1] = F[0, -1]
    if ny > 1:
        F_bg[1, 0] = F[1, 0]
        F_bg[-1, 0] = F[-1, 0]

    bg = np.fft.ifft2(F_bg).real
    return bg, Z - bg


def maybe_blur(Z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return Z
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return Z
    return gaussian_filter(Z, sigma=float(sigma))


def _hanning_edge_1d(n: int, edge: int) -> np.ndarray:
    w = np.ones(n, dtype=float)
    if edge <= 0:
        return w
    edge = min(edge, n // 2)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, edge, endpoint=True)))
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return w


def apply_edge_hanning_taper(Z: np.ndarray, edge: int) -> np.ndarray:
    if edge <= 0:
        return Z
    ny, nx = Z.shape
    wx = _hanning_edge_1d(nx, edge)
    wy = _hanning_edge_1d(ny, edge)
    W = np.outer(wy, wx)
    return Z * W


def compute_cluster_power_table(cwt: dict, iwave: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dec = cwt["decomposition"]
    power4d = np.abs(dec) ** 2

    cluster_ids = np.unique(iwave[iwave >= 0]).astype(int)
    if cluster_ids.size == 0:
        return cluster_ids, np.array([]), np.array([])

    P_abs = np.zeros(cluster_ids.size, dtype=float)
    for i, kcl in enumerate(cluster_ids):
        P_abs[i] = float(np.sum(power4d[iwave == kcl]))

    total = float(np.sum(P_abs))
    if total > 0 and np.isfinite(total):
        P_pct = 100.0 * P_abs / total
    else:
        P_pct = np.full_like(P_abs, np.nan)

    return cluster_ids, P_abs, P_pct


def cluster_amp_theta_maps(
    cwt: dict,
    iwave: np.ndarray,
    kcl: int,
    amp_min_fraction_of_cluster_max: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    dec = cwt["decomposition"]
    power = np.abs(dec) ** 2

    mask4d = (iwave == kcl)
    if not np.any(mask4d):
        ny = dec.shape[3]
        nx = dec.shape[2]
        return np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)

    ns, nt, nx, ny = power.shape
    power2 = power.reshape(ns * nt, nx * ny)
    mask2 = mask4d.reshape(ns * nt, nx * ny)

    p = np.where(mask2, power2, -np.inf)
    pmax = np.max(p, axis=0)
    valid = np.isfinite(pmax) & (pmax > 0)

    A = np.full(nx * ny, np.nan, dtype=float)
    theta = np.full(nx * ny, np.nan, dtype=float)

    if np.any(valid):
        arg = np.argmax(p[:, valid], axis=0)
        t_idx = arg % nt
        A_valid = np.sqrt(pmax[valid])
        theta_valid = np.asarray(cwt["theta"], dtype=float)[t_idx]
        A[valid] = A_valid
        theta[valid] = theta_valid

    A_xy = A.reshape(nx, ny).T
    theta_xy = theta.reshape(nx, ny).T

    if np.isfinite(np.nanmax(A_xy)):
        thr = float(amp_min_fraction_of_cluster_max) * float(np.nanmax(A_xy))
        low = A_xy < thr
        A_xy[low] = np.nan
        theta_xy[low] = np.nan

    return A_xy, theta_xy


def apply_red_noise_filter_cwt(
    cwt: dict,
    white_noise_threshold: float = 0.1,
    wps_noise_threshold: float = 8.0,
) -> Tuple[dict, Dict[str, Any]]:
    diag: Dict[str, Any] = {}
    if stats is None:
        diag["skipped"] = True
        diag["reason"] = "scipy not available"
        return cwt, diag

    cwt_copy = dict(cwt)
    dec0 = cwt["decomposition"]
    dec = dec0.copy()
    cwt_copy["decomposition"] = dec

    WPS0 = np.abs(dec) ** 2
    diag["n_coeff_total"] = int(WPS0.size)

    mask_1 = WPS0 < (float(white_noise_threshold) ** 2)
    dec[mask_1] = 0

    WPS1 = np.abs(dec) ** 2
    diag["n_zero_after_white"] = int(np.count_nonzero(WPS1 == 0))

    median_WPS = np.median(WPS1, axis=(1, 2, 3))
    mad = stats.median_abs_deviation(WPS1, axis=(1, 2, 3), scale=1.0)
    sMAD_WPS = 1.4826 * mad

    sMAD_safe = np.where(sMAD_WPS > 0, sMAD_WPS, np.nan)
    WPS_scaled = (WPS1 - median_WPS[:, None, None, None]) / sMAD_safe[:, None, None, None]

    mask = WPS_scaled < float(wps_noise_threshold)
    dec[mask] = 0

    WPS2 = np.abs(dec) ** 2
    diag["n_zero_after_scaled"] = int(np.count_nonzero(WPS2 == 0))

    finite_scaled = np.isfinite(WPS_scaled)

    diag["sMAD_min"] = float(np.nanmin(sMAD_WPS))
    diag["sMAD_median"] = float(np.nanmedian(sMAD_WPS))

    if np.any(finite_scaled):
        diag["WPS_scaled_min"] = float(np.nanmin(WPS_scaled[finite_scaled]))
        diag["WPS_scaled_median"] = float(np.nanmedian(WPS_scaled[finite_scaled]))
        diag["WPS_scaled_max"] = float(np.nanmax(WPS_scaled[finite_scaled]))
    else:
        diag["WPS_scaled_min"] = np.nan
        diag["WPS_scaled_median"] = np.nan
        diag["WPS_scaled_max"] = np.nan

    return cwt_copy, diag


# --------------------------
# Quicklook plots
# --------------------------
def quicklook_save(
    x1d: np.ndarray,
    y1d: np.ndarray,
    Z: np.ndarray,
    title: str,
    outpath: Path,
    *,
    save_plots: bool,
    save_dpi: int,
) -> None:
    if not save_plots:
        return
    fig, ax = plt.subplots()
    im = ax.pcolormesh(x1d, y1d, Z, shading="auto", cmap="gray")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("arb")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_fig(fig, outpath, enable=True, save_dpi=save_dpi)


def quicklook_background_decomp_4panel_save(
    x1d_km: np.ndarray,
    y1d_km: np.ndarray,
    original: np.ndarray,
    background: np.ndarray,
    remaining: np.ndarray,
    remaining_tapered: np.ndarray,
    taper_edge_px: int,
    outpath: Path,
    *,
    wavy_after_noise_xy: np.ndarray | None = None,
    save_plots: bool,
    save_dpi: int,
) -> None:
    if not save_plots:
        return

    med = np.nanmedian(original)
    orig_c = original - med
    bg_c = background - med
    rem_c = remaining
    rem_tapered_c = remaining_tapered

    arrays = [orig_c, bg_c, rem_c, rem_tapered_c]
    if wavy_after_noise_xy is not None:
        arrays.append(wavy_after_noise_xy)

    vmin, vmax = diverging_limits(*arrays, p_lo=1, p_hi=99)

    fig, axs = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axs = axs.flatten()

    panels = [
        (orig_c, "Original (median sub)"),
        (bg_c, "Background (plane + first Fourier)"),
        (rem_c, "Remaining = original - background"),
        (rem_tapered_c, f"Remaining with edge taper (width={taper_edge_px})"),
    ]

    if wavy_after_noise_xy is not None:
        panels.append((wavy_after_noise_xy, "Wavy stuff (wavelet recon. after noise filtering)"))

    im0 = None
    for ax, (Zp, title) in zip(axs, panels):
        im0 = ax.pcolormesh(x1d_km, y1d_km, Zp, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal", adjustable="box")

    for ax in axs[len(panels) :]:
        ax.axis("off")

    cbar = fig.colorbar(im0, ax=axs[: len(panels)], orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("arb")
    save_fig(fig, outpath, enable=True, save_dpi=save_dpi)


# --------------------------
# Cluster products for outputs
# --------------------------
def cluster_dominant_lambda_map(
    cwt: dict,
    iwave: np.ndarray,
    kcl: int,
    amp_mask_xy: np.ndarray | None = None,
) -> np.ndarray:
    dec = cwt["decomposition"]
    power0 = np.abs(dec) ** 2
    mask4d = (iwave == kcl)
    if not np.any(mask4d):
        ny = dec.shape[3]
        nx = dec.shape[2]
        return np.full((ny, nx), np.nan)

    ns, nt, nx, ny = power0.shape
    power2 = power0.reshape(ns * nt, nx * ny)
    mask2 = mask4d.reshape(ns * nt, nx * ny)

    p = np.where(mask2, power2, -np.inf)
    pmax = np.max(p, axis=0)
    valid = np.isfinite(pmax) & (pmax > 0)

    lam_xy = np.full(nx * ny, np.nan, dtype=float)
    if np.any(valid):
        arg = np.argmax(p[:, valid], axis=0)
        s_idx = arg // nt
        t_idx = arg % nt

        lamx = np.asarray(cwt["wavelength_x"], dtype=float)[s_idx, t_idx]
        lamy = np.asarray(cwt["wavelength_y"], dtype=float)[s_idx, t_idx]

        inv2 = np.zeros_like(lamx, dtype=float)
        goodx = np.isfinite(lamx) & (lamx != 0)
        goody = np.isfinite(lamy) & (lamy != 0)
        inv2[goodx] += (1.0 / lamx[goodx]) ** 2
        inv2[goody] += (1.0 / lamy[goody]) ** 2

        lam = np.full_like(inv2, np.nan, dtype=float)
        ok = inv2 > 0
        lam[ok] = 1.0 / np.sqrt(inv2[ok])

        lam_xy[valid] = lam

    lam_xy = lam_xy.reshape(nx, ny).T

    if amp_mask_xy is not None:
        lam_xy[~np.isfinite(amp_mask_xy)] = np.nan

    return lam_xy


def make_output_cluster_path(input_path: str | Path) -> Path:
    p = Path(input_path)
    name = p.name
    if "_l5c_" in name:
        name = name.replace("_l5c_", "_l6c_")
    elif "_l3c_" in name:
        name = name.replace("_l3c_", "_l6c_")
    elif "_l3a_" in name:
        name = name.replace("_l3a_", "_l6a_")
    else:
        name = p.stem + "_clusters.nc"
    return p.with_name(name)


def copy_lxc_to_l6x_with_clusters(
    lxc_path: str | Path,
    l6x_path: str | Path,
    clusters: List[Dict[str, Any]],
    y_dim: int,
    x_dim: int,
    unbin_factor: int = 1,
    product_type: str = "l5c",
) -> None:
    def _unbin_repeat_xy_local(A: np.ndarray, factor: int) -> np.ndarray:
        if factor is None or int(factor) <= 1:
            return A
        f = int(factor)
        return np.repeat(np.repeat(A, f, axis=0), f, axis=1)

    lxc_path = Path(lxc_path)
    l6x_path = Path(l6x_path)
    l6x_path.parent.mkdir(parents=True, exist_ok=True)

    ncl = int(len(clusters))
    has_rec = any((c.get("rec_xy", None) is not None) for c in clusters)

    fup = int(unbin_factor) if (unbin_factor is not None and int(unbin_factor) > 1) else 1
    y_dim_w = int(y_dim) * fup
    x_dim_w = int(x_dim) * fup

    with Dataset(lxc_path, "r") as src, Dataset(l6x_path, "w", format="NETCDF4") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        for dname, dim in src.dimensions.items():
            dst.createDimension(dname, (len(dim) if not dim.isunlimited() else None))

        for vname, var in src.variables.items():
            out = dst.createVariable(
                vname,
                var.datatype,
                var.dimensions,
                zlib=True,
                complevel=4,
                fill_value=getattr(var, "_FillValue", None),
            )
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            out[:] = var[:]

        dst.createDimension("cluster", ncl)
        dst.createDimension("y_cluster", y_dim_w)
        dst.createDimension("x_cluster", x_dim_w)

        v_slice = dst.createVariable("SlicesNo", "i4", ("cluster",), zlib=True, complevel=4)
        v_area = dst.createVariable("Area", "f4", ("cluster",), zlib=True, complevel=4)
        v_pabs = dst.createVariable("ClusterPowerAbs", "f8", ("cluster",), zlib=True, complevel=4)
        v_ppct = dst.createVariable("ClusterPowerPct", "f4", ("cluster",), zlib=True, complevel=4)

        v_amp = dst.createVariable(
            "Amplitude",
            "f4",
            ("cluster", "y_cluster", "x_cluster"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        v_lam = dst.createVariable(
            "DominantWavelength",
            "f4",
            ("cluster", "y_cluster", "x_cluster"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        v_ang = dst.createVariable(
            "Angle",
            "f4",
            ("cluster", "y_cluster", "x_cluster"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        v_rec = None
        if has_rec:
            v_rec = dst.createVariable(
                "ClusterReconstruction",
                "f4",
                ("cluster", "y_cluster", "x_cluster"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            v_rec.units = "arb"
            v_rec.long_name = "Per cluster wavelet reconstruction map (relative; median subtracted)"

        v_area.units = "km2"
        v_lam.units = "km"
        v_ang.units = "rad"
        v_slice.long_name = "Slice index (starts at 0) for each cluster"
        v_pabs.units = "arb"
        v_ppct.units = "percent"
        v_amp.units = "K" if product_type == "l3a" else "kR"

        if ncl == 0:
            return

        v_slice[:] = np.asarray([c["slice_no"] for c in clusters], dtype=np.int32)
        v_area[:] = np.asarray([c["area_km2"] for c in clusters], dtype=np.float32)
        v_pabs[:] = np.asarray([c.get("power_abs", np.nan) for c in clusters], dtype=np.float64)
        v_ppct[:] = np.asarray([c.get("power_pct", np.nan) for c in clusters], dtype=np.float32)

        amp_stack = np.full((ncl, y_dim_w, x_dim_w), np.nan, dtype=np.float32)
        lam_stack = np.full((ncl, y_dim_w, x_dim_w), np.nan, dtype=np.float32)
        ang_stack = np.full((ncl, y_dim_w, x_dim_w), np.nan, dtype=np.float32)
        rec_stack = np.full((ncl, y_dim_w, x_dim_w), np.nan, dtype=np.float32) if has_rec else None

        def _force_shape(M: np.ndarray) -> np.ndarray:
            if M.shape == (y_dim_w, x_dim_w):
                return M.astype(np.float32)
            tmp = np.full((y_dim_w, x_dim_w), np.nan, dtype=np.float32)
            yy = min(y_dim_w, M.shape[0])
            xx = min(x_dim_w, M.shape[1])
            tmp[:yy, :xx] = M[:yy, :xx]
            return tmp

        for i, c in enumerate(clusters):
            A = _unbin_repeat_xy_local(c["A_xy"], fup)
            L = _unbin_repeat_xy_local(c["lam_xy"], fup)
            T = _unbin_repeat_xy_local(c["theta_xy"], fup)

            amp_stack[i] = _force_shape(A)
            lam_stack[i] = _force_shape(L)
            ang_stack[i] = _force_shape(T)

            if has_rec and rec_stack is not None and v_rec is not None:
                R = c.get("rec_xy", None)
                if R is not None:
                    R = _force_shape(_unbin_repeat_xy_local(R, fup))
                    rec_stack[i] = R

        v_amp[:] = amp_stack
        v_lam[:] = lam_stack
        v_ang[:] = ang_stack
        if has_rec and v_rec is not None and rec_stack is not None:
            v_rec[:] = rec_stack


# --------------------------
# Slice pipeline (Option A)
# --------------------------
def run_pipeline_for_slice(
    rad2d_raw: np.ndarray,
    x_slice: slice,
    outdir: Path,
    core_start_unbinned: int,
    core_end_unbinned: int,
    x_base_unbinned: int,
    cfg: Dict[str, Any],
    slice_no: int,
    product_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    SAVE_DPI = int(params["SAVE_DPI"])
    SAVE_SLICE_PLOTS = bool(params["SAVE_SLICE_PLOTS"])
    SAVE_REDNOISE_DIAGNOSTICS = bool(params["SAVE_REDNOISE_DIAGNOSTICS"])
    APPLY_RED_NOISE_FILTER = bool(params["APPLY_RED_NOISE_FILTER"])

    REMOVE_TREND = bool(params["REMOVE_TREND"])
    STANDARDIZE = bool(params["STANDARDIZE"])
    GAUSS_BLUR_SIGMA = float(params["GAUSS_BLUR_SIGMA"])

    APPLY_BINNING = bool(params["APPLY_BINNING"])
    BIN_FACTOR = int(params["BIN_FACTOR"])

    APPLY_TAPER = bool(params["APPLY_TAPER"])
    TAPER_EDGE_PIXELS = int(params["TAPER_EDGE_PIXELS"])

    S0 = float(params["S0"])
    DJ = float(params["DJ"])
    JS = int(params["JS"])
    JT = int(params["JT"])
    ASPECT = float(params["ASPECT"])

    DX_KM_BASE = float(params["DX_KM_BASE"])
    DY_KM_BASE = float(params["DY_KM_BASE"])

    AMP_MIN_FRACTION_OF_CLUSTER_MAX = float(params["AMP_MIN_FRACTION_OF_CLUSTER_MAX"])

    min_amp = float(cfg.get("MIN_AMP"))
    thr = float(cfg.get("THR"))
    white_noise_threshold = float(cfg.get("WHITE_NOISE_THRESHOLD"))
    wps_noise_threshold = float(cfg.get("WPS_NOISE_THRESHOLD"))
    run_name = str(cfg.get("name", "run"))

    STORE_CLUSTER_RECON = True

    if APPLY_BINNING and BIN_FACTOR > 1:
        f = int(BIN_FACTOR)
        dx_km = DX_KM_BASE * f
        dy_km = DY_KM_BASE * f
    else:
        f = 1
        dx_km = DX_KM_BASE
        dy_km = DY_KM_BASE

    core_width_unbinned_target = int(core_end_unbinned - core_start_unbinned)
    core_w_binned_target = int(core_width_unbinned_target // f)

    Y_SLICE = params["Y_SLICE"]

    ext_start_unbinned = int(x_slice.start or 0)
    ext_end_unbinned = int(x_slice.stop)

    raw_win_full = np.asarray(rad2d_raw[Y_SLICE, x_slice], dtype=float)

    raw_win_trim, trimL_cols, trimR_cols = trim_window_all_nan_edges(raw_win_full)

    ext_start_eff = ext_start_unbinned + int(trimL_cols)
    ext_end_eff = ext_end_unbinned - int(trimR_cols)

    core_start = int(core_start_unbinned)
    core_end = int(core_end_unbinned)

    core0_eff = max(core_start, ext_start_eff)
    core1_eff = min(core_end, ext_end_eff)

    pad_left_unbinned = max(0, core0_eff - core_start)
    pad_right_unbinned = max(0, core_end - core1_eff)

    pad_left_binned = int(pad_left_unbinned // f)
    pad_right_binned = int(pad_right_unbinned // f)

    rel0_unbinned_eff = int(core0_eff - ext_start_eff)
    rel0_binned_eff = int(rel0_unbinned_eff // f)

    # flatten AFTER trimming (still has NaNs when trimming runs)
    raw_win_flat = flatten_swath(raw_win_trim, target_ny=300)

    if APPLY_BINNING and BIN_FACTOR > 1:
        raw_win_flat = bin2d_block_mean(raw_win_flat, BIN_FACTOR)

    ny, nx = raw_win_flat.shape
    x1d_km = np.arange(nx, dtype=float) * dx_km
    y1d_km = np.arange(ny, dtype=float) * dy_km

    def crop_pad_core(A: np.ndarray) -> np.ndarray:
        return crop_pad_core_from_window(
            A,
            rel0_binned=rel0_binned_eff,
            core_w_binned_target=core_w_binned_target,
            pad_left_binned=pad_left_binned,
            pad_right_binned=pad_right_binned,
        )

    bin_tag = f" (binned {BIN_FACTOR}x{BIN_FACTOR})" if (APPLY_BINNING and BIN_FACTOR > 1) else ""
    label0 = "Temperature" if product_type == "l3a" else "Radiance"

    quicklook_save(
        x1d_km,
        y1d_km,
        raw_win_flat,
        (
            f"{label0} (window, per-slice trim then flatten){bin_tag}\n"
            f"{run_name} MIN_AMP={min_amp} THR={thr}\n"
            f"trimL={trimL_cols} trimR={trimR_cols} padL={pad_left_unbinned} padR={pad_right_unbinned}"
        ),
        outdir / "00_window.png",
        save_plots=SAVE_SLICE_PLOTS,
        save_dpi=SAVE_DPI,
    )

    proc0 = raw_win_flat.astype(float, copy=True)

    if REMOVE_TREND:
        plane = fit_plane(proc0)
        detrended = proc0 - plane
    else:
        plane = np.zeros_like(proc0)
        detrended = proc0

    fourier_bg, _ = extract_first_fourier_bg(detrended)
    background = plane + fourier_bg
    remaining = proc0 - background

    if APPLY_TAPER:
        remaining_tapered = apply_edge_hanning_taper(remaining, TAPER_EDGE_PIXELS)
    else:
        remaining_tapered = remaining.copy()

    proc = remaining_tapered.copy()
    if STANDARDIZE:
        m = float(np.median(proc))
        s = float(np.std(proc))
        proc = (proc - m) / s if s > 0 else (proc - m)

    proc = maybe_blur(proc, GAUSS_BLUR_SIGMA)

    wave_xy = proc.T

    cwt_raw = transform.decompose2d(
        wave_xy, dx=dx_km, dy=dy_km, s0=S0, dj=DJ, js=JS, jt=JT, aspect=ASPECT, dtype=np.complex64
    )

    if APPLY_RED_NOISE_FILTER:
        cwt, diag = apply_red_noise_filter_cwt(
            cwt_raw,
            white_noise_threshold=white_noise_threshold,
            wps_noise_threshold=wps_noise_threshold,
        )

        if SAVE_REDNOISE_DIAGNOSTICS:
            lines = []
            lines.append("Red noise filter diagnostics\n\n")
            lines.append(f"name                  = {run_name}\n")
            lines.append(f"WHITE_NOISE_THRESHOLD = {white_noise_threshold}\n")
            lines.append(f"WPS_NOISE_THRESHOLD   = {wps_noise_threshold}\n\n")
            if diag.get("skipped", False):
                lines.append(f"SKIPPED: {diag.get('reason','unknown')}\n")
            else:
                lines.append(f"n_coeff_total        = {diag.get('n_coeff_total')}\n")
                lines.append(f"n_zero_after_white   = {diag.get('n_zero_after_white')}\n")
                lines.append(f"n_zero_after_scaled  = {diag.get('n_zero_after_scaled')}\n")
                lines.append(f"sMAD_WPS min         = {diag.get('sMAD_min')}\n")
                lines.append(f"sMAD_WPS median      = {diag.get('sMAD_median')}\n")
                lines.append(f"WPS_scaled min       = {diag.get('WPS_scaled_min')}\n")
                lines.append(f"WPS_scaled median    = {diag.get('WPS_scaled_median')}\n")
                lines.append(f"WPS_scaled max       = {diag.get('WPS_scaled_max')}\n")
            (outdir / "02a_rednoise_diagnostics.txt").write_text("".join(lines))
    else:
        cwt = cwt_raw

    rec_wavy = transform.reconstruct2d(cwt)
    wavy_xy = rec_wavy.T - np.median(rec_wavy)

    quicklook_background_decomp_4panel_save(
        x1d_km,
        y1d_km,
        proc0,
        background,
        remaining,
        remaining_tapered,
        TAPER_EDGE_PIXELS,
        outdir / "01_background_decomp_5panel.png",
        wavy_after_noise_xy=wavy_xy,
        save_plots=SAVE_SLICE_PLOTS,
        save_dpi=SAVE_DPI,
    )

    amps, idxs, iwave = utils.identify_cluster2d(cwt, min_amp=min_amp, thr=thr)
    decomposition = cwt["decomposition"]
    orig_decomp = decomposition.copy()

    if SAVE_SLICE_PLOTS:
        cmap_cwt = matplotlib.cm.turbo
        norm_cwt = matplotlib.colors.BoundaryNorm(np.exp(np.linspace(np.log(0.05), np.log(1.0), 10)), cmap_cwt.N)
        fig_cwt, _ = utils.plot_decomposition2d(cwt, redux_s=2, redux_t=2, cmap=cmap_cwt, norm=norm_cwt)
        save_fig(fig_cwt, outdir / "02_decomposition_overview.png", enable=True, save_dpi=SAVE_DPI)

        plot_utils.plot_fov_wavelet_spectrum(cwt, stat="max")
        fig_spec = plt.gcf()
        save_fig(fig_spec, outdir / "03_fov_wavelet_spectrum.png", enable=True, save_dpi=SAVE_DPI)

    cluster_ids, P_abs, P_pct = compute_cluster_power_table(cwt, iwave)
    if cluster_ids.size == 0:
        (outdir / "NO_CLUSTERS.txt").write_text(
            "No clusters found.\n\n"
            f"Run name: {run_name}\n"
            f"Used MIN_AMP={min_amp}, THR={thr}\n"
            f"Used WHITE_NOISE_THRESHOLD={white_noise_threshold}, WPS_NOISE_THRESHOLD={wps_noise_threshold}\n\n"
            f"trimL={trimL_cols} trimR={trimR_cols} padL={pad_left_unbinned} padR={pad_right_unbinned}\n"
        )
        remaining_core_xy = crop_pad_core(proc)
        nan_core = np.full_like(remaining_core_xy, np.nan)

        cleanup_memory()
        return {
            "has_clusters": False,
            "x_start_unbinned": int(core_start_unbinned),
            "x_start_unbinned_global": int(core_start_unbinned + x_base_unbinned),
            "dx_km": dx_km,
            "dy_km": dy_km,
            "remaining_xy": remaining_core_xy,
            "wavy_xy": nan_core,
            "rec_all_xy": nan_core,
            "cluster_records": [],
        }

    power_abs_map = {int(k): float(p) for k, p in zip(cluster_ids, P_abs)}
    power_pct_map = {int(k): float(p) for k, p in zip(cluster_ids, P_pct)}

    cluster_records: List[Dict[str, Any]] = []
    for kcl in cluster_ids.astype(int):
        kcl = int(kcl)

        A_xy, theta_xy = cluster_amp_theta_maps(
            cwt,
            iwave,
            kcl,
            amp_min_fraction_of_cluster_max=AMP_MIN_FRACTION_OF_CLUSTER_MAX,
        )
        lam_xy = cluster_dominant_lambda_map(cwt, iwave, kcl, amp_mask_xy=A_xy)

        rec_core = None
        if STORE_CLUSTER_RECON:
            decomposition[:] = orig_decomp
            decomposition[iwave != kcl] = 0
            rec = transform.reconstruct2d(cwt)
            rec_xy = rec.T
            rec_xy = rec_xy - np.nanmedian(rec_xy)
            rec_core = crop_pad_core(rec_xy)
            decomposition[:] = orig_decomp

        A_core = crop_pad_core(A_xy)
        th_core = crop_pad_core(theta_xy)
        lam_core = crop_pad_core(lam_xy)

        n_pix = int(np.count_nonzero(np.isfinite(A_core)))
        area_km2 = float(n_pix) * float(dx_km) * float(dy_km)

        cluster_records.append(
            dict(
                slice_no=int(slice_no),
                kcl=kcl,
                area_km2=area_km2,
                power_abs=power_abs_map.get(kcl, np.nan),
                power_pct=power_pct_map.get(kcl, np.nan),
                A_xy=A_core,
                lam_xy=lam_core,
                theta_xy=th_core,
                rec_xy=rec_core,
            )
        )

    decomposition[:] = orig_decomp
    decomposition[iwave < 0] = 0
    rec_all = transform.reconstruct2d(cwt)
    rec_all_xy = rec_all.T - np.median(rec_all)
    decomposition[:] = orig_decomp

    remaining_core_xy = crop_pad_core(proc)
    wavy_core_xy = crop_pad_core(wavy_xy)
    rec_all_core_xy = crop_pad_core(rec_all_xy)

    cleanup_memory()
    return {
        "has_clusters": True,
        "x_start_unbinned": int(core_start_unbinned),
        "x_start_unbinned_global": int(core_start_unbinned + x_base_unbinned),
        "dx_km": dx_km,
        "dy_km": dy_km,
        "remaining_xy": remaining_core_xy,
        "wavy_xy": wavy_core_xy,
        "rec_all_xy": rec_all_core_xy,
        "cluster_records": cluster_records,
    }


def stitch_and_save_final_09(
    slice_results: List[Dict[str, Any]],
    stitched_dir: Path,
    filename_prefix: str,
    tag: str,
    product_type: str,
    params: Dict[str, Any],
) -> None:
    if not bool(params["SAVE_STITCHED_PLOT"]):
        return
    if not slice_results:
        return

    stitched_dir.mkdir(parents=True, exist_ok=True)

    SAVE_DPI = int(params["SAVE_DPI"])
    DX_KM_BASE = float(params["DX_KM_BASE"])

    slice_results = sorted(slice_results, key=lambda d: int(d.get("x_start_unbinned_global", 0)))

    dx_km = float(slice_results[0]["dx_km"])
    dy_km = float(slice_results[0]["dy_km"])
    x0_unbinned_global = int(slice_results[0].get("x_start_unbinned_global", 0))
    x0_km = x0_unbinned_global * DX_KM_BASE

    remaining_stitched = np.concatenate([d["remaining_xy"] for d in slice_results], axis=1)
    wavy_stitched = np.concatenate([d["wavy_xy"] for d in slice_results], axis=1)
    rec_all_stitched = np.concatenate([d["rec_all_xy"] for d in slice_results], axis=1)

    ny, nx_tot = remaining_stitched.shape
    x1d_km = x0_km + np.arange(nx_tot, dtype=float) * dx_km
    y1d_km = np.arange(ny, dtype=float) * dy_km

    vmin, vmax = diverging_limits(remaining_stitched, wavy_stitched, rec_all_stitched, p_lo=2, p_hi=98)

    figR, (axT, axM, axB) = plt.subplots(3, 1, figsize=(18, 6), constrained_layout=True, sharex=True, sharey=True)

    imT = axT.pcolormesh(x1d_km, y1d_km, remaining_stitched, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axT.set_title(f"STITCHED (core tiles) {tag}: Remaining (after preprocessing, input to CWT)")
    axT.set_xlabel("x (km)")
    axT.set_ylabel("y (km)")
    axT.set_aspect("equal", adjustable="box")

    imM = axM.pcolormesh(x1d_km, y1d_km, wavy_stitched, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axM.set_title(f"STITCHED (core tiles) {tag}: Wavy stuff (wavelet reconstruction after noise filtering)")
    axM.set_xlabel("x (km)")
    axM.set_ylabel("y (km)")
    axM.set_aspect("equal", adjustable="box")

    imB = axB.pcolormesh(x1d_km, y1d_km, rec_all_stitched, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axB.set_title(f"STITCHED (core tiles) {tag}: Reconstruction after clustering (all clusters combined)")
    axB.set_xlabel("x (km)")
    axB.set_ylabel("y (km)")
    axB.set_aspect("equal", adjustable="box")

    cbar = figR.colorbar(imT, ax=[axT, axM, axB], orientation="vertical", fraction=0.012, pad=0.01)
    cbar.set_label("K (Relative)" if product_type == "l3a" else "kRayleigh (Relative)")

    out_png = stitched_dir / f"{filename_prefix}_FINAL_09_STITCHED_{tag}.png"
    save_fig(figR, out_png, enable=True, save_dpi=SAVE_DPI)

    cleanup_memory()


# --------------------------
# Per file runner (worker)
# --------------------------
def run_one_file_worker(file_path_str: str, params: Dict[str, Any]) -> Dict[str, Any]:
    file_path = Path(file_path_str)

    OUTPUT_ROOT = Path(params["OUTPUT_ROOT"])
    OUTPUT_L6_FOLDER = Path(params["OUTPUT_L6_FOLDER"])

    FRAME_IDX = int(params["FRAME_IDX"])
    X_CHUNK = int(params["X_CHUNK"])
    SLICE_PAD = int(params["SLICE_PAD"])
    X_OFFSET = int(params["X_OFFSET"])

    APPLY_BINNING = bool(params["APPLY_BINNING"])
    BIN_FACTOR = int(params["BIN_FACTOR"])
    Y_SLICE = params["Y_SLICE"]

    try:
        product_type = detect_product_type(file_path)
        runs = get_runs_for_product(product_type)

        print(f"\n=== [PID {os.getpid()}] File: {file_path.name} | product={product_type}", flush=True)

        rad2d = read_frame(file_path, FRAME_IDX, product_type=product_type)
        ny_raw, nx_total = rad2d.shape

        x_base_unbinned = 0

        x_offset_global = int(X_OFFSET)
        x_offset = x_offset_global
        if x_offset < 0:
            x_offset = 0
        if x_offset >= nx_total:
            raise ValueError(
                f"X_OFFSET={x_offset_global} maps to {x_offset} but nx_total is {nx_total} in file {file_path.name}"
            )

        for cfg in runs:
            run_name = str(cfg.get("name", "run"))

            outroot = OUTPUT_ROOT / file_path.stem / f"{run_name}_offset_{x_offset_global:04d}"
            outroot.mkdir(parents=True, exist_ok=True)

            (outroot / "RUN_CONFIG.txt").write_text(
                "\n".join([f"{k} = {v}" for k, v in cfg.items()])
                + f"\nproduct_type = {product_type}\nfile = {file_path.name}\n"
                + f"x_base_unbinned = {x_base_unbinned}\n"
                + f"X_OFFSET_global = {x_offset_global}\n"
                + f"X_OFFSET_local = {x_offset}\n"
                + "note = per-slice trim on raw, then flatten per slice\n"
            )

            core_starts = list(range(x_offset, nx_total, X_CHUNK))
            stitch_results: List[Dict[str, Any]] = []
            all_cluster_records: List[Dict[str, Any]] = []

            f = BIN_FACTOR if (APPLY_BINNING and BIN_FACTOR > 1) else 1

            ny_used = int(rad2d[Y_SLICE, :].shape[0])
            ny_flat = min(int(300), ny_used) if ny_used > 300 else ny_used
            y_dim = int(ny_flat // f)
            x_dim = int(X_CHUNK // f)

            for slice_no, core_start in enumerate(core_starts):
                core_end = min(core_start + X_CHUNK, nx_total)

                ext_start = max(0, core_start - SLICE_PAD)
                ext_end = min(nx_total, core_end + SLICE_PAD)
                x_slice = slice(ext_start, ext_end)

                outdir = outroot / f"xcore_{core_start:05d}_{core_end:05d}_xext_{ext_start:05d}_{ext_end:05d}"
                print(
                    f"[PID {os.getpid()} | {file_path.name} | {run_name}] core={core_start}:{core_end} ext={ext_start}:{ext_end}",
                    flush=True,
                )

                res = run_pipeline_for_slice(
                    rad2d_raw=rad2d,
                    x_slice=x_slice,
                    outdir=outdir,
                    core_start_unbinned=core_start,
                    core_end_unbinned=core_end,
                    x_base_unbinned=x_base_unbinned,
                    cfg=cfg,
                    slice_no=slice_no,
                    product_type=product_type,
                    params=params,
                )

                stitch_results.append(res)
                all_cluster_records.extend(res.get("cluster_records", []))
                cleanup_memory()

            stitched_dir = OUTPUT_ROOT / DEFAULT_STITCHED_FOLDERNAME
            filename_prefix = f"{file_path.stem}_{run_name}_offset_{x_offset_global:04d}"
            
            stitch_and_save_final_09(
                stitch_results,
                stitched_dir=stitched_dir,
                filename_prefix=filename_prefix,
                tag=f"{run_name}_offset_{x_offset_global:04d}",
                product_type=product_type,
                params=params,
            )

            OUTPUT_L6_FOLDER.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_L6_FOLDER / make_output_cluster_path(file_path).name

            copy_lxc_to_l6x_with_clusters(
                lxc_path=file_path,
                l6x_path=out_path,
                clusters=all_cluster_records,
                y_dim=y_dim,
                x_dim=x_dim,
                unbin_factor=(BIN_FACTOR if (APPLY_BINNING and BIN_FACTOR > 1) else 1),
                product_type=product_type,
            )

            print(f"[PID {os.getpid()}] Saved L6* clusters: {out_path}", flush=True)

        cleanup_memory()
        return {"file": file_path.name, "ok": True}

    except Exception as e:
        cleanup_memory()
        return {"file": file_path.name, "ok": False, "error": repr(e)}


# --------------------------
# Main
# --------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FOLDER), help="Input folder (e.g. l3)")
    ap.add_argument("--glob", type=str, default=DEFAULT_FILE_GLOB, help='Glob pattern (e.g. "*.nc")')
    ap.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Plot output root folder")
    ap.add_argument("--output-l6", type=str, default=str(DEFAULT_OUTPUT_L6_FOLDER), help="Output L6 folder")
    ap.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto)")
    ap.add_argument("--frame-idx", type=int, default=DEFAULT_FRAME_IDX)
    ap.add_argument("--x-offset", type=int, default=DEFAULT_X_OFFSET)
    ap.add_argument("--save-slice-plots", action="store_true", default=DEFAULT_SAVE_SLICE_PLOTS)
    ap.add_argument("--save-stitched-plot", action="store_true", default=DEFAULT_SAVE_STITCHED_PLOT)
    ap.add_argument("--save-rednoise-diagnostics", action="store_true", default=DEFAULT_SAVE_REDNOISE_DIAGNOSTICS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    input_folder = Path(args.input)
    output_root = Path(args.output_root)
    output_l6 = Path(args.output_l6)

    files = sorted(input_folder.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {input_folder.resolve()}")

    if args.workers and args.workers > 0:
        n_workers = int(args.workers)
    else:
        cpu = os.cpu_count() or 1
        n_workers = max(1, cpu - 1)

    params: Dict[str, Any] = dict(
        INPUT_FOLDER=str(input_folder),
        FILE_GLOB=args.glob,
        OUTPUT_ROOT=str(output_root),
        OUTPUT_L6_FOLDER=str(output_l6),
        SAVE_DPI=DEFAULT_SAVE_DPI,
        FRAME_IDX=int(args.frame_idx),
        X_CHUNK=DEFAULT_X_CHUNK,
        Y_SLICE=DEFAULT_Y_SLICE,
        SLICE_PAD=DEFAULT_SLICE_PAD,
        X_OFFSET=int(args.x_offset),
        REMOVE_TREND=DEFAULT_REMOVE_TREND,
        STANDARDIZE=DEFAULT_STANDARDIZE,
        GAUSS_BLUR_SIGMA=DEFAULT_GAUSS_BLUR_SIGMA,
        APPLY_BINNING=DEFAULT_APPLY_BINNING,
        BIN_FACTOR=DEFAULT_BIN_FACTOR,
        APPLY_TAPER=DEFAULT_APPLY_TAPER,
        TAPER_EDGE_PIXELS=DEFAULT_TAPER_EDGE_PIXELS,
        S0=DEFAULT_S0,
        DJ=DEFAULT_DJ,
        JS=DEFAULT_JS,
        JT=DEFAULT_JT,
        ASPECT=DEFAULT_ASPECT,
        DX_KM_BASE=DEFAULT_DX_KM_BASE,
        DY_KM_BASE=DEFAULT_DY_KM_BASE,
        N_SHOW=DEFAULT_N_SHOW,
        AMP_MIN_FRACTION_OF_CLUSTER_MAX=DEFAULT_AMP_MIN_FRACTION_OF_CLUSTER_MAX,
        QUIVER_STEP_X=DEFAULT_QUIVER_STEP_X,
        QUIVER_STEP_Y=DEFAULT_QUIVER_STEP_Y,
        QUIVER_SCALE=DEFAULT_QUIVER_SCALE,
        APPLY_RED_NOISE_FILTER=DEFAULT_APPLY_RED_NOISE_FILTER,
        SAVE_REDNOISE_DIAGNOSTICS=bool(args.save_rednoise_diagnostics),
        SAVE_SLICE_PLOTS=bool(args.save_slice_plots),
        SAVE_STITCHED_PLOT=bool(args.save_stitched_plot),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    output_l6.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} files in {input_folder.resolve()}")
    print(f"Workers: {n_workers}")
    print(f"L6 output folder: {output_l6.resolve()}")
    print(f"Plot output root: {output_root.resolve()}")
    print(f"SAVE_SLICE_PLOTS = {params['SAVE_SLICE_PLOTS']}")
    print(f"SAVE_STITCHED_PLOT = {params['SAVE_STITCHED_PLOT']}")
    print(f"SAVE_REDNOISE_DIAGNOSTICS = {params['SAVE_REDNOISE_DIAGNOSTICS']}")
    print("")

    ok = 0
    fail = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one_file_worker, str(fp), params): fp for fp in files}

        for fut in as_completed(futs):
            fp = futs[fut]
            res = fut.result()
            if res.get("ok", False):
                ok += 1
                print(f"[DONE] {fp.name}", flush=True)
            else:
                fail += 1
                print(f"[FAILED] {fp.name}  reason={res.get('error')}", flush=True)

    print(f"\nAll done. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()