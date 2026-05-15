#!/usr/bin/env python3
"""
run_batch_parallel_MF.py

BATCH: ALL SLICES matching + per slice combined L6A recon + FULL ORBIT stitched recon
+ slice Temperature + momentum flux for PASSED clusters only
+ sum MF per slice + stitch MF across orbit
+ output NetCDF contains original L6A global attrs + variables, but ONLY PASSED clusters
  are kept for cluster dimension variables.

Parallel behavior:
- Parallelizes across (L6C, L6A) file pairs (each pair is independent).
- Each worker writes its own outputs folder: outputs_ROOT/<key>/...
- Each worker writes one L7A NetCDF into L7_DIR.

Extra filter (L6A only, per slice):
- If an L6A cluster has: median wavelength < 45 km AND maximum amplitude > 4 K AND median angle (0..180 symmetry) between 80 and 100 degrees or if it has maximum amplitude > 6 K then ignore this cluster AND also ignore any other L6A clusters in that slice that overlap it. Overlap means at least 1 shared finite pixel (can change overlap_min_pixels).

Example:
  python run_batch_MF.py --l6 l6 --l7 l7 --out outputs_BATCH --nproc 6

for %m in (04 07 08) do python run_batch_parallel_MF.py --l6 l6/2024/%m --l7 l7/2024/%m --out outputs_matching/2024/%m
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import os
import traceback
import gc

import numpy as np

# IMPORTANT: set backend before importing pyplot in worker context
import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

from netCDF4 import Dataset


# ============================================================
# Defaults (can be overridden by CLI)
# ============================================================
DEFAULT_L6_DIR = Path("l6/2024/03_v23")
DEFAULT_L7_DIR = Path("l7/2024/03_v23")
DEFAULT_OUTROOT = Path("outputs_matching")

SAVE_DPI = 200
DX_KM = 2.0
DY_KM = 2.0

# Synthetic u, v, N^2 settings
TEMP_VAR_NAME = "Temperature"
TIME_INDEX = 0
N2_VALUE = 4e-4
U_VALUE = 0.0
V_VALUE = 0.0

# Momentum flux constants
G_MS2 = 9.52
LAMBDA_Z_KM = 10.0
C_CANCEL = 1.0

# Matching thresholds (tune)
MIN_IOU = 0.05
MIN_INTERSECTION_PIX = 50
MAX_LAMBDA_MED_RATIO = 1.25
MAX_ANGLE_DIFF_DEG = 15.0

# Save best match even if FAIL (does NOT affect passing)
ASSIGN_BEST_EVEN_IF_FAILS = True

# Plot saving switches (batch friendly)
SAVE_PAIR_PLOTS = True          # side by side A vs C match PNGs per slice (heavy)
SAVE_PER_CLUSTER_PANELS = True  # big panels per passed cluster (heavy)
SAVE_STITCHED_PLOTS = True       # MF_stitched.png + MF_stitched_components.png

# Debug option: save each L6A and L6C cluster individually before matching.
# This does not change matching, filtering, MF calculation, or output NetCDF.
SAVE_PREMATCH_CLUSTER_PLOTS = True

# If True, save stitched PNGs into one shared folder (under outroot) instead of per pair folder
SAVE_STITCHED_TO_COMMON_DIR = True
COMMON_STITCHED_SUBDIRNAME = "_stitched_pngs"

# L6A cluster filter settings
FILTER_L6A_BAD_CLUSTERS = True
BAD_LAM_MED_KM_LT = 45.0
BAD_AMP_MAX_K_GT = 4.0
BAD_ANGLE_DEG_BETWEEN = (80.0, 100.0)  # inclusive, on 0..180 symmetry median
BAD_OVERLAP_MIN_PIXELS = 1             # overlap definition for removing neighbors


# ============================================================
# Helpers
# ============================================================
def save_fig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def robust_limits(A: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> Tuple[float, float]:
    a = np.asarray(A)
    good = np.isfinite(a)
    if not np.any(good):
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(a[good], [p_lo, p_hi])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
        lo = float(np.nanmin(a[good]))
        hi = float(np.nanmax(a[good]))
    return (float(lo), float(hi))


def nanpercentile_safe(a: np.ndarray, q: float) -> float:
    good = np.isfinite(a)
    if not np.any(good):
        return np.nan
    return float(np.nanpercentile(a[good], q))


def circ_diff_deg_180(a_deg: float, b_deg: float) -> float:
    if (not np.isfinite(a_deg)) or (not np.isfinite(b_deg)):
        return np.nan
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def circ_median_deg_180(theta_rad: np.ndarray, mask: np.ndarray) -> float:
    th = np.asarray(theta_rad, dtype=float)
    m = mask & np.isfinite(th)
    if not np.any(m):
        return np.nan
    ang = th[m]
    z = np.exp(1j * 2.0 * ang)
    zmean = np.mean(z)
    if (not np.isfinite(zmean.real)) or (not np.isfinite(zmean.imag)) or abs(zmean) == 0:
        return np.nan
    th_mean = 0.5 * np.angle(zmean)
    deg = (np.degrees(th_mean) + 180.0) % 180.0
    return float(deg)


def pad_right_to(A: np.ndarray, target_nx: int, fill=np.nan) -> np.ndarray:
    """
    Pad a 2D array on the right so it becomes (ny, target_nx).
    """
    A = np.asarray(A)
    ny, nx = A.shape
    if nx >= target_nx:
        return A
    out = np.full((ny, target_nx), fill, dtype=A.dtype)
    out[:, :nx] = A
    return out


def compute_expected_full_nx(slices_no_all: np.ndarray, x_tile: int) -> int:
    """
    Expected full swath width implied by slice numbers and tile width.
    """
    s = np.asarray(slices_no_all, dtype=float)
    s = s[np.isfinite(s)]
    s = s[s >= 0]
    if s.size == 0:
        return int(x_tile)
    max_slice = int(np.max(s))
    return (max_slice + 1) * int(x_tile)


# ============================================================
# L6A filter helpers (bad cluster + overlapping neighbors)
# ============================================================
def cluster_stats_for_filter(A: np.ndarray, L: np.ndarray, T_rad: np.ndarray) -> Dict[str, float]:
    """
    Stats computed over the cluster mask = finite amplitude pixels.
    Returns: lam_med_km, amp_max_k, ang_med_deg_180
    """
    mask = np.isfinite(A)
    if not np.any(mask):
        return dict(lam_med_km=np.nan, amp_max_k=np.nan, ang_med_deg_180=np.nan)

    lam_med = nanpercentile_safe(L[mask], 50)
    amp_max = float(np.nanmax(A[mask])) if np.any(np.isfinite(A[mask])) else np.nan
    ang_med_deg_180 = circ_median_deg_180(T_rad, mask)

    return dict(lam_med_km=float(lam_med), amp_max_k=float(amp_max), ang_med_deg_180=float(ang_med_deg_180))


def is_bad_l6a_cluster(stats: Dict[str, float]) -> bool:
    lam_med = stats["lam_med_km"]
    amp_max = stats["amp_max_k"]
    ang_med = stats["ang_med_deg_180"]

    if (not np.isfinite(amp_max)):
        return False

    # Rule 2: always bad if amp_max > 6 K (no other conditions needed)
    if amp_max > 6.0:
        return True

    # Rule 1: bad if (lam_med < 45 km) AND (amp_max > 4 K) AND (angle in [80, 100])
    if (not np.isfinite(lam_med)) or (not np.isfinite(ang_med)):
        return False

    lo, hi = BAD_ANGLE_DEG_BETWEEN
    return (lam_med < BAD_LAM_MED_KM_LT) and (amp_max > BAD_AMP_MAX_K_GT) and (lo <= ang_med <= hi)

def filter_l6a_clusters_and_overlaps_in_slice(
    l6a: Dict[str, Any],
    idx_a_in_slice: np.ndarray,
    overlap_min_pixels: int = 1,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]], List[int]]:
    """
    Returns:
      kept_indices: L6A global indices in this slice after removing bad clusters and overlaps
      stats_by_idx: stats for each cluster index in slice
      removed_indices: list of removed indices (bad + overlaps)
    """
    idx = np.asarray(idx_a_in_slice, dtype=int)
    if idx.size == 0:
        return idx, {}, []

    masks: Dict[int, np.ndarray] = {}
    stats_by_idx: Dict[int, Dict[str, float]] = {}

    for a_idx in idx:
        a_idx_i = int(a_idx)
        A = np.asarray(l6a["Amplitude"][a_idx_i], dtype=float)
        L = np.asarray(l6a["DominantWavelength"][a_idx_i], dtype=float)
        T = np.asarray(l6a["Angle"][a_idx_i], dtype=float)
        masks[a_idx_i] = np.isfinite(A)
        stats_by_idx[a_idx_i] = cluster_stats_for_filter(A, L, T)

    bad = set()
    for a_idx in idx:
        a_idx_i = int(a_idx)
        if is_bad_l6a_cluster(stats_by_idx[a_idx_i]):
            bad.add(a_idx_i)

    if not bad:
        return idx, stats_by_idx, []

    all_idx = [int(x) for x in idx]
    for bad_idx in list(bad):
        m_bad = masks[bad_idx]
        for other_idx in all_idx:
            if other_idx in bad:
                continue
            inter = int(np.count_nonzero(m_bad & masks[other_idx]))
            if inter >= overlap_min_pixels:
                bad.add(other_idx)

    removed = sorted(bad)
    kept = np.asarray([i for i in all_idx if i not in bad], dtype=int)
    return kept, stats_by_idx, removed


# ============================================================
# Geodesic bearing: 0 = East, CCW
# ============================================================
def initial_bearing_deg_north_cw(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    az = np.arctan2(x, y)
    return (np.rad2deg(az) + 360.0) % 360.0


def bearing_deg_east_ccw_from_points(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    az_north_cw = initial_bearing_deg_north_cw(lat1_deg, lon1_deg, lat2_deg, lon2_deg)
    return (90.0 - az_north_cw) % 360.0


def compute_bearing_to_x_neighbor_east_ccw(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat2d, dtype=float)
    lon = np.asarray(lon2d, dtype=float)

    lat2 = np.empty_like(lat)
    lon2 = np.empty_like(lon)

    lat2[:, :-1] = lat[:, 1:]
    lon2[:, :-1] = lon[:, 1:]
    lat2[:, -1] = lat[:, -2]
    lon2[:, -1] = lon[:, -2]

    bearing = bearing_deg_east_ccw_from_points(lat, lon, lat2, lon2)
    good = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(lat2) & np.isfinite(lon2)
    return np.where(good, bearing, np.nan)


# ============================================================
# Matching IO (memory friendly): do NOT load ClusterReconstruction upfront
# ============================================================
def load_l6_for_matching(path: Path) -> Dict[str, Any]:
    with Dataset(path, "r") as nc:
        return {
            "path": str(path),
            "SlicesNo": np.asarray(nc.variables["SlicesNo"][:], dtype=int),
            "Amplitude": np.asarray(nc.variables["Amplitude"][:], dtype=float),
            "DominantWavelength": np.asarray(nc.variables["DominantWavelength"][:], dtype=float),
            "Angle": np.asarray(nc.variables["Angle"][:], dtype=float),
        }


def get_cluster_maps(data: Dict[str, Any], idx: int) -> Dict[str, np.ndarray]:
    """
    Lazy load ClusterReconstruction only for requested idx (only used if SAVE_PAIR_PLOTS=True).
    """
    A = np.asarray(data["Amplitude"][idx], dtype=float)
    L = np.asarray(data["DominantWavelength"][idx], dtype=float)
    T = np.asarray(data["Angle"][idx], dtype=float)

    R = np.full_like(A, np.nan)
    with Dataset(data["path"], "r") as nc:
        if "ClusterReconstruction" in nc.variables:
            R = np.asarray(nc.variables["ClusterReconstruction"][idx], dtype=float)

    return dict(R=R, A=A, L=L, T=T)


def compute_match_metrics(
    A_src: np.ndarray, L_src: np.ndarray, T_src: np.ndarray,
    A_cand: np.ndarray, L_cand: np.ndarray, T_cand: np.ndarray,
) -> Dict[str, Any]:
    mask_src = np.isfinite(A_src)
    mask_cand = np.isfinite(A_cand)
    inter = mask_src & mask_cand
    union = mask_src | mask_cand

    n_inter = int(np.count_nonzero(inter))
    n_union = int(np.count_nonzero(union))
    iou = (n_inter / n_union) if n_union > 0 else 0.0

    lam_src = L_src[inter]
    lam_cand = L_cand[inter]
    lam_src_med = nanpercentile_safe(lam_src, 50)
    lam_cand_med = nanpercentile_safe(lam_cand, 50)

    if np.isfinite(lam_src_med) and np.isfinite(lam_cand_med) and lam_src_med > 0 and lam_cand_med > 0:
        lam_ratio = max(lam_src_med, lam_cand_med) / min(lam_src_med, lam_cand_med)
    else:
        lam_ratio = np.nan

    ang_src_deg = circ_median_deg_180(T_src, inter)
    ang_cand_deg = circ_median_deg_180(T_cand, inter)
    ang_diff_deg = circ_diff_deg_180(ang_src_deg, ang_cand_deg)

    return dict(
        n_intersection=n_inter,
        iou=float(iou),
        lam_src_med=float(lam_src_med),
        lam_cand_med=float(lam_cand_med),
        lam_ratio=float(lam_ratio),
        ang_src_deg=float(ang_src_deg),
        ang_cand_deg=float(ang_cand_deg),
        ang_diff_deg=float(ang_diff_deg),
    )


def passes_thresholds(m: Dict[str, Any]) -> bool:
    if m["iou"] < MIN_IOU:
        return False
    if m["n_intersection"] < MIN_INTERSECTION_PIX:
        return False
    if (not np.isfinite(m["lam_ratio"])) or (m["lam_ratio"] > MAX_LAMBDA_MED_RATIO):
        return False
    if (not np.isfinite(m["ang_diff_deg"])) or (m["ang_diff_deg"] > MAX_ANGLE_DIFF_DEG):
        return False
    return True


def score(m: Dict[str, Any]) -> float:
    ang = m["ang_diff_deg"]
    if not np.isfinite(ang):
        ang = 999.0
    r = m["lam_ratio"]
    if (not np.isfinite(r)) or r <= 0:
        return 0.0
    return float(m["iou"]) * (1.0 / float(r)) / (1.0 + float(ang))


def plot_side_by_side_4rows(
    left: Dict[str, np.ndarray],
    right: Dict[str, np.ndarray],
    title_left: str,
    title_right: str,
    outpath: Path,
    dx_km: float,
    dy_km: float,
) -> None:
    ny, nx = left["A"].shape
    x1d_km = np.arange(nx, dtype=float) * dx_km
    y1d_km = np.arange(ny, dtype=float) * dy_km

    Rl_vmin, Rl_vmax = robust_limits(left["R"])
    Rr_vmin, Rr_vmax = robust_limits(right["R"])
    Al_vmin, Al_vmax = robust_limits(left["A"])
    Ar_vmin, Ar_vmax = robust_limits(right["A"])

    L_vmin, L_vmax = robust_limits(np.concatenate([left["L"].ravel(), right["L"].ravel()]), 5, 95)
    T_vmin, T_vmax = robust_limits(np.concatenate([left["T"].ravel(), right["T"].ravel()]))

    fig, axs = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True, sharex=True, sharey=True)

    def _panel(ax, Z, vmin, vmax, label, cmap=None):
        im = ax.pcolormesh(x1d_km, y1d_km, Z, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(label)
        ax.set_aspect("equal", adjustable="box")

    axs[0, 0].set_title(title_left)
    axs[0, 1].set_title(title_right)

    _panel(axs[0, 0], left["R"], Rl_vmin, Rl_vmax, "arb", cmap="RdBu_r")
    _panel(axs[0, 1], right["R"], Rr_vmin, Rr_vmax, "arb", cmap="RdBu_r")
    axs[0, 0].set_ylabel("Recon\ny (km)")

    _panel(axs[1, 0], left["A"], Al_vmin, Al_vmax, "K")
    _panel(axs[1, 1], right["A"], Ar_vmin, Ar_vmax, "kR")
    axs[1, 0].set_ylabel("Amp\ny (km)")

    _panel(axs[2, 0], left["L"], L_vmin, L_vmax, "km")
    _panel(axs[2, 1], right["L"], L_vmin, L_vmax, "km")
    axs[2, 0].set_ylabel("Lambda\ny (km)")

    _panel(axs[3, 0], left["T"], T_vmin, T_vmax, "rad")
    _panel(axs[3, 1], right["T"], T_vmin, T_vmax, "rad")
    axs[3, 0].set_ylabel("Angle\ny (km)")

    for r in range(4):
        for c in range(2):
            axs[r, c].set_xlabel("x (km)")

    save_fig(fig, outpath)


def plot_each_cluster_before_matching(
    data: Dict[str, Any],
    idx_list: np.ndarray,
    slice_no: int,
    label: str,
    outdir: Path,
    dx_km: float = DX_KM,
    dy_km: float = DY_KM,
) -> None:
    """
    Debug plot only: save each cluster as its own PNG before matching.

    This does not affect matching, filtering, MF calculation, or NetCDF output.
    It only reads the Amplitude map already loaded by load_l6_for_matching().
    """
    outdir.mkdir(parents=True, exist_ok=True)
    idx_list = np.asarray(idx_list, dtype=int)

    if idx_list.size == 0:
        return

    for cluster_idx in idx_list:
        cluster_idx = int(cluster_idx)
        with Dataset(data["path"], "r") as nc:
            A = np.asarray(nc.variables["ClusterReconstruction"][cluster_idx], dtype=float)

        ny, nx = A.shape
        x1d_km = np.arange(nx, dtype=float) * dx_km
        y1d_km = np.arange(ny, dtype=float) * dy_km

        vmin, vmax = robust_limits(A, 2, 98)
        n_pix = int(np.count_nonzero(np.isfinite(A)))

        fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
        im = ax.pcolormesh(
            x1d_km,
            y1d_km,
            A,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"{label} | slice {slice_no:03d} | cluster {cluster_idx:04d} | n={n_pix}")
        ax.set_xlabel("x km")
        ax.set_ylabel("y km")
        ax.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Amplitude")

        save_fig(
            fig,
            outdir / f"slice_{slice_no:03d}_{label}_cluster_{cluster_idx:04d}_before_matching.png",
        )



# ============================================================
# Flatten swath BEFORE slicing
# ============================================================
def flatten_swath(Z: np.ndarray, target_ny: int | None = None) -> np.ndarray:
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


# ============================================================
# L6A cloning with PASSED ONLY cluster variables + new vars
# ============================================================
def create_passed_only_l6a_with_new_vars(
    src_l6a: Path,
    dst_l7a: Path,
    passed_cluster_indices: np.ndarray,
    time_index: int,
) -> Tuple[str, str, str]:
    passed_cluster_indices = np.asarray(passed_cluster_indices, dtype=int)
    if passed_cluster_indices.size == 0:
        raise ValueError("No passed clusters. Output would be empty.")

    dst_l7a.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(src_l6a, "r") as src:
        amp_var = src.variables["Amplitude"]
        cluster_dim, ytile_dim, xtile_dim = amp_var.dimensions[:3]
        _, y_tile, x_tile = amp_var.shape

        temp_var = src.variables[TEMP_VAR_NAME]
        temp_dims = temp_var.dimensions
        temp_shape = temp_var.shape
        temp_raw_2d = np.asarray(temp_var[time_index], dtype=float)

        # Expected full width implied by slice numbering, used to pad short swaths
        slices_no_all = np.asarray(src.variables["SlicesNo"][:], dtype=int)
        expected_nx = compute_expected_full_nx(slices_no_all, x_tile)

        u_raw_2d = (
            np.asarray(src.variables["u"][time_index], dtype=float)
            if "u" in src.variables
            else np.full_like(temp_raw_2d, U_VALUE, dtype=float)
        )
        v_raw_2d = (
            np.asarray(src.variables["v"][time_index], dtype=float)
            if "v" in src.variables
            else np.full_like(temp_raw_2d, V_VALUE, dtype=float)
        )
        n2_raw_2d = (
            np.asarray(src.variables["NSquared"][time_index], dtype=float)
            if "NSquared" in src.variables
            else np.full_like(temp_raw_2d, N2_VALUE, dtype=float)
        )

        # Latitude, Longitude optional, bearing uses RAW geo
        if ("Latitude" in src.variables) and ("Longitude" in src.variables):
            lat_raw_2d = np.asarray(src.variables["Latitude"][time_index], dtype=float)
            lon_raw_2d = np.asarray(src.variables["Longitude"][time_index], dtype=float)

            bearing_raw_2d = compute_bearing_to_x_neighbor_east_ccw(lat_raw_2d, lon_raw_2d)

            lat_full_2d = flatten_swath(lat_raw_2d, target_ny=y_tile)
            lon_full_2d = flatten_swath(lon_raw_2d, target_ny=y_tile)
        else:
            lat_raw_2d = None
            lon_raw_2d = None
            bearing_raw_2d = np.full_like(temp_raw_2d, np.nan, dtype=float)
            lat_full_2d = np.full((y_tile, temp_raw_2d.shape[1]), np.nan, dtype=float)
            lon_full_2d = np.full((y_tile, temp_raw_2d.shape[1]), np.nan, dtype=float)

        temp_full_2d = flatten_swath(temp_raw_2d, target_ny=y_tile)
        u_full_2d = flatten_swath(u_raw_2d, target_ny=y_tile)
        v_full_2d = flatten_swath(v_raw_2d, target_ny=y_tile)
        n2_full_2d = flatten_swath(n2_raw_2d, target_ny=y_tile)
        bearing_full_2d = flatten_swath(bearing_raw_2d, target_ny=y_tile)

        # Pad right so every tile slice is always exactly x_tile wide even if swath is short
        temp_full_2d = pad_right_to(temp_full_2d, expected_nx, fill=np.nan)
        u_full_2d = pad_right_to(u_full_2d, expected_nx, fill=np.nan)
        v_full_2d = pad_right_to(v_full_2d, expected_nx, fill=np.nan)
        n2_full_2d = pad_right_to(n2_full_2d, expected_nx, fill=np.nan)
        bearing_full_2d = pad_right_to(bearing_full_2d, expected_nx, fill=np.nan)
        lat_full_2d = pad_right_to(lat_full_2d, expected_nx, fill=np.nan)
        lon_full_2d = pad_right_to(lon_full_2d, expected_nx, fill=np.nan)

        n_passed = int(passed_cluster_indices.size)

        with Dataset(dst_l7a, "w") as dst:
            for aname in src.ncattrs():
                dst.setncattr(aname, src.getncattr(aname))

            for dname, dim in src.dimensions.items():
                if dname == cluster_dim:
                    dst.createDimension(dname, n_passed)
                else:
                    dst.createDimension(dname, (None if dim.isunlimited() else len(dim)))

            # Copy all variables, but subset cluster_dim variables to PASSED ONLY
            for vname, vsrc in src.variables.items():
                fill_value = vsrc.getncattr("_FillValue") if "_FillValue" in vsrc.ncattrs() else None
                vdst = dst.createVariable(
                    vname,
                    vsrc.dtype,
                    vsrc.dimensions,
                    zlib=True,
                    complevel=4,
                    fill_value=fill_value,
                )
                for aname in vsrc.ncattrs():
                    if aname != "_FillValue":
                        vdst.setncattr(aname, vsrc.getncattr(aname))

                if cluster_dim in vsrc.dimensions:
                    ax = vsrc.dimensions.index(cluster_dim)
                    vdst[:] = np.take(vsrc[:], passed_cluster_indices, axis=ax)
                else:
                    vdst[:] = vsrc[:]

            # Bearing full field (same dims as Temperature)
            bearing_full = np.zeros(temp_shape, dtype=np.float32)
            ny_out = temp_shape[1]
            nx_out = temp_shape[2]
            ny_copy = min(y_tile, ny_out)
            nx_copy = min(bearing_full_2d.shape[1], nx_out)
            bearing_full[time_index, 0:ny_copy, 0:nx_copy] = bearing_full_2d[0:ny_copy, 0:nx_copy].astype(np.float32)

            if "Bearing" in dst.variables:
                dst.variables["Bearing"][:] = bearing_full
            else:
                vB = dst.createVariable("Bearing", "f4", temp_dims, zlib=True, complevel=4)
                vB.setncattr("long_name", "Geodesic bearing to +x neighbor (0=East, CCW)")
                vB.setncattr("units", "degree")
                vB[:] = bearing_full

            # Allocate per cluster masked tiles
            temp_cluster = np.full((n_passed, y_tile, x_tile), np.nan, dtype=np.float32)
            u_cluster = np.full_like(temp_cluster, np.nan)
            v_cluster = np.full_like(temp_cluster, np.nan)
            n2_cluster = np.full_like(temp_cluster, np.nan)
            bearing_cluster = np.full_like(temp_cluster, np.nan)

            mf_cluster = np.full_like(temp_cluster, np.nan)
            mfz_cluster = np.full_like(temp_cluster, np.nan)
            mfm_cluster = np.full_like(temp_cluster, np.nan)

            amp_passed = np.asarray(dst.variables["Amplitude"][:], dtype=float)
            lam_passed = np.asarray(dst.variables["DominantWavelength"][:], dtype=float)
            slice_passed = np.asarray(dst.variables["SlicesNo"][:], dtype=int)
            ang_rad_passed = np.asarray(dst.variables["Angle"][:], dtype=float)

            for j in range(n_passed):
                s = int(slice_passed[j])
                x0 = s * x_tile
                x1 = x0 + x_tile

                mask = np.isfinite(amp_passed[j])

                # Tiles are now guaranteed to be (y_tile, x_tile) because of padding above
                Ttile = temp_full_2d[:, x0:x1]
                Utile = u_full_2d[:, x0:x1]
                Vtile = v_full_2d[:, x0:x1]
                N2tile = n2_full_2d[:, x0:x1]
                Btile = bearing_full_2d[:, x0:x1]

                temp_cluster[j][mask] = Ttile[mask]
                u_cluster[j][mask] = Utile[mask]
                v_cluster[j][mask] = Vtile[mask]
                n2_cluster[j][mask] = N2tile[mask]
                bearing_cluster[j][mask] = Btile[mask]

                A = amp_passed[j]
                Lx = lam_passed[j]
                ang_deg_tile = (np.rad2deg(ang_rad_passed[j]) % 360.0)

                good = mask
                good = good & np.isfinite(Ttile) & np.isfinite(N2tile) & np.isfinite(Lx) & np.isfinite(Btile)
                good = good & (Ttile != 0.0) & (N2tile > 0.0) & (Lx > 0.0)

                if np.any(good):
                    mf_val = (
                        0.5
                        * (G_MS2 ** 2)
                        / N2tile[good]
                        * (LAMBDA_Z_KM / Lx[good])
                        * (A[good] / Ttile[good]) ** 2
                        * (1.0 / (C_CANCEL ** 2))
                    )
                    mf_cluster[j][good] = mf_val

                    phi_deg = (Btile[good] + ang_deg_tile[good]) % 360.0
                    phi_rad = np.deg2rad(phi_deg)

                    mfz_cluster[j][good] = mf_val * (np.cos(phi_rad) ** 2)
                    mfm_cluster[j][good] = mf_val * (np.sin(phi_rad) ** 2)

            def _write(name: str, arr: np.ndarray, dims: Tuple[str, ...], long_name: str, units: str):
                if name in dst.variables:
                    dst.variables[name][:] = arr
                else:
                    v = dst.createVariable(name, "f4", dims, zlib=True, complevel=4)
                    v.setncattr("long_name", long_name)
                    v.setncattr("units", units)
                    v[:] = arr

            _write("Temp_cluster", temp_cluster, (cluster_dim, ytile_dim, xtile_dim), "Temperature, cluster masked", "K")
            _write("u_cluster", u_cluster, (cluster_dim, ytile_dim, xtile_dim), "u wind, cluster masked", "m/s")
            _write("v_cluster", v_cluster, (cluster_dim, ytile_dim, xtile_dim), "v wind, cluster masked", "m/s")
            _write("NSquared_cluster", n2_cluster, (cluster_dim, ytile_dim, xtile_dim), "N^2, cluster masked", "s^-2")
            _write("Bearing_cluster", bearing_cluster, (cluster_dim, ytile_dim, xtile_dim), "Bearing (0=East, CCW), cluster masked", "degree")

            _write("MF_cluster", mf_cluster, (cluster_dim, ytile_dim, xtile_dim), "MF proxy per pixel, cluster masked", "m2 s-2")
            _write("MFz_cluster", mfz_cluster, (cluster_dim, ytile_dim, xtile_dim), "MF cos^2(phi)", "m2 s-2")
            _write("MFm_cluster", mfm_cluster, (cluster_dim, ytile_dim, xtile_dim), "MF sin^2(phi)", "m2 s-2")

            # Slice sums on tile grid
            smin, smax = int(np.min(slice_passed)), int(np.max(slice_passed))
            full_slices = np.arange(smin, smax + 1)

            if "slice" not in dst.dimensions:
                dst.createDimension("slice", len(full_slices))
            if "SliceNo_passed_slices" not in dst.variables:
                dst.createVariable("SliceNo_passed_slices", "i4", ("slice",))[:] = full_slices
            else:
                dst.variables["SliceNo_passed_slices"][:] = full_slices

            mf_slice = np.zeros((len(full_slices), y_tile, x_tile), dtype=np.float32)
            mfz_slice = np.zeros_like(mf_slice)
            mfm_slice = np.zeros_like(mf_slice)

            for k, s in enumerate(full_slices):
                idx2 = np.where(slice_passed == s)[0]
                if idx2.size > 0:
                    mf_slice[k] = np.nansum(mf_cluster[idx2], axis=0)
                    mfz_slice[k] = np.nansum(mfz_cluster[idx2], axis=0)
                    mfm_slice[k] = np.nansum(mfm_cluster[idx2], axis=0)

            _write("MF_slice", mf_slice, ("slice", ytile_dim, xtile_dim), "Per slice MF sum (passed clusters), tile grid", "m2 s-2")
            _write("MFz_slice", mfz_slice, ("slice", ytile_dim, xtile_dim), "Per slice zonal MF sum (passed clusters), tile grid", "m2 s-2")
            _write("MFm_slice", mfm_slice, ("slice", ytile_dim, xtile_dim), "Per slice meridional MF sum (passed clusters), tile grid", "m2 s-2")

            # Stitch onto full x (top tile rows only)
            MF_top = np.full(temp_shape, np.nan, dtype=np.float32)
            MFz_top = np.full(temp_shape, np.nan, dtype=np.float32)
            MFm_top = np.full(temp_shape, np.nan, dtype=np.float32)

            for k, s in enumerate(full_slices):
                x0 = s * x_tile
                x1 = x0 + x_tile
                # Clamp to output width
                nx_out = temp_shape[2]
                x0c = max(0, min(x0, nx_out))
                x1c = max(0, min(x1, nx_out))
                if x1c <= x0c:
                    continue
                w = x1c - x0c
                MF_top[time_index, 0:y_tile, x0c:x1c] = mf_slice[k][:, :w]
                MFz_top[time_index, 0:y_tile, x0c:x1c] = mfz_slice[k][:, :w]
                MFm_top[time_index, 0:y_tile, x0c:x1c] = mfm_slice[k][:, :w]

            _write("MF_top", MF_top, temp_dims, "Stitched MF sum (passed clusters) on top tile rows only", "m2 s-2")
            _write("MFz_top", MFz_top, temp_dims, "Stitched zonal MF share on top tile rows only", "m2 s-2")
            _write("MFm_top", MFm_top, temp_dims, "Stitched meridional MF share on top tile rows only", "m2 s-2")

            # Flattened toprows Temperature, Latitude, Longitude aligned with MF_top
            T_top = np.full(temp_shape, np.nan, dtype=np.float32)
            Lat_top = np.full(temp_shape, np.nan, dtype=np.float32)
            Lon_top = np.full(temp_shape, np.nan, dtype=np.float32)

            nx_out = temp_shape[2]
            nx_copy = min(nx_out, temp_full_2d.shape[1])

            T_top[time_index, 0:y_tile, 0:nx_copy] = temp_full_2d[:, 0:nx_copy].astype(np.float32)
            Lat_top[time_index, 0:y_tile, 0:nx_copy] = lat_full_2d[:, 0:nx_copy].astype(np.float32)
            Lon_top[time_index, 0:y_tile, 0:nx_copy] = lon_full_2d[:, 0:nx_copy].astype(np.float32)

            _write("T_top", T_top, temp_dims, "Flattened Temperature on MF_top grid (top tile rows only)", "K")
            _write("Lat_top", Lat_top, temp_dims, "Flattened Latitude on MF_top grid (top tile rows only)", "degree_north")
            _write("Lon_top", Lon_top, temp_dims, "Flattened Longitude on MF_top grid (top tile rows only)", "degree_east")

        del temp_raw_2d, u_raw_2d, v_raw_2d, n2_raw_2d
        del temp_full_2d, u_full_2d, v_full_2d, n2_full_2d
        del bearing_raw_2d, bearing_full_2d
        del temp_cluster, u_cluster, v_cluster, n2_cluster, bearing_cluster
        del mf_cluster, mfz_cluster, mfm_cluster
        gc.collect()

    return cluster_dim, ytile_dim, xtile_dim


# ============================================================
# Plotting
# ============================================================
def plot_stitched_mf_components(nc_path: Path, outpng: Path, time_index: int) -> None:
    outpng.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(nc_path, "r") as nc:
        MF = np.asarray(nc.variables["MF_top"][time_index], dtype=float)
        MFz = np.asarray(nc.variables["MFz_top"][time_index], dtype=float)
        MFm = np.asarray(nc.variables["MFm_top"][time_index], dtype=float)

    ny, nx = MF.shape
    x1d_km = np.arange(nx, dtype=float) * DX_KM
    y1d_km = np.arange(ny, dtype=float) * DY_KM

    stacked = np.concatenate([MF.ravel(), MFz.ravel(), MFm.ravel()])
    vmin, vmax = robust_limits(stacked, 2, 99.9)

    fig, axs = plt.subplots(3, 1, figsize=(22, 8), constrained_layout=True, sharex=True, sharey=True)

    axs[0].pcolormesh(x1d_km, y1d_km, MF, shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Stitched MF (passed clusters summed)")
    axs[0].set_ylabel("y km")
    axs[0].set_aspect("equal", adjustable="box")

    axs[1].pcolormesh(x1d_km, y1d_km, MFz, shading="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Stitched MF zonal share cos2")
    axs[1].set_ylabel("y km")
    axs[1].set_aspect("equal", adjustable="box")

    im2 = axs[2].pcolormesh(x1d_km, y1d_km, MFm, shading="auto", vmin=vmin, vmax=vmax)
    axs[2].set_title("Stitched MF meridional share sin2")
    axs[2].set_xlabel("x km")
    axs[2].set_ylabel("y km")
    axs[2].set_aspect("equal", adjustable="box")

    for ax in axs:
        ax.set_ylim(0, 600)

    cbar = fig.colorbar(im2, ax=axs, fraction=0.02, pad=0.01)
    cbar.set_label("MF m2 s-2")
    save_fig(fig, outpng)

    del MF, MFz, MFm, stacked, fig, axs
    gc.collect()


# ============================================================
# Pairing utilities
# ============================================================
def _key_from_stem(stem: str) -> str:
    """
    Turn:
      awe_l6c_q20_2024165T2120_03175_v01
      awe_l6a_tmp_2024165T2120_03175_v01
    into the common key:
      2024165T2120_03175_v01
    """
    parts = stem.split("_")
    if len(parts) < 5:
        return stem
    kind = parts[1]  # l6a or l6c
    if kind in {"l6c", "l6a"}:
        return "_".join(parts[3:])
    return "_".join(parts[2:])


def find_l6_pairs(l6_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Returns list of (l6c_path, l6a_path, key)
    """
    l6_dir = Path(l6_dir)
    l6c_files = sorted(l6_dir.glob("*l6c*.nc"))
    l6a_files = sorted(l6_dir.glob("*l6a*.nc"))

    l6c_by_key: Dict[str, Path] = {_key_from_stem(p.stem): p for p in l6c_files}
    l6a_by_key: Dict[str, Path] = {_key_from_stem(p.stem): p for p in l6a_files}

    keys = sorted(set(l6c_by_key) & set(l6a_by_key))
    return [(l6c_by_key[k], l6a_by_key[k], k) for k in keys]


def make_l7a_name_from_l6a(l6a_path: Path) -> str:
    """
    Rename l6a -> l7a in the filename, keep everything else.
    """
    name = l6a_path.name
    name = name.replace("_l6a_", "_l7a_")
    name = name.replace("l6a", "l7a")
    return name


# ============================================================
# One pair runner
# ============================================================
def run_one_pair(
    l6c_path: Path,
    l6a_path: Path,
    outdir: Path,
    l7_dir: Path,
    stitched_png_dir: Path | None = None,
    key: str | None = None,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    l7_dir.mkdir(parents=True, exist_ok=True)

    out_l7a = l7_dir / make_l7a_name_from_l6a(l6a_path)

    l6c = load_l6_for_matching(l6c_path)
    l6a = load_l6_for_matching(l6a_path)

    all_slices_a = np.unique(l6a["SlicesNo"]).astype(int)
    all_slices_a = all_slices_a[all_slices_a >= 0]
    all_slices_a = np.sort(all_slices_a)

    summary_lines: List[str] = []
    summary_lines.append(f"L6A: {l6a_path}\n")
    summary_lines.append(f"L6C: {l6c_path}\n\n")
    summary_lines.append("slice  nA_raw  nA_kept  nC  nBestSaved  nPassed  note\n")

    passed_global_indices: List[int] = []

    for slice_no in all_slices_a:
        idx_a_raw = np.where(l6a["SlicesNo"] == int(slice_no))[0]
        idx_c = np.where(l6c["SlicesNo"] == int(slice_no))[0]

        slice_dir = outdir / f"slice_{slice_no:03d}"
        pairs_dir = slice_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        # Optional debug plots BEFORE matching/filtering.
        # Saves each L6A and L6C cluster individually so you can inspect them by eye.
        # This does not change the pipeline.
        if SAVE_PREMATCH_CLUSTER_PLOTS:
            prematch_dir = slice_dir / "prematch_clusters"

            plot_each_cluster_before_matching(
                data=l6a,
                idx_list=idx_a_raw,
                slice_no=int(slice_no),
                label="L6A",
                outdir=prematch_dir / "L6A",
            )

            plot_each_cluster_before_matching(
                data=l6c,
                idx_list=idx_c,
                slice_no=int(slice_no),
                label="L6C",
                outdir=prematch_dir / "L6C",
            )

        # Apply L6A bad cluster filter per slice (optional)
        if FILTER_L6A_BAD_CLUSTERS and idx_a_raw.size > 0:
            idx_a, stats_by_idx, removed = filter_l6a_clusters_and_overlaps_in_slice(
                l6a=l6a,
                idx_a_in_slice=idx_a_raw,
                overlap_min_pixels=BAD_OVERLAP_MIN_PIXELS,
            )
            if removed:
                lines = []
                lines.append(f"Slice {slice_no}\n")
                lines.append(f"Raw L6A clusters: {idx_a_raw.size}\n")
                lines.append(f"Kept after filter: {idx_a.size}\n")
                lines.append(f"Removed (bad + overlaps): {len(removed)}\n\n")
                lines.append("idx  lam_med_km  amp_max_k  ang_med_deg_180  removed\n")
                for a_idx in sorted([int(x) for x in idx_a_raw]):
                    st = stats_by_idx.get(int(a_idx), {})
                    lam = st.get("lam_med_km", np.nan)
                    amp = st.get("amp_max_k", np.nan)
                    ang = st.get("ang_med_deg_180", np.nan)
                    tag = 1 if int(a_idx) in removed else 0
                    lines.append(f"{int(a_idx):4d}  {lam:10.2f}  {amp:9.2f}  {ang:14.2f}  {tag:7d}\n")
                (slice_dir / "L6A_FILTER_REPORT.txt").write_text("".join(lines))
        else:
            idx_a = np.asarray(idx_a_raw, dtype=int)

        if idx_a.size == 0:
            if idx_a_raw.size == 0:
                note = "no L6A clusters"
            else:
                note = "all L6A filtered"
            summary_lines.append(
                f"{slice_no:5d}  {idx_a_raw.size:6d}  {idx_a.size:7d}  {idx_c.size:2d}  {0:10d}  {0:7d}  {note}\n"
            )
            continue

        if idx_c.size == 0:
            (slice_dir / "NO_L6C_CLUSTERS.txt").write_text(f"No L6C clusters for slice {slice_no}\n")
            summary_lines.append(
                f"{slice_no:5d}  {idx_a_raw.size:6d}  {idx_a.size:7d}  {0:2d}  {0:10d}  {0:7d}  no L6C clusters\n"
            )
            continue

        all_candidate_rows: List[Dict[str, Any]] = []
        best_rows: List[Dict[str, Any]] = []

        for a_idx in idx_a:
            a_idx = int(a_idx)
            A_a = l6a["Amplitude"][a_idx]
            L_a = l6a["DominantWavelength"][a_idx]
            T_a = l6a["Angle"][a_idx]

            candidates: List[Dict[str, Any]] = []
            for c_idx in idx_c:
                c_idx = int(c_idx)
                A_c = l6c["Amplitude"][c_idx]
                L_c = l6c["DominantWavelength"][c_idx]
                T_c = l6c["Angle"][c_idx]

                m = compute_match_metrics(A_a, L_a, T_a, A_c, L_c, T_c)
                m["cluster_a"] = a_idx
                m["cluster_c"] = c_idx
                m["passes"] = bool(passes_thresholds(m))
                m["score"] = float(score(m))
                candidates.append(m)
                all_candidate_rows.append(m)

            passed = [c for c in candidates if c["passes"]]
            best = max(passed, key=lambda d: d["score"]) if passed else max(candidates, key=lambda d: d["score"])

            if best["passes"] or ASSIGN_BEST_EVEN_IF_FAILS:
                best_rows.append(best)

                if SAVE_PAIR_PLOTS:
                    left_maps = get_cluster_maps(l6a, best["cluster_a"])
                    right_maps = get_cluster_maps(l6c, best["cluster_c"])

                    pass_tag = "PASS" if best["passes"] else "FAIL"
                    outpng = pairs_dir / f"slice_{slice_no:03d}_A_{best['cluster_a']:04d}_to_C_{best['cluster_c']:04d}_{pass_tag}.png"

                    title_left = (
                        f"L6A Temp | slice {slice_no} | idx {best['cluster_a']} | "
                        f"n={np.count_nonzero(np.isfinite(left_maps['A']))}"
                    )
                    title_right = (
                        f"L6C Rad | slice {slice_no} | idx {best['cluster_c']} | "
                        f"IoU={best['iou']:.3f} inter={best['n_intersection']} "
                        f"ratio={best['lam_ratio']:.3f} dTheta={best['ang_diff_deg']:.1f}deg"
                    )

                    plot_side_by_side_4rows(
                        left=left_maps,
                        right=right_maps,
                        title_left=title_left,
                        title_right=title_right,
                        outpath=outpng,
                        dx_km=DX_KM,
                        dy_km=DY_KM,
                    )

                    del left_maps, right_maps
                    gc.collect()

        (slice_dir / f"slice_{slice_no:03d}_ALL_candidates_ratio.txt").write_text(
            "cluster_a  cluster_c  passes  score    iou    inter  lam_ratio  ang_diff_deg  thetaA_deg  thetaC_deg  lamA_med  lamC_med\n"
            + "".join(
                f"{m['cluster_a']:9d}  {m['cluster_c']:9d}  {int(m['passes']):6d}  {m['score']:6.3f}  "
                f"{m['iou']:.3f}  {m['n_intersection']:5d}  "
                f"{m['lam_ratio']:8.3f}  {m['ang_diff_deg']:12.1f}  "
                f"{m['ang_src_deg']:10.1f}  {m['ang_cand_deg']:10.1f}  "
                f"{m['lam_src_med']:8.1f}  {m['lam_cand_med']:8.1f}\n"
                for m in sorted(all_candidate_rows, key=lambda d: (d["cluster_a"], -d["score"]))
            )
        )

        (slice_dir / f"slice_{slice_no:03d}_BEST_matches_A_to_C_ratio.txt").write_text(
            "cluster_a  best_cluster_c  passes  score    iou    inter  lam_ratio  ang_diff_deg  thetaA_deg  thetaC_deg  lamA_med  lamC_med\n"
            + "".join(
                f"{m['cluster_a']:9d}  {m['cluster_c']:14d}  {int(m['passes']):6d}  {m['score']:6.3f}  "
                f"{m['iou']:.3f}  {m['n_intersection']:5d}  "
                f"{m['lam_ratio']:8.3f}  {m['ang_diff_deg']:12.1f}  "
                f"{m['ang_src_deg']:10.1f}  {m['ang_cand_deg']:10.1f}  "
                f"{m['lam_src_med']:8.1f}  {m['lam_cand_med']:8.1f}\n"
                for m in sorted(best_rows, key=lambda d: d["cluster_a"])
            )
        )

        passed_here = [int(m["cluster_a"]) for m in best_rows if bool(m["passes"])]
        passed_global_indices.extend(passed_here)

        summary_lines.append(
            f"{slice_no:5d}  {idx_a_raw.size:6d}  {idx_a.size:7d}  {idx_c.size:2d}  {len(best_rows):10d}  {len(passed_here):7d}  ok\n"
        )

        del all_candidate_rows, best_rows
        gc.collect()

    summary_path = outdir / "SUMMARY_all_slices.txt"
    summary_path.write_text("".join(summary_lines))

    passed_global_indices = sorted(set(passed_global_indices))
    passed_idx = np.asarray(passed_global_indices, dtype=int)

    if passed_idx.size == 0:
        return dict(
            ok=False,
            l6c=str(l6c_path),
            l6a=str(l6a_path),
            outdir=str(outdir),
            l7a=str(out_l7a),
            n_passed=0,
            summary=str(summary_path),
            note="No passed clusters",
        )

    create_passed_only_l6a_with_new_vars(
        src_l6a=l6a_path,
        dst_l7a=out_l7a,
        passed_cluster_indices=passed_idx,
        time_index=TIME_INDEX,
    )

    # Decide where stitched PNGs go
    if stitched_png_dir is None:
        stitched_png_dir = outdir
    stitched_png_dir = Path(stitched_png_dir)
    stitched_png_dir.mkdir(parents=True, exist_ok=True)

    # Safe prefix so filenames do not collide
    prefix = key if (key is not None and str(key).strip()) else out_l7a.stem

    if SAVE_STITCHED_PLOTS:
        with Dataset(out_l7a, "r") as nc:
            MF2 = np.asarray(nc.variables["MF_top"][TIME_INDEX], dtype=float)
            ny, nx = MF2.shape
            x1d_km = np.arange(nx, dtype=float) * DX_KM
            y1d_km = np.arange(ny, dtype=float) * DY_KM
            vmin, vmax = robust_limits(MF2, 2, 99.9)

            fig, ax = plt.subplots(figsize=(22, 3), constrained_layout=True)
            im = ax.pcolormesh(x1d_km, y1d_km, MF2, shading="auto", vmin=vmin, vmax=vmax)
            ax.set_title("Momentum Flux (passed clusters summed)")
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
            ax.set_ylim(0, 600)
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("MF (m2 s-2)")

            save_fig(fig, stitched_png_dir / f"{prefix}_MF_stitched.png")

        plot_stitched_mf_components(
            out_l7a,
            stitched_png_dir / f"{prefix}_MF_stitched_components.png",
            TIME_INDEX,
        )

    if SAVE_PER_CLUSTER_PANELS:
        pass

    del l6c, l6a, passed_idx
    mpl.pyplot.close("all")
    gc.collect()

    return dict(
        ok=True,
        l6c=str(l6c_path),
        l6a=str(l6a_path),
        outdir=str(outdir),
        l7a=str(out_l7a),
        n_passed=int(len(passed_global_indices)),
        summary=str(summary_path),
        note="ok",
    )


# ============================================================
# Parallel wrapper
# ============================================================
def _worker_init(matplotlib_backend: str = "Agg") -> None:
    """
    Run once per worker process.
    """
    mpl.use(matplotlib_backend, force=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _run_pair_task(args: Tuple[str, str, str, str, str, str]) -> Dict[str, Any]:
    """
    args: (l6c_path, l6a_path, key, outroot, l7_dir, stitched_png_dir)
    """
    l6c_s, l6a_s, key, outroot_s, l7_s, stitched_s = args
    l6c_path = Path(l6c_s)
    l6a_path = Path(l6a_s)
    outroot = Path(outroot_s)
    l7_dir = Path(l7_s)
    stitched_dir = Path(stitched_s)

    pair_outdir = outroot / key
    try:
        res = run_one_pair(
            l6c_path,
            l6a_path,
            pair_outdir,
            l7_dir=l7_dir,
            stitched_png_dir=stitched_dir,
            key=key,
        )
        res["key"] = key
        return res
    except Exception as e:
        return dict(
            ok=False,
            key=key,
            l6c=str(l6c_path),
            l6a=str(l6a_path),
            outdir=str(pair_outdir),
            error=str(e),
            traceback=traceback.format_exc(),
        )


# ============================================================
# CLI + main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel batch L6A/L6C matching -> L7A with MF + stitched plots.")
    p.add_argument("--l6", type=str, default=str(DEFAULT_L6_DIR), help="Input folder containing L6A/L6C NetCDF files.")
    p.add_argument("--l7", type=str, default=str(DEFAULT_L7_DIR), help="Output folder for L7A NetCDF files.")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUTROOT), help="Output root for per pair plots text.")
    p.add_argument("--nproc", type=int, default=6, help="Number of worker processes.")
    p.add_argument("--chunksize", type=int, default=1, help="Multiprocessing chunksize (usually keep 1 for big tasks).")
    p.add_argument("--no-stitch-plots", action="store_true", help="Disable stitched MF plots (faster).")
    p.add_argument("--no-l6a-filter", action="store_true", help="Disable L6A bad cluster + overlap filtering.")
    p.add_argument(
        "--save-prematch-cluster-plots",
        action="store_true",
        help="Debug only: save each L6A/L6C cluster individually before matching.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    global SAVE_STITCHED_PLOTS
    if args.no_stitch_plots:
        SAVE_STITCHED_PLOTS = False

    global FILTER_L6A_BAD_CLUSTERS
    if args.no_l6a_filter:
        FILTER_L6A_BAD_CLUSTERS = False

    global SAVE_PREMATCH_CLUSTER_PLOTS
    if args.save_prematch_cluster_plots:
        SAVE_PREMATCH_CLUSTER_PLOTS = True

    l6_dir = Path(args.l6)
    l7_dir = Path(args.l7)
    outroot = Path(args.out)

    l7_dir.mkdir(parents=True, exist_ok=True)
    outroot.mkdir(parents=True, exist_ok=True)

    # Shared stitched png folder
    if SAVE_STITCHED_TO_COMMON_DIR:
        stitched_png_dir = outroot / COMMON_STITCHED_SUBDIRNAME
    else:
        stitched_png_dir = outroot
    stitched_png_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_l6_pairs(l6_dir)
    print(f"Found {len(pairs)} L6C/L6A pairs in: {l6_dir.resolve()}")
    if len(pairs) == 0:
        print("No pairs found. Check filenames contain l6a and l6c and share the same key.")
        return 2

    tasks: List[Tuple[str, str, str, str, str, str]] = [
        (str(l6c_path), str(l6a_path), key, str(outroot), str(l7_dir), str(stitched_png_dir))
        for (l6c_path, l6a_path, key) in pairs
    ]

    mpl.use("Agg", force=True)

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"Running with nproc={args.nproc}  chunksize={args.chunksize}")
    print(f"L6A filter enabled: {FILTER_L6A_BAD_CLUSTERS}")
    print(f"Pre-match individual cluster plots enabled: {SAVE_PREMATCH_CLUSTER_PLOTS}")
    print(f"Stitched PNG folder: {stitched_png_dir.resolve()}")

    with mp.Pool(processes=args.nproc, initializer=_worker_init) as pool:
        for res in pool.imap_unordered(_run_pair_task, tasks, chunksize=args.chunksize):
            key = res.get("key", "UNKNOWN")
            ok = bool(res.get("ok"))
            if ok:
                results.append(res)
                print(f"[OK]   {key}  n_passed={res.get('n_passed', 'NA')}  l7a={Path(res.get('l7a','')).name}")
            else:
                errors.append(res)
                print(f"[FAIL] {key}  {res.get('error')}")

    summary_txt = outroot / "BATCH_SUMMARY.txt"
    lines: List[str] = []
    lines.append(f"Input dir: {l6_dir.resolve()}\n")
    lines.append(f"Output nc dir (l7): {l7_dir.resolve()}\n")
    lines.append(f"Outroot: {outroot.resolve()}\n")
    lines.append(f"Stitched png dir: {stitched_png_dir.resolve()}\n")
    lines.append(f"Pairs found: {len(pairs)}\n")
    lines.append(f"Success: {sum(1 for r in results if r.get('ok'))}\n")
    lines.append(f"Fail: {len(errors)}\n\n")

    lines.append("Per pair:\n")
    for r in sorted(results, key=lambda d: d.get("key", "")):
        lines.append(f"{r.get('key')}  ok={r.get('ok')}  n_passed={r.get('n_passed', 0)}  l7a={r.get('l7a')}\n")

    if errors:
        lines.append("\nErrors:\n")
        for e in sorted(errors, key=lambda d: d.get("key", "")):
            lines.append(f"{e.get('key')}  {e.get('error')}\n")

    summary_txt.write_text("".join(lines))

    print("\nBatch done.")
    print("Batch outroot:", outroot.resolve())
    print("L7 output folder:", l7_dir.resolve())
    print("Stitched png folder:", stitched_png_dir.resolve())
    print("Batch summary:", summary_txt.resolve())

    if errors:
        err_dir = outroot / "_errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        for e in errors:
            key = e.get("key", "UNKNOWN")
            (err_dir / f"{key}.txt").write_text(e.get("traceback", "") or str(e))

    return 0 if len(errors) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())