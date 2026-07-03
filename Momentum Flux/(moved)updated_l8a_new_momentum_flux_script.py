#!/usr/bin/env python3
# %%
"""
Run phase-speed calculation for selected/all slices and selected/all clusters.

What this script does:
    - Choose a list of slices: TARGET_SLICES
      Use TARGET_SLICES = None to run all slices.
    - Choose a list of cluster numbers within each slice: TARGET_CLUSTERS_IN_SLICE
      Use TARGET_CLUSTERS_IN_SLICE = None to run all clusters in each slice.
    - If a slice has fewer clusters than requested, it runs available requested clusters only
    - Reads NAVGEM U, V, and NN from NAVGEM_NC_PATH
    - For each cluster, extracts U, V, and NN using the same per-slice logic as the L6 pipeline:
        1) take raw slice with padding
        2) trim all-NaN edge columns inside that slice window
        3) flatten swath on the trimmed window
        4) crop back to the core slice
        5) pad back to the cluster width
    - Saves phase-speed output separately for each cluster
    - Saves all cluster info plots together in one folder per slice
    - Adds amplitude evolution and cosine-fit panel to the main info figure
    - Calculates m² = N² / |c - w|² and λz = 2π/m
    - Calculates new momentum flux using actual λz instead of fixed λz
    - Uses corrected phase-speed direction for signed New MF_u and New MF_v
    - Adds λz, New MF_u, and New MF_v panels to the main info figure
    - Saves one combined summary CSV
    - Copies the old L7A file to a new L8A file and appends the new variables

Example:
    TARGET_SLICES = None
    TARGET_CLUSTERS_IN_SLICE = None

This means:
    run all clusters in all slices.

Example subset:
    TARGET_SLICES = [3, 4]
    TARGET_CLUSTERS_IN_SLICE = [0, 1, 2, 3, 4]
"""

from pathlib import Path
import csv
import shutil
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ============================================================
# Settings
# ============================================================
NC_PATH = Path(
    "l7/2024/03_v23/awe_l7a_tmp_2024075T0538_01770_v23.nc"
)

# NAVGEM file with U, V, and NN/N2/NSquared.
NAVGEM_NC_PATH = Path(r"D:\navgem_2024075T0538_01770.nc")

# Choose slices and clusters here.
TARGET_SLICES = [3]
TARGET_CLUSTERS_IN_SLICE = [0]

DX_KM = 2.0
DY_KM = 2.0
DPI = 200

# Momentum-flux constants.
# Same formula as the L7A creation function, but the fixed lambda_z is replaced
# by the actual lambda_z map calculated from phase speed and NAVGEM winds.
G_MS2 = 9.52
C_CANCEL = 1.0

# These should match the slicing logic used when making the cluster file.
X_CHUNK = 600
SLICE_PAD = 50
Y_SLICE = slice(None, None)
NAVGEM_TARGET_NY = 300

# If True, use the saved representative point from the L7A file.
# If False, use the raw maximum-amplitude pixel from Amplitude.
USE_SAVED_REPRESENTATIVE_POINT = True

# Save one info plot per cluster.
SAVE_FIG = True
PLOT_OUT_BASE_DIR = Path("plots_l7a_cluster_debug")

# ============================================================
# Phase speed settings
# ============================================================
CALCULATE_PHASE_SPEED = True

# L1A/remap file used for the phase-speed calculation.
# This should match the remap altitude you want, e.g. remap85.
L1A_REMAP_NC_PATH = Path(
    r"D:\awe_l1a_q20_2024075T0538_01770_v23_remap85.nc"
)

PHASE_SPEED_BASE_OUT_DIR = Path("AWE/01770/remap85_from_l7a")
PHASE_SPEED_ROI_LABEL = 22
PHASE_SPEED_DT_FRAME_S = 1.1
PHASE_SPEED_SAVE_OVERLAY = True

# Set False to avoid the long repeated print block from calculate_phase_speed_for_l7a_point.
PHASE_SPEED_VERBOSE = False

# Combined summary CSV.
COMBINED_SUMMARY_CSV = PHASE_SPEED_BASE_OUT_DIR / "selected_slices_clusters_phase_speed_summary.csv"

# ============================================================
# L8A output settings
# ============================================================
# If True, copy the input L7A file to a new L8A file and append the new variables.
SAVE_L8A_FILE = True

# By default, keep the file next to NC_PATH and change l7a -> l8a in the filename.
L8A_OUT_PATH = NC_PATH.with_name(
    NC_PATH.name.replace("_l7a_", "_l8a_")
    if "_l7a_" in NC_PATH.name
    else NC_PATH.stem + "_l8a.nc"
)

# Filled while processing clusters, then written to the L8A file at the end.
L8A_CLUSTER_OUTPUTS = []


# ============================================================
# Helpers
# ============================================================
def robust_limits(a, p_lo=2, p_hi=98):
    a = np.asarray(a, dtype=float)
    good = np.isfinite(a)

    if not np.any(good):
        return -1.0, 1.0

    lo, hi = np.nanpercentile(a[good], [p_lo, p_hi])

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo == hi):
        lo = np.nanmin(a[good])
        hi = np.nanmax(a[good])

    return float(lo), float(hi)


def read_cluster_var(nc, name, cluster_idx, shape_like=None):
    """
    Read one cluster-level 2D variable if it exists.
    Otherwise return a NaN array with shape_like.
    """
    if name in nc.variables:
        return np.asarray(nc.variables[name][cluster_idx], dtype=float)

    if shape_like is None:
        return None

    return np.full_like(shape_like, np.nan, dtype=float)


def read_scalar_var(nc, name, cluster_idx, default=np.nan):
    """
    Read one cluster-level 1D scalar if it exists.
    Otherwise return default.
    """
    if name in nc.variables:
        return float(np.asarray(nc.variables[name][cluster_idx]))

    return default


def read_scalar_int_var(nc, name, cluster_idx, default=-1):
    """
    Read one cluster-level 1D integer if it exists.
    Otherwise return default.
    """
    if name in nc.variables:
        return int(np.asarray(nc.variables[name][cluster_idx]))

    return default


def get_point_value(arr, yy, xx):
    if arr is None:
        return np.nan

    if yy < 0 or xx < 0:
        return np.nan

    if yy >= arr.shape[0] or xx >= arr.shape[1]:
        return np.nan

    return float(arr[yy, xx])


def trim_all_nan_edge_columns(Z: np.ndarray):
    """
    Trim only all-NaN columns on the left/right edge.
    This is the same helper logic used by your slicing pipeline.
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D array, got {Z.shape}")

    col_has_data = np.any(np.isfinite(Z), axis=0)
    if not np.any(col_has_data):
        return Z, 0, Z.shape[1]

    x0 = int(np.argmax(col_has_data))
    x1 = int(len(col_has_data) - np.argmax(col_has_data[::-1]))
    return Z[:, x0:x1], x0, x1


def trim_window_all_nan_edges(win: np.ndarray):
    """
    Return trimmed window and number of columns trimmed on each side.
    """
    win_trim, x0, x1 = trim_all_nan_edge_columns(win)
    trim_left = int(x0)
    trim_right = int(win.shape[1] - x1)
    return win_trim, trim_left, trim_right


def flatten_swath(Z: np.ndarray, target_ny: int = 300) -> np.ndarray:
    ny, nx = Z.shape
    out = np.full_like(Z, np.nan, dtype=float)

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


def find_var(nc, possible_names):
    """
    Find the first variable name that exists in the NAVGEM file.
    Add names to the candidate lists below if your file uses different names.
    """
    for name in possible_names:
        if name in nc.variables:
            return name
    raise KeyError(f"Could not find any of these variables: {possible_names}")


def read_navgem_2d_var(nc, possible_names):
    """
    Read one NAVGEM variable and force it to 2D.
    It handles singleton dimensions by squeezing.
    """
    name = find_var(nc, possible_names)
    arr = np.asarray(nc.variables[name][:], dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"NAVGEM variable {name} is not 2D after squeeze. Shape = {arr.shape}"
        )

    print(f"Read NAVGEM {name}: shape={arr.shape}")
    return arr


def read_navgem_data(navgem_nc_path):
    """
    Read raw NAVGEM U, V, and NN fields.
    These are NOT flattened globally. They are flattened per slice later.
    """
    with Dataset(navgem_nc_path, "r") as nc:
        print("=" * 78)
        print(f"NAVGEM file: {navgem_nc_path}")
        print(nc)

        print("\nNAVGEM Dimensions:")
        for name, dim in nc.dimensions.items():
            print(f"  {name}: {len(dim)}")

        print("\nNAVGEM Variables:")
        for name, var in nc.variables.items():
            print(f"  {name}: shape={var.shape}, dtype={var.dtype}")
            if hasattr(var, "units"):
                print(f"      units: {var.units}")
            if hasattr(var, "long_name"):
                print(f"      long_name: {var.long_name}")

        U_raw = read_navgem_2d_var(nc, ["U", "u", "U_top", "u_top", "zonal_wind", "ZonalWind"])
        V_raw = read_navgem_2d_var(nc, ["V", "v", "V_top", "v_top", "meridional_wind", "MeridionalWind"])
        NN_raw = read_navgem_2d_var(nc, ["NN", "N2", "NSquared", "NSquared_top", "nsquared", "N_squared"])

    if U_raw.shape != V_raw.shape or U_raw.shape != NN_raw.shape:
        raise ValueError(
            f"NAVGEM shape mismatch: U={U_raw.shape}, V={V_raw.shape}, NN={NN_raw.shape}"
        )

    print("\nRaw NAVGEM shapes:")
    print(f"  U_raw  : {U_raw.shape}")
    print(f"  V_raw  : {V_raw.shape}")
    print(f"  NN_raw : {NN_raw.shape}")
    print("=" * 78)

    return {
        "U_raw": U_raw,
        "V_raw": V_raw,
        "NN_raw": NN_raw,
    }


def extract_navgem_slice_like_pipeline(
    Z_raw: np.ndarray,
    *,
    slice_no: int,
    cluster_shape: tuple[int, int],
    x_chunk: int = X_CHUNK,
    slice_pad: int = SLICE_PAD,
    y_slice=Y_SLICE,
    target_ny: int = NAVGEM_TARGET_NY,
) -> np.ndarray:
    """
    Extract a NAVGEM field for one L7A/L6 cluster slice using the same slicing logic
    as the decomposition pipeline:

        core_start = slice_no * X_CHUNK
        core_end   = core_start + X_CHUNK
        ext_start  = core_start - SLICE_PAD
        ext_end    = core_end + SLICE_PAD

    Then:
        1) slice raw field with padding
        2) trim all-NaN edge columns within that padded window
        3) flatten swath on the trimmed window
        4) crop the flattened window back to the core region
        5) pad to match cluster_shape

    This avoids the too-simple slice_no * 600 direct crop.
    """
    target_y, target_x = cluster_shape

    nx_total = Z_raw.shape[1]
    core_start = int(slice_no) * int(x_chunk)
    core_end = min(core_start + int(x_chunk), nx_total)

    ext_start = max(0, core_start - int(slice_pad))
    ext_end = min(nx_total, core_end + int(slice_pad))

    if core_start >= nx_total:
        raise ValueError(
            f"slice_no={slice_no} gives core_start={core_start}, but NAVGEM nx={nx_total}"
        )

    raw_win_full = np.asarray(Z_raw[y_slice, ext_start:ext_end], dtype=float)
    raw_win_trim, trim_left, trim_right = trim_window_all_nan_edges(raw_win_full)

    ext_start_eff = ext_start + int(trim_left)
    ext_end_eff = ext_end - int(trim_right)

    core0_eff = max(core_start, ext_start_eff)
    core1_eff = min(core_end, ext_end_eff)

    pad_left = max(0, core0_eff - core_start)
    pad_right = max(0, core_end - core1_eff)

    rel0 = int(core0_eff - ext_start_eff)
    inner_w = int((core_end - core_start) - pad_left - pad_right)
    inner_w = max(inner_w, 0)

    Z_flat = flatten_swath(raw_win_trim, target_ny=target_ny)

    # Match the cluster y size.
    if Z_flat.shape[0] < target_y:
        tmp = np.full((target_y, Z_flat.shape[1]), np.nan, dtype=float)
        tmp[: Z_flat.shape[0], :] = Z_flat
        Z_flat = tmp
    elif Z_flat.shape[0] > target_y:
        Z_flat = Z_flat[:target_y, :]

    core = Z_flat[:, rel0:rel0 + inner_w]

    out = np.full((target_y, target_x), np.nan, dtype=float)

    # If the cluster width is different from X_CHUNK, use the cluster width as final output.
    insert0 = int(pad_left)
    insert1 = min(target_x, insert0 + core.shape[1])

    if insert0 < target_x and insert1 > insert0:
        out[:, insert0:insert1] = core[:, : insert1 - insert0]

    return out


def get_navgem_fields_for_cluster(navgem_data, slice_no, cluster_shape):
    """
    Get U, V, and NN for the slice that this cluster belongs to.
    """
    U = extract_navgem_slice_like_pipeline(
        navgem_data["U_raw"],
        slice_no=slice_no,
        cluster_shape=cluster_shape,
    )
    V = extract_navgem_slice_like_pipeline(
        navgem_data["V_raw"],
        slice_no=slice_no,
        cluster_shape=cluster_shape,
    )
    NN = extract_navgem_slice_like_pipeline(
        navgem_data["NN_raw"],
        slice_no=slice_no,
        cluster_shape=cluster_shape,
    )
    return U, V, NN


def calculate_lambda_z_map(U, V, NN, true_angle_deg, phase_speed_m_per_s, phase_direction):
    """
    Calculate vertical wavelength from NAVGEM wind and phase-speed vector.

    Formula used here:
        m² = N² / |c - w|²
        λz = 2π / m

    Since the phase-speed estimate is along the wave propagation direction,
    this projects the horizontal wind vector w = (U, V) onto the same direction.

    Inputs:
        U, V: NAVGEM wind components in m/s
        NN: N² in s⁻²
        true_angle_deg: propagation angle map in degrees.
            This follows the same convention as the rest of this script:
            0° = east, counterclockwise positive.
        phase_speed_m_per_s: phase speed magnitude/sign from phase-speed fit
        phase_direction: +1 or -1 from phase-speed fit

    Outputs:
        LambdaZ_km: vertical wavelength in km
        M2: vertical wavenumber squared in m⁻²
        IntrinsicSpeed: scalar projection of (c - w), in m/s
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    NN = np.asarray(NN, dtype=float)
    theta = np.deg2rad(np.asarray(true_angle_deg, dtype=float))

    # Wind projected along the phase propagation direction.
    w_along = U * np.cos(theta) + V * np.sin(theta)

    # Build signed c along the same direction.
    c_val = float(phase_speed_m_per_s) if np.isfinite(phase_speed_m_per_s) else np.nan
    
    try:
        direction = float(phase_direction)
    except (TypeError, ValueError):
        direction = np.nan
    
    if not np.isfinite(direction) or direction == 0:
        direction = np.sign(c_val) if np.isfinite(c_val) and c_val != 0 else 1.0
    
    # FLIPPED SIGN FOR MY CONVENTION
    c_along = -direction * abs(c_val) if np.isfinite(c_val) else np.nan
    
    # Intrinsic phase speed
    intrinsic_speed = c_along - w_along

    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = NN / intrinsic_speed**2
        m = np.sqrt(m2)
        lambda_z_km = (2.0 * np.pi / m) / 1000.0

    bad = (
        (~np.isfinite(lambda_z_km))
        | (~np.isfinite(m2))
        | (m2 <= 0)
        | (~np.isfinite(intrinsic_speed))
    )
    lambda_z_km = np.asarray(lambda_z_km, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    intrinsic_speed = np.asarray(intrinsic_speed, dtype=float)

    lambda_z_km[bad] = np.nan
    m2[bad] = np.nan
    intrinsic_speed[~np.isfinite(intrinsic_speed)] = np.nan

    return lambda_z_km, m2, intrinsic_speed



def calculate_new_momentum_flux_map(
    A,
    Temp,
    Lx_km,
    N2,
    LambdaZ_km,
    true_angle_deg,
    phase_direction,
):
    """
    Calculate new momentum flux using actual lambda_z.

    Same magnitude formula as the old L7A MF calculation:

        MF = 0.5 * g² / N² * (lambda_z / lambda_h)
             * (Amplitude / Temperature)² * (1 / C_CANCEL²)

    Difference:
        old MF uses fixed lambda_z
        new MF uses LambdaZ_km from m² = N² / |c - w|²

    Direction:
        The magnitude is positive.
        New_MF_u and New_MF_v are signed using the corrected phase-speed direction.

        Since calculate_lambda_z_map uses:
            c_along = -phase_direction * abs(c)

        the corrected propagation sign is:
            phase_sign_corrected = -phase_direction
    """
    A = np.asarray(A, dtype=float)
    Temp = np.asarray(Temp, dtype=float)
    Lx_km = np.asarray(Lx_km, dtype=float)
    N2 = np.asarray(N2, dtype=float)
    LambdaZ_km = np.asarray(LambdaZ_km, dtype=float)
    theta = np.deg2rad(np.asarray(true_angle_deg, dtype=float))

    try:
        direction = float(phase_direction)
    except (TypeError, ValueError):
        direction = np.nan

    if not np.isfinite(direction) or direction == 0:
        direction = 1.0

    phase_sign_corrected = -np.sign(direction)

    good = (
        np.isfinite(A)
        & np.isfinite(Temp)
        & np.isfinite(Lx_km)
        & np.isfinite(N2)
        & np.isfinite(LambdaZ_km)
        & np.isfinite(theta)
        & (Temp != 0.0)
        & (Lx_km > 0.0)
        & (N2 > 0.0)
        & (LambdaZ_km > 0.0)
    )

    NewMF = np.full_like(A, np.nan, dtype=float)
    NewMF_u = np.full_like(A, np.nan, dtype=float)
    NewMF_v = np.full_like(A, np.nan, dtype=float)

    if np.any(good):
        NewMF[good] = (
            0.5
            * (G_MS2 ** 2)
            / N2[good]
            * (LambdaZ_km[good] / Lx_km[good])
            * (A[good] / Temp[good]) ** 2
            * (1.0 / (C_CANCEL ** 2))
        )

        NewMF_u[good] = NewMF[good] * phase_sign_corrected * np.cos(theta[good])
        NewMF_v[good] = NewMF[good] * phase_sign_corrected * np.sin(theta[good])

    return NewMF, NewMF_u, NewMF_v

def plot_panel(fig, ax, Z, title, x1d_km, y1d_km, xx=None, yy=None, cmap=None, units_label=""):
    if Z is None:
        ax.axis("off")
        ax.set_title(f"{title}\nnot found")
        return None

    vmin, vmax = robust_limits(Z)

    im = ax.pcolormesh(
        x1d_km,
        y1d_km,
        Z,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    if xx is not None and yy is not None and xx >= 0 and yy >= 0:
        ax.plot(
            xx * DX_KM,
            yy * DY_KM,
            "ro",
            markersize=7,
            markeredgecolor="k",
        )

    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if units_label:
        cbar.set_label(units_label)

    return im


def plot_amplitude_evolution_panel(ax, phase_summary_csv):
    """
    Plot crest relative amplitude evolution and cosine fits inside the main figure.

    This uses the files already created by phase_speed_from_l7a_point.py:
        crest_relative_amplitude_evolution.csv
        phase_speed_from_l7a_point_summary.csv
    """
    if not phase_summary_csv:
        ax.axis("off")
        ax.set_title("Amplitude evolution\nnot available")
        return

    phase_summary_csv = Path(phase_summary_csv)
    crest_csv = phase_summary_csv.parent / "crest_relative_amplitude_evolution.csv"

    if not crest_csv.exists():
        ax.axis("off")
        ax.set_title("Amplitude evolution\nCSV not found")
        ax.text(
            0.02,
            0.95,
            str(crest_csv),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            wrap=True,
        )
        return

    try:
        from phase_speed_from_l7a_point import load_roi_csv, fit_phase_grid_search

        t, wavelength_km, wavelength_error_km, angle_deg, angle_error_deg, Y, pt_names = load_roi_csv(crest_csv)

        if Y.size == 0:
            ax.axis("off")
            ax.set_title("Amplitude evolution\nempty CSV")
            return

        best_a_deg, b_best, sse_min = fit_phase_grid_search(t, Y)
        phase_rate_rad = np.deg2rad(best_a_deg)

        for j in range(Y.shape[1]):
            label_data = pt_names[j] if j < len(pt_names) else f"pt{j + 1}"
            ax.plot(t, Y[:, j], "o", markersize=3, alpha=0.75, label=label_data)
            ax.plot(t, np.cos(phase_rate_rad * t + b_best[j]), linewidth=1.2, alpha=0.75)

        ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_title(
            f"Crest relative amplitude evolution\n"
            f"fit rate = {best_a_deg:.3g} deg/frame | SSE = {sse_min:.3g}"
        )
        ax.set_xlabel("Image index")
        ax.set_ylabel("Relative amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

        # Keep the legend small. If there are many points, it can cover the plot.
        if Y.shape[1] <= 7:
            ax.legend(fontsize=7, loc="upper right", ncol=1)

    except Exception as exc:
        ax.axis("off")
        ax.set_title("Amplitude evolution\nplot failed")
        ax.text(
            0.02,
            0.98,
            str(exc),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            wrap=True,
        )


def get_true_angle_map(nc, cluster_idx, T_rad, Bearing):
    """
    Read true angle map from file if present.
    Otherwise calculate true angle = dominant angle + bearing.
    """
    true_angle_var_candidates = [
        "PeakAmplitude_TrueAngle_deg",
        "PeakAmplitude_true_angle_deg",
        "TrueAngle_cluster",
        "TrueAngle_deg_cluster",
        "TrueAngle",
    ]

    for name in true_angle_var_candidates:
        if name in nc.variables:
            var = nc.variables[name]
            if var.ndim >= 3:
                return np.asarray(var[cluster_idx], dtype=float)

    return (np.rad2deg(T_rad) + Bearing) % 360.0


def get_saved_true_angle(nc, cluster_idx):
    for name in [
        "PeakAmplitude_TrueAngle_deg",
        "PeakAmplitude_true_angle_deg",
        "PeakAmplitude_TrueAngle",
        "PeakAmplitude_true_angle",
    ]:
        if name in nc.variables:
            return read_scalar_var(nc, name, cluster_idx, default=np.nan)
    return np.nan


def get_saved_bearing(nc, cluster_idx):
    for name in [
        "PeakAmplitude_Bearing_deg",
        "PeakAmplitude_Bearing",
        "PeakAmplitude_BearingAngle_deg",
    ]:
        if name in nc.variables:
            return read_scalar_var(nc, name, cluster_idx, default=np.nan)
    return np.nan


def save_cluster_plot(
    *,
    NC_PATH,
    TARGET_SLICE,
    TARGET_CLUSTER_IN_SLICE,
    cluster_idx,
    A,
    L,
    T_deg,
    R,
    Temp,
    U,
    V,
    N2,
    Bearing,
    TrueAngle,
    LambdaZ,
    NewMF,
    NewMF_u,
    NewMF_v,
    MF,
    MFz,
    MFm,
    xx,
    yy,
    raw_yy,
    raw_xx,
    global_x_saved,
    point_source,
    selection_method,
    lat_saved,
    lon_saved,
    amp_point,
    lambda_point,
    angle_deg_point,
    bearing_point,
    true_angle_point,
    temp_point,
    u_point,
    v_point,
    n2_point,
    intrinsic_speed_point,
    m2_point,
    lambda_z_point,
    new_mf_point,
    new_mf_u_point,
    new_mf_v_point,
    mf_point,
    mfz_point,
    mfm_point,
    phase_speed_point,
    phase_speed_error_point,
    phase_direction,
    phase_avg_wavelength,
    phase_avg_wavelength_error,
    phase_avg_angle,
    phase_avg_angle_error,
    phase_closest_frame,
    phase_summary_csv,
):
    ny, nx = A.shape
    x1d_km = np.arange(nx) * DX_KM
    y1d_km = np.arange(ny) * DY_KM

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)

    gs = fig.add_gridspec(
        nrows=4,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 1.25],
    )

    # Left side: all plot panels
    axs = []
    for r in range(4):
        for c in range(4):
            axs.append(fig.add_subplot(gs[r, c]))

    # Right side: one tall info panel
    ax_info = fig.add_subplot(gs[:, 4])
    ax_info.axis("off")

    plot_panel(fig, axs[0], R, "Cluster Reconstruction", x1d_km, y1d_km, xx, yy, cmap="RdBu_r")
    plot_panel(fig, axs[1], A, "Amplitude (K)", x1d_km, y1d_km, xx, yy, units_label="K")
    plot_panel(fig, axs[2], L, "Dominant Wavelength (km)", x1d_km, y1d_km, xx, yy, units_label="km")
    plot_panel(fig, axs[3], T_deg, "Dominant Angle (deg)", x1d_km, y1d_km, xx, yy, units_label="deg")

    plot_panel(fig, axs[4], Temp, "Temperature (K)", x1d_km, y1d_km, xx, yy, units_label="K")
    plot_panel(fig, axs[5], U, "u NAVGEM (m/s)", x1d_km, y1d_km, xx, yy, units_label="m/s")
    plot_panel(fig, axs[6], V, "v NAVGEM (m/s)", x1d_km, y1d_km, xx, yy, units_label="m/s")
    plot_panel(fig, axs[7], N2, "NN/N² NAVGEM", x1d_km, y1d_km, xx, yy)

    plot_panel(fig, axs[8], Bearing, "Bearing Angle (deg)", x1d_km, y1d_km, xx, yy, units_label="deg")
    plot_panel(fig, axs[9], TrueAngle, "True Angle = Angle + Bearing (deg)", x1d_km, y1d_km, xx, yy, units_label="deg")

    # Replacing MF with component plots
    plot_panel(fig, axs[10], MFz, "MF_u / Zonal Momentum Flux", x1d_km, y1d_km, xx, yy, units_label="m² s⁻²")
    plot_panel(fig, axs[11], MFm, "MF_v / Meridional Momentum Flux", x1d_km, y1d_km, xx, yy, units_label="m² s⁻²")

    plot_amplitude_evolution_panel(axs[12], phase_summary_csv)

    plot_panel(
        fig,
        axs[13],
        LambdaZ,
        "Vertical Wavelength λz (km)",
        x1d_km,
        y1d_km,
        xx,
        yy,
        units_label="km",
    )

    plot_panel(
        fig,
        axs[14],
        NewMF_u,
        "New MF_u using actual λz",
        x1d_km,
        y1d_km,
        xx,
        yy,
        units_label="m² s⁻²",
    )

    plot_panel(
        fig,
        axs[15],
        NewMF_v,
        "New MF_v using actual λz",
        x1d_km,
        y1d_km,
        xx,
        yy,
        units_label="m² s⁻²",
    )

    summary_text = (
        f"Selected point\n"
        f"source: {point_source}\n"
        f"selection method: {selection_method}\n"
        f"  -1: not saved / older file\n"
        f"   1: max amp inside y=150-450\n"
        f"   2: closest to y=150-450\n\n"
        f"y, x = {yy}, {xx}\n"
        f"global x = {global_x_saved}\n"
        f"raw peak y, x = {raw_yy}, {raw_xx}\n\n"
        f"lat = {lat_saved:.5f}\n"
        f"lon = {lon_saved:.5f}\n\n"
        f"amp = {amp_point:.3g} K\n"
        f"lambda = {lambda_point:.3g} km\n"
        f"dominant angle = {angle_deg_point:.3g} deg\n"
        f"bearing = {bearing_point:.3g} deg\n"
        f"true angle = {true_angle_point:.3g} deg\n\n"
        f"phase speed = {phase_speed_point:.3g} ± {phase_speed_error_point:.3g} m/s\n"
        f"phase dir = {phase_direction}\n"
        f"phase avg λ = {phase_avg_wavelength:.3g} ± {phase_avg_wavelength_error:.3g} km\n"
        f"phase avg angle = {phase_avg_angle:.3g} ± {phase_avg_angle_error:.3g} deg\n"
        f"closest frame = {phase_closest_frame}\n\n"
        f"T = {temp_point:.3g} K\n"
        f"u NAVGEM = {u_point:.3g} m/s\n"
        f"v NAVGEM = {v_point:.3g} m/s\n"
        f"NN/N² NAVGEM = {n2_point:.3g}\n"
        f"intrinsic speed c-w = {intrinsic_speed_point:.3g} m/s\n"
        f"m² = {m2_point:.3g} m⁻²\n"
        f"lambda_z = {lambda_z_point:.3g} km\n\n"
        f"New MF = {new_mf_point:.3g}\n"
        f"New MF_u = {new_mf_u_point:.3g}\n"
        f"New MF_v = {new_mf_v_point:.3g}\n\n"
        f"Old MF_u / MFz = {mfz_point:.3g}\n"
        f"Old MF_v / MFm = {mfm_point:.3g}\n"
    )

    ax_info.text(
        0.02,
        0.98,
        summary_text,
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle(
        (
            f"{NC_PATH.name}\n"
            f"Slice {TARGET_SLICE} | "
            f"Cluster {TARGET_CLUSTER_IN_SLICE} "
            f"(global idx {cluster_idx}) | "
            f"c={phase_speed_point:.2f}±{phase_speed_error_point:.2f} m/s | "
            f"λz={lambda_z_point:.2f} km"
        ),
        fontsize=14,
    )

    slice_plot_dir = PLOT_OUT_BASE_DIR / f"slice_{TARGET_SLICE:03d}"
    slice_plot_dir.mkdir(parents=True, exist_ok=True)

    out_png = slice_plot_dir / (
        f"{NC_PATH.stem}_slice_{TARGET_SLICE:03d}"
        f"_cluster_{TARGET_CLUSTER_IN_SLICE:03d}"
        f"_global_{cluster_idx:05d}_info.png"
    )

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return out_png

def process_one_cluster(nc, slices_all, TARGET_SLICE, TARGET_CLUSTER_IN_SLICE, navgem_data):
    idx_in_slice = np.where(slices_all == TARGET_SLICE)[0]

    if idx_in_slice.size == 0:
        return {
            "slice": TARGET_SLICE,
            "cluster_in_slice": TARGET_CLUSTER_IN_SLICE,
            "global_cluster_index": -1,
            "status": "skipped_no_clusters_in_slice",
        }

    if TARGET_CLUSTER_IN_SLICE >= idx_in_slice.size:
        return {
            "slice": TARGET_SLICE,
            "cluster_in_slice": TARGET_CLUSTER_IN_SLICE,
            "global_cluster_index": -1,
            "status": "skipped_cluster_not_available",
            "n_clusters_in_slice": int(idx_in_slice.size),
        }

    cluster_idx = int(idx_in_slice[TARGET_CLUSTER_IN_SLICE])

    phase_speed_out_dir = (
        PHASE_SPEED_BASE_OUT_DIR
        / f"slice_{TARGET_SLICE:03d}"
        / f"cluster_{TARGET_CLUSTER_IN_SLICE:03d}_global_{cluster_idx:05d}"
    )

    print(
        f"Running slice {TARGET_SLICE}, "
        f"cluster {TARGET_CLUSTER_IN_SLICE}, "
        f"global {cluster_idx}"
    )

    # --------------------------------------------------------
    # Required variables
    # --------------------------------------------------------
    A = np.asarray(nc.variables["Amplitude"][cluster_idx], dtype=float)
    L = np.asarray(nc.variables["DominantWavelength"][cluster_idx], dtype=float)
    T_rad = np.asarray(nc.variables["Angle"][cluster_idx], dtype=float)
    T_deg = np.rad2deg(T_rad) % 180.0

    if "ClusterReconstruction" in nc.variables:
        R = np.asarray(nc.variables["ClusterReconstruction"][cluster_idx], dtype=float)
    else:
        R = np.full_like(A, np.nan)

    # --------------------------------------------------------
    # Optional cluster variables
    # --------------------------------------------------------
    Temp = read_cluster_var(nc, "Temp_cluster", cluster_idx, shape_like=A)

    # U, V, and N2/NN are read from the NAVGEM file and extracted using the
    # same per-slice trim/flatten/crop/pad logic used when the clusters were made.
    slice_no = int(slices_all[cluster_idx])
    U, V, N2 = get_navgem_fields_for_cluster(navgem_data, slice_no, A.shape)

    Bearing = read_cluster_var(nc, "Bearing_cluster", cluster_idx, shape_like=A)
    MF = read_cluster_var(nc, "MF_cluster", cluster_idx, shape_like=A)
    MFz = read_cluster_var(nc, "MFz_cluster", cluster_idx, shape_like=A)
    MFm = read_cluster_var(nc, "MFm_cluster", cluster_idx, shape_like=A)

    TrueAngle = get_true_angle_map(nc, cluster_idx, T_rad, Bearing)

    # --------------------------------------------------------
    # Select point
    # --------------------------------------------------------
    if not np.any(np.isfinite(A)):
        raise ValueError("Cluster contains no finite amplitude pixels.")

    raw_yy, raw_xx = np.unravel_index(np.nanargmax(A), A.shape)

    saved_yy = read_scalar_int_var(nc, "PeakAmplitude_y_index", cluster_idx, default=-1)
    saved_xx = read_scalar_int_var(nc, "PeakAmplitude_x_index", cluster_idx, default=-1)

    use_saved = (
        USE_SAVED_REPRESENTATIVE_POINT
        and saved_yy >= 0
        and saved_xx >= 0
        and saved_yy < A.shape[0]
        and saved_xx < A.shape[1]
    )

    if use_saved:
        yy, xx = saved_yy, saved_xx
        point_source = "saved representative point"
    else:
        yy, xx = int(raw_yy), int(raw_xx)
        point_source = "raw maximum-amplitude point"

    # --------------------------------------------------------
    # Values at selected point
    # --------------------------------------------------------
    amp_point = get_point_value(A, yy, xx)
    lambda_point = get_point_value(L, yy, xx)
    angle_rad_point = get_point_value(T_rad, yy, xx)
    angle_deg_point = float(np.rad2deg(angle_rad_point) % 180.0) if np.isfinite(angle_rad_point) else np.nan

    temp_point = get_point_value(Temp, yy, xx)
    u_point = get_point_value(U, yy, xx)
    v_point = get_point_value(V, yy, xx)
    n2_point = get_point_value(N2, yy, xx)
    bearing_point = get_point_value(Bearing, yy, xx)

    true_angle_point = get_point_value(TrueAngle, yy, xx)
    if not np.isfinite(true_angle_point) and np.isfinite(angle_rad_point) and np.isfinite(bearing_point):
        true_angle_point = (np.rad2deg(angle_rad_point) + bearing_point) % 360.0

    mf_point = get_point_value(MF, yy, xx)
    mfz_point = get_point_value(MFz, yy, xx)
    mfm_point = get_point_value(MFm, yy, xx)

    # --------------------------------------------------------
    # Saved scalar values from file, if available
    # --------------------------------------------------------
    lat_saved = read_scalar_var(nc, "PeakAmplitude_Latitude", cluster_idx, default=np.nan)
    lon_saved = read_scalar_var(nc, "PeakAmplitude_Longitude", cluster_idx, default=np.nan)
    global_x_saved = read_scalar_int_var(nc, "PeakAmplitude_global_x_index", cluster_idx, default=-1)
    selection_method = read_scalar_int_var(nc, "PeakAmplitude_selection_method", cluster_idx, default=-1)

    saved_true_angle = get_saved_true_angle(nc, cluster_idx)
    if np.isfinite(saved_true_angle):
        true_angle_point = saved_true_angle

    saved_bearing = get_saved_bearing(nc, cluster_idx)
    if np.isfinite(saved_bearing):
        bearing_point = saved_bearing

    n_pix = int(np.count_nonzero(np.isfinite(A)))

    # --------------------------------------------------------
    # Calculate phase speed
    # --------------------------------------------------------
    phase = None
    phase_speed_point = np.nan
    phase_speed_error_point = np.nan
    phase_direction = np.nan
    phase_avg_wavelength = np.nan
    phase_avg_wavelength_error = np.nan
    phase_avg_angle = np.nan
    phase_avg_angle_error = np.nan
    phase_closest_frame = -1
    phase_closest_distance = np.nan
    phase_summary_csv = ""
    status = "ok"

    if CALCULATE_PHASE_SPEED:
        if not np.isfinite(lat_saved) or not np.isfinite(lon_saved):
            status = "skipped_phase_speed_bad_lat_lon"
        elif not np.isfinite(true_angle_point):
            status = "skipped_phase_speed_bad_true_angle"
        elif not np.isfinite(lambda_point):
            status = "skipped_phase_speed_bad_wavelength"
        else:
            import importlib
            import phase_speed_from_l7a_point

            importlib.reload(phase_speed_from_l7a_point)
            from phase_speed_from_l7a_point import calculate_phase_speed_for_l7a_point

            phase = calculate_phase_speed_for_l7a_point(
                l1a_nc_path=L1A_REMAP_NC_PATH,
                target_lat=lat_saved,
                target_lon=lon_saved,
                target_angle_deg=true_angle_point,
                target_wavelength_km=lambda_point,
                out_dir=phase_speed_out_dir,
                roi_label=PHASE_SPEED_ROI_LABEL,
                dt_frame_s=PHASE_SPEED_DT_FRAME_S,
                save_overlay=PHASE_SPEED_SAVE_OVERLAY,
                verbose=PHASE_SPEED_VERBOSE,
            )

            phase_speed_point = phase.get("phase_speed_m_per_s", np.nan)
            phase_speed_error_point = phase.get("phase_speed_error_m_per_s", np.nan)
            phase_direction = phase.get("direction", np.nan)
            phase_avg_wavelength = phase.get("avg_wavelength_km", np.nan)
            phase_avg_wavelength_error = phase.get("avg_wavelength_error_km", np.nan)
            phase_avg_angle = phase.get("avg_angle_deg", np.nan)
            phase_avg_angle_error = phase.get("avg_angle_error_deg", np.nan)
            phase_closest_frame = phase.get("closest_frame", -1)
            phase_closest_distance = phase.get("closest_distance_km", np.nan)
            phase_summary_csv = phase.get("summary_csv_path", "")

    # --------------------------------------------------------
    # Calculate vertical wavelength from m² = N² / |c - w|²
    # --------------------------------------------------------
    LambdaZ, M2, IntrinsicSpeed = calculate_lambda_z_map(
        U,
        V,
        N2,
        TrueAngle,
        phase_speed_point,
        phase_direction,
    )

    lambda_z_point = get_point_value(LambdaZ, yy, xx)
    m2_point = get_point_value(M2, yy, xx)
    intrinsic_speed_point = get_point_value(IntrinsicSpeed, yy, xx)

    # --------------------------------------------------------
    # Calculate new momentum flux using actual lambda_z
    # --------------------------------------------------------
    NewMF, NewMF_u, NewMF_v = calculate_new_momentum_flux_map(
        A=A,
        Temp=Temp,
        Lx_km=L,
        N2=N2,
        LambdaZ_km=LambdaZ,
        true_angle_deg=TrueAngle,
        phase_direction=phase_direction,
    )

    new_mf_point = get_point_value(NewMF, yy, xx)
    new_mf_u_point = get_point_value(NewMF_u, yy, xx)
    new_mf_v_point = get_point_value(NewMF_v, yy, xx)

    # --------------------------------------------------------
    # Store arrays/scalars for the new L8A file.
    # The L8A writer copies the original L7A file, so all old variables are kept.
    # These new variables are appended by global cluster index.
    # --------------------------------------------------------
    if SAVE_L8A_FILE:
        L8A_CLUSTER_OUTPUTS.append({
            "global_cluster_index": int(cluster_idx),
            "slice_no": int(slice_no),
            "LambdaZ_cluster": np.asarray(LambdaZ, dtype=np.float32),
            "M2_cluster": np.asarray(M2, dtype=np.float32),
            "IntrinsicSpeed_cluster": np.asarray(IntrinsicSpeed, dtype=np.float32),
            "NewMF_cluster": np.asarray(NewMF, dtype=np.float32),
            "NewMF_u_cluster": np.asarray(NewMF_u, dtype=np.float32),
            "NewMF_v_cluster": np.asarray(NewMF_v, dtype=np.float32),
            "PhaseSpeed_cluster": float(phase_speed_point),
            "PhaseSpeedError_cluster": float(phase_speed_error_point),
            "PhaseDirection_cluster": float(phase_direction) if np.isfinite(phase_direction) else np.nan,
            "PhaseAvgWavelength_cluster": float(phase_avg_wavelength),
            "PhaseAvgWavelengthError_cluster": float(phase_avg_wavelength_error),
            "PhaseAvgAngle_cluster": float(phase_avg_angle),
            "PhaseAvgAngleError_cluster": float(phase_avg_angle_error),
        })

    # --------------------------------------------------------
    # Save plot with all plots for this slice together
    # --------------------------------------------------------
    plot_path = ""
    if SAVE_FIG:
        out_png = save_cluster_plot(
            NC_PATH=NC_PATH,
            TARGET_SLICE=TARGET_SLICE,
            TARGET_CLUSTER_IN_SLICE=TARGET_CLUSTER_IN_SLICE,
            cluster_idx=cluster_idx,
            A=A,
            L=L,
            T_deg=T_deg,
            R=R,
            Temp=Temp,
            U=U,
            V=V,
            N2=N2,
            Bearing=Bearing,
            TrueAngle=TrueAngle,
            LambdaZ=LambdaZ,
            NewMF=NewMF,
            NewMF_u=NewMF_u,
            NewMF_v=NewMF_v,
            MF=MF,
            MFz=MFz,
            MFm=MFm,
            xx=xx,
            yy=yy,
            raw_yy=raw_yy,
            raw_xx=raw_xx,
            global_x_saved=global_x_saved,
            point_source=point_source,
            selection_method=selection_method,
            lat_saved=lat_saved,
            lon_saved=lon_saved,
            amp_point=amp_point,
            lambda_point=lambda_point,
            angle_deg_point=angle_deg_point,
            bearing_point=bearing_point,
            true_angle_point=true_angle_point,
            temp_point=temp_point,
            u_point=u_point,
            v_point=v_point,
            n2_point=n2_point,
            intrinsic_speed_point=intrinsic_speed_point,
            m2_point=m2_point,
            lambda_z_point=lambda_z_point,
            new_mf_point=new_mf_point,
            new_mf_u_point=new_mf_u_point,
            new_mf_v_point=new_mf_v_point,
            mf_point=mf_point,
            mfz_point=mfz_point,
            mfm_point=mfm_point,
            phase_speed_point=phase_speed_point,
            phase_speed_error_point=phase_speed_error_point,
            phase_direction=phase_direction,
            phase_avg_wavelength=phase_avg_wavelength,
            phase_avg_wavelength_error=phase_avg_wavelength_error,
            phase_avg_angle=phase_avg_angle,
            phase_avg_angle_error=phase_avg_angle_error,
            phase_closest_frame=phase_closest_frame,
            phase_summary_csv=phase_summary_csv,
        )
        plot_path = str(out_png)

    print(
        f"  c = {phase_speed_point:.2f} ± {phase_speed_error_point:.2f} m/s | "
        f"lat/lon = {lat_saved:.3f}, {lon_saved:.3f} | "
        f"lambda = {lambda_point:.2f} km | "
        f"true angle = {true_angle_point:.2f} deg | "
        f"u/v/NN = {u_point:.2f}, {v_point:.2f}, {n2_point:.3g} | "
        f"lambda_z = {lambda_z_point:.2f} km | "
        f"New MF_u/v = {new_mf_u_point:.3g}, {new_mf_v_point:.3g} | "
        f"plot = {plot_path}"
    )

    return {
        "slice": int(TARGET_SLICE),
        "cluster_in_slice": int(TARGET_CLUSTER_IN_SLICE),
        "global_cluster_index": int(cluster_idx),
        "status": status,
        "point_source": point_source,
        "selection_method": int(selection_method),
        "selected_y": int(yy),
        "selected_x": int(xx),
        "raw_peak_y": int(raw_yy),
        "raw_peak_x": int(raw_xx),
        "selected_global_x": int(global_x_saved),
        "latitude": float(lat_saved),
        "longitude": float(lon_saved),
        "amplitude_K": float(amp_point),
        "target_wavelength_km": float(lambda_point),
        "target_dominant_angle_deg": float(angle_deg_point),
        "target_bearing_deg": float(bearing_point),
        "target_true_angle_deg": float(true_angle_point),
        "temperature_K": float(temp_point),
        "u_navgem_m_per_s": float(u_point),
        "v_navgem_m_per_s": float(v_point),
        "NN_or_N2_navgem": float(n2_point),
        "intrinsic_speed_c_minus_w_m_per_s": float(intrinsic_speed_point),
        "m2_1_per_m2": float(m2_point),
        "lambda_z_km": float(lambda_z_point),
        "New_MF_m2_s2": float(new_mf_point),
        "New_MF_u_m2_s2": float(new_mf_u_point),
        "New_MF_v_m2_s2": float(new_mf_v_point),
        "MF_m2_s2": float(mf_point),
        "MFz_m2_s2": float(mfz_point),
        "MFm_m2_s2": float(mfm_point),
        "finite_pixels": int(n_pix),
        "phase_speed_m_per_s": float(phase_speed_point),
        "phase_speed_error_m_per_s": float(phase_speed_error_point),
        "phase_direction": phase_direction,
        "phase_avg_wavelength_km": float(phase_avg_wavelength),
        "phase_avg_wavelength_error_km": float(phase_avg_wavelength_error),
        "phase_avg_angle_deg": float(phase_avg_angle),
        "phase_avg_angle_error_deg": float(phase_avg_angle_error),
        "phase_closest_frame": int(phase_closest_frame),
        "phase_closest_distance_km": float(phase_closest_distance),
        "phase_summary_csv": phase_summary_csv,
        "phase_output_dir": str(phase_speed_out_dir),
        "plot_path": plot_path,
    }



def save_l8a_with_new_variables(src_l7a_path, dst_l8a_path, cluster_outputs):
    """
    Create a new L8A NetCDF file that keeps all variables from the input L7A file
    and appends the new phase-speed-based variables.

    New cluster variables:
        LambdaZ_cluster
        M2_cluster
        IntrinsicSpeed_cluster
        NewMF_cluster
        NewMF_u_cluster
        NewMF_v_cluster

    New scalar-per-cluster variables:
        PhaseSpeed_cluster
        PhaseSpeedError_cluster
        PhaseDirection_cluster
        PhaseAvgWavelength_cluster
        PhaseAvgWavelengthError_cluster
        PhaseAvgAngle_cluster
        PhaseAvgAngleError_cluster

    New summed/stiched variables when possible:
        NewMF_slice, NewMF_u_slice, NewMF_v_slice
        NewMF_top, NewMF_u_top, NewMF_v_top

    Only processed clusters are filled. Unprocessed clusters remain NaN.
    """
    src_l7a_path = Path(src_l7a_path)
    dst_l8a_path = Path(dst_l8a_path)

    if not cluster_outputs:
        print("No L8A cluster outputs to save.")
        return ""

    dst_l8a_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy old L7A first, preserving old variables and attributes.
    shutil.copy2(src_l7a_path, dst_l8a_path)

    with Dataset(dst_l8a_path, "r+") as nc:
        if "Amplitude" not in nc.variables:
            raise KeyError("Could not find Amplitude in copied L8A file.")

        amp_var = nc.variables["Amplitude"]
        cluster_dim, ytile_dim, xtile_dim = amp_var.dimensions[:3]
        n_cluster, y_tile, x_tile = amp_var.shape[:3]

        def _make_full_cluster_array(key):
            arr = np.full((n_cluster, y_tile, x_tile), np.nan, dtype=np.float32)
            for item in cluster_outputs:
                idx = int(item["global_cluster_index"])
                if 0 <= idx < n_cluster and key in item:
                    Z = np.asarray(item[key], dtype=np.float32)
                    ny = min(y_tile, Z.shape[0])
                    nx = min(x_tile, Z.shape[1])
                    arr[idx, :ny, :nx] = Z[:ny, :nx]
            return arr

        def _make_full_scalar_array(key):
            arr = np.full((n_cluster,), np.nan, dtype=np.float32)
            for item in cluster_outputs:
                idx = int(item["global_cluster_index"])
                if 0 <= idx < n_cluster and key in item:
                    arr[idx] = item[key]
            return arr

        def _write_or_update(name, arr, dims, long_name, units):
            arr = np.asarray(arr)
            name_to_write = name

            if name in nc.variables and nc.variables[name].shape != arr.shape:
                name_to_write = name + "_L8A"
                print(
                    f"Variable {name} already exists with shape {nc.variables[name].shape}; "
                    f"writing {name_to_write} instead."
                )

            if name_to_write in nc.variables:
                v = nc.variables[name_to_write]
                v[:] = arr
            else:
                v = nc.createVariable(
                    name_to_write,
                    "f4",
                    dims,
                    zlib=True,
                    complevel=4,
                    fill_value=np.float32(np.nan),
                )
                v.setncattr("long_name", long_name)
                v.setncattr("units", units)
                v[:] = arr

        cluster_dims = (cluster_dim, ytile_dim, xtile_dim)

        lambda_z_cluster = _make_full_cluster_array("LambdaZ_cluster")
        m2_cluster = _make_full_cluster_array("M2_cluster")
        intrinsic_cluster = _make_full_cluster_array("IntrinsicSpeed_cluster")
        new_mf_cluster = _make_full_cluster_array("NewMF_cluster")
        new_mf_u_cluster = _make_full_cluster_array("NewMF_u_cluster")
        new_mf_v_cluster = _make_full_cluster_array("NewMF_v_cluster")

        _write_or_update("LambdaZ_cluster", lambda_z_cluster, cluster_dims,
                         "Vertical wavelength from m^2 = N^2 / |c-w|^2, cluster masked", "km")
        _write_or_update("M2_cluster", m2_cluster, cluster_dims,
                         "Vertical wavenumber squared from phase speed and NAVGEM wind, cluster masked", "m-2")
        _write_or_update("IntrinsicSpeed_cluster", intrinsic_cluster, cluster_dims,
                         "Intrinsic phase speed projection c-w, cluster masked", "m s-1")
        _write_or_update("NewMF_cluster", new_mf_cluster, cluster_dims,
                         "Momentum flux using actual lambda_z, cluster masked", "m2 s-2")
        _write_or_update("NewMF_u_cluster", new_mf_u_cluster, cluster_dims,
                         "Signed zonal momentum flux using actual lambda_z and corrected phase-speed direction", "m2 s-2")
        _write_or_update("NewMF_v_cluster", new_mf_v_cluster, cluster_dims,
                         "Signed meridional momentum flux using actual lambda_z and corrected phase-speed direction", "m2 s-2")

        scalar_dims = (cluster_dim,)
        scalar_defs = [
            ("PhaseSpeed_cluster", "Phase speed from L1A/remap fit", "m s-1"),
            ("PhaseSpeedError_cluster", "Phase speed uncertainty from L1A/remap fit", "m s-1"),
            ("PhaseDirection_cluster", "Direction value from phase-speed fit", "1"),
            ("PhaseAvgWavelength_cluster", "Average wavelength from phase-speed ROI", "km"),
            ("PhaseAvgWavelengthError_cluster", "Average wavelength uncertainty from phase-speed ROI", "km"),
            ("PhaseAvgAngle_cluster", "Average angle from phase-speed ROI", "degree"),
            ("PhaseAvgAngleError_cluster", "Average angle uncertainty from phase-speed ROI", "degree"),
        ]
        for key, long_name, units in scalar_defs:
            _write_or_update(key, _make_full_scalar_array(key), scalar_dims, long_name, units)

        # Slice sums and stitched top grids, similar to old MF variables.
        if "SlicesNo" in nc.variables:
            slices_all = np.asarray(nc.variables["SlicesNo"][:], dtype=int)
            processed_idx = np.array(
                [int(item["global_cluster_index"]) for item in cluster_outputs],
                dtype=int,
            )
            processed_idx = processed_idx[(processed_idx >= 0) & (processed_idx < n_cluster)]

            if processed_idx.size > 0:
                full_slices = np.unique(slices_all[processed_idx]).astype(int)
                slice_dim_name = "slice_l8a"

                if slice_dim_name not in nc.dimensions:
                    nc.createDimension(slice_dim_name, len(full_slices))

                if "SliceNo_l8a_slices" not in nc.variables:
                    v = nc.createVariable("SliceNo_l8a_slices", "i4", (slice_dim_name,))
                    v.setncattr("long_name", "Slice numbers for L8A new variables")
                    v[:] = full_slices

                new_mf_slice = np.full((len(full_slices), y_tile, x_tile), np.nan, dtype=np.float32)
                new_mf_u_slice = np.full_like(new_mf_slice, np.nan)
                new_mf_v_slice = np.full_like(new_mf_slice, np.nan)

                for k, s in enumerate(full_slices):
                    idx2 = processed_idx[slices_all[processed_idx] == s]
                    if idx2.size > 0:
                        new_mf_slice[k] = np.nansum(new_mf_cluster[idx2], axis=0)
                        new_mf_u_slice[k] = np.nansum(new_mf_u_cluster[idx2], axis=0)
                        new_mf_v_slice[k] = np.nansum(new_mf_v_cluster[idx2], axis=0)

                slice_dims = (slice_dim_name, ytile_dim, xtile_dim)
                _write_or_update("NewMF_slice", new_mf_slice, slice_dims,
                                 "Per-slice sum of NewMF for processed clusters", "m2 s-2")
                _write_or_update("NewMF_u_slice", new_mf_u_slice, slice_dims,
                                 "Per-slice sum of signed zonal NewMF for processed clusters", "m2 s-2")
                _write_or_update("NewMF_v_slice", new_mf_v_slice, slice_dims,
                                 "Per-slice sum of signed meridional NewMF for processed clusters", "m2 s-2")

                # Stitch onto top grid if T_top or Temperature exists.
                top_template_name = "T_top" if "T_top" in nc.variables else None
                if top_template_name is None:
                    for candidate in ["Temperature", "Temp", "T"]:
                        if candidate in nc.variables:
                            top_template_name = candidate
                            break

                if top_template_name is not None:
                    template_var = nc.variables[top_template_name]
                    top_dims = template_var.dimensions
                    top_shape = template_var.shape

                    if len(top_shape) == 3:
                        new_mf_top = np.full(top_shape, np.nan, dtype=np.float32)
                        new_mf_u_top = np.full(top_shape, np.nan, dtype=np.float32)
                        new_mf_v_top = np.full(top_shape, np.nan, dtype=np.float32)

                        time_index_for_top = 0
                        nx_out = top_shape[2]
                        for k, s in enumerate(full_slices):
                            x0 = int(s) * x_tile
                            x1 = x0 + x_tile
                            x0c = max(0, min(x0, nx_out))
                            x1c = max(0, min(x1, nx_out))
                            if x1c <= x0c:
                                continue
                            w = x1c - x0c
                            ny_copy = min(y_tile, top_shape[1])
                            new_mf_top[time_index_for_top, 0:ny_copy, x0c:x1c] = new_mf_slice[k, 0:ny_copy, :w]
                            new_mf_u_top[time_index_for_top, 0:ny_copy, x0c:x1c] = new_mf_u_slice[k, 0:ny_copy, :w]
                            new_mf_v_top[time_index_for_top, 0:ny_copy, x0c:x1c] = new_mf_v_slice[k, 0:ny_copy, :w]

                        _write_or_update("NewMF_top", new_mf_top, top_dims,
                                         "Stitched NewMF sum on top grid for processed clusters", "m2 s-2")
                        _write_or_update("NewMF_u_top", new_mf_u_top, top_dims,
                                         "Stitched signed zonal NewMF sum on top grid for processed clusters", "m2 s-2")
                        _write_or_update("NewMF_v_top", new_mf_v_top, top_dims,
                                         "Stitched signed meridional NewMF sum on top grid for processed clusters", "m2 s-2")

        nc.setncattr(
            "l8a_note",
            "Copied from L7A and appended phase-speed-based lambda_z and new momentum flux variables.",
        )

    print(f"Saved L8A file with old + new variables: {dst_l8a_path}")
    return str(dst_l8a_path)

def save_combined_summary(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No rows to save.")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved combined summary: {out_csv}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 78)
    print(f"File                    : {NC_PATH}")
    print(f"NAVGEM file             : {NAVGEM_NC_PATH}")
    print(f"Target slices           : {TARGET_SLICES if TARGET_SLICES is not None else 'ALL'}")
    print(f"Requested clusters      : {TARGET_CLUSTERS_IN_SLICE if TARGET_CLUSTERS_IN_SLICE is not None else 'ALL'}")
    print(f"Plot folder base        : {PLOT_OUT_BASE_DIR}")
    print(f"Phase speed folder base : {PHASE_SPEED_BASE_OUT_DIR}")
    print(f"Save L8A file           : {SAVE_L8A_FILE}")
    print(f"L8A output file         : {L8A_OUT_PATH}")
    print("=" * 78)

    # Read NAVGEM once. Each cluster extracts the correct slice from these raw 2D fields.
    navgem_data = read_navgem_data(NAVGEM_NC_PATH)

    results = []

    with Dataset(NC_PATH, "r") as nc:
        slices_all = np.asarray(nc.variables["SlicesNo"][:], dtype=int)

        if TARGET_SLICES is None:
            slices_to_run = sorted(np.unique(slices_all).astype(int))
        else:
            slices_to_run = TARGET_SLICES

        for target_slice in slices_to_run:
            idx_in_slice = np.where(slices_all == target_slice)[0]
            n_available = int(idx_in_slice.size)

            print("-" * 78)
            print(f"Slice {target_slice}: {n_available} clusters available")

            if n_available == 0:
                results.append({
                    "slice": int(target_slice),
                    "cluster_in_slice": -1,
                    "global_cluster_index": -1,
                    "status": "skipped_no_clusters_in_slice",
                })
                continue

            if TARGET_CLUSTERS_IN_SLICE is None:
                clusters_to_run = list(range(n_available))
            else:
                clusters_to_run = [c for c in TARGET_CLUSTERS_IN_SLICE if c < n_available]

            print(f"Clusters to run         : {clusters_to_run}")

            if not clusters_to_run:
                results.append({
                    "slice": int(target_slice),
                    "cluster_in_slice": -1,
                    "global_cluster_index": -1,
                    "status": "skipped_no_requested_clusters_available",
                    "n_clusters_in_slice": n_available,
                })
                continue

            for target_cluster in clusters_to_run:
                try:
                    row = process_one_cluster(
                        nc,
                        slices_all,
                        target_slice,
                        target_cluster,
                        navgem_data,
                    )
                    results.append(row)
                except Exception as exc:
                    print(
                        f"  ERROR for slice {target_slice}, "
                        f"cluster {target_cluster}: {exc}"
                    )
                    traceback.print_exc()
                    results.append({
                        "slice": int(target_slice),
                        "cluster_in_slice": int(target_cluster),
                        "global_cluster_index": -1,
                        "status": "error",
                        "error_message": str(exc),
                    })

    save_combined_summary(results, COMBINED_SUMMARY_CSV)

    if SAVE_L8A_FILE:
        save_l8a_with_new_variables(
            src_l7a_path=NC_PATH,
            dst_l8a_path=L8A_OUT_PATH,
            cluster_outputs=L8A_CLUSTER_OUTPUTS,
        )

    print("Done.")


if __name__ == "__main__":
    main()
