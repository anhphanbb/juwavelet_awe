#!/usr/bin/env python3
"""
phase_speed_from_l7a_point.py

Callable version of the notebook phase-speed workflow.

Main use:
    from phase_speed_from_l7a_point import calculate_phase_speed_for_l7a_point

    result = calculate_phase_speed_for_l7a_point(
        l1a_nc_path=r"D:/awe_l1a_q20_2024075T0538_01770_v23_remap85.nc",
        target_lat=lat_saved,
        target_lon=lon_saved,
        target_angle_deg=true_angle_point,
        target_wavelength_km=lambda_point,
        out_dir="AWE/01770/remap85_from_l7a/cluster_001",
        cluster_id="cluster_001",
    )

Batch use:
    from phase_speed_from_l7a_point import calculate_phase_speed_for_multiple_l7a_points

    clusters = [
        {
            "cluster_id": "slice04_cluster01",
            "target_lat": lat1,
            "target_lon": lon1,
            "target_angle_deg": angle1,
            "target_wavelength_km": wavelength1,
            "roi_label": 22,
        },
        {
            "cluster_id": "slice04_cluster02",
            "target_lat": lat2,
            "target_lon": lon2,
            "target_angle_deg": angle2,
            "target_wavelength_km": wavelength2,
            "roi_label": 22,
        },
    ]

    results = calculate_phase_speed_for_multiple_l7a_points(
        l1a_nc_path=r"D:/awe_l1a_q20_2024075T0538_01770_v23_remap85.nc",
        clusters=clusters,
        base_out_dir="AWE/01770/remap85_from_l7a",
    )

The function uses the L7A point values as:
    target_lat            -> target lat
    target_lon            -> target lon
    target_angle_deg      -> target angle filter
    target_wavelength_km  -> target wavelength / band-pass center

It returns a dictionary with phase speed information, and optionally saves
intermediate PNG frames, ROI CSVs, and summary CSV tables.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import imageio.v3 as iio
from scipy.ndimage import maximum_filter, minimum_filter, median_filter
from scipy import ndimage
from skimage import morphology

from juwavelet import transform, utils


def _silent_print(*args, **kwargs):
    pass

# Silence juwavelet internal print messages.
utils.print = _silent_print


@dataclass
class PhaseSpeedResult:
    roi: str
    target_lat: float
    target_lon: float
    target_angle_deg: float
    target_wavelength_km: float
    closest_frame: int
    closest_distance_km: float
    t_start: int
    t_end: int
    n_frames_used: int
    n_points: int
    best_a_deg_per_frame: float
    avg_wavelength_km: float
    avg_wavelength_error_km: float
    avg_angle_deg: float
    avg_angle_error_deg: float
    direction: int
    phase_slope_deg_per_point: float
    phase_speed_m_per_s: float
    phase_speed_error_m_per_s: float
    sse_min: float
    csv_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    v = float(v)
    if abs(v) >= 1000 or abs(v) < 0.1:
        return f"{v:.2e}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def robust_limits(a, p_lo=2, p_hi=98):
    a = np.asarray(a, dtype=float)
    good = np.isfinite(a)
    if not np.any(good):
        return -1.0, 1.0
    lo, hi = np.nanpercentile(a[good], [p_lo, p_hi])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
        lo = np.nanmin(a[good])
        hi = np.nanmax(a[good])
    return float(lo), float(hi)


# -----------------------------------------------------------------------------
# Frame creation from L1A/remap file
# -----------------------------------------------------------------------------
def save_preinterp_frames_near_target(
    nc_path: str | Path,
    out_dir: str | Path,
    target_lat: float,
    target_lon: float,
    *,
    center_y: int = 150,
    center_x: int = 150,
    n_frames_before_after: int = 12,
    r_plot_km: float = 200.0,
    vmin: float = 3.0,
    vmax: float = 14.0,
    dpi: int = 65,
    force_rebuild: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Save PNG frames around the target location.

    Returns:
        frame_dir, metadata
    """
    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    frame_dir = out_dir / "awe_radiance_pre_interpolation_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frame_dir.glob("*.png"))
    if existing and not force_rebuild:
        # Still inspect the file to return the frame metadata.
        pass

    with Dataset(nc_path, "r") as ds:
        rad_all = np.asarray(ds.variables["Radiance"][:], dtype=float)
        lat_all = np.asarray(ds.variables["Latitude"][:], dtype=float)
        lon_raw = np.asarray(ds.variables["Longitude"][:], dtype=float)

    lon_all = ((lon_raw + 180) % 360) - 180
    nt = rad_all.shape[0]

    center_lat = lat_all[:, center_y, center_x]
    center_lon = lon_all[:, center_y, center_x]
    dist_km = haversine_km(center_lat, center_lon, target_lat, target_lon)

    valid_center = np.isfinite(center_lat) & np.isfinite(center_lon) & np.isfinite(dist_km)
    if not np.any(valid_center):
        raise ValueError("No valid center lat/lon values found in the L1A/remap file.")

    valid_indices = np.where(valid_center)[0]
    closest_frame = int(valid_indices[np.argmin(dist_km[valid_center])])
    closest_distance = float(dist_km[closest_frame])

    t_start = max(0, closest_frame - n_frames_before_after)
    t_end = min(nt - 1, closest_frame + n_frames_before_after)
    frames = list(range(t_start, t_end + 1))

    if existing and not force_rebuild:
        meta = {
            "closest_frame": closest_frame,
            "closest_distance_km": closest_distance,
            "t_start": t_start,
            "t_end": t_end,
            "frames": frames,
        }
        return frame_dir, meta

    # Clear old PNGs only when rebuilding.
    for p in frame_dir.glob("*.png"):
        p.unlink()

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(target_lat))
    dlat_deg = r_plot_km / km_per_deg_lat
    dlon_deg = r_plot_km / km_per_deg_lon

    lat_min = target_lat - dlat_deg
    lat_max = target_lat + dlat_deg
    lon_min = target_lon - dlon_deg
    lon_max = target_lon + dlon_deg

    for frame_id in frames:
        lat_t = lat_all[frame_id]
        lon_t = lon_all[frame_id]
        rad_t = rad_all[frame_id]

        lat_flat = lat_t.ravel()
        lon_flat = lon_t.ravel()
        rad_flat = rad_t.ravel()

        valid = np.isfinite(lat_flat) & np.isfinite(lon_flat) & np.isfinite(rad_flat)
        inside_box = (
            (lat_flat >= lat_min) & (lat_flat <= lat_max)
            & (lon_flat >= lon_min) & (lon_flat <= lon_max)
        )
        mask = valid & inside_box

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(
            lon_flat[mask],
            lat_flat[mask],
            c=rad_flat[mask],
            s=5,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.axis("off")
        out_path = frame_dir / f"awe_preinterp_{frame_id:04d}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)

    meta = {
        "closest_frame": closest_frame,
        "closest_distance_km": closest_distance,
        "t_start": t_start,
        "t_end": t_end,
        "frames": frames,
    }
    return frame_dir, meta


# -----------------------------------------------------------------------------
# ROI helpers
# -----------------------------------------------------------------------------
def parse_roi_label(label: int | str) -> Tuple[int, int]:
    s = str(label).strip()
    if not s.isdigit() or len(s) != 2:
        raise ValueError(f"ROI label must have exactly 2 digits, got: {label}")
    return int(s[0]), int(s[1])


def roi_label_to_slices(
    label: int | str,
    *,
    target_wavelength_km: Optional[float] = None,
    img_nx: int = 200,
    img_ny: int = 200,
    roi_width: Optional[int] = None,
    roi_height: Optional[int] = None,
) -> Tuple[str, slice, slice]:

    """
    ROI logic:

    λ < 64 km  -> ROI = 80 x 80
    λ ≥ 64 km  -> ROI = 120 x 120

    ROI stays centered around (100, 100).

    ROI labels shift the ROI center by ±25 pixels:
        11 = upper-left
        22 = center
        33 = lower-right

    Example for 100x100 ROI:

        Center ROI (22):
            x0 = 100 - 50 = 50
            y0 = 100 - 50 = 50

        ROI 33:
            x0 = 50 + 25 = 75
            y0 = 50 + 25 = 75
    """

    a, b = parse_roi_label(label)

    if not (1 <= a <= 3 and 1 <= b <= 3):
        raise ValueError(f"ROI label {label}: a,b must be 1..3")

    # ROI size
    if roi_width is None or roi_height is None:
        if target_wavelength_km is not None and target_wavelength_km < 64:
            roi_width = 80
            roi_height = 80
        else:
            roi_width = 120
            roi_height = 120

    # Image center
    center_x = img_nx // 2
    center_y = img_ny // 2

    # Shift grid
    # a,b = 1,2,3 -> shifts = -25,0,+25
    shift_x = 25 * (a - 2)
    shift_y = 25 * (b - 2)

    # Centered ROI with shift
    x0 = int(center_x - roi_width // 2 + shift_x)
    y0 = int(center_y - roi_height // 2 + shift_y)

    x1 = x0 + roi_width
    y1 = y0 + roi_height

    # Clip to image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_nx, x1)
    y1 = min(img_ny, y1)

    roi_name = f"ROI_{label}_{roi_width}x{roi_height}"

    return roi_name, slice(x0, x1), slice(y0, y1)


def choose_roi_label_from_peak(
    peak_x: float,
    peak_y: float,
    *,
    roi_width: int,
    roi_height: int,
    threshold_px: float = 25.0,
) -> int:
    """
    Choose a 3x3 ROI label from the peak position found inside ROI 22.

    The first digit controls x shift:
        1 = left, 2 = center, 3 = right

    The second digit controls y shift:
        1 = up, 2 = center, 3 = down

    Example:
        peak is 30 px right of ROI center -> 32
        peak is 35 px above ROI center    -> 21
    """
    center_x = roi_width / 2.0
    center_y = roi_height / 2.0

    dx = float(peak_x) - center_x
    dy = float(peak_y) - center_y

    a = 2
    b = 2

    if dx > threshold_px:
        a = 3
    elif dx < -threshold_px:
        a = 1

    # Image coordinates: smaller y is higher/up.
    if dy > threshold_px:
        b = 3
    elif dy < -threshold_px:
        b = 1

    return int(f"{a}{b}")


def _roi_size_for_wavelength(
    target_wavelength_km: Optional[float],
    *,
    roi_width: Optional[int] = None,
    roi_height: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Keep the ROI-size rule in one place so smart ROI and roi_label_to_slices agree.
    """
    if roi_width is not None and roi_height is not None:
        return int(roi_width), int(roi_height)

    if target_wavelength_km is not None and target_wavelength_km < 64:
        return 80, 80
    return 120, 120


def find_peak_in_roi_for_file(
    img_path: Path,
    x_slice: slice,
    y_slice: slice,
    *,
    target_wavelength_km: float,
    target_angle_deg: float,
    dx: float = 2.0,
    dy: float = 2.0,
    s0: Optional[float] = None,
    dj: float = 1 / 8,
    js: int = 6,
    jt: int = 18,
    aspect: float = 1.0,
    bandpass_factor: float = 2.0,
    angle_half_width_deg: float = 20.0,
    cluster_min_amp: float = 0.25,
    cluster_thr: float = 0.1,
    standardize: bool = True,
    gauss_blur_sigma: float = 0.0,
    apply_taper: bool = True,
    taper_edge_pixels: int = 6,
) -> Tuple[Optional[int], Optional[int], float, float]:
    """
    Run the same preprocessing + wavelet filtering on one image/ROI and return
    the strongest crest/peak position inside that ROI.

    This is used only for smart ROI selection:
        1. Try ROI 22 on the first frame.
        2. Find peak position inside ROI 22.
        3. Pick the better ROI label.
    """
    if s0 is None:
        s0 = float(target_wavelength_km / (np.sqrt(2)))

    full = load_image(img_path)
    raw = full[y_slice, x_slice].astype(float)

    plane = fit_plane(raw)
    detrended = raw - plane
    fourier_bg, _ = extract_first_fourier_bg(detrended)
    background = plane + fourier_bg
    remaining = raw - background
    remaining = apply_edge_hanning_taper(remaining, taper_edge_pixels) if apply_taper else remaining

    proc = remaining.copy()
    if standardize:
        proc = proc - np.median(proc)
        s_proc = np.std(proc)
        if s_proc > 0:
            proc = proc / s_proc
    proc = maybe_blur(proc, gauss_blur_sigma)

    cwt = transform.decompose2d(proc.T, dx=dx, dy=dy, s0=s0, dj=dj, js=js, jt=jt, aspect=aspect)
    decomposition = cwt["decomposition"]

    lam_eff, theta_deg = compute_lambda_theta_fields(cwt)
    mask_keep = (
        np.isfinite(lam_eff)
        & (lam_eff >= target_wavelength_km / bandpass_factor)
        & (lam_eff <= target_wavelength_km * bandpass_factor)
        & angle180_window_mask(theta_deg, target_angle_deg, angle_half_width_deg)
    )
    decomposition *= mask_keep[:, :, None, None].astype(decomposition.dtype)

    amps, idxs, iwave = utils.identify_cluster2d(cwt, min_amp=cluster_min_amp, thr=cluster_thr)
    n_clusters = len(amps) if amps is not None else 0
    if n_clusters == 0:
        return None, None, np.nan, np.nan

    order = np.argsort(amps)[::-1]
    kcl = int(order[0])
    orig_decomp = decomposition.copy()
    decomposition[:] = orig_decomp
    decomposition[iwave != kcl] = 0

    rec = transform.reconstruct2d(cwt)
    rec_centered = rec.T - np.median(rec)

    wavelength_km, wavelength_error_km, angle_deg, angle_error_deg, _ = compute_cluster_weighted_lambda_angle(cwt, iwave, kcl)

    crests = find_top_crests(rec_centered, num=1, neighborhood=7, min_rel_amp=0.2)
    if crests:
        peak_y, peak_x, _ = crests[0]
        return int(peak_x), int(peak_y), float(wavelength_km), float(angle_deg)

    peak_y, peak_x = np.unravel_index(np.nanargmax(rec_centered), rec_centered.shape)
    return int(peak_x), int(peak_y), float(wavelength_km), float(angle_deg)


def choose_smart_roi_label_from_first_frame(
    frame_dir: str | Path,
    *,
    target_wavelength_km: float,
    target_angle_deg: float,
    default_roi_label: int | str = "smart",
    threshold_px: float = 25.0,
    verbose: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    Two-pass ROI selection.

    First, use ROI 22 as the scouting ROI on the first saved frame.
    Then use the detected peak location inside that ROI to choose the final ROI.
    The full CSV/phase-speed analysis is then run only on the final ROI.
    """
    frame_dir = Path(frame_dir)
    files = sorted(frame_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG image files found in {frame_dir}")

    scout_roi_label = default_roi_label
    scout_roi_name, scout_x_slice, scout_y_slice = roi_label_to_slices(
        scout_roi_label,
        target_wavelength_km=target_wavelength_km,
    )

    roi_width = scout_x_slice.stop - scout_x_slice.start
    roi_height = scout_y_slice.stop - scout_y_slice.start

    peak_x, peak_y, scout_lam, scout_angle = find_peak_in_roi_for_file(
        files[0],
        scout_x_slice,
        scout_y_slice,
        target_wavelength_km=target_wavelength_km,
        target_angle_deg=target_angle_deg,
    )

    if peak_x is None or peak_y is None:
        final_roi_label = int(scout_roi_label)
        dx_from_center = np.nan
        dy_from_center = np.nan
    else:
        dx_from_center = float(peak_x) - roi_width / 2.0
        dy_from_center = float(peak_y) - roi_height / 2.0
        final_roi_label = choose_roi_label_from_peak(
            peak_x,
            peak_y,
            roi_width=roi_width,
            roi_height=roi_height,
            threshold_px=threshold_px,
        )

    info = {
        "scout_roi_label": int(scout_roi_label),
        "scout_roi_name": scout_roi_name,
        "scout_frame": str(files[0]),
        "scout_peak_x": peak_x,
        "scout_peak_y": peak_y,
        "scout_roi_width": roi_width,
        "scout_roi_height": roi_height,
        "scout_peak_dx_from_center_px": dx_from_center,
        "scout_peak_dy_from_center_px": dy_from_center,
        "scout_wavelength_km": scout_lam,
        "scout_angle_deg": scout_angle,
        "smart_roi_label": int(final_roi_label),
        "smart_roi_threshold_px": float(threshold_px),
    }

    if verbose:
        print("Smart ROI scouting:")
        print(f"  scout ROI                  : {scout_roi_name}")
        print(f"  first frame                 : {files[0].name}")
        print(f"  peak in scout ROI           : x={peak_x}, y={peak_y}")
        print(f"  offset from scout ROI center: dx={dx_from_center:.1f}, dy={dy_from_center:.1f} px")
        print(f"  final ROI label             : {final_roi_label}")

    return int(final_roi_label), info


# -----------------------------------------------------------------------------
# Image and preprocessing helpers
# -----------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 3:
        img = np.mean(img, axis=-1)
    if img.ndim != 2:
        raise ValueError(f"Expected single-frame 2D image; got shape {img.shape}")
    Z = np.asarray(img, dtype=np.float32)
    med = float(np.median(Z))
    return np.nan_to_num(Z, nan=med, posinf=med, neginf=med).astype(np.float32, copy=False)


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
    ny, nx = Z.shape
    if edge <= 0:
        return Z
    return Z * np.outer(_hanning_edge_1d(ny, edge), _hanning_edge_1d(nx, edge))


def maybe_blur(Z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return Z
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(Z, sigma=float(sigma))


def angle180_window_mask(theta_deg, target_angle_deg, half_width_deg):
    theta_deg = np.asarray(theta_deg, dtype=float)
    target_angle_deg = float(target_angle_deg) % 180.0
    diff = np.abs((theta_deg - target_angle_deg + 90.0) % 180.0 - 90.0)
    return np.isfinite(theta_deg) & (diff <= float(half_width_deg))


def compute_lambda_theta_fields(cwt: dict) -> Tuple[np.ndarray, np.ndarray]:
    lamx = np.asarray(cwt["wavelength_x"], float)
    lamy = np.asarray(cwt["wavelength_y"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_lamx2 = np.where(np.isfinite(lamx) & (lamx != 0.0), 1.0 / lamx**2, 0.0)
        inv_lamy2 = np.where(np.isfinite(lamy) & (lamy != 0.0), 1.0 / lamy**2, 0.0)
        inv2 = inv_lamx2 + inv_lamy2
        lam_eff = np.where(inv2 > 0, 1.0 / np.sqrt(inv2), np.inf)
        kx = np.where(np.isfinite(lamx) & (lamx != 0.0), 1.0 / lamx, 0.0)
        ky = np.where(np.isfinite(lamy) & (lamy != 0.0), 1.0 / lamy, 0.0)
        theta_deg = np.degrees(np.arctan2(ky, kx))
        theta_deg = np.mod(180 - theta_deg, 180.0)
    return lam_eff, theta_deg


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan
    v = values[mask]
    w = weights[mask]
    wsum = np.sum(w)
    mean = np.sum(w * v) / wsum
    var = np.sum(w * (v - mean) ** 2) / wsum
    return float(mean), float(np.sqrt(var))


def weighted_circular_mean_std_deg(theta_deg_values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    theta = np.asarray(theta_deg_values, float)
    weights = np.asarray(weights, float)
    mask = np.isfinite(theta) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan
    theta = theta[mask]
    w = weights[mask]
    wsum = np.sum(w)
    ang = np.deg2rad(2.0 * theta)
    mean_cos = np.sum(w * np.cos(ang)) / wsum
    mean_sin = np.sum(w * np.sin(ang)) / wsum
    mean_angle = 0.5 * np.rad2deg(np.arctan2(mean_sin, mean_cos))
    mean_angle = np.mod(mean_angle, 180.0)
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    std_angle = np.nan if (not np.isfinite(R) or R <= 0) else np.rad2deg(np.sqrt(-2.0 * np.log(min(R, 1.0)))) / 2.0
    return float(mean_angle), float(std_angle)


def compute_cluster_weighted_lambda_angle(cwt: dict, iwave: np.ndarray, kcl: int) -> Tuple[float, float, float, float, int]:
    lam_eff, theta_deg = compute_lambda_theta_fields(cwt)
    decomposition = np.asarray(cwt["decomposition"])
    cluster_mask = iwave == kcl
    cluster_st = np.any(cluster_mask, axis=(2, 3))
    n_points = int(np.sum(cluster_st))
    if n_points == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    amp_st = np.sum(np.abs(decomposition) * cluster_mask, axis=(2, 3))
    lam_vals = lam_eff[cluster_st]
    theta_vals = theta_deg[cluster_st]
    weights = amp_st[cluster_st]
    lam_mean, lam_err = weighted_mean_std(lam_vals, weights)
    theta_mean, theta_err = weighted_circular_mean_std_deg(theta_vals, weights)
    return lam_mean, lam_err, theta_mean, theta_err, n_points


def find_top_crests(Z: np.ndarray, num: int = 1, neighborhood: int = 7, min_rel_amp: float = 0.2):
    Z = np.asarray(Z, float)
    if Z.size == 0:
        return []
    max_val = np.nanmax(Z)
    if not np.isfinite(max_val) or max_val <= 0:
        return []
    local_max = maximum_filter(Z, size=max(int(neighborhood), 1), mode="nearest")
    mask = (Z == local_max) & (Z > 0) & (Z >= min_rel_amp * max_val)
    ys, xs = np.where(mask)
    if ys.size == 0:
        iy, ix = np.unravel_index(np.nanargmax(Z), Z.shape)
        return [(int(iy), int(ix), float(Z[iy, ix]))][:num]
    vals = Z[ys, xs]
    order = np.argsort(vals)[::-1]
    return [(int(ys[idx]), int(xs[idx]), float(vals[idx])) for idx in order[:num]]


def compute_ridge_mask(Z: np.ndarray, level_frac: float = 0.6) -> np.ndarray:
    Z = np.asarray(Z, float)
    if Z.size == 0:
        return np.zeros_like(Z, dtype=bool)
    max_val = np.nanmax(Z)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.zeros_like(Z, dtype=bool)
    labels, num = ndimage.label(Z >= level_frac * max_val)
    if num == 0:
        return np.zeros_like(Z, dtype=bool)
    iy_max, ix_max = np.unravel_index(np.nanargmax(Z), Z.shape)
    label_peak = labels[iy_max, ix_max]
    if label_peak == 0:
        return np.zeros_like(Z, dtype=bool)
    return morphology.skeletonize(labels == label_peak)


# -----------------------------------------------------------------------------
# Wavelet ROI CSV creation
# -----------------------------------------------------------------------------
def run_pipeline_for_file(
    img_path: Path,
    overlay_dir: Path,
    x_slice: slice,
    y_slice: slice,
    *,
    target_wavelength_km: float,
    target_angle_deg: float,
    dx: float = 2.0,
    dy: float = 2.0,
    s0: Optional[float] = None,
    dj: float = 1 / 8,
    js: int = 6,
    jt: int = 18,
    aspect: float = 1.0,
    bandpass_factor: float = 2.0,
    angle_half_width_deg: float = 20.0,
    cluster_min_amp: float = 0.25,
    cluster_thr: float = 0.1,
    standardize: bool = True,
    gauss_blur_sigma: float = 0.0,
    apply_taper: bool = True,
    taper_edge_pixels: int = 6,
    save_overlay: bool = True,
    crest_iy: Optional[np.ndarray] = None,
    crest_ix: Optional[np.ndarray] = None,
    init_crests_from_this_file: bool = False,
    n_sample_points: int = 7,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, float, float, float]:

    if s0 is None:
        s0 = float(target_wavelength_km / (np.sqrt(2)))

    full = load_image(img_path)
    raw = full[y_slice, x_slice].astype(float)

    plane = fit_plane(raw)
    detrended = raw - plane
    fourier_bg, _ = extract_first_fourier_bg(detrended)
    background = plane + fourier_bg
    remaining = raw - background
    remaining = apply_edge_hanning_taper(remaining, taper_edge_pixels) if apply_taper else remaining

    proc = remaining.copy()
    if standardize:
        proc = proc - np.median(proc)
        s_proc = np.std(proc)
        if s_proc > 0:
            proc = proc / s_proc
    proc = maybe_blur(proc, gauss_blur_sigma)

    cwt = transform.decompose2d(proc.T, dx=dx, dy=dy, s0=s0, dj=dj, js=js, jt=jt, aspect=aspect)
    decomposition = cwt["decomposition"]

    lam_eff, theta_deg = compute_lambda_theta_fields(cwt)
    mask_keep = (
        np.isfinite(lam_eff)
        & (lam_eff >= target_wavelength_km / bandpass_factor)
        & (lam_eff <= target_wavelength_km * bandpass_factor)
        & angle180_window_mask(theta_deg, target_angle_deg, angle_half_width_deg)
    )
    decomposition *= mask_keep[:, :, None, None].astype(decomposition.dtype)

    amps, idxs, iwave = utils.identify_cluster2d(cwt, min_amp=cluster_min_amp, thr=cluster_thr)
    n_clusters = len(amps) if amps is not None else 0
    if n_clusters == 0:
        return None, crest_iy, crest_ix, np.nan, np.nan, np.nan, np.nan

    order = np.argsort(amps)[::-1]
    kcl = int(order[0])
    orig_decomp = decomposition.copy()
    decomposition[:] = orig_decomp
    decomposition[iwave != kcl] = 0

    rec = transform.reconstruct2d(cwt)
    rec_centered = rec.T - np.median(rec)
    ny_img, nx_img = rec_centered.shape

    wavelength_km, wavelength_error_km, angle_deg, angle_error_deg, _ = compute_cluster_weighted_lambda_angle(cwt, iwave, kcl)

    if np.isfinite(wavelength_km) and wavelength_km > 0:
        radius_pix = max(int(np.round((wavelength_km / 2.0) / dx)), 1)
    else:
        radius_pix = 1
    win_size = 2 * radius_pix + 1

    local_max = maximum_filter(rec_centered, size=win_size, mode="nearest")
    local_min = minimum_filter(rec_centered, size=win_size, mode="nearest")
    denom_safe = np.where(local_max - local_min != 0, local_max - local_min, np.nan)
    rel_amp = -1.0 + 2.0 * (rec_centered - local_min) / denom_safe
    rel_amp = np.where(np.isfinite(rel_amp), rel_amp, 0.0)
    rel_amp = np.clip(rel_amp, -1.0, 1.0)

    if init_crests_from_this_file and crest_iy is None and crest_ix is None:
        crests = find_top_crests(rec_centered, num=1, neighborhood=7, min_rel_amp=0.2)
        if crests:
            crest0_iy, crest0_ix, _ = crests[0]
            # The angle saved from compute_lambda_theta_fields() is flipped as 180 - theta.
            # For choosing the 7 points, we want the actual propagation/normal direction.
            sample_angle_deg = (180.0 - angle_deg) % 180.0
            
            theta_rad = np.deg2rad(sample_angle_deg)
            ux = np.cos(theta_rad)
            uy = np.sin(theta_rad)
            step_pix = max(radius_pix // 4, 1)
            offsets = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
            if offsets.size != n_sample_points:
                offsets = np.linspace(-3, 3, n_sample_points)
            crest_ix = crest0_ix + np.round(step_pix * offsets * ux).astype(int)
            crest_iy = crest0_iy + np.round(step_pix * offsets * uy).astype(int)
            crest_ix = np.clip(crest_ix, 0, nx_img - 1)
            crest_iy = np.clip(crest_iy, 0, ny_img - 1)

    vals_on_crests = None
    if crest_iy is not None and crest_ix is not None:
        if crest_iy.max() < ny_img and crest_ix.max() < nx_img:
            vals_on_crests = rel_amp[crest_iy, crest_ix]

    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        x1d = np.arange(nx_img, dtype=float)
        y1d = np.arange(ny_img, dtype=float)[::-1]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
        bg = remaining - np.median(remaining)
        bg_vmin, bg_vmax = robust_limits(bg)
        axs[0].pcolormesh(x1d, y1d, bg, shading="auto", cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
        axs[0].set_title("Background removed")
        axs[1].pcolormesh(x1d, y1d, bg, shading="auto", cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
        axs[1].pcolormesh(x1d, y1d, rec_centered, shading="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6, alpha=0.6)
        axs[1].set_title(f"Cluster λ={_fmt(wavelength_km)} km, θ={angle_deg:.1f}°")
        im = axs[2].pcolormesh(x1d, y1d, rel_amp, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axs[2].set_title("Local relative amplitude")
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        for ax in axs:
            ax.set_aspect("equal")
            if crest_iy is not None and crest_ix is not None:
                ax.plot(crest_ix, ny_img - 1 - crest_iy, "mo", ms=3)
        fig.tight_layout()
        fig.savefig(overlay_dir / f"{img_path.stem}_overlay_clusters_crests_phase.png", dpi=200)
        plt.close(fig)

    return vals_on_crests, crest_iy, crest_ix, float(wavelength_km), float(angle_deg), float(wavelength_error_km), float(angle_error_deg)


def build_roi_csv(
    frame_dir: str | Path,
    roi_label: int | str,
    *,
    target_wavelength_km: float,
    target_angle_deg: float,
    overlay_base_dir: Optional[str | Path] = None,
    save_overlay: bool = True,
    force_rebuild: bool = True,
    smart_roi_threshold_px: float = 25.0,
    verbose: bool = True,
) -> Tuple[Path, int, Dict[str, Any]]:
    frame_dir = Path(frame_dir)

    smart_info: Dict[str, Any] = {}
    roi_label_in = str(roi_label).strip().lower()
    if roi_label_in in {"smart", "auto"}:
        roi_label, smart_info = choose_smart_roi_label_from_first_frame(
            frame_dir,
            target_wavelength_km=target_wavelength_km,
            target_angle_deg=target_angle_deg,
            default_roi_label=22,
            threshold_px=smart_roi_threshold_px,
            verbose=verbose,
        )

    roi_name, x_slice, y_slice = roi_label_to_slices(
        roi_label,
        target_wavelength_km=target_wavelength_km,
    )
    overlay_base = Path(overlay_base_dir) if overlay_base_dir is not None else frame_dir / "overlays_crests_phase"
    overlay_dir = overlay_base / roi_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    csv_path = overlay_dir / "crest_relative_amplitude_evolution.csv"

    if csv_path.exists() and not force_rebuild:
        return csv_path, int(roi_label), smart_info

    files = sorted(frame_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG image files found in {frame_dir}")

    crest_iy = None
    crest_ix = None
    crest_value_series: List[np.ndarray] = []
    frame_indices: List[int] = []
    wavelength_series_km: List[float] = []
    wavelength_error_series_km: List[float] = []
    angle_series_deg: List[float] = []
    angle_error_series_deg: List[float] = []

    for i, img_path in enumerate(files):
        vals, crest_iy, crest_ix, lam, ang, lam_err, ang_err = run_pipeline_for_file(
            img_path,
            overlay_dir,
            x_slice,
            y_slice,
            target_wavelength_km=target_wavelength_km,
            target_angle_deg=target_angle_deg,
            save_overlay=save_overlay,
            crest_iy=crest_iy,
            crest_ix=crest_ix,
            init_crests_from_this_file=(i == 0),
        )
        if vals is not None:
            crest_value_series.append(vals)
            frame_indices.append(i)
            wavelength_series_km.append(lam)
            wavelength_error_series_km.append(lam_err)
            angle_series_deg.append(ang)
            angle_error_series_deg.append(ang_err)

    if not crest_value_series:
        raise RuntimeError(f"No amplitude data collected for {roi_name}.")

    val_arr = np.vstack(crest_value_series)
    n_files_used, n_points = val_arr.shape
    frames = np.array(frame_indices, dtype=int)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_index", "wavelength_km", "wavelength_error_km", "angle_deg", "angle_error_deg"] + [f"pt{p + 1}" for p in range(n_points)]
        writer.writerow(header)
        for irow in range(n_files_used):
            row = [
                int(frames[irow]),
                float(wavelength_series_km[irow]),
                float(wavelength_error_series_km[irow]),
                float(angle_series_deg[irow]),
                float(angle_error_series_deg[irow]),
            ] + [float(x) for x in val_arr[irow, :]]
            writer.writerow(row)

    np.save(overlay_dir / "crest_relative_amplitude_evolution.npy", val_arr)
    return csv_path, int(roi_label), smart_info


# -----------------------------------------------------------------------------
# Phase speed fit helpers
# -----------------------------------------------------------------------------
def combine_mean_and_measurement_error(values, errors=None):
    values = np.asarray(values, float)
    finite_v = values[np.isfinite(values)]
    if finite_v.size == 0:
        return np.nan, np.nan
    mean_value = float(np.nanmean(values))
    spread = float(np.nanstd(values, ddof=1)) if finite_v.size > 1 else 0.0
    if errors is None:
        meas = 0.0
    else:
        errors = np.asarray(errors, float)
        finite_e = errors[np.isfinite(errors)]
        meas = float(np.sqrt(np.nanmean(finite_e**2))) if finite_e.size > 0 else 0.0
    return mean_value, float(np.sqrt(spread**2 + meas**2))


def load_roi_csv(csv_path: str | Path):
    csv_path = Path(csv_path)
    with open(csv_path, "r", newline="") as f:
        header = f.readline().strip().split(",")
    header_clean = [h.strip() for h in header]
    header_lc = [h.lower() for h in header_clean]

    def _find_col(names, required=True):
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name.lower() in header_lc:
                return header_lc.index(name.lower())
        if required:
            raise ValueError(f"Column {names} not found in CSV header: {header_clean}")
        return None

    col_t = _find_col("image_index")
    col_lam = _find_col(["wavelength_km", "dominant_wavelength_km"], required=False)
    col_lam_err = _find_col(["wavelength_error_km", "wavelength_std_km", "wavelength_stdev_km"], required=False)
    col_angle = _find_col(["angle_deg", "dominant_theta_deg", "theta_deg"], required=False)
    col_angle_err = _find_col(["angle_error_deg", "angle_std_deg", "angle_stdev_deg", "theta_error_deg"], required=False)
    metadata_cols = {c for c in [col_t, col_lam, col_lam_err, col_angle, col_angle_err] if c is not None}
    pt_cols = [i for i, h in enumerate(header_lc) if h.startswith("pt")]
    if not pt_cols:
        pt_cols = [i for i in range(len(header_lc)) if i not in metadata_cols]
    if not pt_cols:
        raise ValueError(f"No point columns found in CSV header: {header_clean}")

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]

    t = data[:, col_t].astype(float)
    wavelength_km = data[:, col_lam].astype(float) if col_lam is not None else None
    wavelength_error_km = data[:, col_lam_err].astype(float) if col_lam_err is not None else None
    angle_deg = data[:, col_angle].astype(float) if col_angle is not None else None
    angle_error_deg = data[:, col_angle_err].astype(float) if col_angle_err is not None else None
    Y = data[:, pt_cols].astype(float)
    pt_names = [header_clean[i] for i in pt_cols]
    return t, wavelength_km, wavelength_error_km, angle_deg, angle_error_deg, Y, pt_names


def fit_phase_grid_search(
    t,
    Y,
    *,
    a0_deg: float = 20 / 60,
    f_min: float = 0.25,
    f_max: float = 4.0,
    n_trial: int = 201,
    n_b: int = 720,
):
    n_frames, n_pts = Y.shape
    a_trial_deg = a0_deg * np.linspace(f_min, f_max, n_trial)
    a_trial = np.deg2rad(a_trial_deg)
    b_grid = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    errors = np.zeros(n_trial)

    for k, a_k in enumerate(a_trial):
        base_phase = a_k * t
        phase_mat = base_phase[:, None] + b_grid[None, :]
        cos_mat = np.cos(phase_mat)
        sse_total = 0.0
        for j in range(n_pts):
            y = Y[:, j][:, None]
            sse_b = np.sum((y - cos_mat) ** 2, axis=0)
            sse_total += np.min(sse_b)
        errors[k] = sse_total

    best_idx = int(np.argmin(errors))
    best_a = float(a_trial[best_idx])
    best_a_deg = float(a_trial_deg[best_idx])

    b_best = np.zeros(n_pts)
    base_phase_best = best_a * t
    phase_mat_best = base_phase_best[:, None] + b_grid[None, :]
    cos_mat_best = np.cos(phase_mat_best)
    for j in range(n_pts):
        y = Y[:, j][:, None]
        sse_b = np.sum((y - cos_mat_best) ** 2, axis=0)
        b_best[j] = b_grid[int(np.argmin(sse_b))]

    return best_a_deg, b_best, float(errors[best_idx])


def estimate_direction_from_phase(b_best):
    b_unwrapped = np.unwrap(b_best)
    x_pts = np.arange(len(b_unwrapped))
    slope_rad = np.polyfit(x_pts, b_unwrapped, 1)[0]
    slope_deg = np.degrees(slope_rad)
    direction = -1 if slope_rad > 0 else 1
    return direction, slope_deg


def calculate_phase_speed_from_csv(
    csv_path: str | Path,
    *,
    dt_frame_s: float = 1.1,
    a0_deg: float = 20 / 60,
    f_min: float = 0.25,
    f_max: float = 4.0,
    n_trial: int = 201,
    n_b: int = 720,
) -> Dict[str, Any]:
    t, wavelength_km, wavelength_error_km, angle_deg, angle_error_deg, Y, pt_names = load_roi_csv(csv_path)
    best_a_deg, b_best, sse_min = fit_phase_grid_search(t, Y, a0_deg=a0_deg, f_min=f_min, f_max=f_max, n_trial=n_trial, n_b=n_b)
    direction, phase_slope_deg = estimate_direction_from_phase(b_best)

    avg_wavelength_km, avg_wavelength_error_km = combine_mean_and_measurement_error(wavelength_km, wavelength_error_km)
    phase_speed_mps = 1000.0 * avg_wavelength_km * (best_a_deg / 360.0) / dt_frame_s
    phase_speed_error_mps = 1000.0 * avg_wavelength_error_km * (best_a_deg / 360.0) / dt_frame_s

    if angle_deg is not None:
        avg_angle_deg, avg_angle_error_deg = combine_mean_and_measurement_error(angle_deg, angle_error_deg)
    else:
        avg_angle_deg, avg_angle_error_deg = np.nan, np.nan

    return {
        "best_a_deg_per_frame": float(best_a_deg),
        "avg_wavelength_km": float(avg_wavelength_km),
        "avg_wavelength_error_km": float(avg_wavelength_error_km),
        "avg_angle_deg": float(avg_angle_deg),
        "avg_angle_error_deg": float(avg_angle_error_deg),
        "direction": int(direction),
        "phase_slope_deg_per_point": float(phase_slope_deg),
        "phase_speed_m_per_s": float(direction * phase_speed_mps),
        "phase_speed_error_m_per_s": float(phase_speed_error_mps),
        "sse_min": float(sse_min),
        "n_frames_used": int(Y.shape[0]),
        "n_points": int(Y.shape[1]),
    }



# -----------------------------------------------------------------------------
# Batch-output helpers
# -----------------------------------------------------------------------------
def _safe_path_part(value: Any, default: str = "cluster") -> str:
    """
    Make a string safe to use as a folder/file name.
    """
    s = str(value).strip() if value is not None else default
    if not s:
        s = default
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._")
    return s or default


def _write_dicts_to_csv(rows: Sequence[Dict[str, Any]], csv_path: str | Path) -> Path:
    """
    Write dictionaries to a CSV using the union of keys across all rows.

    This is useful because failed cluster runs may contain error fields while
    successful runs contain phase-speed fields.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        with open(csv_path, "w", newline="") as f:
            f.write("")
        return csv_path

    priority = [
        "cluster_id",
        "cluster_index",
        "status",
        "roi",
        "target_lat",
        "target_lon",
        "target_angle_deg",
        "target_wavelength_km",
        "closest_frame",
        "closest_distance_km",
        "t_start",
        "t_end",
        "n_frames_used",
        "n_points",
        "best_a_deg_per_frame",
        "avg_wavelength_km",
        "avg_wavelength_error_km",
        "avg_angle_deg",
        "avg_angle_error_deg",
        "direction",
        "phase_slope_deg_per_point",
        "phase_speed_m_per_s",
        "phase_speed_error_m_per_s",
        "sse_min",
        "out_dir",
        "csv_path",
        "summary_csv_path",
        "error",
    ]

    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    fieldnames = [k for k in priority if k in all_keys] + [k for k in all_keys if k not in priority]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


# -----------------------------------------------------------------------------
# Main callable function
# -----------------------------------------------------------------------------
def calculate_phase_speed_for_l7a_point(
    l1a_nc_path: str | Path,
    target_lat: float,
    target_lon: float,
    target_angle_deg: float,
    target_wavelength_km: float,
    out_dir: str | Path,
    *,
    cluster_id: Optional[str | int] = None,
    roi_label: int | str = "smart",
    dt_frame_s: float = 1.1,
    n_frames_before_after: int = 12,
    r_plot_km: float = 200.0,
    force_rebuild_frames: bool = True,
    force_rebuild_csv: bool = True,
    save_overlay: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate phase speed using one L7A representative point.

    Parameters
    ----------
    l1a_nc_path:
        L1A/remap NetCDF file with Radiance, Latitude, Longitude.
    target_lat, target_lon:
        L7A selected representative point latitude/longitude.
    target_angle_deg:
        L7A true angle at the selected point. This is used as TARGET_ANGLE.
    target_wavelength_km:
        L7A wavelength at the selected point. This is used as TARGET_WAVELENGTH.
    out_dir:
        Output folder for this cluster. For batch runs, this should normally be
        one unique folder per cluster.
    cluster_id:
        Optional cluster label saved in the result summary.
    roi_label:
        ROI label using your current ROI logic. Default is "smart".
        "smart" first runs ROI 22 on the first frame, finds the peak,
        then switches to 11/12/13/21/22/.../33 if the peak is more
        than 25 px away from the ROI center.

    Returns
    -------
    dict
        Phase speed result fields, including phase_speed_m_per_s.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 78)
        print("Calculating phase speed from L7A-selected point")
        if cluster_id is not None:
            print(f"cluster id           : {cluster_id}")
        print(f"target lat/lon       : {target_lat:.5f}, {target_lon:.5f}")
        print(f"target true angle    : {target_angle_deg:.3f} deg")
        print(f"target wavelength    : {target_wavelength_km:.3f} km")
        print(f"L1A/remap file       : {l1a_nc_path}")
        print(f"output folder        : {out_dir}")
        print("=" * 78)

    frame_dir, meta = save_preinterp_frames_near_target(
        nc_path=l1a_nc_path,
        out_dir=out_dir,
        target_lat=target_lat,
        target_lon=target_lon,
        n_frames_before_after=n_frames_before_after,
        r_plot_km=r_plot_km,
        force_rebuild=force_rebuild_frames,
    )

    csv_path, final_roi_label, smart_roi_info = build_roi_csv(
        frame_dir=frame_dir,
        roi_label=roi_label,
        target_wavelength_km=target_wavelength_km,
        target_angle_deg=target_angle_deg,
        save_overlay=save_overlay,
        force_rebuild=force_rebuild_csv,
        verbose=verbose,
    )

    fit = calculate_phase_speed_from_csv(csv_path, dt_frame_s=dt_frame_s)

    roi_name = f"ROI_{final_roi_label}"
    result = PhaseSpeedResult(
        roi=roi_name,
        target_lat=float(target_lat),
        target_lon=float(target_lon),
        target_angle_deg=float(target_angle_deg),
        target_wavelength_km=float(target_wavelength_km),
        closest_frame=int(meta["closest_frame"]),
        closest_distance_km=float(meta["closest_distance_km"]),
        t_start=int(meta["t_start"]),
        t_end=int(meta["t_end"]),
        csv_path=str(csv_path),
        **fit,
    ).to_dict()

    if smart_roi_info:
        result.update(smart_roi_info)

    if cluster_id is not None:
        result = {"cluster_id": cluster_id, **result}

    result["out_dir"] = str(out_dir)

    # Save a compact one-row summary next to the CSV.
    summary_path = Path(csv_path).parent / "phase_speed_from_l7a_point_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    result["summary_csv_path"] = str(summary_path)

    if verbose:
        print("\nPhase speed result:")
        print(f"  ROI                         : {result['roi']}")
        print(f"  closest frame               : {result['closest_frame']}")
        print(f"  closest center distance     : {result['closest_distance_km']:.2f} km")
        print(f"  avg wavelength              : {result['avg_wavelength_km']:.2f} ± {result['avg_wavelength_error_km']:.2f} km")
        print(f"  avg angle                   : {result['avg_angle_deg']:.2f} ± {result['avg_angle_error_deg']:.2f} deg")
        print(f"  direction                   : {result['direction']}")
        print(f"  phase speed                 : {result['phase_speed_m_per_s']:.2f} ± {result['phase_speed_error_m_per_s']:.2f} m/s")
        print(f"  saved summary               : {summary_path}")

    return result


def calculate_phase_speed_for_multiple_l7a_points(
    l1a_nc_path: str | Path,
    clusters: Sequence[Dict[str, Any]],
    base_out_dir: str | Path,
    *,
    default_roi_label: int | str = "smart",
    dt_frame_s: float = 1.1,
    n_frames_before_after: int = 12,
    r_plot_km: float = 200.0,
    force_rebuild_frames: bool = True,
    force_rebuild_csv: bool = True,
    save_overlay: bool = True,
    continue_on_error: bool = True,
    summary_csv_name: str = "phase_speed_all_clusters_summary.csv",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Calculate phase speed for multiple clusters and save all results.

    Parameters
    ----------
    l1a_nc_path:
        Default L1A/remap NetCDF file. A cluster can override this by including
        "l1a_nc_path" in its dictionary.
    clusters:
        Sequence of dictionaries. Each dictionary must contain:
            target_lat
            target_lon
            target_angle_deg
            target_wavelength_km

        Optional per-cluster keys:
            cluster_id
            roi_label
            l1a_nc_path
            out_dir

        Any extra keys are copied into the output row so you can keep metadata
        like orbit, slice_no, cluster_label, source_file, etc.
    base_out_dir:
        Parent output folder. Each cluster gets its own subfolder unless a
        cluster dictionary provides an explicit "out_dir".
    default_roi_label:
        ROI label used when a cluster does not specify "roi_label".
    continue_on_error:
        If True, failed clusters are recorded in the summary CSV and the batch
        continues. If False, the first error is raised.
    summary_csv_name:
        Name of the combined CSV written inside base_out_dir.

    Returns
    -------
    list of dict
        One result dictionary per cluster. Failed clusters have status="failed".
    """
    base_out_dir = Path(base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for i, cluster in enumerate(clusters):
        cluster = dict(cluster)
        cluster_id = cluster.get("cluster_id", cluster.get("cluster_label", f"cluster_{i + 1:03d}"))
        safe_cluster_id = _safe_path_part(cluster_id, default=f"cluster_{i + 1:03d}")

        cluster_out_dir = Path(cluster.get("out_dir", base_out_dir / safe_cluster_id))
        cluster_l1a_nc_path = cluster.get("l1a_nc_path", l1a_nc_path)
        roi_label = cluster.get("roi_label", default_roi_label)

        if verbose:
            print("\n" + "#" * 78)
            print(f"Batch cluster {i + 1}/{len(clusters)}: {cluster_id}")
            print(f"Output folder: {cluster_out_dir}")
            print("#" * 78)

        metadata = {
            k: v for k, v in cluster.items()
            if k not in {
                "target_lat",
                "target_lon",
                "target_angle_deg",
                "target_wavelength_km",
                "roi_label",
                "l1a_nc_path",
                "out_dir",
            }
        }
        metadata.setdefault("cluster_id", cluster_id)
        metadata["cluster_index"] = i

        try:
            result = calculate_phase_speed_for_l7a_point(
                l1a_nc_path=cluster_l1a_nc_path,
                target_lat=cluster["target_lat"],
                target_lon=cluster["target_lon"],
                target_angle_deg=cluster["target_angle_deg"],
                target_wavelength_km=cluster["target_wavelength_km"],
                out_dir=cluster_out_dir,
                cluster_id=cluster_id,
                roi_label=roi_label,
                dt_frame_s=dt_frame_s,
                n_frames_before_after=n_frames_before_after,
                r_plot_km=r_plot_km,
                force_rebuild_frames=force_rebuild_frames,
                force_rebuild_csv=force_rebuild_csv,
                save_overlay=save_overlay,
                verbose=verbose,
            )
            row = {**metadata, **result, "status": "ok"}
        except Exception as exc:
            if not continue_on_error:
                raise
            row = {
                **metadata,
                "status": "failed",
                "roi": f"ROI_{roi_label}",
                "target_lat": cluster.get("target_lat", np.nan),
                "target_lon": cluster.get("target_lon", np.nan),
                "target_angle_deg": cluster.get("target_angle_deg", np.nan),
                "target_wavelength_km": cluster.get("target_wavelength_km", np.nan),
                "out_dir": str(cluster_out_dir),
                "error": repr(exc),
            }
            if verbose:
                print(f"FAILED: {cluster_id}")
                print(f"  {repr(exc)}")

        results.append(row)

        # Update combined summary after each cluster, so partial results are saved
        # even if a later cluster fails or the run is interrupted.
        _write_dicts_to_csv(results, base_out_dir / summary_csv_name)

    summary_csv_path = base_out_dir / summary_csv_name
    if verbose:
        print("\n" + "=" * 78)
        print(f"Batch complete. Saved combined summary: {summary_csv_path}")
        print("=" * 78)

    return results


if __name__ == "__main__":
    raise SystemExit(
        "Import this file and call calculate_phase_speed_for_l7a_point(...).\n"
        "See the example in the module docstring."
    )
