#!/usr/bin/env python3
"""
save_clusters_recon_and_temp_ranked.py

Read all L7A NetCDF files. For every cluster whose max(Amplitude) > threshold:
- Save a 2-panel PNG:
    Left: cluster reconstruction (ClusterReconstruction if present, else Amplitude)
    Right: corresponding Temperature slice tile (from T_top if present, else Temperature)
- Annotate with:
    max(A) [K], median wavelength [km], median angle (0..180 symmetry) [deg]
- RANK globally by max(A) and put the rank in the filename + write a CSV summary.

All images go into ONE shared output folder.

Example:
  python save_clusters_recon_and_temp_ranked.py --l7 l7 --out outputs_CLUSTER_RECON_TEMP --amp-thr 2.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List, Any
import argparse
import os
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset


# -------------------------
# Helpers
# -------------------------
def robust_limits(A: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> Tuple[float, float]:
    a = np.asarray(A, dtype=float)
    good = np.isfinite(a)
    if not np.any(good):
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(a[good], [p_lo, p_hi])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
        lo = float(np.nanmin(a[good]))
        hi = float(np.nanmax(a[good]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return (-1.0, 1.0)
    return float(lo), float(hi)


def nanpercentile_safe(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float)
    good = np.isfinite(a)
    if not np.any(good):
        return np.nan
    return float(np.nanpercentile(a[good], q))


def circ_median_deg_180(theta_rad: np.ndarray, mask: np.ndarray) -> float:
    th = np.asarray(theta_rad, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(th)
    if not np.any(m):
        return np.nan
    ang = th[m]
    # 180-degree symmetry => double angles, then halve the mean angle
    z = np.exp(1j * 2.0 * ang)
    zmean = np.mean(z)
    if (not np.isfinite(zmean.real)) or (not np.isfinite(zmean.imag)) or abs(zmean) == 0:
        return np.nan
    th_mean = 0.5 * np.angle(zmean)
    deg = (np.degrees(th_mean) + 180.0) % 180.0
    return float(deg)


def save_fig(fig: plt.Figure, outpath: Path, dpi: int) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def key_from_name(name: str) -> str:
    # awe_l7a_tmp_2024165T2120_03175_v01.nc -> 2024165T2120_03175_v01
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 5 and parts[1] == "l7a":
        return "_".join(parts[3:])
    return stem


def cluster_stats(A: np.ndarray, L: np.ndarray, Ang: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(A)
    if not np.any(mask):
        return dict(amp_max_k=np.nan, lam_med_km=np.nan, ang_med_deg_180=np.nan)
    amp_max = float(np.nanmax(A[mask])) if np.any(np.isfinite(A[mask])) else np.nan
    lam_med = nanpercentile_safe(L[mask], 50)
    ang_med = circ_median_deg_180(Ang, mask)
    return dict(amp_max_k=float(amp_max), lam_med_km=float(lam_med), ang_med_deg_180=float(ang_med))


def match_width(A: np.ndarray, target_nx: int) -> np.ndarray:
    """
    Ensure A has width target_nx by trimming or padding with NaNs.
    """
    A = np.asarray(A, dtype=float)
    ny, nx = A.shape
    if nx == target_nx:
        return A
    if nx > target_nx:
        return A[:, :target_nx]
    # pad on the right
    pad = target_nx - nx
    return np.pad(A, ((0, 0), (0, pad)), mode="constant", constant_values=np.nan)


def plot_cluster_and_temp(
    recon: np.ndarray,
    temp_tile: np.ndarray,
    outpng: Path,
    title: str,
    subtitle: str,
    dx_km: float,
    dy_km: float,
    dpi: int,
) -> None:
    R = np.asarray(recon, dtype=float)
    T = np.asarray(temp_tile, dtype=float)

    # Use imshow with extent => no pcolormesh shape headaches.
    rmin, rmax = robust_limits(R, 2, 98)
    tmin, tmax = robust_limits(T, 2, 98)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)

    # Left: recon
    nyR, nxR = R.shape
    extR = [0.0, nxR * dx_km, 0.0, nyR * dy_km]
    im0 = axs[0].imshow(
        R,
        origin="lower",
        extent=extR,
        aspect="equal",
        vmin=rmin,
        vmax=rmax,
        cmap="RdBu_r",
        interpolation="nearest",
    )
    axs[0].set_title("Cluster recon")
    axs[0].set_xlabel("x (km)")
    axs[0].set_ylabel("y (km)")
    fig.colorbar(im0, ax=axs[0], fraction=0.045, pad=0.02)

    # Right: temperature tile
    nyT, nxT = T.shape
    extT = [0.0, nxT * dx_km, 0.0, nyT * dy_km]
    im1 = axs[1].imshow(
        T,
        origin="lower",
        extent=extT,
        aspect="equal",
        vmin=tmin,
        vmax=tmax,
        interpolation="nearest",
    )
    axs[1].set_title("Temperature slice tile")
    axs[1].set_xlabel("x (km)")
    fig.colorbar(im1, ax=axs[1], fraction=0.045, pad=0.02).set_label("K")

    fig.suptitle(title, fontsize=12)
    fig.text(
        0.01,
        0.99,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=3),
    )

    save_fig(fig, outpng, dpi=dpi)


# -------------------------
# Collection pass (fast-ish)
# -------------------------
def collect_candidates(files: List[Path], time_index: int, amp_thr: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for nc_path in files:
        file_key = key_from_name(nc_path.name)
        try:
            with Dataset(nc_path, "r") as nc:
                needed = ["Amplitude", "DominantWavelength", "Angle", "SlicesNo"]
                if any(v not in nc.variables for v in needed):
                    continue

                amp = np.asarray(nc.variables["Amplitude"][:], dtype=float)
                lam = np.asarray(nc.variables["DominantWavelength"][:], dtype=float)
                ang = np.asarray(nc.variables["Angle"][:], dtype=float)
                slices_no = np.asarray(nc.variables["SlicesNo"][:], dtype=int)

                if amp.ndim != 3 or amp.shape != lam.shape or amp.shape != ang.shape:
                    continue
                if slices_no.ndim != 1 or slices_no.shape[0] != amp.shape[0]:
                    continue

                amp_max = np.nanmax(amp, axis=(1, 2))
                keep = np.isfinite(amp_max) & (amp_max > float(amp_thr))
                if not np.any(keep):
                    continue

                for cidx in np.where(keep)[0]:
                    cidx = int(cidx)
                    s = int(slices_no[cidx])
                    if s < 0:
                        continue

                    st = cluster_stats(amp[cidx], lam[cidx], ang[cidx])
                    if (not np.isfinite(st["amp_max_k"])) or (st["amp_max_k"] <= float(amp_thr)):
                        continue

                    out.append(
                        dict(
                            nc_path=str(nc_path),
                            file_key=file_key,
                            cluster_idx=cidx,
                            slice_no=s,
                            amp_max_k=st["amp_max_k"],
                            lam_med_km=st["lam_med_km"],
                            ang_med_deg_180=st["ang_med_deg_180"],
                        )
                    )
        except Exception:
            continue
    return out


# -------------------------
# Plot pass (ranked order)
# -------------------------
def render_ranked(
    items: List[Dict[str, Any]],
    out_dir: Path,
    time_index: int,
    dx_km: float,
    dy_km: float,
    dpi: int,
    overwrite: bool,
) -> int:
    saved = 0
    for i, it in enumerate(items, start=1):
        nc_path = Path(it["nc_path"])
        file_key = str(it["file_key"])
        cidx = int(it["cluster_idx"])
        s = int(it["slice_no"])

        outpng = out_dir / f"{i:05d}_Amax_{it['amp_max_k']:.2f}K_{file_key}_slice_{s:03d}_cluster_{cidx:05d}.png"
        if outpng.exists() and (not overwrite):
            continue

        try:
            with Dataset(nc_path, "r") as nc:
                amp = np.asarray(nc.variables["Amplitude"][cidx], dtype=float)
                y_tile, x_tile = amp.shape

                # Temperature top rows
                if "T_top" in nc.variables:
                    T_top = np.asarray(nc.variables["T_top"][time_index, 0:y_tile, :], dtype=float)
                else:
                    T_raw = np.asarray(nc.variables["Temperature"][time_index], dtype=float)
                    T_top = T_raw[0:y_tile, :]

                x_full = T_top.shape[1]

                # Slice indexing
                x0 = s * x_tile
                x1 = min(x0 + x_tile, x_full)
                if x0 < 0 or x0 >= x_full or x1 <= x0:
                    continue

                temp_tile = T_top[:, x0:x1]
                target_nx = temp_tile.shape[1]

                # Recon (or fallback)
                if "ClusterReconstruction" in nc.variables:
                    recon = np.asarray(nc.variables["ClusterReconstruction"][cidx], dtype=float)
                else:
                    recon = amp  # fallback

                # IMPORTANT: make recon width match temp width (last slice etc.)
                recon = match_width(recon, target_nx)

        except Exception:
            continue

        title = f"#{i} | {file_key} | slice {s} | cluster {cidx}"
        subtitle = (
            f"max(A)={it['amp_max_k']:.2f} K   "
            f"λ_med={it['lam_med_km']:.1f} km   "
            f"θ_med(0..180)={it['ang_med_deg_180']:.1f}°"
        )

        plot_cluster_and_temp(
            recon=recon,
            temp_tile=temp_tile,
            outpng=outpng,
            title=title,
            subtitle=subtitle,
            dx_km=dx_km,
            dy_km=dy_km,
            dpi=dpi,
        )
        saved += 1

    return saved


def write_csv(items: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "amp_max_k", "lam_med_km", "ang_med_deg_180", "file_key", "slice_no", "cluster_idx", "nc_path"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, it in enumerate(items, start=1):
            row = dict(
                rank=r,
                amp_max_k=it["amp_max_k"],
                lam_med_km=it["lam_med_km"],
                ang_med_deg_180=it["ang_med_deg_180"],
                file_key=it["file_key"],
                slice_no=it["slice_no"],
                cluster_idx=it["cluster_idx"],
                nc_path=it["nc_path"],
            )
            w.writerow(row)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank clusters by max amplitude and save (recon + temp) panels.")
    p.add_argument("--l7", type=str, default="l7", help="Folder containing L7A NetCDF files.")
    p.add_argument("--glob", type=str, default="*l7a*.nc", help="File glob to match L7A files.")
    p.add_argument("--out", type=str, default="outputs_CLUSTER_RECON_TEMP", help="One shared output folder for PNGs.")
    p.add_argument("--time-index", type=int, default=0, help="Time index to use (usually 0).")
    p.add_argument("--amp-thr", type=float, default=3.0, help="Keep clusters if max(Amplitude) > this (K).")
    p.add_argument("--dx-km", type=float, default=2.0, help="x pixel spacing for plot axis (km).")
    p.add_argument("--dy-km", type=float, default=2.0, help="y pixel spacing for plot axis (km).")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    p.add_argument("--max-plots", type=int, default=0, help="If >0, only plot top N clusters by amplitude.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    l7_dir = Path(args.l7)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(l7_dir.glob(args.glob))
    print(f"Found {len(files)} files in {l7_dir.resolve()} matching {args.glob}")
    if not files:
        return 2

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("Collecting candidates...")
    items = collect_candidates(files=files, time_index=args.time_index, amp_thr=args.amp_thr)
    if not items:
        print("No clusters found above threshold.")
        return 0

    # Sort descending by amplitude
    items.sort(key=lambda d: float(d.get("amp_max_k", np.nan)), reverse=True)

    if args.max_plots and args.max_plots > 0:
        items = items[: int(args.max_plots)]

    # Write summary CSV
    out_csv = out_dir / "ranked_clusters.csv"
    write_csv(items, out_csv)

    print(f"Found {len(items)} clusters above threshold. Rendering in ranked order...")
    n_saved = render_ranked(
        items=items,
        out_dir=out_dir,
        time_index=args.time_index,
        dx_km=args.dx_km,
        dy_km=args.dy_km,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )

    print(f"\nDone. Saved PNGs: {n_saved}")
    print(f"Output folder: {out_dir.resolve()}")
    print(f"CSV summary: {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())