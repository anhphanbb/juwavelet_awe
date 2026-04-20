# %%
"""
Parallel DOY vs latitude plots for MFz_top and MFm_top from L7A outputs
for all days in 2024.

Features:
1. Group files by DOY
2. Process one DOY per worker
3. Copy each file to a temporary local folder before opening
4. Apply latitude mask
5. Optionally sample points after masking
6. Accumulate latitude-bin sums/counts for that DOY
7. Save one incremental file per DOY
8. Resume by skipping already-saved DOYs
9. Build final yearly grids from saved per-DOY files
10. Make plots and save the final combined arrays
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import re
import gc
import shutil
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# -------------------------
# User params
# -------------------------
YEAR = 2024
BASE_L7_DIR = Path("/Volumes/aweread/socfiles/public/l7")
FILE_GLOB = "*l7a*.nc"
TIME_INDEX = 0

LAT_NAME = "Lat_top"
COMPONENTS = ("MFz_top", "MFm_top")

DISPLAY_NAMES = {
    "MFz_top": "Zonal Momentum Flux",
    "MFm_top": "Meridional Momentum Flux",
}

OUTDIR = Path(f"outputs_MF_doy_latitude/{YEAR}")
OUTDIR.mkdir(parents=True, exist_ok=True)

INCREMENTAL_DIR = OUTDIR / "incremental_doy_rows"
INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)

# Latitude binning
LAT_MIN = -60.0
LAT_MAX = 60.0
LAT_BIN_DEG = 1.0

# Plot scaling
USE_SYMMETRIC_PERCENTILE = True
PCTL = 99.0

# Parallel
MAX_WORKERS = 2   # start small on macOS

# Sampling
SAMPLE_EVERY = 4   # 1 = no sampling, 4 = keep about 1/4 of valid points

# Resume behavior
SKIP_EXISTING_DOYS = True

# Filename DOY pattern like: awe_l7a_tmp_2024005T0917_00688_v01.nc
DOY_RE = re.compile(rf"{YEAR}(\d{{3}})T\d{{4}}")


# -------------------------
# Helpers
# -------------------------
def extract_doy_from_name(path: Path) -> int | None:
    m = DOY_RE.search(path.name)
    if m is None:
        return None
    return int(m.group(1))


def get_var_2d(nc: Dataset, var_name: str, time_index: int = 0) -> np.ndarray | None:
    if var_name not in nc.variables:
        return None

    var = nc.variables[var_name]

    try:
        var.set_auto_mask(False)
    except Exception:
        pass

    if var.ndim == 3:
        if time_index >= var.shape[0]:
            return None
        arr = np.array(var[time_index, :, :], dtype=np.float64, copy=True)
    elif var.ndim == 2:
        arr = np.array(var[:, :], dtype=np.float64, copy=True)
    else:
        return None

    return arr


def robust_symmetric_limit(arr: np.ndarray, percentile: float = 99.0) -> float:
    good = np.isfinite(arr)
    if not np.any(good):
        return 1.0
    vmax = np.nanpercentile(np.abs(arr[good]), percentile)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(np.abs(arr[good]))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return float(vmax)


def copy_to_temp(src_path: Path, temp_dir: Path) -> Path:
    local_path = temp_dir / src_path.name
    shutil.copy2(src_path, local_path)
    return local_path


def doy_save_path(doy: int) -> Path:
    return INCREMENTAL_DIR / f"{YEAR}_DOY{doy:03d}_rows.npz"


def save_doy_result(
    doy: int,
    lat_edges: np.ndarray,
    lat_centers: np.ndarray,
    row_dict: dict[str, np.ndarray],
    n_ok: int,
    n_total: int,
) -> Path:
    outpath = doy_save_path(doy)
    np.savez_compressed(
        outpath,
        year=YEAR,
        doy=doy,
        lat_edges=lat_edges,
        lat_centers=lat_centers,
        MFz_top=row_dict["MFz_top"],
        MFm_top=row_dict["MFm_top"],
        n_ok=n_ok,
        n_total=n_total,
    )
    return outpath


def load_doy_result(doy: int) -> dict | None:
    path = doy_save_path(doy)
    if not path.exists():
        return None

    with np.load(path) as data:
        out = {
            "doy": int(data["doy"]),
            "lat_edges": data["lat_edges"],
            "lat_centers": data["lat_centers"],
            "MFz_top": data["MFz_top"],
            "MFm_top": data["MFm_top"],
            "n_ok": int(data["n_ok"]),
            "n_total": int(data["n_total"]),
        }
    return out


def process_one_doy(args):
    """
    Worker for one DOY.

    Returns
    -------
    doy : int
    row_dict : dict[str, np.ndarray]
        row_dict[comp] has shape (nlat,) with NaN where no data
    n_ok : int
        number of files successfully used
    n_total : int
        total files for this DOY
    """
    (
        doy,
        file_list,
        lat_edges,
        nlat,
        year,
        time_index,
        lat_name,
        components,
        lat_min,
        lat_max,
        sample_every,
    ) = args

    doy_sum = {comp: np.zeros(nlat, dtype=np.float64) for comp in components}
    doy_cnt = {comp: np.zeros(nlat, dtype=np.int64) for comp in components}

    n_ok = 0
    n_total = len(file_list)

    with tempfile.TemporaryDirectory(prefix=f"l7a_{year}_{doy:03d}_") as tmpdir:
        tmpdir = Path(tmpdir)

        for p_str in file_list:
            p = Path(p_str)
            lat = None
            arr = None

            try:
                local_p = copy_to_temp(p, tmpdir)

                with Dataset(local_p, "r") as nc:
                    try:
                        nc.set_auto_mask(False)
                    except Exception:
                        pass
                    try:
                        nc.set_auto_maskandscale(False)
                    except Exception:
                        pass

                    lat = get_var_2d(nc, lat_name, time_index)
                    if lat is None:
                        continue

                    lat_flat = lat.ravel()
                    base = np.isfinite(lat_flat) & (lat_flat >= lat_min) & (lat_flat <= lat_max)
                    if not np.any(base):
                        continue

                    lat_use = lat_flat[base]
                    lat_idx = np.digitize(lat_use, lat_edges) - 1
                    good_lat = (lat_idx >= 0) & (lat_idx < nlat)
                    if not np.any(good_lat):
                        continue

                    lat_idx = lat_idx[good_lat]

                    if sample_every > 1:
                        sel = (np.arange(lat_idx.size) % sample_every) == 0
                        lat_idx = lat_idx[sel]
                    else:
                        sel = slice(None)

                    if lat_idx.size == 0:
                        continue

                    for comp in components:
                        arr = get_var_2d(nc, comp, time_index)
                        if arr is None:
                            continue

                        val = arr.ravel()[base][good_lat]
                        val = val[sel]

                        good = np.isfinite(val)
                        if not np.any(good):
                            continue

                        np.add.at(doy_sum[comp], lat_idx[good], val[good])
                        np.add.at(doy_cnt[comp], lat_idx[good], 1)

                n_ok += 1

            except Exception as e:
                print(f"  Worker skip {p.name}: {e}")

            finally:
                del lat, arr
                gc.collect()

    row_dict = {}
    for comp in components:
        row = np.full(nlat, np.nan, dtype=np.float32)
        mask = doy_cnt[comp] > 0
        if np.any(mask):
            row[mask] = (doy_sum[comp][mask] / doy_cnt[comp][mask]).astype(np.float32)
        row_dict[comp] = row

    return doy, row_dict, n_ok, n_total


def make_doy_lat_plot(
    grid: np.ndarray,
    comp: str,
    outdir: Path,
    year: int,
    doy_edges: np.ndarray,
    lat_edges: np.ndarray,
    ndoy: int,
    lat_min: float,
    lat_max: float,
):
    fig, ax = plt.subplots(figsize=(16, 6))

    if USE_SYMMETRIC_PERCENTILE:
        vmax = robust_symmetric_limit(grid, PCTL)
        vmin = -vmax
    else:
        good = np.isfinite(grid)
        if np.any(good):
            vmin = float(np.nanmin(grid))
            vmax = float(np.nanmax(grid))
            if vmin == vmax:
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0

    im = ax.pcolormesh(
        doy_edges,
        lat_edges,
        grid.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"{DISPLAY_NAMES.get(comp, comp)} in {year}")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim(1, ndoy + 1)
    ax.set_ylim(lat_min, lat_max)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"Momentum Flux (m$^2$ s$^{-2}$)")

    outpath = outdir / f"{year}_{comp}_DOY_vs_latitude.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Saved: {outpath.resolve()}")


def main():
    # -------------------------
    # Build file list grouped by DOY
    # -------------------------
    files_by_doy = defaultdict(list)

    for month in range(1, 13):
        month_dir = BASE_L7_DIR / str(YEAR) / f"{month:02d}"
        if not month_dir.exists():
            print(f"Skip missing folder: {month_dir}")
            continue

        for p in sorted(month_dir.glob(FILE_GLOB)):
            doy = extract_doy_from_name(p)
            if doy is None:
                print(f"Could not parse DOY from filename, skip: {p.name}")
                continue
            files_by_doy[doy].append(p)

    all_doys = sorted(files_by_doy.keys())

    if len(all_doys) == 0:
        raise FileNotFoundError(f"No L7A files found for year {YEAR} in {BASE_L7_DIR}")

    print(f"Found {sum(len(v) for v in files_by_doy.values())} files across {len(all_doys)} DOYs")

    # -------------------------
    # Final grids: DOY x latitude
    # -------------------------
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN_DEG, LAT_BIN_DEG)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    nlat = len(lat_centers)

    ndoy = 366 if ((YEAR % 4 == 0 and YEAR % 100 != 0) or (YEAR % 400 == 0)) else 365
    doy_axis = np.arange(1, ndoy + 1)
    doy_edges = np.arange(1, ndoy + 2)

    final_grids = {
        comp: np.full((ndoy, nlat), np.nan, dtype=np.float32)
        for comp in COMPONENTS
    }

    # -------------------------
    # Build tasks only for DOYs not already saved
    # -------------------------
    tasks = []
    skipped_existing = 0

    for doy in all_doys:
        already_done = doy_save_path(doy).exists()
        if SKIP_EXISTING_DOYS and already_done:
            skipped_existing += 1
            continue

        file_list = [str(p) for p in files_by_doy[doy]]
        tasks.append(
            (
                doy,
                file_list,
                lat_edges,
                nlat,
                YEAR,
                TIME_INDEX,
                LAT_NAME,
                COMPONENTS,
                LAT_MIN,
                LAT_MAX,
                SAMPLE_EVERY,
            )
        )

    print(f"Existing saved DOYs skipped: {skipped_existing}")
    print(f"DOYs to process now: {len(tasks)}")

    # -------------------------
    # Run in parallel and save incrementally
    # -------------------------
    if len(tasks) > 0:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
            futures = [ex.submit(process_one_doy, task) for task in tasks]

            for i, fut in enumerate(as_completed(futures), start=1):
                doy, row_dict, n_ok, n_total = fut.result()

                save_path = save_doy_result(
                    doy=doy,
                    lat_edges=lat_edges,
                    lat_centers=lat_centers,
                    row_dict=row_dict,
                    n_ok=n_ok,
                    n_total=n_total,
                )

                print(f"[{i}/{len(futures)}] DOY {doy:03d}: used {n_ok}/{n_total} files -> saved {save_path.name}")

    # -------------------------
    # Rebuild final grids from saved per-DOY files
    # -------------------------
    loaded_count = 0
    missing_count = 0

    for doy in range(1, ndoy + 1):
        rec = load_doy_result(doy)
        if rec is None:
            missing_count += 1
            continue

        doy_row = doy - 1
        final_grids["MFz_top"][doy_row, :] = rec["MFz_top"]
        final_grids["MFm_top"][doy_row, :] = rec["MFm_top"]
        loaded_count += 1

    print(f"Loaded saved DOYs into final grids: {loaded_count}")
    if missing_count > 0:
        print(f"Missing DOYs with no saved row file: {missing_count}")

    # -------------------------
    # Save final combined arrays BEFORE plotting
    # -------------------------
    final_npz = OUTDIR / f"{YEAR}_MFz_MFm_DOY_latitude_grids.npz"
    np.savez_compressed(
        final_npz,
        doy_axis=doy_axis,
        doy_edges=doy_edges,
        lat_edges=lat_edges,
        lat_centers=lat_centers,
        MFz_top=final_grids["MFz_top"],
        MFm_top=final_grids["MFm_top"],
    )
    print(f"Saved combined yearly arrays: {final_npz.resolve()}")

    # -------------------------
    # Make plots
    # -------------------------
    make_doy_lat_plot(
        final_grids["MFz_top"],
        "MFz_top",
        OUTDIR,
        YEAR,
        doy_edges,
        lat_edges,
        ndoy,
        LAT_MIN,
        LAT_MAX,
    )

    make_doy_lat_plot(
        final_grids["MFm_top"],
        "MFm_top",
        OUTDIR,
        YEAR,
        doy_edges,
        lat_edges,
        ndoy,
        LAT_MIN,
        LAT_MAX,
    )

    print("\nDone.")
    print(f"Outputs saved in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()