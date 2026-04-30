# %%
"""
Batch-run the satellite phase-speed notebook for remap83 through remap87.

What this script does:
1. Uses your original Jupyter notebook as a template.
2. For each remap number, remap83 to remap87, it makes a temporary notebook copy with:
      nc_path = ..._remapXX.nc
      out_dir = AWE/10369/remapXX
3. Executes that temporary notebook.
4. Finds each remap's individual overall_phase_speed_summary.csv.
5. Combines all successful remaps into one big CSV and Excel table.

Important fix in this version:
- Windows paths like Z:\socfiles\... are inserted using a safe regex replacement.
  This avoids the "re.error: bad escape \\s" problem.

Run from the folder that contains your notebook, for example:
    python run_satellite_notebook_remap83_to_87_FIXED.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import traceback

import nbformat
import pandas as pd
from nbclient import NotebookClient


# ============================================================
# User settings
# ============================================================

# Folder where this script is located.
# All relative paths below are interpreted relative to this folder.
SCRIPT_DIR = Path(__file__).resolve().parent

# Notebook template.
# The script will try these names in order and use the first one it finds.
NOTEBOOK_TEMPLATE_CANDIDATES = [
    SCRIPT_DIR / "Satellite Based for 04714 v23 April 27 2026.ipynb",
]

# Remap range to process.
REMAP_START = 83
REMAP_END = 87

# Input NetCDF pattern.
# The {remap} part is replaced with 83, 84, ..., 87.
NC_PATH_PATTERN = r"Z:\socfiles\l1a\remap_alt\awe_l1a_q20_2024265T0540_04714_v23_remap{remap}.nc"

# Output base folder.
OUT_BASE = SCRIPT_DIR / "AWE" / "04714"

# Folder to store the temporary executed notebooks.
EXECUTED_NOTEBOOK_DIR = OUT_BASE / "executed_notebooks"

# Continue to the next remap if one remap fails.
CONTINUE_ON_ERROR = True

# Kernel name.
# If python3 does not work, change this to your environment kernel name, for example:
#     KERNEL_NAME = "juwavelet_env"
KERNEL_NAME = "python3"

# Timeout per notebook cell, in seconds.
# None means no timeout.
CELL_TIMEOUT = None


# ============================================================
# Helper functions
# ============================================================

def choose_notebook_template() -> Path:
    """Use the first notebook template that exists."""
    for p in NOTEBOOK_TEMPLATE_CANDIDATES:
        if p.exists():
            return p

    msg = "Could not find notebook template. Tried:\n"
    msg += "\n".join(f"  {p}" for p in NOTEBOOK_TEMPLATE_CANDIDATES)
    raise FileNotFoundError(msg)


def safe_re_sub(pattern: str, replacement: str, source: str, flags: int = 0) -> str:
    """
    Safe regex replacement.

    Using lambda makes Python insert replacement literally.
    This is important for Windows paths like Z:\socfiles\..., because normal
    re.sub replacement strings treat backslashes as escape sequences.
    """
    return re.sub(pattern, lambda match: replacement, source, flags=flags)


def patch_notebook_source(source: str, remap: int) -> str:
    """
    Patch hardcoded nc_path and out_dir lines inside a notebook code cell.

    This keeps your notebook pipeline the same, but changes the input/output
    for each remap run.
    """
    remap_name = f"remap{remap}"
    nc_path = NC_PATH_PATTERN.format(remap=remap)
    out_dir = OUT_BASE / remap_name

    # Replacement lines to insert into notebook cells.
    # Use r"..." for Windows paths.
    new_nc_line = f'nc_path = r"{nc_path}"'

    # Use a forward-slash path for Path(...). Windows can read this fine.
    # This avoids accidental backslash escapes in the notebook itself.
    new_out_dir_line = f'out_dir = Path(r"{out_dir.as_posix()}")'

    # Replace a full line like:
    #     nc_path = r"..."
    #     nc_path = "..."
    source = safe_re_sub(
        pattern=r'^\s*nc_path\s*=\s*r?["\'].*?["\']\s*$',
        replacement=new_nc_line,
        source=source,
        flags=re.MULTILINE,
    )

    # Replace a full line like:
    #     out_dir = Path("...")
    #     out_dir = Path(r"...")
    source = safe_re_sub(
        pattern=r'^\s*out_dir\s*=\s*Path\(\s*r?["\'].*?["\']\s*\)\s*$',
        replacement=new_out_dir_line,
        source=source,
        flags=re.MULTILINE,
    )

    return source


def make_patched_notebook(template_path: Path, remap: int) -> Path:
    """Create a patched temporary notebook for one remap."""
    EXECUTED_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(template_path, as_version=4)

    found_nc_path = False
    found_out_dir = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        old_source = cell.source
        new_source = patch_notebook_source(old_source, remap)
        cell.source = new_source

        if old_source != new_source:
            if re.search(r'^\s*nc_path\s*=', old_source, flags=re.MULTILINE):
                found_nc_path = True
            if re.search(r'^\s*out_dir\s*=', old_source, flags=re.MULTILINE):
                found_out_dir = True

    if not found_nc_path:
        print("WARNING: Did not find a line starting with nc_path = in the notebook.")
    if not found_out_dir:
        print("WARNING: Did not find a line starting with out_dir = in the notebook.")

    patched_path = EXECUTED_NOTEBOOK_DIR / f"Satellite_Based_10369_v23_remap{remap}_executed.ipynb"
    nbformat.write(nb, patched_path)

    return patched_path


def execute_notebook(notebook_path: Path) -> None:
    """Execute one notebook in-place."""
    nb = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name=KERNEL_NAME,
        allow_errors=False,
    )

    # Execute from SCRIPT_DIR, not from executed_notebooks.
    # This keeps relative paths behaving like they did when you ran the original notebook.
    client.execute(cwd=str(SCRIPT_DIR))

    nbformat.write(nb, notebook_path)


def find_overall_csv(remap: int) -> Path:
    """
    Find the individual overall table generated by the phase-speed notebook.

    Expected somewhere under:
        AWE/10369/remapXX/
    """
    remap_dir = OUT_BASE / f"remap{remap}"

    matches = sorted(remap_dir.rglob("overall_phase_speed_summary.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No overall_phase_speed_summary.csv found under:\n  {remap_dir}"
        )

    # If there are multiple, use the newest one.
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def combine_overall_tables(successful_remaps: list[int]) -> tuple[Path, Path]:
    """Read all individual overall CSV files and combine them into one big table."""
    dfs = []

    for remap in successful_remaps:
        csv_path = find_overall_csv(remap)
        df = pd.read_csv(csv_path)

        # Add tracking columns at the front.
        df.insert(0, "remap", f"remap{remap}")
        df.insert(1, "source_csv", str(csv_path))

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No successful remap tables to combine.")

    big_df = pd.concat(dfs, ignore_index=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    big_csv = OUT_BASE / f"overall_phase_speed_summary_remap{REMAP_START}_to_remap{REMAP_END}.csv"
    big_xlsx = OUT_BASE / f"overall_phase_speed_summary_remap{REMAP_START}_to_remap{REMAP_END}.xlsx"

    big_df.to_csv(big_csv, index=False)
    big_df.to_excel(big_xlsx, index=False)

    print("\nSaved big overall tables:")
    print(f"  CSV : {big_csv}")
    print(f"  XLSX: {big_xlsx}")

    return big_csv, big_xlsx


# ============================================================
# Main batch run
# ============================================================

def main() -> None:
    template_path = choose_notebook_template()
    remaps = list(range(REMAP_START, REMAP_END + 1))

    successful_remaps: list[int] = []
    failed_remaps: list[int] = []

    print("Batch run settings")
    print("==================")
    print(f"Script folder     : {SCRIPT_DIR}")
    print(f"Template notebook : {template_path}")
    print(f"Remaps            : {remaps}")
    print(f"Output base       : {OUT_BASE}")
    print(f"Kernel            : {KERNEL_NAME}")
    print()

    for remap in remaps:
        print("\n" + "=" * 70)
        print(f"Running remap{remap}")
        print("=" * 70)

        try:
            patched_notebook = make_patched_notebook(template_path, remap)
            print(f"Temporary notebook: {patched_notebook}")

            execute_notebook(patched_notebook)
            print(f"Finished notebook: {patched_notebook}")

            csv_path = find_overall_csv(remap)
            print(f"Found individual overall table: {csv_path}")

            successful_remaps.append(remap)

        except Exception as exc:
            failed_remaps.append(remap)
            print(f"\nFAILED remap{remap}: {exc}")
            traceback.print_exc()

            if not CONTINUE_ON_ERROR:
                raise

    print("\n" + "=" * 70)
    print("Batch run finished")
    print("=" * 70)
    print(f"Successful remaps: {successful_remaps}")
    print(f"Failed remaps    : {failed_remaps}")

    if successful_remaps:
        combine_overall_tables(successful_remaps)
    else:
        print("No successful remaps, so no big table was saved.")


if __name__ == "__main__":
    main()
