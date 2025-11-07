#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_summary_plot.py

Reads all CSVs named like:
    threshold_sweep_space{S}_max{M}.csv
Extracts Accuracy and Recall at Threshold = 0.50.
Builds 2 heatmaps (Accuracy, Recall):
    y-axis = space
    x-axis = max
Saves:
    accuracy_heatmap_threshold0p50.png
    recall_heatmap_threshold0p50.png
Also prints a summary table and the best combos.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============= user settings =============

RESULTS_DIR = r"."   # folder with your threshold_sweep_space*_max*.csv
TARGET_THRESHOLD = 0.40

FNAME_RE = re.compile(
    r"threshold_sweep_space(\d+)_max(\d+)\.csv",
    re.IGNORECASE
)

# ============= step 1: collect data =============

files = glob.glob(os.path.join(RESULTS_DIR, "threshold_sweep_space*_max*.csv"))

records = []

for fpath in files:
    base = os.path.basename(fpath)
    m = FNAME_RE.match(base)
    if not m:
        # filename doesn't match pattern, skip
        continue

    space_val = int(m.group(1))
    max_val = int(m.group(2))

    # read the CSV
    df = pd.read_csv(fpath)

    # We expect columns:
    # Threshold, True_Negatives, False_Positives, False_Negatives,
    # True_Positives, Accuracy, Recall

    if "Threshold" not in df.columns:
        print(f"[WARN] 'Threshold' column not found in {base}, skipping.")
        continue
    if "Accuracy" not in df.columns:
        print(f"[WARN] 'Accuracy' column not found in {base}, skipping.")
        continue
    if "Recall" not in df.columns:
        print(f"[WARN] 'Recall' column not found in {base}, skipping.")
        continue

    # get the row with Threshold == 0.50 (allow tiny float tolerance)
    row = df.loc[(df["Threshold"] - TARGET_THRESHOLD).abs() < 1e-9]
    if row.empty:
        # fallback exact match
        row = df.loc[df["Threshold"] == TARGET_THRESHOLD]

    if row.empty:
        print(f"[WARN] No Threshold {TARGET_THRESHOLD} in {base}, skipping.")
        continue

    row = row.iloc[0]

    acc = float(row["Accuracy"])
    rec = float(row["Recall"])

    records.append({
        "space": space_val,
        "max": max_val,
        "Accuracy": acc,
        "Recall": rec,
    })

summary_df = pd.DataFrame(records)
if summary_df.empty:
    raise RuntimeError("No data collected. Check folder/filenames/columns.")

print(f"\nSummary (Threshold = {TARGET_THRESHOLD}):")
print(summary_df.sort_values(["space", "max"]).to_string(index=False))

# ============= step 2: build grids =============

all_spaces = sorted(summary_df["space"].unique())
all_maxes  = sorted(summary_df["max"].unique())

acc_grid = np.full((len(all_spaces), len(all_maxes)), np.nan)
rec_grid = np.full((len(all_spaces), len(all_maxes)), np.nan)

for _, r in summary_df.iterrows():
    s = r["space"]
    m = r["max"]
    s_idx = all_spaces.index(s)
    m_idx = all_maxes.index(m)
    acc_grid[s_idx, m_idx] = r["Accuracy"]
    rec_grid[s_idx, m_idx] = r["Recall"]

# ============= step 3: plotting helper =============

def plot_grid(grid, spaces, maxes, title, cbar_label, outfile):
    """
    grid shape: [len(spaces), len(maxes)]
    y-axis: space
    x-axis: max
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # show NaNs as blanks: we'll mask them
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(
        masked,
        origin="upper",
        aspect="auto",
        interpolation="nearest"
    )

    # x/y tick labels are your actual parameter values
    ax.set_xticks(range(len(maxes)))
    ax.set_xticklabels(maxes)
    ax.set_yticks(range(len(spaces)))
    ax.set_yticklabels(spaces)

    ax.set_xlabel("max radiance cutoff")
    ax.set_ylabel("space (frames)")
    ax.set_title(title)

    # write the numeric value at each valid cell
    for i in range(len(spaces)):
        for j in range(len(maxes)):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.4,
                        boxstyle="round,pad=0.2"
                    )
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}")

# ============= step 4: make plots =============

plot_grid(
    acc_grid,
    all_spaces,
    all_maxes,
    title=f"Accuracy at Threshold {TARGET_THRESHOLD}",
    cbar_label="Accuracy",
    outfile="accuracy_heatmap_threshold0p40.png"
)

plot_grid(
    rec_grid,
    all_spaces,
    all_maxes,
    title=f"Recall at Threshold {TARGET_THRESHOLD}",
    cbar_label="Recall",
    outfile="recall_heatmap_threshold0p40.png"
)

# ============= step 5: best combos =============

best_acc = summary_df.loc[summary_df["Accuracy"].idxmax()]
best_rec = summary_df.loc[summary_df["Recall"].idxmax()]

print("\nBest Accuracy combo:")
print(
    f"  space={best_acc['space']}, "
    f"max={best_acc['max']}, "
    f"Accuracy={best_acc['Accuracy']:.4f}, "
    f"Recall={best_acc['Recall']:.4f}"
)

print("\nBest Recall combo:")
print(
    f"  space={best_rec['space']}, "
    f"max={best_rec['max']}, "
    f"Accuracy={best_rec['Accuracy']:.4f}, "
    f"Recall={best_rec['Recall']:.4f}"
)
