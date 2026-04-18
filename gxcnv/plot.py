"""
gxcnv.plot
==========
High-quality, publication-ready CNV visualisation.

Plots produced
--------------
1. Genome-wide Z-score plot  (<prefix>_genome.png)
   - One panel per chromosome arranged in a 2-column grid
   - Colour-coded by copy-number deviation
   - CBS segments overlaid as horizontal lines
   - Chromosome ideogram-style x-axis

2. Region risk heatmap  (<prefix>_regions.png)
   - Horizontal bar chart of risk % per clinical target region
   - Dual-track pass/fail indicators

3. QC summary panel  (<prefix>_qc.png)
   - Z-score distribution histogram
   - Per-chromosome median Z-score bar chart
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from .utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":          "#FAFAFA",
    "grid":        "#E4E4E7",
    "text":        "#18181B",
    "muted":       "#71717A",
    "accent":      "#EA580C",
    "deletion":    "#2563EB",   # blue
    "amplification":"#DC2626",  # red
    "neutral":     "#94A3B8",   # slate
    "segment":     "#0F172A",   # near-black
    "high_risk":   "#EA580C",
    "low_risk":    "#22C55E",
}

# Custom diverging colormap: blue → white → red
_CMAP = LinearSegmentedColormap.from_list(
    "cnv_div",
    [PALETTE["deletion"], "#FFFFFF", PALETTE["amplification"]],
    N=256,
)


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "text.color":        PALETTE["text"],
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })


# ---------------------------------------------------------------------------
# Helper: load bins.bed
# ---------------------------------------------------------------------------

def _load_bins_tsv(bins_tsv_path):
    """
    Parse <prefix>_bins.tsv (gxcnv-native format) into arrays.

    Expected columns (after ##-headers and #-column-header):
    chrom  start  end  gc_fraction  obs_norm  exp_norm
    z_score  obs_exp_ratio  flag

    Returns
    -------
    chroms  : (N,) str array
    starts  : (N,) int
    ends    : (N,) int
    z_scores: (N,) float
    ratios  : (N,) float  (obs_exp_ratio)
    """
    chroms, starts, ends, zs, ratios = [], [], [], [], []
    with open(bins_tsv_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            chroms.append(parts[0])
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))
            # col 6 = z_score, col 7 = obs_exp_ratio
            zs.append(float(parts[6]) if parts[6] != "NA" else np.nan)
            ratios.append(float(parts[7]))
    return (
        np.array(chroms),
        np.array(starts, dtype=np.int64),
        np.array(ends,   dtype=np.int64),
        np.array(zs,     dtype=float),
        np.array(ratios, dtype=float),
    )


def _load_segments_tsv(segments_tsv_path):
    """
    Parse <prefix>_segments.tsv (gxcnv-native format).

    Columns: chrom  start  end  n_bins  mean_z  copy_number_est  segment_type
    """
    segs = []
    with open(segments_tsv_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            segs.append({
                "chrom":  parts[0],
                "start":  int(parts[1]),
                "end":    int(parts[2]),
                "mean_z": float(parts[4]),
                "cn":     float(parts[5]),
                "type":   parts[6],
            })
    return segs


def _load_regions_tsv(regions_tsv_path):
    """
    Parse <prefix>_regions.tsv (gxcnv-native format).

    Columns: chrom  start  end  region_name  track_a_mean_z
             track_b_mahal_dist  track_b_pvalue  risk_pct
             track_a_result  track_b_result  dual_call
    """
    regions = []
    with open(regions_tsv_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            regions.append({
                "chrom":    parts[0],
                "start":    int(parts[1]),
                "end":      int(parts[2]),
                "name":     parts[3],
                "mean_z":   float(parts[4]),
                "p_value":  float(parts[6]),
                "risk_pct": float(parts[7]),
                "call":     parts[10],
            })
    return regions


# ---------------------------------------------------------------------------
# Plot 1: Genome-wide Z-score
# ---------------------------------------------------------------------------

CANONICAL_ORDER = (
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)


def plot_genome(bins_tsv, segments_tsv, output_path,
                sample_name="Sample", sex="Unknown"):
    """
    Genome-wide Z-score plot with CBS segments.
    """
    _setup_style()

    chroms_arr, starts, ends, z_scores, ratios = _load_bins_tsv(bins_tsv)
    segments = _load_segments_tsv(segments_tsv)

    # Determine chromosomes present in data
    present = [c for c in CANONICAL_ORDER if np.any(chroms_arr == c)]
    n_chroms = len(present)

    n_cols = 2
    n_rows = (n_chroms + 1) // n_cols

    fig = plt.figure(figsize=(22, n_rows * 1.8 + 2))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Title
    fig.suptitle(
        f"gxcnv  ·  Genome-wide CNV Profile\n"
        f"{sample_name}  |  Predicted sex: {sex}",
        fontsize=14, fontweight="bold", color=PALETTE["text"],
        y=0.98,
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55, wspace=0.25,
        left=0.06, right=0.97,
        top=0.93, bottom=0.04,
    )

    z_abs_max = max(5.0, np.nanpercentile(np.abs(z_scores), 99))
    norm = Normalize(vmin=-z_abs_max, vmax=z_abs_max)

    for idx, chrom in enumerate(present):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        mask = chroms_arr == chrom
        s = starts[mask]
        z = z_scores[mask]
        mid = (s + ends[mask]) / 2.0

        # Scatter: colour by Z-score
        colors = _CMAP(norm(np.nan_to_num(z)))
        ax.scatter(mid, z, c=colors, s=2.5, linewidths=0, alpha=0.7, rasterized=True)

        # Reference band (±2 σ)
        ax.axhspan(-2, 2, color=PALETTE["grid"], alpha=0.35, linewidth=0)

        # Zero line
        ax.axhline(0, color=PALETTE["muted"], linewidth=0.8, linestyle="--", alpha=0.6)

        # Threshold lines
        ax.axhline(-3, color=PALETTE["deletion"],      linewidth=0.8, linestyle=":", alpha=0.8)
        ax.axhline( 3, color=PALETTE["amplification"], linewidth=0.8, linestyle=":", alpha=0.8)

        # CBS segments
        chrom_segs = [sg for sg in segments if sg["chrom"] == chrom]
        for sg in chrom_segs:
            seg_color = (
                PALETTE["deletion"]      if sg["mean_z"] < -2 else
                PALETTE["amplification"] if sg["mean_z"] >  2 else
                PALETTE["segment"]
            )
            ax.hlines(
                sg["mean_z"],
                sg["start"], sg["end"],
                colors=seg_color, linewidths=2.5, alpha=0.9,
            )

        # Axis formatting
        chrom_len = int(ends[mask].max()) if mask.sum() > 0 else 1
        ax.set_xlim(0, chrom_len)
        ax.set_ylim(-z_abs_max - 0.5, z_abs_max + 0.5)
        ax.set_title(chrom, fontsize=9, fontweight="bold",
                     color=PALETTE["text"], pad=3)
        ax.set_ylabel("Z-score", fontsize=7, color=PALETTE["muted"])
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=7)

        # Megabase x-axis labels
        mb_ticks = np.arange(0, chrom_len, 50_000_000)
        ax.set_xticks(mb_ticks)
        ax.set_xticklabels([f"{int(t/1e6)}" for t in mb_ticks], fontsize=6)
        ax.set_xlabel("Position (Mb)", fontsize=6, color=PALETTE["muted"])
        ax.grid(axis="y", linewidth=0.4)

    # Colorbar
    sm = ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.97, 0.1, 0.008, 0.8])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Z-score", fontsize=8, color=PALETTE["text"])
    cbar.ax.tick_params(labelsize=7, colors=PALETTE["muted"])

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info(f"Saved genome plot: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Region risk heatmap
# ---------------------------------------------------------------------------

def plot_regions(regions_tsv, output_path,
                 sample_name="Sample", thresh_p=0.05):
    """
    Horizontal bar chart of risk % per clinical target region.
    """
    _setup_style()
    regions = _load_regions_tsv(regions_tsv)
    if not regions:
        logger.warning("No regions found – skipping region plot.")
        return

    # Sort by risk_pct descending
    regions = sorted(regions, key=lambda r: r["risk_pct"], reverse=True)
    names    = [r["name"]     for r in regions]
    risks    = [r["risk_pct"] for r in regions]
    calls    = [r["call"]     for r in regions]
    p_values = [r["p_value"]  for r in regions]

    n = len(regions)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.55 + 1.5)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bar_colors = [
        PALETTE["high_risk"] if c == "HIGH_RISK" else PALETTE["low_risk"]
        for c in calls
    ]

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, risks, color=bar_colors, height=0.65,
                   edgecolor="none", alpha=0.85)

    # Risk threshold line (95%)
    thresh_risk = (1 - thresh_p) * 100
    ax.axvline(thresh_risk, color=PALETTE["accent"], linewidth=1.5,
               linestyle="--", alpha=0.9, label=f"Threshold ({thresh_risk:.0f}%)")

    # Annotate bars
    for i, (bar, r) in enumerate(zip(bars, risks)):
        ax.text(
            min(r + 1.5, 102), i,
            f"{r:.1f}%",
            va="center", ha="left", fontsize=8,
            color=PALETTE["text"], fontweight="bold",
        )
        # p-value annotation
        ax.text(
            -2, i,
            f"p={p_values[i]:.3f}",
            va="center", ha="right", fontsize=7,
            color=PALETTE["muted"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-18, 110)
    ax.set_xlabel("Risk (%)", fontsize=10, color=PALETTE["text"])
    ax.set_title(
        f"gxcnv  ·  Clinical Target Region Risk\n{sample_name}",
        fontsize=12, fontweight="bold", color=PALETTE["text"], pad=10,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.5, alpha=0.5)

    # Legend
    high_patch = mpatches.Patch(color=PALETTE["high_risk"], label="HIGH RISK")
    low_patch  = mpatches.Patch(color=PALETTE["low_risk"],  label="LOW RISK")
    ax.legend(handles=[high_patch, low_patch], loc="lower right",
              fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info(f"Saved region plot: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: QC summary
# ---------------------------------------------------------------------------

def plot_qc(bins_tsv, output_path, sample_name="Sample"):
    """
    QC summary: Z-score distribution + per-chromosome median Z-score.
    """
    _setup_style()

    chroms_arr, starts, ends, z_scores, ratios = _load_bins_tsv(bins_tsv)

    present = [c for c in CANONICAL_ORDER if np.any(chroms_arr == c)]
    chrom_medians = []
    for c in present:
        mask = chroms_arr == c
        chrom_medians.append(float(np.nanmedian(z_scores[mask])))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"gxcnv  ·  QC Summary  |  {sample_name}",
        fontsize=13, fontweight="bold", color=PALETTE["text"], y=1.02,
    )

    # --- Panel A: Z-score histogram ---
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    valid_z = z_scores[np.isfinite(z_scores)]
    ax.hist(valid_z, bins=120, color=PALETTE["neutral"], edgecolor="none",
            alpha=0.8, density=True)

    # Overlay normal distribution
    x = np.linspace(-8, 8, 400)
    from scipy.stats import norm as _norm
    ax.plot(x, _norm.pdf(x, 0, 1), color=PALETTE["accent"],
            linewidth=1.8, label="N(0,1)")

    ax.axvline(-3, color=PALETTE["deletion"],      linewidth=1.2,
               linestyle="--", alpha=0.8, label="Z = ±3")
    ax.axvline( 3, color=PALETTE["amplification"], linewidth=1.2,
               linestyle="--", alpha=0.8)

    ax.set_xlabel("Z-score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Z-score Distribution", fontsize=11, fontweight="bold")
    ax.set_xlim(-8, 8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4)

    # Stats annotation
    med = np.nanmedian(valid_z)
    mad = np.nanmedian(np.abs(valid_z - med))
    ax.text(
        0.97, 0.97,
        f"Median = {med:.3f}\nMAD = {mad:.3f}\nN bins = {len(valid_z):,}",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=8, color=PALETTE["text"],
        bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                  boxstyle="round,pad=0.4"),
    )

    # --- Panel B: Per-chromosome median Z-score ---
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"])
    y_pos = np.arange(len(present))
    bar_colors = [
        PALETTE["deletion"]      if m < -2 else
        PALETTE["amplification"] if m >  2 else
        PALETTE["neutral"]
        for m in chrom_medians
    ]
    ax2.barh(y_pos, chrom_medians, color=bar_colors, height=0.7,
             edgecolor="none", alpha=0.85)
    ax2.axvline(0,  color=PALETTE["muted"],       linewidth=0.8)
    ax2.axvline(-2, color=PALETTE["deletion"],     linewidth=0.8,
                linestyle=":", alpha=0.7)
    ax2.axvline( 2, color=PALETTE["amplification"],linewidth=0.8,
                linestyle=":", alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(present, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Median Z-score", fontsize=10)
    ax2.set_title("Per-chromosome Median Z-score", fontsize=11,
                  fontweight="bold")
    ax2.grid(axis="x", linewidth=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info(f"Saved QC plot: {output_path}")


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def plot_all(output_prefix, sample_name=None, sex="Unknown", thresh_p=0.05):
    """
    Generate all three plots from gxcnv-native TSV output files at `output_prefix`.
    """
    if sample_name is None:
        sample_name = os.path.basename(output_prefix)

    bins_tsv     = f"{output_prefix}_bins.tsv"
    segments_tsv = f"{output_prefix}_segments.tsv"
    regions_tsv  = f"{output_prefix}_regions.tsv"

    if os.path.exists(bins_tsv) and os.path.exists(segments_tsv):
        plot_genome(
            bins_tsv, segments_tsv,
            f"{output_prefix}_genome.png",
            sample_name=sample_name, sex=sex,
        )
        plot_qc(
            bins_tsv,
            f"{output_prefix}_qc.png",
            sample_name=sample_name,
        )
    else:
        logger.warning("bins.tsv or segments.tsv not found – skipping genome/QC plots.")

    if os.path.exists(regions_tsv):
        plot_regions(
            regions_tsv,
            f"{output_prefix}_regions.png",
            sample_name=sample_name,
            thresh_p=thresh_p,
        )
    else:
        logger.warning("regions.tsv not found – skipping region plot.")
