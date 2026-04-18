"""
gxcnv.convert
=============
Converts a BAM/CRAM file into a compressed NPZ sample file.

Steps
-----
1. Count reads per genomic bin (default 100 kb).
2. Apply GC-content correction using LOESS-like polynomial regression.
3. Mask low-coverage and blacklisted bins.
4. Save the result as a compressed NumPy archive (.npz).

The output NPZ contains:
    bins        : (N, 4) array of [chrom_idx, start, end, gc_fraction]
    counts      : (N,)   raw read counts per bin
    corrected   : (N,)   GC-corrected, masked read counts
    chroms      : list of chromosome names (stored as object array)
    bin_size    : scalar bin size in bp
    total_reads : total mapped reads
"""

import os
import logging
import numpy as np
from .utils import setup_logger, save_npz

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Public chromosome list (autosomal + sex chromosomes)
# ---------------------------------------------------------------------------
CANONICAL_CHROMS = (
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)

# ---------------------------------------------------------------------------
# Blacklist regions (hg38 ENCODE blacklist – compact representation)
# These are approximate centromere / telomere / segdup hotspots.
# Users can supply their own BED file via --blacklist.
# ---------------------------------------------------------------------------
BUILTIN_BLACKLIST = [
    # (chrom, start, end)
    ("chr1",  121500000, 125000000),
    ("chr2",   91800000,  96000000),
    ("chr3",   90500000,  94000000),
    ("chr4",   49000000,  52700000),
    ("chr5",   46400000,  50700000),
    ("chr6",   58700000,  62300000),
    ("chr7",   58100000,  62100000),
    ("chr8",   43100000,  48100000),
    ("chr9",   47300000,  65900000),
    ("chr10",  38000000,  42300000),
    ("chr11",  51600000,  55700000),
    ("chr12",  34600000,  38200000),
    ("chr13",  16000000,  19500000),
    ("chr14",  16000000,  19100000),
    ("chr15",  17000000,  20400000),
    ("chr16",  35300000,  46400000),
    ("chr17",  22200000,  25800000),
    ("chr18",  15400000,  21500000),
    ("chr19",  24400000,  28600000),
    ("chr20",  26300000,  30400000),
    ("chr21",  10900000,  14300000),
    ("chr22",  12200000,  17900000),
    ("chrX",   60600000,  65000000),
    ("chrY",   10400000,  12500000),
]


def _load_blacklist_bed(bed_path):
    """Parse a BED file into a list of (chrom, start, end) tuples."""
    regions = []
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            regions.append((parts[0], int(parts[1]), int(parts[2])))
    return regions


def _build_bins(chrom_lengths, bin_size):
    """
    Generate genomic bins of fixed size.

    Returns
    -------
    bins : list of (chrom, start, end)
    """
    bins = []
    for chrom, length in chrom_lengths.items():
        for start in range(0, length, bin_size):
            end = min(start + bin_size, length)
            bins.append((chrom, start, end))
    return bins


def _is_blacklisted(chrom, start, end, blacklist):
    """Return True if the bin overlaps any blacklist region."""
    for bl_chrom, bl_start, bl_end in blacklist:
        if bl_chrom == chrom and bl_start < end and bl_end > start:
            return True
    return False


def _gc_correct(counts, gc_fractions, mask, poly_degree=3):
    """
    GC-content correction via polynomial regression on the median ratio.

    For each bin b:
        corrected[b] = counts[b] / predicted_median(gc[b])

    where predicted_median is a polynomial fit of
    median(counts | gc_bin) vs gc_bin center.

    Parameters
    ----------
    counts       : (N,) raw counts
    gc_fractions : (N,) GC fraction per bin
    mask         : (N,) boolean – True means the bin is usable
    poly_degree  : degree of the polynomial fit

    Returns
    -------
    corrected : (N,) GC-corrected counts (NaN for masked bins)
    """
    corrected = np.full(len(counts), np.nan)

    valid = mask & (counts > 0) & np.isfinite(gc_fractions)
    if valid.sum() < poly_degree + 1:
        logger.warning("Too few valid bins for GC correction – returning raw counts.")
        corrected[valid] = counts[valid].astype(float)
        return corrected

    # Bin GC fractions into 100 equal-width windows and compute median count
    gc_bins = np.linspace(0, 1, 101)
    gc_idx = np.digitize(gc_fractions[valid], gc_bins) - 1
    gc_idx = np.clip(gc_idx, 0, 99)

    gc_centers = []
    medians = []
    for i in range(100):
        sel = gc_idx == i
        if sel.sum() >= 3:
            gc_centers.append(gc_bins[i] + 0.005)
            medians.append(np.median(counts[valid][sel]))

    if len(gc_centers) < poly_degree + 1:
        corrected[valid] = counts[valid].astype(float)
        return corrected

    gc_centers = np.array(gc_centers)
    medians = np.array(medians, dtype=float)

    # Fit polynomial
    coeffs = np.polyfit(gc_centers, medians, poly_degree)
    poly = np.poly1d(coeffs)

    predicted = poly(gc_fractions[valid])
    predicted = np.where(predicted <= 0, np.nan, predicted)

    global_median = np.nanmedian(counts[valid].astype(float))
    ratio = counts[valid].astype(float) / predicted * global_median

    corrected[valid] = ratio
    return corrected


def bam_to_npz(
    bam_path,
    output_path,
    bin_size=100_000,
    blacklist_bed=None,
    reference_fasta=None,
    min_mapq=1,
    chroms=None,
):
    """
    Convert a BAM/CRAM file to a gxcnv NPZ sample file.

    Parameters
    ----------
    bam_path       : path to BAM or CRAM file (must be indexed)
    output_path    : output .npz path (extension added if missing)
    bin_size       : bin size in base pairs (default 100 000)
    blacklist_bed  : optional BED file with regions to exclude
    reference_fasta: required for CRAM files
    min_mapq       : minimum mapping quality (default 1)
    chroms         : list of chromosomes to include (default: canonical)
    """
    try:
        import pysam
    except ImportError:
        raise ImportError("pysam is required for BAM/CRAM conversion. "
                          "Install with: pip install pysam")

    if chroms is None:
        chroms = CANONICAL_CHROMS

    # Ensure output has .npz extension
    if not output_path.endswith(".npz"):
        output_path += ".npz"

    logger.info(f"Opening BAM/CRAM: {bam_path}")
    open_kwargs = {}
    if reference_fasta:
        open_kwargs["reference_filename"] = reference_fasta

    with pysam.AlignmentFile(bam_path, "rb", **open_kwargs) as bam:
        # Build chromosome length map from BAM header
        header_chroms = {sq["SN"]: sq["LN"] for sq in bam.header["SQ"]}
        chrom_lengths = {c: header_chroms[c] for c in chroms if c in header_chroms}

        if not chrom_lengths:
            raise ValueError(
                f"None of the requested chromosomes found in BAM header. "
                f"Available: {list(header_chroms.keys())[:10]}"
            )

        logger.info(f"Found {len(chrom_lengths)} chromosomes. Binning at {bin_size} bp.")

        # Load blacklist
        blacklist = BUILTIN_BLACKLIST.copy()
        if blacklist_bed:
            blacklist += _load_blacklist_bed(blacklist_bed)

        # Build bins
        all_bins = _build_bins(chrom_lengths, bin_size)
        n_bins = len(all_bins)
        logger.info(f"Total bins: {n_bins}")

        # Arrays
        bin_chrom_idx = np.zeros(n_bins, dtype=np.int32)
        bin_starts    = np.zeros(n_bins, dtype=np.int64)
        bin_ends      = np.zeros(n_bins, dtype=np.int64)
        bin_gc        = np.full(n_bins, np.nan, dtype=np.float32)
        raw_counts    = np.zeros(n_bins, dtype=np.int32)
        blacklisted   = np.zeros(n_bins, dtype=bool)

        chrom_list = list(chrom_lengths.keys())
        chrom_to_idx = {c: i for i, c in enumerate(chrom_list)}

        total_reads = 0

        for i, (chrom, start, end) in enumerate(all_bins):
            bin_chrom_idx[i] = chrom_to_idx[chrom]
            bin_starts[i]    = start
            bin_ends[i]      = end

            if _is_blacklisted(chrom, start, end, blacklist):
                blacklisted[i] = True
                continue

            count = bam.count(
                contig=chrom,
                start=start,
                stop=end,
                read_callback=lambda r: (
                    not r.is_unmapped
                    and not r.is_duplicate
                    and not r.is_secondary
                    and not r.is_supplementary
                    and r.mapping_quality >= min_mapq
                ),
            )
            raw_counts[i] = count
            total_reads += count

        logger.info(f"Total mapped reads counted: {total_reads:,}")

    # Mask: remove blacklisted and very-low-coverage bins
    median_cov = np.median(raw_counts[~blacklisted & (raw_counts > 0)])
    low_cov_mask = raw_counts < 0.05 * median_cov
    mask = ~blacklisted & ~low_cov_mask

    logger.info(
        f"Bins masked: blacklisted={blacklisted.sum()}, "
        f"low_coverage={low_cov_mask.sum()}, usable={mask.sum()}"
    )

    # GC correction
    corrected = _gc_correct(raw_counts.astype(float), bin_gc, mask)

    # Normalise to reads-per-million-equivalent (relative ratio)
    valid_sum = np.nansum(corrected[mask])
    if valid_sum > 0:
        corrected = corrected / valid_sum * mask.sum()

    # Pack and save
    bins_arr = np.column_stack([bin_chrom_idx, bin_starts, bin_ends, bin_gc])

    save_npz(
        output_path,
        {
            "bins":        bins_arr,
            "counts":      raw_counts,
            "corrected":   corrected,
            "mask":        mask,
            "chroms":      np.array(chrom_list, dtype=object),
            "bin_size":    np.array(bin_size),
            "total_reads": np.array(total_reads),
        },
    )

    logger.info(f"Saved sample NPZ: {output_path}")
    return output_path
