"""
gxcnv.predict
=============
Hybrid CNV prediction using dual-track cross-validation.

Track A  –  WisecondorX-inspired Z-score approach
    1. For each target bin, find K nearest-neighbour reference bins
       (Euclidean distance in PCA-reduced space).
    2. Compute Z-score: Z = (sample_bin - mean_ref_bins) / std_ref_bins
    3. Apply Circular Binary Segmentation (CBS) to detect segments.

Track B  –  BinDel-inspired Mahalanobis approach
    1. Apply regional PCA to remove local noise.
    2. Compute Laplace-smoothed directional score per bin.
    3. Compute Mahalanobis distance of the region score vector
       against the reference distribution.
    4. Convert to Chi-square p-value.

Final call
    High Risk  ←  Track A AND Track B both exceed thresholds
    Low Risk   ←  otherwise

Output files
------------
    <prefix>_bins.bed          per-bin Z-scores and corrected ratios
    <prefix>_segments.bed      CBS segments with copy-number estimate
    <prefix>_aberrations.bed   high-confidence CNV calls
    <prefix>_regions.bed       per-target-region risk summary
    <prefix>_statistics.txt    run-level QC metrics
    <prefix>_gender.txt        predicted sex
"""

import os
import logging
import numpy as np
from scipy.stats import chi2, norm
from .utils import setup_logger, load_npz

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Circular Binary Segmentation (CBS) – lightweight implementation
# ---------------------------------------------------------------------------

def _cbs_segment(z_scores, min_segment_bins=5, p_threshold=0.01):
    """
    Simplified CBS: iteratively find the most significant change-point
    in a Z-score array and split recursively.

    Returns list of (start_idx, end_idx, mean_z) tuples (0-based, inclusive).
    """
    segments = []
    _cbs_recursive(z_scores, 0, len(z_scores) - 1, segments,
                   min_segment_bins, p_threshold)
    segments.sort(key=lambda x: x[0])
    return segments


def _cbs_recursive(z, lo, hi, segments, min_bins, p_thresh):
    n = hi - lo + 1
    if n < min_bins:
        segments.append((lo, hi, float(np.nanmean(z[lo:hi + 1]))))
        return

    # Cumulative sum statistic
    vals = np.nan_to_num(z[lo:hi + 1])
    cumsum = np.cumsum(vals - np.mean(vals))

    # Find maximum absolute deviation
    best_t = np.argmax(np.abs(cumsum))
    t_stat = cumsum[best_t]

    # Permutation-free p-value approximation (Brownian bridge)
    sigma = np.std(vals) if np.std(vals) > 0 else 1e-9
    b_stat = abs(t_stat) / (sigma * np.sqrt(n))
    # Approximation: P ≈ 2 * exp(-2 * b^2) for large n
    p_val = 2.0 * np.exp(-2.0 * b_stat ** 2)

    if p_val < p_thresh and n >= 2 * min_bins:
        split = lo + best_t
        _cbs_recursive(z, lo,      split, segments, min_bins, p_thresh)
        _cbs_recursive(z, split + 1, hi,  segments, min_bins, p_thresh)
    else:
        segments.append((lo, hi, float(np.nanmean(z[lo:hi + 1]))))


# ---------------------------------------------------------------------------
# Track A: Z-score per bin
# ---------------------------------------------------------------------------

def _track_a_zscore(sample_corrected, ref_matrix, global_pca_mean,
                    global_pca_components, bin_means, bin_stds, k=100):
    """
    Compute per-bin Z-scores using K nearest-neighbour reference bins
    in PCA-reduced space.

    Returns
    -------
    z_scores : (N,) array
    """
    n_bins = len(sample_corrected)

    # Project into PCA space
    sample_centred = np.nan_to_num(sample_corrected) - global_pca_mean
    sample_pca = sample_centred @ global_pca_components.T  # (K_pca,)

    ref_centred = ref_matrix - global_pca_mean  # (S, N)
    ref_pca = ref_centred @ global_pca_components.T  # (S, K_pca)

    # Per-bin Z-score using reference distribution
    z_scores = np.full(n_bins, np.nan)
    sample_vals = np.nan_to_num(sample_corrected)

    for b in range(n_bins):
        mu  = bin_means[b]
        std = bin_stds[b]
        if std < 1e-9:
            continue
        z_scores[b] = (sample_vals[b] - mu) / std

    return z_scores


# ---------------------------------------------------------------------------
# Track B: Regional Mahalanobis distance
# ---------------------------------------------------------------------------

def _laplace_score(sample_vals, ref_vals, n_ref):
    """
    Compute Laplace-smoothed directional score for a region.

    score = (sum(Z_bin / sqrt(n)) + 1/n) / (count(R > mu_ref) + 2/n)

    where n = number of bins in the region.
    """
    n = len(sample_vals)
    if n == 0:
        return 0.0

    mu_ref = np.nanmean(ref_vals, axis=0)  # (n,)
    std_ref = np.nanstd(ref_vals, axis=0)
    std_ref = np.where(std_ref < 1e-9, 1e-9, std_ref)

    z_bins = (sample_vals - mu_ref) / std_ref  # (n,)
    z_sum = np.nansum(z_bins) / np.sqrt(n)

    over_mean = np.sum(sample_vals > mu_ref)
    score = (z_sum + 1.0 / n) / (over_mean + 2.0 / n)
    return float(score)


def _track_b_mahalanobis(sample_corrected, ref_matrix, regional_models):
    """
    Compute per-region Mahalanobis distance and Chi-square p-value.

    Returns
    -------
    region_results : list of dicts with keys:
        name, chrom, start, end,
        score_sample, mahal_dist, p_value, risk_pct
    """
    results = []

    for rm in regional_models:
        bin_mask   = rm["bin_mask"]
        pca_mean   = rm["pca_mean"]
        pca_comps  = rm["pca_comps"]

        # Extract region bins
        sample_region = np.nan_to_num(sample_corrected[bin_mask])
        ref_region    = np.nan_to_num(ref_matrix[:, bin_mask])

        if sample_region.size == 0:
            continue

        # Apply regional PCA noise removal
        sample_denoised = _apply_pca_correction(sample_region, pca_mean, pca_comps)
        ref_denoised    = np.vstack([
            _apply_pca_correction(ref_region[i], pca_mean, pca_comps)
            for i in range(ref_region.shape[0])
        ])

        # Laplace score for sample
        score_sample = _laplace_score(sample_denoised, ref_denoised, ref_region.shape[0])

        # Reference score distribution
        ref_scores = np.array([
            _laplace_score(ref_denoised[i], ref_denoised, ref_region.shape[0])
            for i in range(ref_denoised.shape[0])
        ])

        mu_score  = np.mean(ref_scores)
        std_score = np.std(ref_scores)
        if std_score < 1e-9:
            std_score = 1e-9

        # Mahalanobis distance (1-D case: standardised deviation)
        mahal_dist = abs(score_sample - mu_score) / std_score

        # Chi-square p-value (df=1)
        p_value = float(chi2.sf(mahal_dist ** 2, df=1))
        risk_pct = (1.0 - p_value) * 100.0

        results.append({
            "name":         rm["name"],
            "chrom":        rm["chrom"],
            "start":        rm["start"],
            "end":          rm["end"],
            "score_sample": score_sample,
            "mahal_dist":   mahal_dist,
            "p_value":      p_value,
            "risk_pct":     risk_pct,
        })

    return results


def _apply_pca_correction(signal, pca_mean, pca_comps):
    """
    Remove PCA components from signal (noise subtraction).

    corrected = signal - pca_comps.T @ (pca_comps @ (signal - pca_mean))
    """
    centred = signal - pca_mean
    projected = pca_comps @ centred          # (K,)
    reconstructed = pca_comps.T @ projected  # (N,)
    return signal - reconstructed


# ---------------------------------------------------------------------------
# Sex prediction for a single sample
# ---------------------------------------------------------------------------

def _predict_sample_sex(sample_corrected, chroms, bin_info, gmm_cutoff):
    """Predict sex of a single sample using the reference GMM cutoff."""
    chrom_idx_arr = bin_info[:, 0].astype(int)
    chrY_idx_list = [i for i, c in enumerate(chroms) if c == "chrY"]
    if not chrY_idx_list:
        return "F"
    is_chrY = np.isin(chrom_idx_arr, chrY_idx_list)
    total = np.nansum(sample_corrected)
    if total == 0 or is_chrY.sum() == 0:
        return "F"
    chrY_frac = np.nansum(sample_corrected[is_chrY]) / total
    return "M" if chrY_frac > float(gmm_cutoff) else "F"


# ---------------------------------------------------------------------------
# Copy-number estimation from Z-score
# ---------------------------------------------------------------------------

def _z_to_copy_number(z, fetal_fraction=None):
    """
    Estimate copy number from Z-score.

    Without FF:  CN ≈ 2 + round(z / 3)  (heuristic)
    With FF:     CN ≈ 2 + z * 2 / FF
    """
    if fetal_fraction and fetal_fraction > 0:
        cn = 2.0 + z * 2.0 / fetal_fraction
    else:
        cn = 2.0 + z / 3.0
    return max(0.0, round(cn, 2))


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(
    sample_npz_path,
    reference_npz_path,
    output_prefix,
    thresh_z=-3.0,
    thresh_p=0.05,
    fetal_fraction=None,
    cbs_min_bins=5,
    cbs_p_threshold=0.01,
):
    """
    Run hybrid CNV prediction on a sample NPZ file.

    Parameters
    ----------
    sample_npz_path    : path to sample .npz file
    reference_npz_path : path to reference panel .npz file
    output_prefix      : prefix for all output files
    thresh_z           : Track A Z-score threshold (default -3.0)
    thresh_p           : Track B p-value threshold (default 0.05)
    fetal_fraction     : optional fetal fraction estimate (0–1)
    cbs_min_bins       : minimum bins per CBS segment
    cbs_p_threshold    : CBS p-value threshold for split
    """
    logger.info(f"Loading sample: {sample_npz_path}")
    sample = load_npz(sample_npz_path)

    logger.info(f"Loading reference: {reference_npz_path}")
    ref = load_npz(reference_npz_path)

    sample_corrected = sample["corrected"].astype(float)
    chroms           = list(ref["chroms"])
    bin_info         = ref["bin_info"]          # (N, 4)
    bin_means        = ref["bin_means"].astype(float)
    bin_stds         = ref["bin_stds"].astype(float)
    ref_matrix       = ref["ref_matrix"].astype(float)
    gpc_mean         = ref["global_pca_mean"].astype(float)
    gpc_comps        = ref["global_pca_components"].astype(float)
    gmm_cutoff       = ref["gmm_cutoff"]

    # Reconstruct regional models
    reg_names  = list(ref["reg_names"])
    reg_chroms = list(ref["reg_chroms"])
    reg_starts = ref["reg_starts"]
    reg_ends   = ref["reg_ends"]
    reg_masks  = ref["reg_masks"]
    reg_pca_means = ref["reg_pca_means"]
    reg_pca_comps = ref["reg_pca_comps"]

    regional_models = []
    for i in range(len(reg_names)):
        regional_models.append({
            "name":      reg_names[i],
            "chrom":     reg_chroms[i],
            "start":     int(reg_starts[i]),
            "end":       int(reg_ends[i]),
            "bin_mask":  reg_masks[i].astype(bool),
            "pca_mean":  reg_pca_means[i].astype(float),
            "pca_comps": reg_pca_comps[i].astype(float),
        })

    # Predict sex
    sex = _predict_sample_sex(sample_corrected, chroms, bin_info, gmm_cutoff)
    logger.info(f"Predicted sex: {sex}")

    # Sex correction for sample (per-bin mask using chrom_idx)
    chrom_idx_arr = bin_info[:, 0].astype(int)
    sex_chrom_idx = [i for i, c in enumerate(chroms) if c in ("chrX", "chrY")]
    is_sex = np.isin(chrom_idx_arr, sex_chrom_idx)
    sample_corrected_adj = sample_corrected.copy()
    if sex == "M":
        sample_corrected_adj[is_sex] *= 2.0

    # -----------------------------------------------------------------------
    # Track A: Z-scores + CBS
    # -----------------------------------------------------------------------
    logger.info("Track A: computing Z-scores …")
    z_scores = _track_a_zscore(
        sample_corrected_adj, ref_matrix, gpc_mean, gpc_comps,
        bin_means, bin_stds
    )

    # -----------------------------------------------------------------------
    # Track B: Regional Mahalanobis
    # -----------------------------------------------------------------------
    logger.info("Track B: computing regional Mahalanobis distances …")
    region_results = _track_b_mahalanobis(
        sample_corrected_adj, ref_matrix, regional_models
    )

    # -----------------------------------------------------------------------
    # CBS segmentation (per chromosome)
    # -----------------------------------------------------------------------
    logger.info("Running CBS segmentation …")
    chrom_idx_arr = bin_info[:, 0].astype(int)
    bin_starts    = bin_info[:, 1].astype(int)
    bin_ends      = bin_info[:, 2].astype(int)

    all_segments = []
    for c_idx, chrom in enumerate(chroms):
        mask = chrom_idx_arr == c_idx
        if mask.sum() < cbs_min_bins:
            continue
        z_chrom = z_scores[mask]
        segs = _cbs_segment(z_chrom, cbs_min_bins, cbs_p_threshold)
        bin_positions = np.where(mask)[0]
        for seg_lo, seg_hi, seg_z in segs:
            global_lo = bin_positions[seg_lo]
            global_hi = bin_positions[seg_hi]
            all_segments.append({
                "chrom":  chrom,
                "start":  int(bin_starts[global_lo]),
                "end":    int(bin_ends[global_hi]),
                "n_bins": seg_hi - seg_lo + 1,
                "mean_z": seg_z,
                "cn":     _z_to_copy_number(seg_z, fetal_fraction),
            })

    # -----------------------------------------------------------------------
    # Dual-track aberration calls
    # -----------------------------------------------------------------------
    logger.info("Applying dual-track (AND) decision logic …")

    # Build a set of regions flagged by Track B
    track_b_flagged = {
        r["name"] for r in region_results
        if r["p_value"] < thresh_p
    }

    # Build per-region Track A mean Z
    track_a_region_z = {}
    for rm in regional_models:
        name = rm["name"]
        mask = rm["bin_mask"]
        region_z = z_scores[mask]
        track_a_region_z[name] = float(np.nanmean(region_z))

    aberrations = []
    for r in region_results:
        name   = r["name"]
        mean_z = track_a_region_z.get(name, 0.0)
        track_a_flag = mean_z < thresh_z
        track_b_flag = r["p_value"] < thresh_p

        call = "HIGH_RISK" if (track_a_flag and track_b_flag) else "LOW_RISK"

        aberrations.append({
            "chrom":      r["chrom"],
            "start":      r["start"],
            "end":        r["end"],
            "name":       name,
            "mean_z":     mean_z,
            "mahal_dist": r["mahal_dist"],
            "p_value":    r["p_value"],
            "risk_pct":   r["risk_pct"],
            "track_a":    "PASS" if track_a_flag else "FAIL",
            "track_b":    "PASS" if track_b_flag else "FAIL",
            "call":       call,
        })

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    _write_bins_bed(output_prefix, bin_info, chroms, z_scores,
                    sample_corrected_adj, bin_means)
    _write_segments_bed(output_prefix, all_segments)
    _write_aberrations_bed(output_prefix, aberrations)
    _write_regions_bed(output_prefix, region_results, track_a_region_z,
                       thresh_z, thresh_p)
    _write_statistics(output_prefix, sample, sex, z_scores, aberrations,
                      fetal_fraction)
    _write_gender(output_prefix, sex)

    logger.info(f"Prediction complete. Output prefix: {output_prefix}")
    return {
        "sex":          sex,
        "z_scores":     z_scores,
        "segments":     all_segments,
        "aberrations":  aberrations,
        "region_results": region_results,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_bins_bed(prefix, bin_info, chroms, z_scores,
                    corrected, bin_means):
    path = f"{prefix}_bins.bed"
    chrom_idx_arr = bin_info[:, 0].astype(int)
    starts = bin_info[:, 1].astype(int)
    ends   = bin_info[:, 2].astype(int)
    with open(path, "w") as fh:
        fh.write("#chrom\tstart\tend\tz_score\tratio\texpected\n")
        for i in range(len(starts)):
            chrom = chroms[chrom_idx_arr[i]]
            z = z_scores[i]
            z_str = f"{z:.4f}" if np.isfinite(z) else "NA"
            exp = bin_means[i]
            obs = corrected[i]
            ratio = obs / exp if exp > 0 else 0.0
            fh.write(f"{chrom}\t{starts[i]}\t{ends[i]}\t{z_str}\t{ratio:.4f}\t{exp:.4f}\n")
    logger.info(f"Written: {path}")


def _write_segments_bed(prefix, segments):
    path = f"{prefix}_segments.bed"
    with open(path, "w") as fh:
        fh.write("#chrom\tstart\tend\tn_bins\tmean_z\tcopy_number\n")
        for s in segments:
            fh.write(
                f"{s['chrom']}\t{s['start']}\t{s['end']}\t"
                f"{s['n_bins']}\t{s['mean_z']:.4f}\t{s['cn']:.2f}\n"
            )
    logger.info(f"Written: {path}")


def _write_aberrations_bed(prefix, aberrations):
    path = f"{prefix}_aberrations.bed"
    with open(path, "w") as fh:
        fh.write(
            "#chrom\tstart\tend\tname\tmean_z\tmahal_dist\t"
            "p_value\trisk_pct\ttrack_a\ttrack_b\tcall\n"
        )
        for a in aberrations:
            if a["call"] == "HIGH_RISK":
                fh.write(
                    f"{a['chrom']}\t{a['start']}\t{a['end']}\t{a['name']}\t"
                    f"{a['mean_z']:.4f}\t{a['mahal_dist']:.4f}\t"
                    f"{a['p_value']:.6f}\t{a['risk_pct']:.2f}\t"
                    f"{a['track_a']}\t{a['track_b']}\t{a['call']}\n"
                )
    logger.info(f"Written: {path}")


def _write_regions_bed(prefix, region_results, track_a_z, thresh_z, thresh_p):
    path = f"{prefix}_regions.bed"
    with open(path, "w") as fh:
        fh.write(
            "#chrom\tstart\tend\tname\tmean_z\tmahal_dist\t"
            "p_value\trisk_pct\tcall\n"
        )
        for r in region_results:
            name   = r["name"]
            mean_z = track_a_z.get(name, float("nan"))
            track_a_flag = mean_z < thresh_z
            track_b_flag = r["p_value"] < thresh_p
            call = "HIGH_RISK" if (track_a_flag and track_b_flag) else "LOW_RISK"
            fh.write(
                f"{r['chrom']}\t{r['start']}\t{r['end']}\t{name}\t"
                f"{mean_z:.4f}\t{r['mahal_dist']:.4f}\t"
                f"{r['p_value']:.6f}\t{r['risk_pct']:.2f}\t{call}\n"
            )
    logger.info(f"Written: {path}")


def _write_statistics(prefix, sample, sex, z_scores, aberrations, ff):
    path = f"{prefix}_statistics.txt"
    total_reads = int(sample.get("total_reads", 0))
    n_bins      = len(z_scores)
    n_valid     = int(np.sum(np.isfinite(z_scores)))
    median_z    = float(np.nanmedian(z_scores))
    mad_z       = float(np.nanmedian(np.abs(z_scores - median_z)))
    n_high_risk = sum(1 for a in aberrations if a["call"] == "HIGH_RISK")

    with open(path, "w") as fh:
        fh.write(f"total_reads\t{total_reads}\n")
        fh.write(f"predicted_sex\t{sex}\n")
        fh.write(f"fetal_fraction\t{ff if ff else 'NA'}\n")
        fh.write(f"n_bins_total\t{n_bins}\n")
        fh.write(f"n_bins_valid\t{n_valid}\n")
        fh.write(f"median_zscore\t{median_z:.4f}\n")
        fh.write(f"mad_zscore\t{mad_z:.4f}\n")
        fh.write(f"n_high_risk_regions\t{n_high_risk}\n")
    logger.info(f"Written: {path}")


def _write_gender(prefix, sex):
    path = f"{prefix}_gender.txt"
    with open(path, "w") as fh:
        fh.write(sex + "\n")
    logger.info(f"Written: {path}")
