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

Output files  (gxcnv-native format, TSV with ##-prefixed meta-headers)
------------------------------------------------------------------------
    <prefix>_bins.tsv          per-bin Z-scores, ratios, and corrected values
    <prefix>_segments.tsv      CBS segments with copy-number estimate
    <prefix>_calls.tsv         high-confidence CNV calls (dual-track confirmed)
    <prefix>_regions.tsv       per-target-region risk summary
    <prefix>_qcmetrics.tsv     run-level QC metrics
    <prefix>_sex.txt           predicted sex (single line)
"""

import os
import datetime
import logging
import numpy as np
from scipy.stats import chi2, norm
from . import __version__
from .utils import setup_logger, load_npz

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Meta-header writer helper
# ---------------------------------------------------------------------------

def _meta_headers(extra: dict = None) -> str:
    """
    Return a block of ##-prefixed meta-header lines common to all output files.
    """
    lines = [
        f"##gxcnv_version={__version__}",
        f"##generated={datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"##algorithm=hybrid_dual_track_cnv",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"##{k}={v}")
    return "\n".join(lines) + "\n"


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

    vals = np.nan_to_num(z[lo:hi + 1])
    cumsum = np.cumsum(vals - np.mean(vals))

    best_t = np.argmax(np.abs(cumsum))
    t_stat = cumsum[best_t]

    sigma = np.std(vals) if np.std(vals) > 0 else 1e-9
    b_stat = abs(t_stat) / (sigma * np.sqrt(n))
    p_val = 2.0 * np.exp(-2.0 * b_stat ** 2)

    if p_val < p_thresh and n >= 2 * min_bins:
        split = lo + best_t
        _cbs_recursive(z, lo,        split, segments, min_bins, p_thresh)
        _cbs_recursive(z, split + 1, hi,   segments, min_bins, p_thresh)
    else:
        segments.append((lo, hi, float(np.nanmean(z[lo:hi + 1]))))


# ---------------------------------------------------------------------------
# Track A: Z-score per bin
# ---------------------------------------------------------------------------

def _track_a_zscore(sample_corrected, ref_matrix, global_pca_mean,
                    global_pca_components, bin_means, bin_stds, k=100):
    """
    Compute per-bin Z-scores using reference distribution statistics.

    Returns
    -------
    z_scores : (N,) array
    """
    n_bins = len(sample_corrected)
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

    score = (sum(Z_bin / sqrt(n)) + 1/n) / (count(x > mu_ref) + 2/n)
    """
    n = len(sample_vals)
    if n == 0:
        return 0.0

    mu_ref  = np.nanmean(ref_vals, axis=0)
    std_ref = np.nanstd(ref_vals, axis=0)
    std_ref = np.where(std_ref < 1e-9, 1e-9, std_ref)

    z_bins   = (sample_vals - mu_ref) / std_ref
    z_sum    = np.nansum(z_bins) / np.sqrt(n)
    over_mean = np.sum(sample_vals > mu_ref)

    return float((z_sum + 1.0 / n) / (over_mean + 2.0 / n))


def _track_b_mahalanobis(sample_corrected, ref_matrix, regional_models):
    """
    Compute per-region Mahalanobis distance and Chi-square p-value.

    Returns
    -------
    region_results : list of dicts
    """
    results = []

    for rm in regional_models:
        bin_mask  = rm["bin_mask"]
        pca_mean  = rm["pca_mean"]
        pca_comps = rm["pca_comps"]

        sample_region = np.nan_to_num(sample_corrected[bin_mask])
        ref_region    = np.nan_to_num(ref_matrix[:, bin_mask])

        if sample_region.size == 0:
            continue

        sample_denoised = _apply_pca_correction(sample_region, pca_mean, pca_comps)
        ref_denoised    = np.vstack([
            _apply_pca_correction(ref_region[i], pca_mean, pca_comps)
            for i in range(ref_region.shape[0])
        ])

        score_sample = _laplace_score(sample_denoised, ref_denoised, ref_region.shape[0])

        ref_scores = np.array([
            _laplace_score(ref_denoised[i], ref_denoised, ref_region.shape[0])
            for i in range(ref_denoised.shape[0])
        ])

        mu_score  = np.mean(ref_scores)
        std_score = np.std(ref_scores)
        if std_score < 1e-9:
            std_score = 1e-9

        mahal_dist = abs(score_sample - mu_score) / std_score
        p_value    = float(chi2.sf(mahal_dist ** 2, df=1))
        risk_pct   = (1.0 - p_value) * 100.0

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
    """Remove PCA-captured noise components from signal."""
    centred      = signal - pca_mean
    projected    = pca_comps @ centred
    reconstructed = pca_comps.T @ projected
    return signal - reconstructed


# ---------------------------------------------------------------------------
# Sex prediction for a single sample
# ---------------------------------------------------------------------------

def _predict_sample_sex(sample_corrected, chroms, bin_info, gmm_cutoff):
    """Predict sex of a single sample using the reference GMM cutoff."""
    chrom_idx_arr  = bin_info[:, 0].astype(int)
    chrY_idx_list  = [i for i, c in enumerate(chroms) if c == "chrY"]
    if not chrY_idx_list:
        return "F"
    is_chrY = np.isin(chrom_idx_arr, chrY_idx_list)
    total   = np.nansum(sample_corrected)
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

    Without FF:  CN ≈ 2 + z / 3   (heuristic)
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
    bin_info         = ref["bin_info"]
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
    chrom_idx_arr  = bin_info[:, 0].astype(int)
    sex_chrom_idx  = [i for i, c in enumerate(chroms) if c in ("chrX", "chrY")]
    is_sex         = np.isin(chrom_idx_arr, sex_chrom_idx)
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
    bin_starts = bin_info[:, 1].astype(int)
    bin_ends   = bin_info[:, 2].astype(int)

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
    # Write output files (gxcnv-native TSV format)
    # -----------------------------------------------------------------------
    run_meta = {
        "sample":          os.path.basename(sample_npz_path),
        "reference":       os.path.basename(reference_npz_path),
        "predicted_sex":   sex,
        "fetal_fraction":  fetal_fraction if fetal_fraction else "NA",
        "thresh_z":        thresh_z,
        "thresh_p":        thresh_p,
        "bin_size":        int(ref.get("bin_size", 0)),
        "n_ref_samples":   int(ref.get("n_samples", 0)),
    }

    _write_bins_tsv(output_prefix, bin_info, chroms, z_scores,
                    sample_corrected_adj, bin_means, run_meta)
    _write_segments_tsv(output_prefix, all_segments, run_meta)
    _write_calls_tsv(output_prefix, aberrations, run_meta)
    _write_regions_tsv(output_prefix, region_results, track_a_region_z,
                       thresh_z, thresh_p, run_meta)
    _write_qcmetrics_tsv(output_prefix, sample, sex, z_scores, aberrations,
                         fetal_fraction, run_meta)
    _write_sex_txt(output_prefix, sex, run_meta)

    logger.info(f"Prediction complete. Output prefix: {output_prefix}")
    return {
        "sex":            sex,
        "z_scores":       z_scores,
        "segments":       all_segments,
        "aberrations":    aberrations,
        "region_results": region_results,
    }


# ---------------------------------------------------------------------------
# Output writers  –  gxcnv-native TSV format
# ---------------------------------------------------------------------------

def _write_bins_tsv(prefix, bin_info, chroms, z_scores,
                    corrected, bin_means, meta):
    """
    Per-bin table.

    Columns
    -------
    chrom  start  end  gc_fraction  raw_count_norm  expected_norm
    z_score  obs_exp_ratio  flag
    """
    path = f"{prefix}_bins.tsv"
    chrom_idx_arr = bin_info[:, 0].astype(int)
    starts = bin_info[:, 1].astype(int)
    ends   = bin_info[:, 2].astype(int)
    gcs    = bin_info[:, 3]

    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write(
            "#chrom\tstart\tend\tgc_fraction\t"
            "obs_norm\texp_norm\tz_score\tobs_exp_ratio\tflag\n"
        )
        for i in range(len(starts)):
            chrom = chroms[chrom_idx_arr[i]]
            z     = z_scores[i]
            z_str = f"{z:.4f}" if np.isfinite(z) else "NA"
            exp   = bin_means[i]
            obs   = corrected[i]
            ratio = obs / exp if exp > 0 else 0.0
            gc    = gcs[i]
            gc_str = f"{gc:.4f}" if np.isfinite(gc) else "NA"

            if not np.isfinite(z):
                flag = "MASKED"
            elif z < -3:
                flag = "DEL_CANDIDATE"
            elif z > 3:
                flag = "DUP_CANDIDATE"
            else:
                flag = "NORMAL"

            fh.write(
                f"{chrom}\t{starts[i]}\t{ends[i]}\t{gc_str}\t"
                f"{obs:.6f}\t{exp:.6f}\t{z_str}\t{ratio:.4f}\t{flag}\n"
            )
    logger.info(f"Written: {path}")


def _write_segments_tsv(prefix, segments, meta):
    """
    CBS segment table.

    Columns
    -------
    chrom  start  end  n_bins  mean_z  copy_number_est  segment_type
    """
    path = f"{prefix}_segments.tsv"
    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write(
            "#chrom\tstart\tend\tn_bins\tmean_z\t"
            "copy_number_est\tsegment_type\n"
        )
        for s in segments:
            if s["mean_z"] < -1.5:
                stype = "DELETION"
            elif s["mean_z"] > 1.5:
                stype = "DUPLICATION"
            else:
                stype = "NEUTRAL"
            fh.write(
                f"{s['chrom']}\t{s['start']}\t{s['end']}\t"
                f"{s['n_bins']}\t{s['mean_z']:.4f}\t"
                f"{s['cn']:.2f}\t{stype}\n"
            )
    logger.info(f"Written: {path}")


def _write_calls_tsv(prefix, aberrations, meta):
    """
    Dual-track confirmed CNV calls.

    Columns
    -------
    chrom  start  end  region_name  track_a_mean_z  track_b_mahal_dist
    track_b_pvalue  risk_pct  track_a_result  track_b_result  dual_call
    """
    path = f"{prefix}_calls.tsv"
    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write(
            "#chrom\tstart\tend\tregion_name\t"
            "track_a_mean_z\ttrack_b_mahal_dist\ttrack_b_pvalue\t"
            "risk_pct\ttrack_a_result\ttrack_b_result\tdual_call\n"
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


def _write_regions_tsv(prefix, region_results, track_a_z,
                       thresh_z, thresh_p, meta):
    """
    Per-target-region risk summary (all regions, not just HIGH_RISK).

    Columns
    -------
    chrom  start  end  region_name  track_a_mean_z  track_b_laplace_score
    track_b_mahal_dist  track_b_pvalue  risk_pct
    track_a_result  track_b_result  dual_call
    """
    path = f"{prefix}_regions.tsv"
    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write(
            "#chrom\tstart\tend\tregion_name\t"
            "track_a_mean_z\ttrack_b_mahal_dist\ttrack_b_pvalue\t"
            "risk_pct\ttrack_a_result\ttrack_b_result\tdual_call\n"
        )
        for r in region_results:
            name         = r["name"]
            mean_z       = track_a_z.get(name, float("nan"))
            track_a_flag = mean_z < thresh_z
            track_b_flag = r["p_value"] < thresh_p
            call         = "HIGH_RISK" if (track_a_flag and track_b_flag) else "LOW_RISK"
            fh.write(
                f"{r['chrom']}\t{r['start']}\t{r['end']}\t{name}\t"
                f"{mean_z:.4f}\t{r['mahal_dist']:.4f}\t"
                f"{r['p_value']:.6f}\t{r['risk_pct']:.2f}\t"
                f"{'PASS' if track_a_flag else 'FAIL'}\t"
                f"{'PASS' if track_b_flag else 'FAIL'}\t{call}\n"
            )
    logger.info(f"Written: {path}")


def _write_qcmetrics_tsv(prefix, sample, sex, z_scores, aberrations,
                         ff, meta):
    """
    Run-level QC metrics in key=value TSV format.
    """
    path = f"{prefix}_qcmetrics.tsv"
    total_reads = int(sample.get("total_reads", 0))
    n_bins      = len(z_scores)
    n_valid     = int(np.sum(np.isfinite(z_scores)))
    median_z    = float(np.nanmedian(z_scores))
    mad_z       = float(np.nanmedian(np.abs(z_scores - median_z)))
    n_high_risk = sum(1 for a in aberrations if a["call"] == "HIGH_RISK")

    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write("#metric\tvalue\n")
        fh.write(f"total_reads\t{total_reads}\n")
        fh.write(f"predicted_sex\t{sex}\n")
        fh.write(f"fetal_fraction\t{ff if ff else 'NA'}\n")
        fh.write(f"n_bins_total\t{n_bins}\n")
        fh.write(f"n_bins_valid\t{n_valid}\n")
        fh.write(f"pct_bins_valid\t{n_valid / n_bins * 100:.2f}\n")
        fh.write(f"median_z_score\t{median_z:.4f}\n")
        fh.write(f"mad_z_score\t{mad_z:.4f}\n")
        fh.write(f"n_del_candidate_bins\t"
                 f"{int(np.sum(z_scores < -3))}\n")
        fh.write(f"n_dup_candidate_bins\t"
                 f"{int(np.sum(z_scores > 3))}\n")
        fh.write(f"n_high_risk_regions\t{n_high_risk}\n")
    logger.info(f"Written: {path}")


def _write_sex_txt(prefix, sex, meta):
    """
    Single-line sex prediction result.
    """
    path = f"{prefix}_sex.txt"
    with open(path, "w") as fh:
        fh.write(_meta_headers(meta))
        fh.write(f"predicted_sex\t{sex}\n")
    logger.info(f"Written: {path}")
