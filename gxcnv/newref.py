"""
gxcnv.newref
============
Build a reference panel from a set of normal-sample NPZ files.

Algorithm
---------
1. Load all sample NPZ files and verify bin compatibility.
2. Predict sex of each sample using a Gaussian Mixture Model (GMM)
   fitted to the chrY read-fraction distribution.
3. Apply sex-aware normalisation:
   - Male samples: chrX and chrY counts × 2 (to match diploid level).
4. Compute per-bin statistics (mean, std) across the reference cohort.
5. Fit a Global PCA model (capturing 95 % of variance) on the full
   genome bin matrix.
6. For each clinical target region, fit a Regional PCA model
   (capturing 5–50 % of variance) for fine-grained noise removal.
7. Save the reference panel as a compressed NPZ archive.

Reference NPZ contents
----------------------
    bin_means      : (N,)    per-bin mean across reference samples
    bin_stds       : (N,)    per-bin std across reference samples
    bin_info       : (N, 4)  [chrom_idx, start, end, gc_fraction]
    chroms         : object array of chromosome names
    bin_size       : scalar
    n_samples      : number of reference samples
    sex_labels     : (S,)    predicted sex per sample ('M' / 'F')
    gmm_means      : (2,)    GMM component means (chrY fraction)
    gmm_stds       : (2,)    GMM component stds
    gmm_cutoff     : scalar  decision boundary
    global_pca_components : (K, N) top-K PCA components
    global_pca_mean       : (N,)   mean vector used in PCA
    global_pca_var_ratio  : (K,)   explained variance ratio
    target_regions : object array of dicts (one per clinical target)
    ref_matrix     : (S, N)  normalised reference matrix (after sex correction)
"""

import os
import glob
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from .utils import setup_logger, load_npz, save_npz

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Default clinical target regions (hg38 coordinates)
# Each entry: (name, chrom, start, end)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_REGIONS = [
    ("DiGeorge_22q11",     "chr22", 18_900_000,  21_800_000),
    ("Williams_7q11",      "chr7",  72_700_000,  74_100_000),
    ("Angelman_15q11",     "chr15", 22_800_000,  28_500_000),
    ("PraderWilli_15q11",  "chr15", 22_800_000,  28_500_000),
    ("Wolf_4p16",          "chr4",   1_000_000,   4_000_000),
    ("CriDuChat_5p15",     "chr5",       1_000,  11_800_000),
    ("NF1_17q11",          "chr17", 29_400_000,  30_200_000),
    ("Smith_17p11",        "chr17", 16_700_000,  20_500_000),
    ("Langer_Giedion_8q24","chr8",  116_600_000, 119_400_000),
    ("Miller_Dieker_17p13","chr17",       1_000,   4_000_000),
    ("CHARGE_8q12",        "chr8",   61_500_000,  62_000_000),
    ("Kabuki_12q13",       "chr12",  49_000_000,  50_000_000),
    ("Sotos_5q35",         "chr5",  175_000_000, 177_000_000),
    ("Rubinstein_16p13",   "chr16",   3_700_000,   3_900_000),
    ("Potocki_Lupski_17p11","chr17", 15_000_000,  20_500_000),
]


def _predict_sex_gmm(ref_matrix, chroms, bin_info):
    """
    Predict sample sex using a 2-component GMM on the chrY read fraction.

    Parameters
    ----------
    ref_matrix : (S, N) normalised count matrix
    chroms     : list of chromosome names (one per unique chromosome)
    bin_info   : (N, 4) array [chrom_idx, start, end, gc]

    Returns
    -------
    sex_labels : (S,) array of 'M' or 'F'
    gmm_means  : (2,) component means
    gmm_stds   : (2,) component stds
    cutoff     : float decision boundary
    """
    chrom_arr = np.asarray(chroms)
    # Build per-bin boolean mask using chrom_idx column of bin_info
    chrom_idx_arr = bin_info[:, 0].astype(int)
    chrY_idx_list = [i for i, c in enumerate(chroms) if c == "chrY"]
    if not chrY_idx_list:
        is_chrY = np.zeros(ref_matrix.shape[1], dtype=bool)
    else:
        is_chrY = np.isin(chrom_idx_arr, chrY_idx_list)

    if is_chrY.sum() == 0:
        logger.warning("No chrY bins found – assigning all samples as Female.")
        return (
            np.array(["F"] * ref_matrix.shape[0]),
            np.array([0.0, 0.0]),
            np.array([1e-6, 1e-6]),
            0.0,
        )

    chrY_fraction = ref_matrix[:, is_chrY].sum(axis=1)
    # Normalise by total signal
    total = ref_matrix.sum(axis=1)
    total = np.where(total == 0, 1, total)
    chrY_fraction = chrY_fraction / total

    # Fit 2-component GMM
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
    gmm.fit(chrY_fraction.reshape(-1, 1))

    means = gmm.means_.flatten()
    stds  = np.sqrt(gmm.covariances_.flatten())

    # Sort so component 0 = low (female), component 1 = high (male)
    order = np.argsort(means)
    means = means[order]
    stds  = stds[order]

    # Decision boundary: local minimum between the two Gaussians
    x = np.linspace(means[0], means[1], 1000)
    p0 = gmm.weights_[order[0]] * _gauss_pdf(x, means[0], stds[0])
    p1 = gmm.weights_[order[1]] * _gauss_pdf(x, means[1], stds[1])
    diff = np.abs(p0 - p1)
    cutoff = x[np.argmin(diff)]

    sex_labels = np.where(chrY_fraction > cutoff, "M", "F")
    n_male   = (sex_labels == "M").sum()
    n_female = (sex_labels == "F").sum()
    logger.info(
        f"Sex prediction: {n_male} Male, {n_female} Female "
        f"(cutoff chrY-fraction = {cutoff:.5f})"
    )
    return sex_labels, means, stds, cutoff


def _gauss_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _sex_correct(ref_matrix, sex_labels, chroms, bin_info):
    """
    Multiply chrX and chrY counts by 2 for male samples so that
    sex chromosomes are on the same scale as autosomes (diploid).
    """
    chrom_idx_arr = bin_info[:, 0].astype(int)
    sex_chrom_idx = [i for i, c in enumerate(chroms) if c in ("chrX", "chrY")]
    is_sex = np.isin(chrom_idx_arr, sex_chrom_idx)
    corrected = ref_matrix.copy().astype(float)
    male_idx = np.where(sex_labels == "M")[0]
    corrected[np.ix_(male_idx, is_sex)] *= 2.0
    return corrected


def _fit_global_pca(matrix, variance_threshold=0.95):
    """
    Fit PCA retaining enough components to explain `variance_threshold`
    of total variance.

    Returns
    -------
    pca_mean       : (N,)
    components     : (K, N)
    var_ratio      : (K,)
    """
    n_max = min(matrix.shape) - 1
    pca = PCA(n_components=n_max, svd_solver="full")
    pca.fit(matrix)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_keep = max(1, min(n_keep, n_max))

    logger.info(
        f"Global PCA: keeping {n_keep} components "
        f"({cumvar[n_keep - 1] * 100:.1f}% variance explained)"
    )
    return pca.mean_, pca.components_[:n_keep], pca.explained_variance_ratio_[:n_keep]


def _fit_regional_pca(matrix, bin_mask, variance_range=(0.05, 0.50)):
    """
    Fit a regional PCA on the subset of bins defined by `bin_mask`.

    The number of components is chosen so that the cumulative explained
    variance falls within [variance_range[0], variance_range[1]].

    Returns
    -------
    dict with keys: mean, components, var_ratio
    """
    sub = matrix[:, bin_mask]
    if sub.shape[1] < 2 or sub.shape[0] < 2:
        return None

    n_max = min(sub.shape) - 1
    pca = PCA(n_components=n_max, svd_solver="full")
    pca.fit(sub)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    lo = np.searchsorted(cumvar, variance_range[0])
    hi = np.searchsorted(cumvar, variance_range[1])
    n_keep = max(1, min(hi + 1, n_max))

    return {
        "mean":       pca.mean_,
        "components": pca.components_[:n_keep],
        "var_ratio":  pca.explained_variance_ratio_[:n_keep],
    }


def build_reference(
    npz_paths,
    output_path,
    target_regions=None,
    global_pca_variance=0.95,
    regional_pca_variance_range=(0.05, 0.50),
):
    """
    Build a gxcnv reference panel from a list of normal-sample NPZ files.

    Parameters
    ----------
    npz_paths                 : list of paths to sample NPZ files
    output_path               : output reference .npz path
    target_regions            : list of (name, chrom, start, end) tuples
                                (defaults to DEFAULT_TARGET_REGIONS)
    global_pca_variance       : cumulative variance threshold for global PCA
    regional_pca_variance_range : (lo, hi) variance range for regional PCA
    """
    if target_regions is None:
        target_regions = DEFAULT_TARGET_REGIONS

    if not output_path.endswith(".npz"):
        output_path += ".npz"

    logger.info(f"Loading {len(npz_paths)} sample NPZ files …")

    samples = [load_npz(p) for p in npz_paths]

    # Validate bin compatibility
    ref_bins = samples[0]["bins"]
    ref_chroms = list(samples[0]["chroms"])
    ref_bin_size = int(samples[0]["bin_size"])
    n_bins = ref_bins.shape[0]

    for i, s in enumerate(samples[1:], 1):
        if s["bins"].shape[0] != n_bins:
            raise ValueError(
                f"Sample {npz_paths[i]} has {s['bins'].shape[0]} bins "
                f"but expected {n_bins}."
            )

    # Build reference matrix: (S, N)
    ref_matrix = np.vstack([s["corrected"] for s in samples])  # (S, N)

    # Replace NaN with 0 for PCA
    ref_matrix = np.nan_to_num(ref_matrix, nan=0.0)

    # Sex prediction
    sex_labels, gmm_means, gmm_stds, gmm_cutoff = _predict_sex_gmm(
        ref_matrix, ref_chroms, ref_bins
    )

    # Sex correction
    ref_matrix_corrected = _sex_correct(ref_matrix, sex_labels, ref_chroms, ref_bins)

    # Per-bin statistics
    bin_means = np.mean(ref_matrix_corrected, axis=0)
    bin_stds  = np.std(ref_matrix_corrected, axis=0)
    bin_stds  = np.where(bin_stds == 0, 1e-9, bin_stds)  # avoid division by zero

    # Global PCA
    gpc_mean, gpc_components, gpc_var_ratio = _fit_global_pca(
        ref_matrix_corrected, variance_threshold=global_pca_variance
    )

    # Regional PCA for each target region
    chrom_arr = np.asarray(ref_chroms)
    bin_starts = ref_bins[:, 1].astype(int)
    bin_ends   = ref_bins[:, 2].astype(int)
    chrom_idx_arr = ref_bins[:, 0].astype(int)

    regional_models = []
    for name, chrom, reg_start, reg_end in target_regions:
        if chrom not in ref_chroms:
            logger.warning(f"Target region '{name}' chrom {chrom} not in reference – skipping.")
            continue

        c_idx = ref_chroms.index(chrom)
        bin_mask = (
            (chrom_idx_arr == c_idx)
            & (bin_starts < reg_end)
            & (bin_ends   > reg_start)
        )
        n_region_bins = bin_mask.sum()
        if n_region_bins == 0:
            logger.warning(f"No bins found for target region '{name}' – skipping.")
            continue

        rpca = _fit_regional_pca(
            ref_matrix_corrected,
            bin_mask,
            variance_range=regional_pca_variance_range,
        )
        if rpca is None:
            continue

        regional_models.append({
            "name":       name,
            "chrom":      chrom,
            "start":      reg_start,
            "end":        reg_end,
            "bin_mask":   bin_mask,
            "pca_mean":   rpca["mean"],
            "pca_comps":  rpca["components"],
            "pca_var":    rpca["var_ratio"],
        })
        logger.info(
            f"  Regional PCA '{name}': {n_region_bins} bins, "
            f"{len(rpca['var_ratio'])} components"
        )

    logger.info(f"Saving reference panel to {output_path} …")

    # Serialise regional models into flat arrays for NPZ storage
    reg_names  = np.array([r["name"]  for r in regional_models], dtype=object)
    reg_chroms = np.array([r["chrom"] for r in regional_models], dtype=object)
    reg_starts = np.array([r["start"] for r in regional_models], dtype=np.int64)
    reg_ends   = np.array([r["end"]   for r in regional_models], dtype=np.int64)
    reg_masks  = np.vstack([r["bin_mask"] for r in regional_models]) if regional_models else np.empty((0, n_bins), dtype=bool)
    reg_pca_means  = np.array([r["pca_mean"]  for r in regional_models], dtype=object)
    reg_pca_comps  = np.array([r["pca_comps"] for r in regional_models], dtype=object)
    reg_pca_vars   = np.array([r["pca_var"]   for r in regional_models], dtype=object)

    save_npz(
        output_path,
        {
            "bin_info":              ref_bins,
            "bin_means":             bin_means,
            "bin_stds":              bin_stds,
            "chroms":                np.array(ref_chroms, dtype=object),
            "bin_size":              np.array(ref_bin_size),
            "n_samples":             np.array(len(samples)),
            "sex_labels":            sex_labels,
            "gmm_means":             gmm_means,
            "gmm_stds":              gmm_stds,
            "gmm_cutoff":            np.array(gmm_cutoff),
            "global_pca_mean":       gpc_mean,
            "global_pca_components": gpc_components,
            "global_pca_var_ratio":  gpc_var_ratio,
            "reg_names":             reg_names,
            "reg_chroms":            reg_chroms,
            "reg_starts":            reg_starts,
            "reg_ends":              reg_ends,
            "reg_masks":             reg_masks,
            "reg_pca_means":         reg_pca_means,
            "reg_pca_comps":         reg_pca_comps,
            "reg_pca_vars":          reg_pca_vars,
            "ref_matrix":            ref_matrix_corrected,
        },
    )

    logger.info(
        f"Reference panel built: {len(samples)} samples, "
        f"{n_bins} bins, {len(regional_models)} target regions."
    )
    return output_path
