# gxcnv

**Hybrid sWGS CNV Analysis Algorithm**

gxcnv is an independent, from-scratch implementation of a hybrid Copy Number Variation (CNV) detection engine for shallow Whole Genome Sequencing (sWGS) data, designed for clinical NIPT applications.

It combines the strengths of two published approaches:

- **Track A** – Whole-genome Z-score normalisation with Circular Binary Segmentation (CBS), inspired by the WisecondorX methodology (Raman et al., *NAR* 2019)
- **Track B** – Regional PCA denoising with Laplace-smoothed Mahalanobis distance scoring, inspired by the BinDel methodology (Säks et al., *Bioinformatics* 2024)

A final **AND-gate dual-track decision** is applied: a region is called `HIGH_RISK` only when both Track A and Track B independently exceed their respective thresholds, dramatically reducing false positives.

> **Note:** gxcnv is an original implementation. No source code from WisecondorX or BinDel has been copied or adapted. The mathematical principles are derived from the published literature.

---

## Features

| Command | Description |
|---|---|
| `gxcnv convert` | BAM/CRAM → compressed NPZ sample file with GC correction |
| `gxcnv newref` | Build reference panel with GMM sex prediction + dual PCA |
| `gxcnv predict` | Hybrid dual-track CNV prediction with CBS segmentation |
| `gxcnv plot` | Publication-quality genome-wide, region risk, and QC plots |

---

## Installation

```bash
git clone https://github.com/genolyx/gx-cnv.git
cd gx-cnv
pip install -e .
```

### Requirements

- Python ≥ 3.8
- numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, pysam

---

## Quick Start

### Step 1 — Convert BAM to NPZ

```bash
gxcnv convert sample.bam sample.npz --bin-size 100000
```

| Option | Default | Description |
|---|---|---|
| `--bin-size` | `100000` | Bin size in base pairs |
| `--min-mapq` | `1` | Minimum mapping quality |
| `--blacklist` | *(none)* | BED file of regions to exclude |

---

### Step 2 — Build reference panel

```bash
# From individual NPZ files
gxcnv newref ref1.npz ref2.npz ref3.npz -o reference.npz

# Or from a directory
gxcnv newref /path/to/ref_npz_dir/ -o reference.npz
```

> **Recommendation:** Use ≥ 50 normal female samples for a robust reference panel. Mixed-sex panels are supported; sex is predicted automatically via GMM.

| Option | Default | Description |
|---|---|---|
| `-o` / `--output` | required | Output reference NPZ path |
| `--pca-variance` | `0.95` | Cumulative variance threshold for global PCA |
| `--reg-min-bins` | `2` | Minimum bins required to model a target region |

---

### Step 3 — Predict CNVs

```bash
gxcnv predict sample.npz reference.npz -o results/SAMPLE001
```

| Option | Default | Description |
|---|---|---|
| `-o` / `--output` | required | Output file prefix |
| `--thresh-z` | `-3.0` | Track A Z-score threshold |
| `--thresh-p` | `0.05` | Track B p-value threshold |
| `--fetal-fraction` | *(auto)* | Fetal fraction estimate (0–1) |

---

### Step 4 — Re-generate plots only

```bash
gxcnv plot results/SAMPLE001 --sex F
```

---

## Output Files

All output files use the **gxcnv-native TSV format**: plain tab-separated text with `##`-prefixed meta-header lines at the top, followed by a single `#`-prefixed column-header line.

### Meta-header block (common to all files)

```
##gxcnv_version=0.1.0
##generated=2025-04-18T12:00:00Z
##algorithm=hybrid_dual_track_cnv
##sample=sample.npz
##reference=reference.npz
##predicted_sex=F
##fetal_fraction=NA
##thresh_z=-3.0
##thresh_p=0.05
##bin_size=100000
##n_ref_samples=50
```

---

### `*_bins.tsv` — Per-bin table

One row per genomic bin across the entire genome.

| Column | Type | Description |
|---|---|---|
| `chrom` | str | Chromosome name (e.g. `chr1`) |
| `start` | int | Bin start coordinate (0-based) |
| `end` | int | Bin end coordinate |
| `gc_fraction` | float | GC content fraction of the bin |
| `obs_norm` | float | GC-corrected, normalised observed read count |
| `exp_norm` | float | Expected value from reference panel |
| `z_score` | float \| `NA` | Track A Z-score; `NA` for masked bins |
| `obs_exp_ratio` | float | Observed / expected ratio |
| `flag` | str | `NORMAL` \| `DEL_CANDIDATE` \| `DUP_CANDIDATE` \| `MASKED` |

**Flag thresholds:** `DEL_CANDIDATE` when `z_score < −3`; `DUP_CANDIDATE` when `z_score > 3`.

---

### `*_segments.tsv` — CBS segment table

One row per segment produced by Circular Binary Segmentation.

| Column | Type | Description |
|---|---|---|
| `chrom` | str | Chromosome name |
| `start` | int | Segment start coordinate |
| `end` | int | Segment end coordinate |
| `n_bins` | int | Number of bins in segment |
| `mean_z` | float | Mean Z-score across segment |
| `copy_number_est` | float | Estimated copy number |
| `segment_type` | str | `DELETION` \| `DUPLICATION` \| `NEUTRAL` |

**Segment type thresholds:** `DELETION` when `mean_z < −1.5`; `DUPLICATION` when `mean_z > 1.5`.

---

### `*_calls.tsv` — Dual-track confirmed CNV calls

Only regions that pass **both** Track A and Track B thresholds (`HIGH_RISK`) are written here.

| Column | Type | Description |
|---|---|---|
| `chrom` | str | Chromosome name |
| `start` | int | Region start coordinate |
| `end` | int | Region end coordinate |
| `region_name` | str | Clinical region identifier (e.g. `DiGeorge_22q11`) |
| `track_a_mean_z` | float | Mean Z-score across region bins (Track A) |
| `track_b_mahal_dist` | float | Mahalanobis distance of region score (Track B) |
| `track_b_pvalue` | float | Chi-square p-value (df=1) from Mahalanobis distance |
| `risk_pct` | float | Risk percentage: `(1 − p_value) × 100` |
| `track_a_result` | str | `PASS` if `mean_z < thresh_z`, else `FAIL` |
| `track_b_result` | str | `PASS` if `p_value < thresh_p`, else `FAIL` |
| `dual_call` | str | `HIGH_RISK` (both PASS) |

---

### `*_regions.tsv` — All target regions summary

One row per clinical target region, regardless of call status. Use this file for downstream review of all regions.

| Column | Type | Description |
|---|---|---|
| `chrom` | str | Chromosome name |
| `start` | int | Region start coordinate |
| `end` | int | Region end coordinate |
| `region_name` | str | Clinical region identifier |
| `track_a_mean_z` | float | Mean Z-score across region bins (Track A) |
| `track_b_mahal_dist` | float | Mahalanobis distance (Track B) |
| `track_b_pvalue` | float | Chi-square p-value (Track B) |
| `risk_pct` | float | Risk percentage |
| `track_a_result` | str | `PASS` \| `FAIL` |
| `track_b_result` | str | `PASS` \| `FAIL` |
| `dual_call` | str | `HIGH_RISK` \| `LOW_RISK` |

---

### `*_qcmetrics.tsv` — Run-level QC metrics

Key-value format (`#metric\tvalue`).

| Metric | Description |
|---|---|
| `total_reads` | Total mapped reads in sample |
| `predicted_sex` | GMM-predicted sex (`F` or `M`) |
| `fetal_fraction` | Fetal fraction estimate (`NA` if not provided) |
| `n_bins_total` | Total number of bins |
| `n_bins_valid` | Bins with finite Z-score (not masked) |
| `pct_bins_valid` | Percentage of valid bins |
| `median_z_score` | Genome-wide median Z-score |
| `mad_z_score` | Median absolute deviation of Z-scores |
| `n_del_candidate_bins` | Bins with `z_score < −3` |
| `n_dup_candidate_bins` | Bins with `z_score > 3` |
| `n_high_risk_regions` | Number of `HIGH_RISK` dual-track calls |

---

### `*_sex.txt` — Predicted sex

Single key-value line after meta-headers.

```
##gxcnv_version=0.1.0
...
#metric	value
predicted_sex	F
```

---

### Plot files

| File | Description |
|---|---|
| `*_genome.png` | Genome-wide Z-score plot with CBS segments overlaid, colour-coded by deviation |
| `*_regions.png` | Horizontal bar chart of risk % per clinical target region with dual-track pass/fail indicators |
| `*_qc.png` | Z-score distribution histogram + per-chromosome median Z-score bar chart |

---

## Algorithm Overview

### Sex Prediction (GMM)

A 2-component Gaussian Mixture Model is fitted to the chrY read-fraction distribution of the reference cohort. The decision boundary is the local minimum between the two Gaussian components.

For male samples, chrX and chrY bin counts are multiplied by 2 to place them on the same diploid scale as autosomes before all downstream analyses.

### Track A — Z-score + CBS

For each bin *b*:

```
Z(b) = ( x(b) − μ_ref(b) ) / σ_ref(b)
```

Circular Binary Segmentation (CBS) is applied per chromosome using a Brownian-bridge approximation for the split p-value. Each segment is assigned a copy-number estimate:

```
CN(seg) = 2 + mean_Z(seg) / 3          (without fetal fraction)
CN(seg) = 2 + mean_Z(seg) × 2 / FF     (with fetal fraction FF)
```

### Track B — Regional Mahalanobis

For each clinical target region *r*:

1. **Regional PCA denoising** — noise components explaining 5–50% of variance are removed.
2. **Laplace-smoothed directional score:**

```
score(r) = ( Σ Z_bin / √n  +  1/n )  /  ( count(obs > μ_ref)  +  2/n )
```

3. **Mahalanobis distance** of `score(r)` against the reference score distribution.
4. **Chi-square p-value** (df = 1).

### Dual-Track AND-Gate Decision

```
HIGH_RISK  ←  mean_Z(r) < thresh_Z   AND   p_value(r) < thresh_p
LOW_RISK   ←  otherwise
```

Default thresholds: `thresh_Z = −3.0`, `thresh_p = 0.05`

---

## Clinical Target Regions

gxcnv includes 15 clinically relevant microdeletion/microduplication regions by default (hg38 coordinates):

| Region | Locus | Syndrome |
|---|---|---|
| `DiGeorge_22q11` | 22q11.2 | DiGeorge / velocardiofacial syndrome |
| `Williams_7q11` | 7q11.23 | Williams-Beuren syndrome |
| `Angelman_15q11` | 15q11-q13 | Angelman syndrome |
| `PraderWilli_15q11` | 15q11-q13 | Prader-Willi syndrome |
| `Wolf_4p16` | 4p16.3 | Wolf-Hirschhorn syndrome |
| `CriDuChat_5p15` | 5p15.2 | Cri-du-Chat syndrome |
| `NF1_17q11` | 17q11.2 | Neurofibromatosis type 1 |
| `Smith_17p11` | 17p11.2 | Smith-Magenis syndrome |
| `Langer_Giedion_8q24` | 8q24.1 | Langer-Giedion syndrome |
| `Miller_Dieker_17p13` | 17p13.3 | Miller-Dieker lissencephaly |
| `CHARGE_8q12` | 8q12 | CHARGE syndrome |
| `Kabuki_12q13` | 12q13 | Kabuki syndrome |
| `Sotos_5q35` | 5q35 | Sotos syndrome |
| `Rubinstein_16p13` | 16p13.3 | Rubinstein-Taybi syndrome |
| `Potocki_Lupski_17p11` | 17p11.2 | Potocki-Lupski syndrome |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

1. Raman L, et al. (2019). *Clinical validation of copy number variant detection from shallow whole-genome sequencing applied to non-invasive prenatal testing.* Nucleic Acids Research. [PMC6393301](https://pmc.ncbi.nlm.nih.gov/articles/PMC6393301/)

2. Säks M, et al. (2024). *BinDel: a tool for detecting microdeletion syndromes from non-invasive prenatal testing shallow whole-genome sequencing data.* Bioinformatics. [PMC11893519](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893519/)
