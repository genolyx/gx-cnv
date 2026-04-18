# gxcnv

**Hybrid sWGS CNV Analysis Algorithm**

gxcnv is an independent, from-scratch implementation of a hybrid Copy Number Variation (CNV) detection engine for shallow Whole Genome Sequencing (sWGS) data, designed for clinical NIPT applications.

It combines the strengths of two published approaches:
- **Track A** – Whole-genome Z-score normalisation with Circular Binary Segmentation (CBS), inspired by the WisecondorX methodology (Raman et al., *NAR* 2019)
- **Track B** – Regional PCA denoising with Laplace-smoothed Mahalanobis distance scoring, inspired by the BinDel methodology (Säks et al., *Bioinformatics* 2024)

A final **AND-gate dual-track decision** is applied: a region is called HIGH RISK only when both Track A and Track B independently exceed their respective thresholds, dramatically reducing false positives.

> **Note:** gxcnv is an original implementation. No source code from WisecondorX or BinDel has been copied or adapted. The mathematical principles are derived from the published literature.

---

## Features

| Feature | Description |
|---|---|
| `convert` | BAM/CRAM → compressed NPZ sample file with GC correction |
| `newref` | Build reference panel with GMM sex prediction + dual PCA |
| `predict` | Hybrid dual-track CNV prediction with CBS segmentation |
| `plot` | Publication-quality genome-wide, region risk, and QC plots |

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

### Step 1: Convert BAM to NPZ

```bash
gxcnv convert sample.bam sample.npz --bin-size 100000
```

### Step 2: Build reference panel

```bash
# From individual NPZ files
gxcnv newref ref1.npz ref2.npz ref3.npz -o reference.npz

# Or from a directory
gxcnv newref /path/to/ref_npz_dir/ -o reference.npz
```

Recommended: ≥ 50 normal female samples for a robust reference panel.

### Step 3: Predict CNVs

```bash
gxcnv predict sample.npz reference.npz -o results/SAMPLE001
```

This generates the following output files (gxcnv-native TSV format with `##`-prefixed meta-headers):

| File | Description |
|---|---|
| `SAMPLE001_bins.tsv` | Per-bin Z-scores, GC fraction, obs/exp ratios, and per-bin flags |
| `SAMPLE001_segments.tsv` | CBS segments with copy-number estimate and segment type |
| `SAMPLE001_calls.tsv` | HIGH RISK CNV calls confirmed by dual-track AND-gate |
| `SAMPLE001_regions.tsv` | All target regions: Track A/B scores, Mahalanobis distance, dual_call |
| `SAMPLE001_qcmetrics.tsv` | Run-level QC metrics (reads, valid bins, median Z, MAD) |
| `SAMPLE001_sex.txt` | Predicted sex with meta-headers |
| `SAMPLE001_genome.png` | Genome-wide Z-score plot |
| `SAMPLE001_regions.png` | Region risk bar chart |
| `SAMPLE001_qc.png` | QC summary panel |

All TSV files begin with `##gxcnv_version`, `##generated`, and `##algorithm` meta-header lines, followed by a `#`-prefixed column-header line.

### Step 4: Re-generate plots only

```bash
gxcnv plot results/SAMPLE001 --sex F
```

---

## Algorithm Overview

### Sex Prediction (GMM)

A 2-component Gaussian Mixture Model is fitted to the chrY read-fraction distribution of the reference cohort. The decision boundary is the local minimum between the two Gaussian components.

For male samples, chrX and chrY bin counts are multiplied by 2 to place them on the same diploid scale as autosomes.

### Track A – Z-score + CBS

For each bin *b*:

```
Z(b) = (x(b) - μ_ref(b)) / σ_ref(b)
```

Circular Binary Segmentation is applied per chromosome using a Brownian-bridge approximation for the split p-value.

### Track B – Regional Mahalanobis

For each clinical target region *r*:

1. Regional PCA noise removal (5–50% variance components)
2. Laplace-smoothed directional score:

```
score(r) = (Σ Z_bin / √n  +  1/n) / (count(x > μ_ref)  +  2/n)
```

3. Mahalanobis distance vs. reference score distribution
4. Chi-square p-value (df = 1)

### Dual-Track Decision

```
HIGH RISK  ←  mean_Z(r) < thresh_Z  AND  p_value(r) < thresh_p
LOW RISK   ←  otherwise
```

Default thresholds: `thresh_Z = -3.0`, `thresh_p = 0.05`

---

## Clinical Target Regions

gxcnv includes 15 clinically relevant microdeletion/microduplication regions by default (hg38):

DiGeorge (22q11.2), Williams-Beuren (7q11.23), Angelman/Prader-Willi (15q11-q13), Wolf-Hirschhorn (4p16.3), Cri-du-Chat (5p15.2), NF1 (17q11.2), Smith-Magenis (17p11.2), Langer-Giedion (8q24.1), Miller-Dieker (17p13.3), CHARGE (8q12), Kabuki (12q13), Sotos (5q35), Rubinstein-Taybi (16p13.3), Potocki-Lupski (17p11.2).

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## References

1. Raman L, et al. (2019). *Clinical validation of copy number variant detection from shallow whole-genome sequencing applied to non-invasive prenatal testing.* Nucleic Acids Research. [PMC6393301](https://pmc.ncbi.nlm.nih.gov/articles/PMC6393301/)

2. Säks M, et al. (2024). *BinDel: a tool for detecting microdeletion syndromes from non-invasive prenatal testing shallow whole-genome sequencing data.* Bioinformatics. [PMC11893519](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893519/)
