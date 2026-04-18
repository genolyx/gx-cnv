"""
Synthetic end-to-end test for gxcnv.

Creates fake NPZ files (no real BAM needed) and runs:
  newref → predict → plot
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gxcnv.utils import save_npz
from gxcnv.newref import build_reference
from gxcnv.predict import predict
from gxcnv.plot import plot_all

# ---------------------------------------------------------------------------
# Synthetic bin layout: 24 chromosomes, 50 bins each
# ---------------------------------------------------------------------------
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
BIN_SIZE = 100_000
BINS_PER_CHROM = 50
N_BINS = len(CHROMS) * BINS_PER_CHROM
N_REF  = 20   # reference samples

rng = np.random.default_rng(42)


def _make_bins():
    chrom_idx, starts, ends, gc = [], [], [], []
    for c_idx, chrom in enumerate(CHROMS):
        for b in range(BINS_PER_CHROM):
            chrom_idx.append(c_idx)
            starts.append(b * BIN_SIZE)
            ends.append((b + 1) * BIN_SIZE)
            gc.append(rng.uniform(0.35, 0.65))
    return np.column_stack([chrom_idx, starts, ends, gc]).astype(float)


def _make_sample_npz(bins, corrected, path):
    mask = np.ones(N_BINS, dtype=bool)
    save_npz(path, {
        "bins":        bins,
        "counts":      (corrected * 100).astype(int),
        "corrected":   corrected,
        "mask":        mask,
        "chroms":      np.array(CHROMS, dtype=object),
        "bin_size":    np.array(BIN_SIZE),
        "total_reads": np.array(int(corrected.sum() * 100)),
    })


def make_reference_npz_files(tmpdir):
    bins = _make_bins()
    paths = []
    for i in range(N_REF):
        # Normal female: chrY ≈ 0
        corrected = rng.normal(1.0, 0.05, N_BINS).clip(0)
        # chrY near zero for females
        chrom_arr = np.array(CHROMS)
        is_chrY = np.repeat(chrom_arr == "chrY", BINS_PER_CHROM)
        corrected[is_chrY] = rng.normal(0.01, 0.005, is_chrY.sum()).clip(0)
        path = os.path.join(tmpdir, f"ref_{i:03d}.npz")
        _make_sample_npz(bins, corrected, path)
        paths.append(path)
    return paths, bins


def make_affected_sample_npz(tmpdir, bins):
    """Sample with a deletion in chr22 bins 0-9 (DiGeorge region)."""
    corrected = rng.normal(1.0, 0.05, N_BINS).clip(0)
    chrom_arr = np.array(CHROMS)
    is_chr22 = np.repeat(chrom_arr == "chr22", BINS_PER_CHROM)
    idx_chr22 = np.where(is_chr22)[0]
    # Simulate ~50% deletion in first 10 bins of chr22
    corrected[idx_chr22[:10]] *= 0.5
    path = os.path.join(tmpdir, "affected_sample.npz")
    _make_sample_npz(bins, corrected, path)
    return path


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[TEST] Working directory: {tmpdir}")

        # 1. Build reference
        ref_paths, bins = make_reference_npz_files(tmpdir)
        ref_npz = os.path.join(tmpdir, "reference.npz")
        build_reference(ref_paths, ref_npz)
        assert os.path.exists(ref_npz), "Reference NPZ not created"
        print("[TEST] Reference panel built OK")

        # 2. Predict on affected sample
        sample_npz = make_affected_sample_npz(tmpdir, bins)
        out_prefix = os.path.join(tmpdir, "SAMPLE_TEST")
        result = predict(
            sample_npz_path=sample_npz,
            reference_npz_path=ref_npz,
            output_prefix=out_prefix,
            thresh_z=-3.0,
            thresh_p=0.05,
        )
        assert result is not None
        print(f"[TEST] Predicted sex: {result['sex']}")
        print(f"[TEST] Segments: {len(result['segments'])}")
        print(f"[TEST] HIGH RISK regions: "
              f"{sum(1 for a in result['aberrations'] if a['call'] == 'HIGH_RISK')}")

        # Check output files exist
        for suffix in ["_bins.bed", "_segments.bed", "_aberrations.bed",
                       "_regions.bed", "_statistics.txt", "_gender.txt"]:
            path = out_prefix + suffix
            assert os.path.exists(path), f"Missing output: {path}"
            print(f"[TEST] {suffix} OK ({os.path.getsize(path)} bytes)")

        # 3. Generate plots
        plot_all(out_prefix, sample_name="SAMPLE_TEST", sex=result["sex"])
        for suffix in ["_genome.png", "_regions.png", "_qc.png"]:
            path = out_prefix + suffix
            assert os.path.exists(path), f"Missing plot: {path}"
            print(f"[TEST] {suffix} OK ({os.path.getsize(path)} bytes)")

        print("\n[TEST] All checks passed!")


if __name__ == "__main__":
    test_full_pipeline()
