"""
gxcnv.cli
=========
Command-line interface for gxcnv.

Sub-commands
------------
    convert   BAM/CRAM → NPZ sample file
    newref    Build reference panel from NPZ files
    predict   Run hybrid CNV prediction
    plot      Generate plots from existing output files
"""

import argparse
import sys
import os
from . import __version__


def cmd_convert(args):
    from .convert import bam_to_npz
    bam_to_npz(
        bam_path=args.bam,
        output_path=args.output,
        bin_size=args.bin_size,
        blacklist_bed=args.blacklist,
        reference_fasta=args.reference,
        min_mapq=args.min_mapq,
        chroms=args.chroms.split(",") if args.chroms else None,
    )


def cmd_newref(args):
    from .newref import build_reference
    npz_paths = []
    for item in args.inputs:
        if os.path.isdir(item):
            npz_paths += sorted(
                [os.path.join(item, f) for f in os.listdir(item)
                 if f.endswith(".npz")]
            )
        else:
            npz_paths.append(item)

    if not npz_paths:
        print("ERROR: No NPZ files found.", file=sys.stderr)
        sys.exit(1)

    build_reference(
        npz_paths=npz_paths,
        output_path=args.output,
        global_pca_variance=args.pca_variance,
    )


def cmd_predict(args):
    from .predict import predict
    from .plot import plot_all

    result = predict(
        sample_npz_path=args.sample,
        reference_npz_path=args.reference,
        output_prefix=args.output,
        thresh_z=args.thresh_z,
        thresh_p=args.thresh_p,
        fetal_fraction=args.fetal_fraction,
        cbs_min_bins=args.cbs_min_bins,
        cbs_p_threshold=args.cbs_p,
    )

    if not args.no_plot:
        sample_name = os.path.basename(args.output)
        plot_all(
            output_prefix=args.output,
            sample_name=sample_name,
            sex=result["sex"],
            thresh_p=args.thresh_p,
        )


def cmd_plot(args):
    from .plot import plot_all
    plot_all(
        output_prefix=args.prefix,
        sample_name=args.sample_name or os.path.basename(args.prefix),
        sex=args.sex,
        thresh_p=args.thresh_p,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="gxcnv",
        description=f"gxcnv v{__version__} – Hybrid sWGS CNV Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Convert BAM to NPZ
  gxcnv convert sample.bam sample.npz --bin-size 100000

  # 2. Build reference panel
  gxcnv newref ref1.npz ref2.npz ref3.npz -o reference.npz

  # 3. Predict CNVs
  gxcnv predict sample.npz reference.npz -o results/SAMPLE001

  # 4. Re-generate plots only
  gxcnv plot results/SAMPLE001
""",
    )
    parser.add_argument("--version", action="version", version=f"gxcnv {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ convert
    p_conv = subparsers.add_parser("convert", help="Convert BAM/CRAM to NPZ")
    p_conv.add_argument("bam",    help="Input BAM or CRAM file")
    p_conv.add_argument("output", help="Output NPZ file path")
    p_conv.add_argument("--bin-size",  type=int, default=100_000,
                        metavar="BP",  help="Bin size in bp (default: 100000)")
    p_conv.add_argument("--blacklist", default=None,
                        metavar="BED", help="BED file with regions to exclude")
    p_conv.add_argument("--reference", default=None,
                        metavar="FA",  help="Reference FASTA (required for CRAM)")
    p_conv.add_argument("--min-mapq",  type=int, default=1,
                        metavar="Q",   help="Minimum mapping quality (default: 1)")
    p_conv.add_argument("--chroms",    default=None,
                        metavar="LIST",help="Comma-separated chromosome list")
    p_conv.set_defaults(func=cmd_convert)

    # ------------------------------------------------------------------ newref
    p_ref = subparsers.add_parser("newref", help="Build reference panel")
    p_ref.add_argument("inputs", nargs="+",
                       help="NPZ files or directory containing NPZ files")
    p_ref.add_argument("-o", "--output", required=True,
                       help="Output reference NPZ path")
    p_ref.add_argument("--pca-variance", type=float, default=0.95,
                       metavar="F",
                       help="Cumulative variance for global PCA (default: 0.95)")
    p_ref.set_defaults(func=cmd_newref)

    # ----------------------------------------------------------------- predict
    p_pred = subparsers.add_parser("predict", help="Run CNV prediction")
    p_pred.add_argument("sample",    help="Sample NPZ file")
    p_pred.add_argument("reference", help="Reference panel NPZ file")
    p_pred.add_argument("-o", "--output", required=True,
                        help="Output prefix (e.g. results/SAMPLE001)")
    p_pred.add_argument("--thresh-z",   type=float, default=-3.0,
                        metavar="Z",
                        help="Track A Z-score threshold (default: -3.0)")
    p_pred.add_argument("--thresh-p",   type=float, default=0.05,
                        metavar="P",
                        help="Track B p-value threshold (default: 0.05)")
    p_pred.add_argument("--fetal-fraction", type=float, default=None,
                        metavar="FF",
                        help="Fetal fraction estimate 0–1 (optional)")
    p_pred.add_argument("--cbs-min-bins", type=int, default=5,
                        metavar="N",
                        help="Minimum bins per CBS segment (default: 5)")
    p_pred.add_argument("--cbs-p",    type=float, default=0.01,
                        metavar="P",
                        help="CBS split p-value threshold (default: 0.01)")
    p_pred.add_argument("--no-plot",  action="store_true",
                        help="Skip plot generation")
    p_pred.set_defaults(func=cmd_predict)

    # -------------------------------------------------------------------- plot
    p_plot = subparsers.add_parser("plot", help="Generate plots from output files")
    p_plot.add_argument("prefix",      help="Output prefix used in predict step")
    p_plot.add_argument("--sample-name", default=None,
                        help="Sample name for plot titles")
    p_plot.add_argument("--sex",       default="Unknown",
                        help="Predicted sex (M/F/Unknown)")
    p_plot.add_argument("--thresh-p",  type=float, default=0.05,
                        metavar="P",
                        help="Risk threshold for region plot (default: 0.05)")
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
