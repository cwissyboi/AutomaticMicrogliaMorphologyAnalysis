"""
combine_results.py

Combines the per-scan CSV files produced by the SLURM array job into a
single CSV file.

Usage (from initial_pipeline/):
    python combine_results.py

Optional arguments:
    --input_pattern   Glob pattern for input CSVs
                      (default: morphology/morphology_outputs/pipeline_run_*.csv)
    --output_path     Path for the combined output CSV
                      (default: morphology/morphology_outputs/all_scans_combined.csv)
"""

import argparse
import glob
import sys
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Combine per-scan morphology CSVs")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="morphology/morphology_outputs/pipeline_run_*.csv",
        help="Glob pattern matching the per-scan CSV files to combine"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="morphology/morphology_outputs/all_scans_combined.csv",
        help="Path for the combined output CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(args.input_pattern))

    if not files:
        sys.exit(f"ERROR: No files matched pattern: {args.input_pattern}")

    print(f"Found {len(files)} CSV files to combine:")
    for f in files:
        print(f"  {f}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined.to_csv(args.output_path, index=False)

    print(f"\nCombined {len(files)} files -> {args.output_path}")
    print(f"Total rows : {len(combined)}")
    print(f"Scans      : {combined['scan_name'].nunique()}")


if __name__ == "__main__":
    main()
