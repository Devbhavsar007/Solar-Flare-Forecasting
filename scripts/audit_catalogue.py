#!/usr/bin/env python3
"""
Provenance audit tool for master_catalogue.csv.

Given a peak_time, prints the provenance fields so any misclassified
alert can be traced back to its exact source data and model version.

Usage:
    python scripts/audit_catalogue.py --peak_time "2024-02-22T10:33:00"
"""
import argparse
from pathlib import Path

import pandas as pd


CATALOGUE_PATH = Path("data/processed/master_catalogue.csv")

PROVENANCE_COLS = [
    "solexs_fits_path",
    "hel1os_fits_path",
    "model_version",
    "pipeline_run_id",
]


def audit(peak_time_str: str) -> None:
    """Look up provenance for events near the given peak_time."""
    if not CATALOGUE_PATH.exists():
        print(f"ERROR: {CATALOGUE_PATH} does not exist.")
        return

    df = pd.read_csv(CATALOGUE_PATH, parse_dates=["peak_time"])
    target = pd.Timestamp(peak_time_str)
    window = pd.Timedelta(minutes=2)

    mask = (df["peak_time"] >= target - window) & (
        df["peak_time"] <= target + window
    )
    matches = df.loc[mask]

    if matches.empty:
        print(f"No events found within ±2 min of {peak_time_str}")
        return

    print(f"Found {len(matches)} event(s) near {peak_time_str}:\n")
    for _, row in matches.iterrows():
        print(f"  peak_time:        {row['peak_time']}")
        print(f"  flare_class:      {row['flare_class']}")
        print(f"  source:           {row['source']}")
        print(f"  confidence:       {row['confidence']:.3f}")
        for col in PROVENANCE_COLS:
            print(f"  {col:20s} {row[col]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit provenance of master catalogue entries."
    )
    parser.add_argument(
        "--peak_time",
        required=True,
        help='ISO-format peak time, e.g. "2024-02-22T10:33:00"',
    )
    args = parser.parse_args()
    audit(args.peak_time)


if __name__ == "__main__":
    main()
