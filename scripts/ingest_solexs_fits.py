"""
SoLEXS Batch FITS Ingest (scripts/ingest_solexs_fits.py)

Unzips all 767 SoLEXS zip files, reads SDD2 .lc.gz lightcurve FITS files
via read_solexs(), deduplicates timestamps within and across files,
validates schemas, and saves a unified parquet.

BUG 4 FIX:
  - Drops duplicate timestamps within each file before appending.
  - Drops duplicates across files after concatenation; logs count.
  - Asserts each DataFrame has exactly columns ['counts'] before concat.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import zipfile
import shutil
import pandas as pd
import numpy as np
import traceback
from pathlib import Path

from src.ingestion.fits_reader import read_solexs

SOLEXS_DATA_DIR = r"d:\Users\Nitro 5\Downloads\SOLEXS_Data"
STAGING_DIR = os.path.join("data", "raw", "solexs_fits")
OUTPUT_DIR = os.path.join("data", "raw", "solexs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "solexs_all.parquet")


def main():
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Find all zips
    zip_files = sorted(glob.glob(os.path.join(SOLEXS_DATA_DIR, "**", "*.zip"), recursive=True))
    print(f"Found {len(zip_files)} zip files in {SOLEXS_DATA_DIR}")

    if len(zip_files) == 0:
        print("ERROR: No zip files found. Exiting.")
        return

    # 2. Extract and read each file
    all_dfs = []
    schema_fail_files = []
    read_fail_files = []
    total_rows_before_intra_dedup = 0
    total_rows_after_intra_dedup = 0

    for i, zip_path in enumerate(zip_files):
        zip_name = os.path.basename(zip_path)

        # Extract SDD2 .lc.gz from zip
        lc_file = None
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for entry in zf.namelist():
                    if 'SDD2' in entry and entry.endswith('.lc.gz'):
                        zf.extract(entry, STAGING_DIR)
                        lc_file = os.path.join(STAGING_DIR, entry)
                        break
        except (zipfile.BadZipFile, OSError) as e:
            read_fail_files.append((zip_name, f"BadZip: {e}"))
            continue

        if lc_file is None:
            read_fail_files.append((zip_name, "No SDD2 .lc.gz found in zip"))
            continue

        # Read via fits_reader
        try:
            df = read_solexs(lc_file)
        except Exception as e:
            read_fail_files.append((zip_name, str(e)))
            # Clean up extracted file
            try:
                os.remove(lc_file)
            except OSError:
                pass
            continue

        # BUG 4 FIX (a): Assert schema — columns must be exactly ['counts', 'pradan_version']
        expected_cols = ['counts', 'pradan_version']
        if list(df.columns) != expected_cols:
            schema_fail_files.append((zip_name, f"Got columns {list(df.columns)}, expected {expected_cols}"))
            continue

        # BUG 4 FIX (b): Drop duplicate timestamps within this file
        rows_before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        rows_after = len(df)

        total_rows_before_intra_dedup += rows_before
        total_rows_after_intra_dedup += rows_after

        all_dfs.append(df)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(zip_files):
            print(f"  Processed {i+1}/{len(zip_files)} zips "
                  f"({len(all_dfs)} OK, {len(read_fail_files)} failed)")

        # Clean up extracted file to save disk space
        try:
            os.remove(lc_file)
        except OSError:
            pass

    if not all_dfs:
        print("ERROR: No DataFrames produced. Check read failures above.")
        return

    # 3. Concatenate all DataFrames
    print(f"\nConcatenating {len(all_dfs)} DataFrames...")
    combined = pd.concat(all_dfs).sort_index()

    # BUG 4 FIX (c): Drop duplicates across files
    rows_before_cross_dedup = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    rows_after_cross_dedup = len(combined)
    cross_dedup_removed = rows_before_cross_dedup - rows_after_cross_dedup

    # 4. Save to parquet
    combined.to_parquet(OUTPUT_FILE)

    # 5. Report
    print(f"\n{'='*60}")
    print(f"INGEST REPORT")
    print(f"{'='*60}")
    print(f"Total zip files found:           {len(zip_files)}")
    print(f"Files successfully read:         {len(all_dfs)}")
    print(f"Files failed (read/extract):     {len(read_fail_files)}")
    print(f"Files failed (schema mismatch):  {len(schema_fail_files)}")
    print(f"")
    print(f"Total rows (before intra-file dedup): {total_rows_before_intra_dedup:,}")
    print(f"Total rows (after intra-file dedup):  {total_rows_after_intra_dedup:,}")
    print(f"Intra-file duplicates removed:        {total_rows_before_intra_dedup - total_rows_after_intra_dedup:,}")
    print(f"")
    print(f"Total rows (before cross-file dedup): {rows_before_cross_dedup:,}")
    print(f"Total rows (after cross-file dedup):  {rows_after_cross_dedup:,}")
    print(f"Cross-file duplicates removed:        {cross_dedup_removed:,}")
    print(f"")
    print(f"Final time range: {combined.index.min()} to {combined.index.max()}")
    print(f"Output saved to:  {OUTPUT_FILE}")
    print(f"{'='*60}")

    if read_fail_files:
        print(f"\n--- Read/Extract Failures ({len(read_fail_files)}) ---")
        for fname, reason in read_fail_files[:20]:
            print(f"  {fname}: {reason}")
        if len(read_fail_files) > 20:
            print(f"  ... and {len(read_fail_files) - 20} more")

    if schema_fail_files:
        print(f"\n--- Schema Mismatch Failures ({len(schema_fail_files)}) ---")
        for fname, reason in schema_fail_files:
            print(f"  {fname}: {reason}")

    # Cleanup staging dir
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
        print("\nCleaned up staging directory.")


if __name__ == "__main__":
    main()
