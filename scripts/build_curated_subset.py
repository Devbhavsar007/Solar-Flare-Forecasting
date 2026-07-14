#!/usr/bin/env python3
"""
build_curated_subset.py

Creates a curated subset of HEL1OS ZIP files by matching their observation windows
against GOES catalog flare events. Also includes a balanced sample of quiet-sun data.

STRICT TIMESTAMP HANDLING:
1. No silent fallbacks: Files with unparseable timestamps are logged and skipped.
2. Window overlap: A zip is included if its [start, end] window overlaps with a flare's [start, end].
3. UTC standard: HEL1OS filenames are treated as UTC, which matches the FITS internal MJD.
"""

import os
import glob
import re
import shutil
import argparse
import pandas as pd
from datetime import timedelta

def parse_goes_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses GOES catalog 'date', 'start', and 'end' strings into proper UTC Timestamps.
    Handles midnight boundary crossings.
    """
    # Create copies to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Basic parse: date + time string
    # E.g. date='20240101', start='2350' -> 2024-01-01 23:50:00 UTC
    try:
        df['start_time'] = pd.to_datetime(df['date'] + df['start'], format='%Y%m%d%H%M', utc=True)
        # End time initially assumes the same date
        df['end_time'] = pd.to_datetime(df['date'] + df['end'], format='%Y%m%d%H%M', utc=True)
        
        # Handle midnight crossings: if end_time < start_time, it ended the next day
        midnight_cross = df['end_time'] < df['start_time']
        df.loc[midnight_cross, 'end_time'] += pd.Timedelta(days=1)
        
        # Validate no NaNs introduced
        if df['start_time'].isna().any() or df['end_time'].isna().any():
            print(f"[WARNING] Some GOES catalog times could not be parsed and are NaT.")
            df = df.dropna(subset=['start_time', 'end_time'])
            
    except Exception as e:
        raise ValueError(f"Failed to parse GOES catalog times: {e}")
        
    return df

def get_hel1os_windows(base_dir: str) -> pd.DataFrame:
    """
    Scans HEL1OS zip files and parses their observation windows from the filename.
    Filename format: HLS_YYYYMMDD_HHMMSS_<duration>sec_lev1_V<version>.zip
    """
    zip_files = glob.glob(os.path.join(base_dir, "**", "HLS_*.zip"), recursive=True)
    if not zip_files:
        raise FileNotFoundError(f"No HEL1OS zip files found in {base_dir}")

    pattern = re.compile(r'HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V\d+')
    
    records = []
    failed_parses = 0
    
    for z in zip_files:
        filename = os.path.basename(z)
        match = pattern.match(filename)
        if not match:
            print(f"[WARNING] Unparseable filename format, skipping: {filename}")
            failed_parses += 1
            continue
            
        date_str, time_str, duration_str = match.groups()
        
        try:
            # Parse strictly as UTC to match FITS internal time
            start_time = pd.to_datetime(f"{date_str}{time_str}", format='%Y%m%d%H%M%S', utc=True)
            duration = int(duration_str)
            end_time = start_time + pd.Timedelta(seconds=duration)
            
            records.append({
                'filename': filename,
                'filepath': z,
                'start_time': start_time,
                'end_time': end_time,
                'duration_sec': duration
            })
        except Exception as e:
            print(f"[WARNING] Failed to parse timestamp for {filename}: {e}")
            failed_parses += 1
            continue

    if failed_parses > 0:
        print(f"[INFO] Skipped {failed_parses} files due to parse failures.")
        
    return pd.DataFrame(records)

def build_subset(hel1os_dir: str, catalog_path: str, out_dir: str, target_classes: list, quiet_sun_ratio: float = 1.0, dry_run: bool = False):
    """
    Cross-references HEL1OS windows against GOES flare windows and copies the matched files.
    """
    print("--- 1. Loading GOES Catalog ---")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"GOES catalog not found: {catalog_path}")
        
    goes_df = pd.read_parquet(catalog_path)
    
    # Filter to Aditya-L1 era (Sep 2023 onwards)
    goes_df = goes_df[goes_df['peak_time'] >= pd.Timestamp('2023-09-01', tz='UTC')]
    goes_df = parse_goes_times(goes_df)
    
    # Filter by requested flare classes (e.g., ['M', 'X'])
    if target_classes:
        goes_df = goes_df[goes_df['xray_class'].str[0].isin(target_classes)]
        
    print(f"Found {len(goes_df)} target flares ({','.join(target_classes)}) in Aditya-L1 era.")
    
    print("\n--- 2. Parsing HEL1OS Observation Windows ---")
    hls_df = get_hel1os_windows(hel1os_dir)
    print(f"Successfully parsed {len(hls_df)} HEL1OS zip files.")
    
    print("\n--- 3. Matching Overlaps ---")
    # We need to find HEL1OS files where [hls_start, hls_end] overlaps [flare_start, flare_end]
    # Overlap condition: (hls_start <= flare_end) AND (hls_end >= flare_start)
    
    matched_hls_indices = set()
    
    for _, flare in goes_df.iterrows():
        overlaps = hls_df[
            (hls_df['start_time'] <= flare['end_time']) & 
            (hls_df['end_time'] >= flare['start_time'])
        ]
        matched_hls_indices.update(overlaps.index.tolist())
        
    matched_flares_df = hls_df.loc[list(matched_hls_indices)].copy()
    print(f"Found {len(matched_flares_df)} HEL1OS files overlapping with target flares.")
    
    print("\n--- 4. Selecting Quiet-Sun Sample ---")
    # Quiet sun = files that DO NOT overlap with ANY flare in the catalog (even C or B class)
    # Load all flares to ensure absolute quietness
    all_flares = pd.read_parquet(catalog_path)
    all_flares = all_flares[all_flares['peak_time'] >= pd.Timestamp('2023-09-01', tz='UTC')]
    all_flares = parse_goes_times(all_flares)
    
    # Find files that overlap with ANY flare
    any_flare_indices = set()
    for _, flare in all_flares.iterrows():
        overlaps = hls_df[
            (hls_df['start_time'] <= flare['end_time']) & 
            (hls_df['end_time'] >= flare['start_time'])
        ]
        any_flare_indices.update(overlaps.index.tolist())
        
    # Strictly quiet files
    quiet_df = hls_df[~hls_df.index.isin(any_flare_indices)].copy()
    print(f"Found {len(quiet_df)} strictly quiet-sun HEL1OS files.")
    
    num_quiet_to_sample = int(len(matched_flares_df) * quiet_sun_ratio)
    if num_quiet_to_sample > len(quiet_df):
        num_quiet_to_sample = len(quiet_df)
        
    sampled_quiet_df = quiet_df.sample(n=num_quiet_to_sample, random_state=42)
    print(f"Sampled {len(sampled_quiet_df)} quiet-sun files for balance (ratio={quiet_sun_ratio}).")
    
    # Combine
    final_subset = pd.concat([matched_flares_df, sampled_quiet_df]).drop_duplicates(subset=['filename'])
    total_size_mb = sum([os.path.getsize(f) for f in final_subset['filepath']]) / (1024*1024)
    
    print(f"\n--- 5. Subset Summary ---")
    print(f"Total files: {len(final_subset)}")
    print(f"Total size:  {total_size_mb/1024:.2f} GB")
    
    print(f"\n--- 6. Copying Files ---")
    
    if dry_run:
        print("DRY RUN ONLY. No files were copied.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    for _, row in final_subset.iterrows():
        src = row['filepath']
        dst = os.path.join(out_dir, row['filename'])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            
    print(f"Done. Subset staged at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a curated subset of HEL1OS data.")
    parser.add_argument("--hel1os-dir", default=r"d:\Users\Nitro 5\Downloads\HEL1OS_Data")
    parser.add_argument("--catalog", default="data/raw/noaa_catalog.parquet")
    parser.add_argument("--out-dir", default=r"d:\solar_flare_subset\hel1os")
    parser.add_argument("--classes", nargs="+", default=["M", "X"], help="Flare classes to include (e.g. M X)")
    parser.add_argument("--quiet-ratio", type=float, default=1.0, help="Ratio of quiet-sun files to flare files")
    parser.add_argument("--dry-run", action="store_true", help="Print stats but do not copy files")
    
    args = parser.parse_args()
    build_subset(args.hel1os_dir, args.catalog, args.out_dir, args.classes, args.quiet_ratio, args.dry_run)
