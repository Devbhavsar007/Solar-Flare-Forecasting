"""
Verification script for User Checks (Feb 2024 dates, row counts, and parquet describe).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import zipfile
import tempfile
import pandas as pd
import numpy as np
from astropy.io import fits

from src.ingestion.fits_reader import read_solexs

SOLEXS_DATA_DIR = r"d:\Users\Nitro 5\Downloads\SOLEXS_Data"
PARQUET_FILE = os.path.join("data", "raw", "solexs", "solexs_all.parquet")

def main():
    print("="*60)
    print("CHECK 1: EARLIEST DATE MATCH (Filename vs Header vs Computed)")
    print("="*60)
    zip_files = sorted(glob.glob(os.path.join(SOLEXS_DATA_DIR, "**", "*.zip"), recursive=True))
    
    if not zip_files:
        print("No zip files found.")
        return

    # Take the first 3 files (which should be the earliest based on sorting)
    earliest_zips = zip_files[:3]
    for zpath in earliest_zips:
        zname = os.path.basename(zpath)
        print(f"\nInspecting: {zname}")
        
        with zipfile.ZipFile(zpath, 'r') as zf:
            lc_entry = next((name for name in zf.namelist() if 'SDD2' in name and name.endswith('.lc.gz')), None)
            if not lc_entry:
                print("  No SDD2 .lc.gz found.")
                continue
                
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract(lc_entry, tmpdir)
                lc_file = os.path.join(tmpdir, lc_entry)
                
                # Get raw header DATE-OBS
                with fits.open(lc_file) as hdul:
                    hdr = hdul[0].header
                    date_obs = hdr.get("DATE-OBS", "Missing")
                    date_end = hdr.get("DATE-END", "Missing")
                    print(f"  FITS Header DATE-OBS : {date_obs}")
                    print(f"  FITS Header DATE-END : {date_end}")
                
                # Get computed timestamps via read_solexs
                df = read_solexs(lc_file)
                print(f"  Computed First Time  : {df.index.min()}")
                print(f"  Computed Last Time   : {df.index.max()}")

    print("\n" + "="*60)
    print("CHECK 2: MIN/MAX ROWS PER FILE (NAXIS2)")
    print("="*60)
    
    row_counts = []
    print(f"Scanning all {len(zip_files)} files for NAXIS2 (this will take ~30s)...")
    for i, zpath in enumerate(zip_files):
        with zipfile.ZipFile(zpath, 'r') as zf:
            lc_entry = next((name for name in zf.namelist() if 'SDD2' in name and name.endswith('.lc.gz')), None)
            if not lc_entry:
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    zf.extract(lc_entry, tmpdir)
                    lc_file = os.path.join(tmpdir, lc_entry)
                    hdr1 = fits.getheader(lc_file, ext=1)
                    rows = hdr1.get('NAXIS2', 0)
                    row_counts.append({'file': os.path.basename(zpath), 'rows': rows})
                except Exception as e:
                    print(f"Error reading {zpath}: {e}")
                    
        if (i+1) % 100 == 0:
            print(f"  Scanned {i+1}/{len(zip_files)}")
            
    if row_counts:
        rc_df = pd.DataFrame(row_counts)
        print(f"\nStats across {len(rc_df)} parsed files:")
        print(f"  Mean rows per file : {rc_df['rows'].mean():.1f}")
        print(f"  Median rows per file: {rc_df['rows'].median():.1f}")
        print(f"  Min rows per file  : {rc_df['rows'].min()} (File: {rc_df.loc[rc_df['rows'].idxmin(), 'file']})")
        print(f"  Max rows per file  : {rc_df['rows'].max()} (File: {rc_df.loc[rc_df['rows'].idxmax(), 'file']})")
        
        # Show histogram of row counts
        print("\nRow count distribution:")
        bins = [0, 10000, 40000, 80000, 86000, 86400, 100000]
        hist = pd.cut(rc_df['rows'], bins=bins).value_counts().sort_index()
        for interval, count in hist.items():
            print(f"  {interval}: {count} files")
            
        # Files with exactly 0 rows?
        zero_rows = rc_df[rc_df['rows'] == 0]
        if len(zero_rows) > 0:
            print(f"\nWARNING: {len(zero_rows)} files have 0 rows!")

    print("\n" + "="*60)
    print("CHECK 3: PARQUET .DESCRIBE() on COUNTS")
    print("="*60)
    
    if os.path.exists(PARQUET_FILE):
        pq_df = pd.read_parquet(PARQUET_FILE)
        print(f"Parquet loaded. Shape: {pq_df.shape}")
        
        # Force pandas to print all decimal places for floats in describe
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        print(pq_df.describe())
        
        # Check for negative counts or exactly zero
        neg_counts = (pq_df['counts'] < 0).sum()
        zero_counts = (pq_df['counts'] == 0).sum()
        print(f"\nNegative counts total: {neg_counts}")
        print(f"Zero counts total:     {zero_counts} ({zero_counts/len(pq_df):.2%} of data)")
        
        # Check 99th percentile and max
        p99 = pq_df['counts'].quantile(0.99)
        p999 = pq_df['counts'].quantile(0.999)
        print(f"\nPercentiles:")
        print(f"  99th  : {p99:.6f}")
        print(f"  99.9th: {p999:.6f}")
        print(f"  Max   : {pq_df['counts'].max():.6f}")
    else:
        print(f"ERROR: Parquet file not found at {PARQUET_FILE}")

if __name__ == "__main__":
    main()
