"""
Orchestrator script for downloading multi-year GOES XRS data in monthly chunks.
Prevents memory exhaustion and API timeouts during large historical pulls.
"""

import argparse
import pandas as pd
import time
import sys
import os

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.goes_downloader import download_goes_xrs

def batch_download(start_year: int, end_year: int):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Create a list of month starts
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    failed_chunks = []
    
    print(f"Starting batch GOES download from {start_year} to {end_year}")
    print(f"Total monthly chunks to process: {len(months)}")
    
    for i, month_start in enumerate(months):
        month_end = month_start + pd.offsets.MonthEnd(1)
        
        start_str = month_start.strftime("%Y-%m-%d")
        end_str = month_end.strftime("%Y-%m-%d")
        
        start_str_compact = month_start.strftime('%Y%m%d')
        end_str_compact = month_end.strftime('%Y%m%d')
        out_file = os.path.join("data/raw/goes", f"goes_xrs_{start_str_compact}_{end_str_compact}.parquet")
        
        if os.path.exists(out_file):
            print(f"\n[{i+1}/{len(months)}] Skipping {start_str} to {end_str} (already exists)")
            continue
            
        print(f"\n[{i+1}/{len(months)}] Fetching {start_str} to {end_str}...")
        
        try:
            download_goes_xrs(start_str, end_str)
            # Sleep briefly to be nice to the NOAA API
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Failed to download {start_str} to {end_str}: {e}")
            failed_chunks.append((start_str, end_str, str(e)))
            
    if failed_chunks:
        print("\n--- BATCH COMPLETED WITH ERRORS ---")
        print(f"{len(failed_chunks)} chunks failed:")
        for chunk in failed_chunks:
            print(f"  {chunk[0]} to {chunk[1]}: {chunk[2]}")
    else:
        print("\n--- BATCH COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download GOES XRS data by month.")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (inclusive)")
    
    args = parser.parse_args()
    batch_download(args.start_year, args.end_year)
