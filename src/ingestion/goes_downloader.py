"""
Downloads GOES XRS 1-minute flux data using sunpy.net.Fido.
"""

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
import warnings

# Suppress sunpy warnings about metadata
warnings.filterwarnings("ignore", category=UserWarning, module="sunpy")

def download_goes_xrs(start_date: str, end_date: str, output_dir: str = "data/raw/goes"):
    """
    Download GOES XRS 1-minute flux data and save to parquet.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching GOES XRS data from {start_date} to {end_date}...")
    
    # Determine primary satellite based on start year (GOES-15 pre-2017, GOES-16 2017+)
    start_dt = pd.Timestamp(start_date)
    sat_number = 16 if start_dt.year >= 2017 else 15
    
    result = Fido.search(
        a.Time(start_date, end_date),
        a.Instrument.xrs,
        a.Resolution("avg1m"),
        a.goes.SatelliteNumber(sat_number),
    )
    
    if len(result) == 0:
        print("No GOES data found for the specified period.")
        return
        
    print(f"Found {len(result[0])} files. Downloading...")
    # NOAA servers often timeout on concurrent downloads of large GOES-16+ NetCDF files.
    # We restrict max_conn to 2 to prevent "Timeout on reading data from socket".
    downloaded_files = Fido.fetch(result, path=output_dir, max_conn=2)
    
    if not downloaded_files:
        print("Failed to download files.")
        return
        
    print("Parsing FITS files...")
    ts = TimeSeries(downloaded_files, concatenate=True)
    
    # Extract dataframe
    df = ts.to_dataframe()
    
    # Drop rows where xrsa and xrsb are NaN
    df = df.dropna(subset=['xrsa', 'xrsb'], how='all')
    
    # Resample to strict 1-minute cadence
    df = df.resample('1min').mean()
    
    # Rename columns to standardized format
    df = df.rename(columns={'xrsa': 'xrs_a', 'xrsb': 'xrs_b'})
    
    # Check for gaps > 1 hour
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
    
    start_str = pd.Timestamp(start_date).strftime('%Y%m%d')
    end_str = pd.Timestamp(end_date).strftime('%Y%m%d')
    out_file = os.path.join(output_dir, f"goes_xrs_{start_str}_{end_str}.parquet")
    
    df.to_parquet(out_file)
    
    print(f"\n--- GOES Download Complete ---")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Rows:       {len(df)}")
    print(f"Gaps > 1h:  {len(gaps)}")
    print(f"Saved to:   {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GOES XRS data.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Download last N days")
    
    args = parser.parse_args()
    
    if args.days:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
    elif args.start and args.end:
        start_str = args.start
        end_str = args.end
    else:
        print("Error: Must provide either --days or both --start and --end.")
        parser.print_help()
        exit(1)
        
    download_goes_xrs(start_str, end_str)
