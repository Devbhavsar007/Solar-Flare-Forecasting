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
    
    # Handle boundaries: GOES-15 pre-2017, GOES-16 2017 to 2025-04-06, GOES-19 from 2025-04-07
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    queries = []
    if start_dt < pd.Timestamp('2017-01-01'):
        sub_end = min(end_dt, pd.Timestamp('2016-12-31 23:59:59'))
        queries.append((start_dt, sub_end, 15))
    if end_dt >= pd.Timestamp('2017-01-01') and start_dt < pd.Timestamp('2025-04-07'):
        sub_start = max(start_dt, pd.Timestamp('2017-01-01'))
        sub_end = min(end_dt, pd.Timestamp('2025-04-06 23:59:59'))
        queries.append((sub_start, sub_end, 16))
    if end_dt >= pd.Timestamp('2025-04-07'):
        sub_start = max(start_dt, pd.Timestamp('2025-04-07'))
        queries.append((sub_start, end_dt, 19))
        
    from sunpy.net.dataretriever.sources.goes import XRSClient
    client = XRSClient()
    
    downloaded_files = []
    
    for sub_start, sub_end, sat_number in queries:
        res = client.search(
            a.Time(sub_start.strftime('%Y-%m-%d %H:%M:%S'), sub_end.strftime('%Y-%m-%d %H:%M:%S')),
            a.Instrument.xrs,
            a.Resolution("avg1m"),
            a.goes.SatelliteNumber(sat_number),
        )
        if len(res) == 0:
            print(f"No GOES data found for {sub_start} to {sub_end} (sat={sat_number}).")
            continue
            
        print(f"Found {len(res)} files for sat {sat_number}. Downloading...")
        files = Fido.fetch(res, path=output_dir, max_conn=2)
        if files:
            # Fido.fetch returns a parfive.Results object which is iterable
            downloaded_files.extend(list(files))
    
    if not downloaded_files:
        print("Failed to download any files or no data available.")
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
