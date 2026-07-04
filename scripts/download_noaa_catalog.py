import requests
import pandas as pd
from pathlib import Path
from io import StringIO
import argparse

NOAA_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-flares/x-rays/goes/xrs/"
    "goes-xrs-report_{year}.txt"
)

def download_noaa_catalog(start_year: int = 2010,
                           end_year:   int = 2024,
                           out_path:   str = "data/raw/noaa_catalog.parquet"
                           ) -> pd.DataFrame:
    '''
    Download NOAA GOES flare catalog for start_year..end_year.
    Parses the fixed-width text format. Saves to parquet.
    WHY: merger.py's check_noaa_confirmed() needs this file.
         CI/staging use synthetic data so NOAA access must be
         optional (skip if URL unavailable - log warning, return empty df).
    '''
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            resp = requests.get(NOAA_URL.format(year=year), timeout=30)
            resp.raise_for_status()
            
            # Parse fixed-width: cols = [date, time, class, location, ...]
            df = pd.read_fwf(
                StringIO(resp.text), 
                skiprows=4,
                colspecs=[(0,8),(9,14),(15,21),(22,27),(28,30),(31,34),(35,40)],
                names=["date","start","peak","end","goes","xray_class","noaa_region"]
            )
            
            df["peak_time"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["peak"].astype(str),
                format="%Y%m%d %H%M", utc=True, errors="coerce"
            )
            df = df.dropna(subset=["peak_time"])
            frames.append(df)
            
        except Exception as exc:
            print(f"[WARNING] NOAA {year} unavailable: {exc}. Skipping.")
            
    if not frames:
        print("[WARNING] No NOAA data downloaded. check_noaa_confirmed() will return False for all events.")
        return pd.DataFrame()
        
    out = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"NOAA catalog: {len(out)} flares saved to {out_path}")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year",   type=int, default=2024)
    args = p.parse_args()
    download_noaa_catalog(args.start_year, args.end_year)
