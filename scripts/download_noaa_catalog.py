"""
Downloads NOAA GOES flare event catalog.

Data sources:
  - 2010-2016: NGDC text reports (goes-xrs-report_{year}.txt)
  - 2017+:     SWPC JSON API (xray-flares-7-day.json for recent,
               NCEI warehouse text files for historical)

Since NGDC stopped publishing annual text reports after 2016,
and the SWPC JSON only covers the last 7 days, we use SunPy's
HEK (Heliophysics Event Knowledgebase) client for 2017+ data
as the reliable, science-quality source.
"""

import requests
import pandas as pd
from pathlib import Path
from io import StringIO
import argparse

NGDC_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-flares/x-rays/goes/xrs/"
    "goes-xrs-report_{year}.txt"
)


def _parse_ngdc_text(year: int) -> pd.DataFrame:
    """Parse an NGDC fixed-width text report for a single year (2010-2016)."""
    resp = requests.get(NGDC_URL.format(year=year), timeout=30)
    resp.raise_for_status()

    df = pd.read_fwf(
        StringIO(resp.text),
        skiprows=4,
        colspecs=[(0, 8), (9, 14), (15, 21), (22, 27), (28, 30), (31, 34), (35, 40)],
        names=["date", "start", "peak", "end", "goes", "xray_class", "noaa_region"],
    )

    df["peak_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["peak"].astype(str),
        format="%Y%m%d %H%M",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["peak_time"])
    return df


def _fetch_hek_flares(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch flare events from the HEK (Heliophysics Event Knowledgebase)
    via SunPy for years where NGDC text files are unavailable.
    """
    try:
        from sunpy.net import attrs as a
        from sunpy.net import hek
    except ImportError:
        print("[WARNING] sunpy not installed. Cannot fetch HEK flare data.")
        return pd.DataFrame()

    client = hek.HEKClient()
    frames = []

    date_ranges = []
    for y in range(start_year, end_year + 1):
        date_ranges.append((f"{y}-01-01", f"{y+1}-01-01"))

    for date_range in date_ranges:
        try:
            print(f"Querying HEK for {date_range[0]} to {date_range[1]}...")
            result = client.search(
                a.Time(date_range[0], date_range[1]),
                a.hek.EventType("FL"),
                a.hek.FRM.Name == "SWPC",
            )
            if len(result) == 0:
                print(f"[WARNING] HEK returned 0 flares for {date_range}.")
                continue

            records = []
            for event in result:
                # SunPy v7 HEK client returns astropy.time.Time objects for dates
                start_str = str(event.get("event_starttime", ""))
                peak_str = str(event.get("event_peaktime", ""))
                end_str = str(event.get("event_endtime", ""))
                
                records.append({
                    "date": start_str[:10].replace("-", ""),
                    "start": start_str[11:16].replace(":", ""),
                    "peak": peak_str[11:16].replace(":", ""),
                    "end": end_str[11:16].replace(":", ""),
                    "goes": "16",
                    "xray_class": event.get("fl_goescls", ""),
                    "noaa_region": event.get("ar_noaanum", ""),
                    "peak_time": pd.Timestamp(str(event.get("event_peaktime", "")), tz="UTC"),
                })
            
            df = pd.DataFrame(records)
            frames.append(df)
            
        except Exception as exc:
            print(f"[WARNING] HEK {date_range} failed: {exc}. Skipping.")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_noaa_catalog(
    start_year: int = 2010,
    end_year: int = 2024,
    out_path: str = "data/raw/noaa_catalog.parquet",
) -> pd.DataFrame:
    """
    Download NOAA GOES flare catalog for start_year..end_year.

    Uses NGDC text files for 2010-2016, HEK for 2017+.
    Saves to parquet.
    """
    frames = []

    # Phase 1: NGDC text reports (2010-2016)
    ngdc_end = min(end_year, 2016)
    for year in range(start_year, ngdc_end + 1):
        try:
            df = _parse_ngdc_text(year)
            frames.append(df)
            print(f"  NGDC {year}: {len(df)} flares")
        except Exception as exc:
            print(f"[WARNING] NGDC {year} unavailable: {exc}. Skipping.")

    # Phase 2: HEK for 2017+
    hek_start = max(start_year, 2017)
    if hek_start <= end_year:
        hek_df = _fetch_hek_flares(hek_start, end_year)
        if len(hek_df) > 0:
            frames.append(hek_df)

    if not frames:
        print(
            "[WARNING] No flare data downloaded from any source. "
            "check_noaa_confirmed() will return False for all events."
        )
        empty = pd.DataFrame(
            columns=["date", "start", "peak", "end", "goes",
                     "xray_class", "noaa_region", "peak_time"]
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out_path)
        return empty

    out = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"NOAA catalog: {len(out)} flares saved to {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    args = p.parse_args()
    download_noaa_catalog(args.start_year, args.end_year)
