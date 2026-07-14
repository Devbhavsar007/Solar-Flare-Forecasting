"""
Catalog Audit Script: Full 2010-2026 flare counts + class breakdown.

Three checks:
1. Full per-year flare counts from a single unified query (2010-2026)
2. Flare-class breakdown (B/C/M/X) for 2023 specifically (overlap year)
3. Compare against old catalog if it exists on disk
"""
import sys, os
sys.path.insert(0, '.')

import pandas as pd
from scripts.download_noaa_catalog import download_noaa_catalog

# --- Check 1: Fresh unified pull, full range ---
print("=" * 60)
print("CHECK 1: Fresh 2010-2026 pull (single query path)")
print("=" * 60)

df_new = download_noaa_catalog(start_year=2010, end_year=2026, out_path="data/raw/noaa_catalog_audit.parquet")

df_new['start_time'] = pd.to_datetime(
    df_new['date'].astype(str) + ' ' + df_new['start'].astype(str),
    format='%Y%m%d %H%M', utc=True, errors='coerce'
)

# Extract the leading letter of xray_class for C/M/X breakdown
df_new['class_letter'] = df_new['xray_class'].astype(str).str[0].str.upper()

print("\n--- Per-Year Flare Counts ---")
yearly = df_new['start_time'].dt.year.value_counts().sort_index()
for yr, cnt in yearly.items():
    print(f"  {int(yr)}: {cnt}")
print(f"  TOTAL: {yearly.sum()}")

# --- Check 2: Class breakdown for 2023 (overlap year) ---
print("\n" + "=" * 60)
print("CHECK 2: Class breakdown for 2023 (new catalog)")
print("=" * 60)

mask_2023 = df_new['start_time'].dt.year == 2023
breakdown_2023 = df_new.loc[mask_2023, 'class_letter'].value_counts().sort_index()
print(breakdown_2023.to_string())

# Also do 2024 for comparison
print("\n--- Class breakdown for 2024 (new catalog) ---")
mask_2024 = df_new['start_time'].dt.year == 2024
breakdown_2024 = df_new.loc[mask_2024, 'class_letter'].value_counts().sort_index()
print(breakdown_2024.to_string())

# --- Check 3: Compare against old on-disk catalog if it exists ---
print("\n" + "=" * 60)
print("CHECK 3: Old catalog comparison")
print("=" * 60)

old_path = "data/raw/noaa_catalog.parquet"
if os.path.exists(old_path):
    df_old = pd.read_parquet(old_path)
    if 'start_time' not in df_old.columns:
        df_old['start_time'] = pd.to_datetime(
            df_old['date'].astype(str) + ' ' + df_old['start'].astype(str),
            format='%Y%m%d %H%M', utc=True, errors='coerce'
        )
    
    df_old['class_letter'] = df_old['xray_class'].astype(str).str[0].str.upper()
    
    old_yearly = df_old['start_time'].dt.year.value_counts().sort_index()
    print("\n--- Old catalog per-year counts ---")
    for yr, cnt in old_yearly.items():
        print(f"  {int(yr)}: {cnt}")
    print(f"  TOTAL: {old_yearly.sum()}")
    
    print("\n--- Old catalog 2023 class breakdown ---")
    old_2023 = df_old.loc[df_old['start_time'].dt.year == 2023, 'class_letter'].value_counts().sort_index()
    print(old_2023.to_string())
    
    # Side-by-side comparison for overlapping years
    print("\n--- Side-by-side (Old vs New) ---")
    all_years = sorted(set(old_yearly.index) | set(yearly.index))
    print(f"  {'Year':>6}  {'Old':>6}  {'New':>6}  {'Ratio':>8}")
    for yr in all_years:
        o = old_yearly.get(yr, 0)
        n = yearly.get(yr, 0)
        ratio = f"{n/o:.2f}x" if o > 0 else "n/a"
        print(f"  {int(yr):>6}  {o:>6}  {n:>6}  {ratio:>8}")
else:
    print(f"  No old catalog found at {old_path}")

print("\n--- Audit complete ---")
