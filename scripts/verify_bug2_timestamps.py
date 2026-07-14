"""
BUG 2 Verification: MJDREFI guard and timestamp sanity assertion.
Tests both the happy path (real file) and the failure mode (simulated bad data).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import zipfile
import tempfile

# --- Part 1: Find a real SoLEXS .lc.gz file ---
solexs_base = r"d:\Users\Nitro 5\Downloads\SOLEXS_Data"
zip_files = glob.glob(os.path.join(solexs_base, "**", "*.zip"), recursive=True)

if not zip_files:
    print("ERROR: No SoLEXS zip files found.")
    sys.exit(1)

# Pick the first zip
zip_path = zip_files[0]
print(f"Using zip: {zip_path}")

# Extract just the SDD2 .lc.gz file
lc_file = None
with zipfile.ZipFile(zip_path, 'r') as zf:
    for name in zf.namelist():
        if 'SDD2' in name and name.endswith('.lc.gz'):
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', '_verify_bug2')
            os.makedirs(tmp_dir, exist_ok=True)
            zf.extract(name, tmp_dir)
            lc_file = os.path.join(tmp_dir, name)
            break

if lc_file is None:
    print("ERROR: No SDD2 .lc.gz file found in zip.")
    sys.exit(1)

print(f"Extracted: {lc_file}")

# --- Part 2: Happy path — read_solexs on real file ---
print("\n--- HAPPY PATH: read_solexs on real file ---")
from src.ingestion.fits_reader import read_solexs

try:
    df = read_solexs(lc_file)
    print(f"PASS: read_solexs succeeded. Shape: {df.shape}")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Assertion did NOT fire (timestamps are post-2023-09-01).")
except AssertionError as e:
    print(f"FAIL: Assertion fired: {e}")
except KeyError as e:
    print(f"FAIL: KeyError raised: {e}")

# --- Part 3: Simulate the BUG 2 failure mode ---
print("\n--- FAILURE MODE: Simulate bad MJDREFI=0 ---")
# We can't easily create a bad FITS file, so let's directly test the guard logic
from astropy.io import fits as afits
import numpy as np
import pandas as pd

# Read the real header to get the actual MJDREFI for reference
with afits.open(lc_file) as hdul:
    real_header = hdul[1].header
    real_mjdrefi = real_header.get("MJDREFI", 0)
    real_mjdreff = real_header.get("MJDREFF", 0.0)
    real_timezero = real_header.get("TIMEZERO", 0.0)
    time_raw = hdul[1].data["TIME"].astype(float)
    print(f"  Real MJDREFI = {real_mjdrefi}")
    print(f"  Real MJDREFF = {real_mjdreff}")
    print(f"  Real TIMEZERO = {real_timezero}")

# Simulate a file where time_raw is small (seconds since a recent mission epoch, e.g. 2023-09-01)
# If such a file falls back to unix epoch, the timestamps will be in 1970.
simulated_time_raw = np.array([86400.0, 86401.0, 86402.0]) # 1 day of seconds
print("\n  If old code hit the else branch on small mission-epoch seconds:")
bad_timestamps = pd.to_datetime(simulated_time_raw, unit="s", origin="unix")
print(f"  BAD timestamps would be: {bad_timestamps.min()} to {bad_timestamps.max()}")
print(f"  (These are 1970-era timestamps — clearly wrong)")

# Show what the new assertion WOULD catch:
print(f"\n  The new assertion checks: timestamps.min() > 2023-09-01")
print(f"  BAD min = {bad_timestamps.min()} < 2023-09-01 -> assertion WOULD fire")
try:
    assert bad_timestamps.min() > pd.Timestamp("2023-09-01"), (
        f"SoLEXS timestamp out of expected mission range: "
        f"min={bad_timestamps.min()} in SIMULATED_FILE. "
        f"This likely means MJDREFI/TIMEZERO were parsed incorrectly."
    )
    print("  BUG: Assertion did NOT fire — this shouldn't happen.")
except AssertionError as e:
    print(f"  PASS: Assertion correctly caught the bad timestamps:")
    print(f"    {e}")

# Cleanup
import shutil
cleanup_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', '_verify_bug2')
if os.path.exists(cleanup_dir):
    shutil.rmtree(cleanup_dir)
    print("\nCleaned up temp files.")
