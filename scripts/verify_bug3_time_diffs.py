"""
BUG 3 Verification: Confirm TIME column diff = 1.0s on a real SoLEXS file.
Shows first 5 diffs and confirms mean() is the correct aggregation.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import zipfile
import numpy as np
from astropy.io import fits

# Find a real SoLEXS file
solexs_base = r"d:\Users\Nitro 5\Downloads\SOLEXS_Data"
zip_files = glob.glob(os.path.join(solexs_base, "**", "*.zip"), recursive=True)

zip_path = zip_files[0]
print(f"Using zip: {os.path.basename(zip_path)}")

lc_file = None
tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', '_verify_bug3')
os.makedirs(tmp_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    for name in zf.namelist():
        if 'SDD2' in name and name.endswith('.lc.gz'):
            zf.extract(name, tmp_dir)
            lc_file = os.path.join(tmp_dir, name)
            break

print(f"File: {os.path.basename(lc_file)}")
print()

with fits.open(lc_file) as hdul:
    data = hdul[1].data
    header = hdul[1].header
    time_raw = data["TIME"].astype(float)
    
    diffs = np.diff(time_raw)
    
    print("--- TIME column analysis ---")
    print(f"Total rows: {len(time_raw)}")
    print(f"Expected rows for 1-day @ 1s cadence: 86400")
    print()
    
    print("First 5 TIME diffs (seconds):")
    for i in range(min(5, len(diffs))):
        print(f"  TIME[{i+1}] - TIME[{i}] = {diffs[i]:.6f} s")
    
    print(f"\nMean diff:   {np.mean(diffs):.6f} s")
    print(f"Median diff: {np.median(diffs):.6f} s")
    print(f"Min diff:    {np.min(diffs):.6f} s")
    print(f"Max diff:    {np.max(diffs):.6f} s")
    
    # Check HDU name
    print(f"\nHDU 1 name: '{hdul[1].name}'")
    
    # Check TUNIT if present
    for key in header.keys():
        if 'TUNIT' in key:
            print(f"{key} = {header[key]}")
    
    # Conclusion
    is_1s = np.abs(np.median(diffs) - 1.0) < 0.01
    print(f"\n--- Conclusion ---")
    if is_1s:
        print("Bin width = 1.0s (confirmed)")
        print("At 1s cadence: counts/bin == counts/sec numerically.")
        print("Column is a RATE -> .mean() is correct for resampling.")
    else:
        print(f"WARNING: Bin width = {np.median(diffs):.4f}s (NOT 1.0s)")
        print("counts/bin ≠ counts/sec — resampling aggregation needs review!")

# Cleanup
import shutil
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
    print("\nCleaned up temp files.")
