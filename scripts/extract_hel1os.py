"""
JWALA HEL1OS Extraction (scripts/extract_hel1os.py)

Extracts HEL1OS FITS files from raw ZIP archives, reads them, and concatenates
into a single Parquet dataset for training.
"""

import os
import zipfile
import glob
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.fits_reader import read_hel1os

def extract_and_process():
    zip_dir = r"d:\Users\Nitro 5\Downloads\HEL1OS_Data"
    raw_dir = "data/raw/hel1os"
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_files = glob.glob(os.path.join(zip_dir, "**", "*.zip"), recursive=True)
    
    if not zip_files:
        print(f"No ZIP files found in {zip_dir}")
        return
        
    print(f"Found {len(zip_files)} ZIP files. Extracting FITS files...")
    
    dfs = []
    extracted_fits = []
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits')]
                for fits_name in fits_names:
                    # Extract locally to data/raw/hel1os/
                    extracted_path = zf.extract(fits_name, path=raw_dir)
                    extracted_fits.append(extracted_path)
                    print(f"Extracted {fits_name}")
        except zipfile.BadZipFile:
            print(f"Failed to extract {zip_path} - Bad ZIP file")
            
    print(f"Processing {len(extracted_fits)} FITS files...")
    
    for fits_path in extracted_fits:
        try:
            df = read_hel1os(fits_path)
            dfs.append(df)
            print(f"Successfully processed {os.path.basename(fits_path)} (rows: {len(df)})")
        except Exception as e:
            print(f"Error reading {fits_path}: {e}")
            
    if not dfs:
        print("No valid HEL1OS data extracted. Exiting.")
        return
        
    print("Concatenating into single DataFrame...")
    hel1os_all = pd.concat(dfs).sort_index()
    
    # Drop duplicates in case of overlapping FITS files
    hel1os_all = hel1os_all[~hel1os_all.index.duplicated(keep='first')]
    
    parquet_path = os.path.join(raw_dir, "hel1os_all.parquet")
    hel1os_all.to_parquet(parquet_path)
    print(f"Saved {len(hel1os_all)} rows to {parquet_path}")

if __name__ == "__main__":
    extract_and_process()
