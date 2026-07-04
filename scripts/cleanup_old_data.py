import os
import shutil
import time
from pathlib import Path

def cleanup_old_data():
    """
    Implements the DATA RETENTION POLICY:
    - Raw FITS files: Retain 90 days.
    - Processed parquet cache: Retain 365 days.
    - master_catalogue.csv: PERMANENT.
    - logs/: Retain 30 days. Max 1GB total.
    """
    now = time.time()
    
    # Define thresholds in seconds
    days_90 = 90 * 86400
    days_365 = 365 * 86400
    days_30 = 30 * 86400
    
    # 1. Clean raw FITS (> 90 days)
    fits_dir = Path("data/raw")
    if fits_dir.exists():
        for f in fits_dir.rglob("*.fits"):
            if now - f.stat().st_mtime > days_90:
                print(f"Deleting old FITS file: {f}")
                f.unlink()
                
    # 2. Clean parquet cache (> 365 days)
    parquet_dir = Path("data/processed")
    if parquet_dir.exists():
        for f in parquet_dir.rglob("*.parquet"):
            if now - f.stat().st_mtime > days_365:
                print(f"Deleting old parquet cache: {f}")
                f.unlink()
                
    # 3. Clean logs (> 30 days)
    logs_dir = Path("logs")
    total_size = 0
    if logs_dir.exists():
        for f in logs_dir.rglob("*.log*"):
            if now - f.stat().st_mtime > days_30:
                print(f"Deleting old log file: {f}")
                f.unlink()
            else:
                total_size += f.stat().st_size
                
        # Enforce 1GB limit (1024^3 bytes)
        max_size = 1024**3
        if total_size > max_size:
            print(f"Log directory size ({total_size} bytes) exceeds 1GB limit. Truncating old logs.")
            # Sort by oldest first and delete until under limit
            log_files = sorted(logs_dir.rglob("*.log*"), key=lambda x: x.stat().st_mtime)
            for f in log_files:
                if total_size <= max_size:
                    break
                sz = f.stat().st_size
                print(f"Deleting {f} ({sz} bytes) to free space.")
                f.unlink()
                total_size -= sz

if __name__ == "__main__":
    cleanup_old_data()
