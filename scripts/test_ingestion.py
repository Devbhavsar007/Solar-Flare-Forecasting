"""Test script to verify that fits_reader.py correctly parses PRADAN FITS files."""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.fits_reader import read_hel1os, read_solexs
from src.ingestion.schemas import HEL1OS_SCHEMA, SOLEXS_SCHEMA


def test_hel1os(filepath: str):
    print(f"\n{'='*60}")
    print(f"Testing HEL1OS reader: {filepath}")
    print(f"{'='*60}")

    df = read_hel1os(filepath)
    print(f"\nParsed DataFrame head:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Index type: {type(df.index).__name__}")
    print(f"Index range: {df.index.min()} -> {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    HEL1OS_SCHEMA.validate(df)
    print("\n✅ HEL1OS schema validation PASSED!")


def test_solexs(filepath: str):
    print(f"\n{'='*60}")
    print(f"Testing SoLEXS reader: {filepath}")
    print(f"{'='*60}")

    df = read_solexs(filepath)
    print(f"\nParsed DataFrame head:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Index type: {type(df.index).__name__}")
    print(f"Index range: {df.index.min()} -> {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    SOLEXS_SCHEMA.validate(df)
    print("\n✅ SoLEXS schema validation PASSED!")


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_ingestion.py <hel1os_lightcurve.fits> <solexs_lightcurve.lc.gz>")
        print("\nExample:")
        print('  python scripts/test_ingestion.py \\')
        print('    "data/raw/pradan_download/2026/07/01/HLS_.../cdte/lightcurve_cdte1.fits" \\')
        print('    "data/raw/pradan_download/AL1_SLX_.../SDD2/AL1_SOLEXS_..._SDD2_L1.lc.gz"')
        sys.exit(1)

    hel1os_path = sys.argv[1]
    solexs_path = sys.argv[2]

    try:
        test_hel1os(hel1os_path)
    except Exception as e:
        print(f"\n❌ HEL1OS test FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_solexs(solexs_path)
    except Exception as e:
        print(f"\n❌ SoLEXS test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("All tests complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
