import sys
import os

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.fits_reader import read_hel1os
from src.ingestion.schemas import HEL1OS_SCHEMA

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hel1os_reader.py <path_to_hel1os_lightcurve.fits>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    print(f"Reading HEL1OS file: {filepath}")
    
    try:
        df = read_hel1os(filepath)
        print("\nSuccessfully parsed DataFrame:")
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
        print(f"Index type: {type(df.index)}")
        
        # Validate against schema
        print("\nValidating against HEL1OS_SCHEMA...")
        HEL1OS_SCHEMA.validate(df)
        print("✅ Validation PASSED! The data matches the required schema.")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
