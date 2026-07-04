import sys
import json
from astropy.io import fits

def verify_fits_columns(filepaths):
    output = {}
    for filepath in filepaths:
        try:
            print(f"Opening: {filepath}")
            with fits.open(filepath) as hdul:
                hdul.info()
                for i, hdu in enumerate(hdul):
                    if hasattr(hdu, 'columns'):
                        cols = hdu.columns
                        output[filepath] = {
                            "extension": i,
                            "columns": [{"name": c.name, "format": c.format, "unit": str(c.unit)} for c in cols]
                        }
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print("\n--- JSON OUTPUT FOR configs/fits_columns.yaml ---")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_fits_columns.py <fits_file1> [<fits_file2> ...]")
    else:
        verify_fits_columns(sys.argv[1:])
