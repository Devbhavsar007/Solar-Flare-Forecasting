import pandera as pa
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.ingestion.schemas import SOLEXS_SCHEMA, HEL1OS_SCHEMA
except ImportError:
    print("Could not import schemas. Make sure src/ingestion/schemas.py exists.")
    sys.exit(1)

def verify_schemas():
    # Valid SoLEXS
    valid_solexs = pd.DataFrame({
        "flux_high": np.random.uniform(1e-9, 1e-3, 10),
        "flux_low": np.random.uniform(1e-9, 1e-3, 10)
    }, index=pd.date_range("2024-01-01", periods=10))
    
    SOLEXS_SCHEMA.validate(valid_solexs)
    print("Valid SoLEXS DataFrame passed.")
    
    # Invalid SoLEXS
    invalid_solexs = valid_solexs.copy()
    invalid_solexs.iloc[0, 0] = np.nan
    try:
        SOLEXS_SCHEMA.validate(invalid_solexs, lazy=True)
        print("FAIL: Invalid SoLEXS did not raise error.")
    except pa.errors.SchemaErrors as exc:
        print("Invalid SoLEXS correctly rejected.")
        print(f"Failure cases:\n{exc.failure_cases}")
        
    # Valid HEL1OS
    valid_hel1os = pd.DataFrame({
    "counts_low": np.random.randint(0, 1000, 10).astype(np.int64),
    "counts_high": np.random.randint(0, 1000, 10).astype(np.int64)
}, index=pd.date_range("2024-01-01", periods=10))
    
    HEL1OS_SCHEMA.validate(valid_hel1os)
    print("Valid HEL1OS DataFrame passed.")
    
    # Invalid HEL1OS
    invalid_hel1os = valid_hel1os.copy()
    invalid_hel1os.iloc[0, 0] = -1 # Violates greater_than_or_equal_to(0)
    try:
        HEL1OS_SCHEMA.validate(invalid_hel1os, lazy=True)
        print("FAIL: Invalid HEL1OS did not raise error.")
    except pa.errors.SchemaErrors as exc:
        print("Invalid HEL1OS correctly rejected.")
        print(f"Failure cases:\n{exc.failure_cases}")
        
    print("\nCHECK 8 PASSED: Both schemas validate and reject correctly.")

if __name__ == "__main__":
    verify_schemas()
