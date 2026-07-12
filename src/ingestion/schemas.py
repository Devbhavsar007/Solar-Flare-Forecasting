import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd

def _monotonic_index_check(index: pd.Index) -> bool:
    return index.is_monotonic_increasing

# SoLEXS: Single "counts" column from SDD2 lightcurve
SOLEXS_SCHEMA = DataFrameSchema(
    columns={
        "counts": Column(float, checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than(1e9)
        ], nullable=False),
        "pradan_version": Column(str, nullable=False),
    },
    index=pa.Index(
        pa.DateTime,
        checks=pa.Check(
            _monotonic_index_check,
            element_wise=False,
            error=(
                "FITS DatetimeIndex is not monotonically increasing. "
                "Cause: instrument clock glitch or duplicate timestamps. "
                "Fix: df = df[~df.index.duplicated()].sort_index() "
                "BEFORE calling schema.validate()."
            )
        )
    ),
    coerce=False,
)

# HEL1OS: Two energy bands extracted from separate HDU extensions
HEL1OS_SCHEMA = DataFrameSchema(
    columns={
        "counts_low": Column(float, checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than(1e9)
        ], nullable=False),
        "counts_high": Column(float, checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than(1e9)
        ], nullable=False),
    },
    index=pa.Index(
        pa.DateTime,
        checks=pa.Check(
            _monotonic_index_check,
            element_wise=False,
            error=(
                "HEL1OS DatetimeIndex is not monotonically increasing. "
                "Cause: instrument reset during high-energy burst. "
                "Fix: df = df.sort_index().drop_duplicates() before validate()."
            )
        )
    ),
    coerce=False,
)
