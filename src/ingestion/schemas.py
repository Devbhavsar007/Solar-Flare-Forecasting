import pandera as pa
from pandera import Column, DataFrameSchema

SOLEXS_SCHEMA = DataFrameSchema({
    "flux_high": Column(float, pa.Check.greater_than_or_equal_to(0)),
    "flux_low": Column(float, pa.Check.greater_than_or_equal_to(0)),
    # ADD remaining columns after CHECK 1 reveals real names
}, index=pa.Index(pa.DateTime), coerce=False)

HEL1OS_SCHEMA = DataFrameSchema({
    "counts_low": Column(int, pa.Check.greater_than_or_equal_to(0)),
    "counts_high": Column(int, pa.Check.greater_than_or_equal_to(0)),
    # ADD remaining columns after CHECK 1 reveals real names
}, index=pa.Index(pa.DateTime), coerce=False)
