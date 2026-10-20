"""F-026 / F-056: `pa.Schema.empty_table()` imports pandas (110-160 ms per process, Q11 failed test_no_pandas_tpch).
Build an empty table with `np_arrow.empty(schema)` (`pa.Table.from_batches([], schema)`). Twice a slip: now a check."""
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent


def test_no_module_of_the_package_calls_empty_table():
    offenders = [f"{p.name}:{n}" for p in sorted(ROOT.glob("*.py")) for n, line in enumerate(p.read_text().splitlines(), 1)
                 if ".empty_table(" in line]
    assert offenders == [], f"use np_arrow.empty(schema): {offenders}"


def test_np_arrow_empty_keeps_the_schema_and_has_no_rows():
    from omxcore import np_arrow
    schema = pa.schema([("a", pa.int64()), ("b", pa.string()), ("c", pa.dictionary(pa.int32(), pa.string()))])
    t = np_arrow.empty(schema)
    assert t.num_rows == 0 and t.schema == schema
