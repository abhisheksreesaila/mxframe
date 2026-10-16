"""F-065: the 22 TPC-H queries as the specification's SQL text (scripts/tpch_sql/queries.py) through
``mx.sql``, against DuckDB running the same text over the same Parquet files (small scale factor; the SF1
run is ``scripts/tpch_sql/run.py``). A query is either equal to DuckDB or raises for a reason named here:
when a gap closes, move the query out of ``UNSUPPORTED`` (the test fails until you do)."""
import sys
from pathlib import Path

import pytest

from conftest import TPCH_DATA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tpch_sql import run  # noqa: E402

SF = "0.017"
DATA = TPCH_DATA / f"tpch_sf{SF}_exact"

UNSUPPORTED = {
    13: "Cannot parse JOIN ON",                      # a non-equality condition in a LEFT JOIN's ON
    21: "l2.l_suppkey <> l1.l_suppkey",              # correlated <> (EXISTS with an inequality)
}


@pytest.mark.skipif(not (DATA / "lineitem.parquet").exists(), reason="TPC-H small data not present")
@pytest.mark.parametrize("q", range(1, 23))
def test_tpch_sql_text(q):
    out = run.one(q, SF)
    if q in UNSUPPORTED:
        assert out["status"] == "raised" and UNSUPPORTED[q] in out["error"], out
    else:
        assert out["status"] == "equal", out
