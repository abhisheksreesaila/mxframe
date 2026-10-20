"""The synthetic-data benchmark (`benchmark_tpch.py main()`, `benchmark_all_22.py`, ...) still runs:
every query's omxcore (cpu) and pandas version accepts the tables its `make_tpch_qN_tables`
produces and returns the same number of rows.  Row counts only; answers are checked against
DuckDB on real TPC-H data in test_tpch_duckdb.py."""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def bm():
    sys.path.insert(0, str(SCRIPTS))
    import benchmark_tpch
    return benchmark_tpch


def _cases(b):
    li = b.make_lineitem(50_000)
    c3, o3, l3 = b.make_tpch_q3_tables(3_000, 30_000, 120_000)
    return {
        1: [li], 6: [li], 3: [c3, o3, l3],
        12: list(b.make_tpch_q12_tables(10_000, 60_000)), 14: list(b.make_tpch_q14_tables(4_000, 40_000)),
        4: list(b.make_tpch_q4_tables(30_000, 120_000)),
        9: list(b.make_tpch_q9_tables(n_orders=20_000, n_lineitem=50_000)),
        11: list(b.make_tpch_q11_tables()), 18: list(b.make_tpch_q18_tables(3_000, 12_000, 60_000)),
        16: list(b.make_tpch_q16_tables()), 17: list(b.make_tpch_q17_tables(n_lineitem=50_000)),
        2: list(b.make_tpch_q2_tables()), 15: list(b.make_tpch_q15_tables(n_lineitem=50_000)),
        20: list(b.make_tpch_q20_tables(n_lineitem=30_000)),
        21: list(b.make_tpch_q21_tables(n_orders=15_000, n_lineitem=50_000)),
        22: list(b.make_tpch_q22_tables(20_000, 50_000)),
        5: list(b.make_tpch_q5_tables(3_000, 20_000, 80_000)), 10: list(b.make_tpch_q10_tables(3_000, 30_000, 120_000)),
        7: list(b.make_tpch_q7_tables(n_orders=16_000, n_li=40_000)),
        8: list(b.make_tpch_q8_tables(n_orders=16_000, n_li=40_000)),
        13: list(b.make_tpch_q13_tables(20_000, 80_000)), 19: list(b.make_tpch_q19_tables(4_000, 40_000)),
    }


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
def test_synthetic_tables_feed_every_query(q, bm):
    tables = _cases(bm)[q]
    kw = {"supplier": bm.make_synthetic_supplier()} if q in (5, 8) else {}
    got = getattr(bm, f"run_q{q}_mxframe")(*tables, device="cpu", **kw)
    want = getattr(bm, f"run_q{q}_pandas")(*tables, **kw)
    n_got = got.num_rows if hasattr(got, "num_rows") else 1
    n_want = len(want) if hasattr(want, "__len__") else 1
    assert n_got == n_want
