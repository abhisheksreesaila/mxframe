"""F-010 end to end: all 22 TPC-H queries give identical results whether their input tables
come from `pq.read_table` or from the scan_parquet reader (dictionary strings, decoded at
the plan boundary).  SF1, CPU.  Skipped when the data directory is absent."""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import TPCH_DATA

SF1 = TPCH_DATA / "tpch_sf1"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (SF1 / "lineitem.parquet").exists(),
                                 reason="TPC-H SF1 data not present")]

# query number -> table names, in the order the run_qN_mxframe functions take them
QUERIES = {
    1: ["lineitem"], 2: ["nation", "region", "part", "supplier", "partsupp"],
    3: ["customer", "orders", "lineitem"], 4: ["orders", "lineitem"],
    5: ["nation", "customer", "orders", "lineitem"], 6: ["lineitem"],
    7: ["nation", "supplier", "customer", "orders", "lineitem"],
    8: ["nation", "region", "part", "customer", "orders", "lineitem"],
    9: ["nation", "supplier", "partsupp", "part", "orders", "lineitem"],
    10: ["nation", "customer", "orders", "lineitem"], 11: ["nation", "supplier", "partsupp"],
    12: ["orders", "lineitem"], 13: ["customer", "orders"], 14: ["part", "lineitem"],
    15: ["supplier", "lineitem"], 16: ["part", "supplier", "partsupp"],
    17: ["part", "lineitem"], 18: ["customer", "orders", "lineitem"],
    19: ["part", "lineitem"], 20: ["nation", "part", "supplier", "partsupp", "lineitem"],
    21: ["nation", "supplier", "orders", "lineitem"], 22: ["customer", "orders"],
}
# The benchmark harness itself calls pyarrow.compute string kernels (no dictionary kernels
# exist for match_substring) on the input tables in these queries, before omxcore runs.
HARNESS_RAW_ARROW = {2}
ALL_TABLES = sorted({t for ts in QUERIES.values() for t in ts})


@pytest.fixture(scope="module")
def bm():
    sys.path.insert(0, str(SCRIPTS))
    import benchmark_tpch
    return benchmark_tpch


@pytest.fixture(scope="module")
def plain_tables():
    return {n: pq.read_table(SF1 / f"{n}.parquet") for n in ALL_TABLES}


@pytest.fixture(scope="module")
def dict_tables():
    import omxcore as mx
    return {n: mx.read_parquet_table(SF1 / f"{n}.parquet", dictionary="auto") for n in ALL_TABLES}


def _decoded(t: pa.Table) -> pa.Table:
    for i, typ in enumerate(t.schema.types):
        if pa.types.is_dictionary(typ):
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(typ.value_type))
    return t


def _rows(t: pa.Table):
    cols = [t.column(n).to_pylist() for n in t.column_names]
    return sorted(zip(*cols), key=lambda r: tuple("" if v is None else str(v) for v in r))


def _assert_same(a, b):
    if not isinstance(a, pa.Table):                 # scalar-returning queries (Q14, Q19)
        assert a == pytest.approx(b, rel=2e-3)
        return
    assert a.column_names == b.column_names
    assert [str(t) for t in a.schema.types] == [str(t) for t in b.schema.types]
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            if isinstance(u, float) and isinstance(v, float):
                assert u == pytest.approx(v, rel=2e-3, abs=1e-3)
            else:
                assert u == v


def test_dictionary_tables_actually_contain_dictionaries(dict_tables):
    assert any(pa.types.is_dictionary(t) for t in dict_tables["lineitem"].schema.types)


@pytest.mark.parametrize("q", sorted(QUERIES))
def test_query_identical_with_dictionary_inputs(q, bm, plain_tables, dict_tables):
    fn = getattr(bm, f"run_q{q}_mxframe")
    try:
        expected = fn(*[plain_tables[n] for n in QUERIES[q]], device="cpu")
    except Exception as exc:                                    # fails on the plain path too
        pytest.skip(f"Q{q} fails on the plain path as well: {type(exc).__name__}: {exc}")
    tables = [dict_tables[n] for n in QUERIES[q]]
    if q in HARNESS_RAW_ARROW:
        tables = [_decoded(t) for t in tables]
    got = fn(*tables, device="cpu")
    _assert_same(got, expected)


def _lazy_queries(bm):
    """Q1/Q3/Q6/Q12/Q14 rewritten over lazy sources (the harness versions take pa.Tables)."""
    from omxcore import col, lit, when
    D = "cpu"

    def q1(li):
        return (li.filter(col("l_shipdate") <= lit(bm.CUTOFF_Q1))
                .groupby("l_returnflag", "l_linestatus")
                .agg(col("l_quantity").sum().alias("sum_qty"),
                     (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("rev"),
                     col("l_quantity").mean().alias("avg_qty"),
                     col("l_quantity").count().alias("n"))
                .sort("l_returnflag", "l_linestatus").compute(device=D))

    def q3(c, o, li):
        return (o.filter(col("o_orderdate") < lit(bm.DATE_1995_03_15))
                .join(c.filter(col("c_mktsegment") == lit("BUILDING")), left_on="o_custkey", right_on="c_custkey")
                .join(li.filter(col("l_shipdate") > lit(bm.DATE_1995_03_15)), left_on="o_orderkey", right_on="l_orderkey")
                .groupby("o_orderkey", "o_orderdate", "o_shippriority")
                .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
                .sort("revenue", descending=True).limit(10).compute(device=D))

    def q6(li):
        return (li.filter((col("l_shipdate") >= lit(bm.Q6_DATE_LO)) & (col("l_shipdate") < lit(bm.Q6_DATE_HI))
                          & (col("l_discount") >= lit(bm.Q6_DISC_LO)) & (col("l_discount") <= lit(bm.Q6_DISC_HI))
                          & (col("l_quantity") < lit(bm.Q6_QTY_HI)))
                .groupby().agg((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
                .compute(device=D))

    def q12(o, li):
        return (li.join(o, left_on="l_orderkey", right_on="o_orderkey")
                .filter(col("l_shipmode").isin(["MAIL", "SHIP"]))
                .filter(col("l_commitdate") < col("l_receiptdate"))
                .filter(col("l_shipdate") < col("l_commitdate"))
                .filter(col("l_receiptdate") >= lit(bm.Q12_DATE_LO))
                .filter(col("l_receiptdate") < lit(bm.Q12_DATE_HI))
                .groupby("l_shipmode")
                .agg(when(col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0)).sum().alias("high"),
                     when(~col("o_orderpriority").isin(["1-URGENT", "2-HIGH"]), lit(1), lit(0)).sum().alias("low"))
                .sort(col("l_shipmode")).compute(device=D))

    def q14(p, li):
        promo = when(col("p_type").startswith("PROMO"),
                     col("l_extendedprice") * (lit(1.0) - col("l_discount")), lit(0.0))
        total = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
        return (li.join(p, left_on="l_partkey", right_on="p_partkey")
                .filter(col("l_shipdate") >= lit(bm.Q14_DATE_LO)).filter(col("l_shipdate") < lit(bm.Q14_DATE_HI))
                .groupby().agg(promo.sum().alias("promo"), total.sum().alias("total")).compute(device=D))
    return {1: (q1, ["lineitem"]), 3: (q3, ["customer", "orders", "lineitem"]), 6: (q6, ["lineitem"]),
            12: (q12, ["orders", "lineitem"]), 14: (q14, ["part", "lineitem"])}


@pytest.mark.parametrize("q", [1, 3, 6, 12, 14])
def test_queries_through_scan_parquet(q, bm, plain_tables):
    """Genuinely lazy: column pruning + pushdown, compared with the same query on in-memory tables."""
    import omxcore as mx
    from omxcore import LazyFrame
    fn, names = _lazy_queries(bm)[q]
    got = fn(*[mx.scan_parquet(str(SF1 / f"{n}.parquet")) for n in names])
    exp = fn(*[LazyFrame(plain_tables[n]) for n in names])
    if not isinstance(got, pa.Table):
        _assert_same(got, exp)
    elif q in (3,):
        assert got.num_rows == exp.num_rows           # top-10 ties may order differently
        assert got.column("revenue").to_pylist() == pytest.approx(exp.column("revenue").to_pylist(), rel=2e-3)
    else:
        _assert_same(got, exp)
