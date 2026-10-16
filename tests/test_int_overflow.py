"""F-061: integer arithmetic and SUM past int64 never wrap silently. DuckDB raises
"Overflow in multiplication of INT64" (and friends) for + - *; we raise OverflowError. A SUM
DuckDB widens to HUGEINT; our SUM column is int64, so a total past int64 raises too."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

BIG = 1 << 30


@pytest.fixture(params=["1", "0"], ids=["native", "legacy"])
def mode(request, monkeypatch):
    if request.param == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", request.param)
    return request.param


def _con(t):
    con = duckdb.connect()
    con.register("t", t)
    return con


def _t(n=1000):
    a = np.where(np.arange(n) % 2 == 0, BIG, 3).astype(np.int64)      # even rows: 2^30, so a*a*a = 2^90
    return pa.table({"a": A(a), "g": A((np.arange(n) % 4).astype(np.int64)), "v": A(np.ones(n))})


def _both_raise(t, sql, lf, match):
    with pytest.raises(duckdb.OutOfRangeException, match=match):
        _con(t).execute(sql).fetchall()
    with pytest.raises(OverflowError, match=match):
        lf.compute(device="cpu")


def test_filter_on_an_overflowing_product_raises_like_duckdb(mode):
    t = _t()
    lf = LazyFrame(t).filter((col("a") * col("a") * col("a")) > lit(0)).groupby().agg(col("v").sum().alias("s"))
    _both_raise(t, "select sum(v) from t where a * a * a > 0", lf, "Overflow in multiplication of INT64")


def test_an_overflowing_sum_of_products_raises_like_duckdb(mode):
    t = _t()
    e = (col("a") * col("a") * lit(1 << 30)) + (col("a") * col("a") * lit(1 << 30))
    lf = LazyFrame(t).filter(e > lit(0)).groupby().agg(col("v").sum().alias("s"))
    _both_raise(t, "select sum(v) from t where a * a * 1073741824 + a * a * 1073741824 > 0", lf,
                "Overflow in multiplication of INT64")


def test_an_overflowing_addition_and_subtraction_raise_like_duckdb(mode):
    top = (1 << 63) - 1
    t = pa.table({"a": A(np.array([top, 1], dtype=np.int64)), "v": A(np.ones(2))})
    lf = LazyFrame(t).filter((col("a") + col("a")) > lit(0)).groupby().agg(col("v").sum().alias("s"))
    _both_raise(t, "select sum(v) from t where a + a > 0", lf, "Overflow in addition of INT64")
    lf = LazyFrame(t).filter((lit(-5) - col("a") - col("a")) < lit(0)).groupby().agg(col("v").sum().alias("s"))
    _both_raise(t, "select sum(v) from t where -5 - a - a < 0", lf, "Overflow in subtraction of INT64")


def test_an_overflowing_aggregate_expression_raises(mode):
    t = _t()
    lf = LazyFrame(t).groupby("g").agg((col("a") * col("a") * col("a")).max().alias("m"))
    _both_raise(t, "select g, max(a*a*a) from t group by g", lf, "Overflow in multiplication of INT64")


def test_the_largest_product_that_fits_is_exact(mode):
    t = pa.table({"a": A(np.array([3037000499, 1, 2], dtype=np.int64)), "v": A(np.ones(3))})
    got = LazyFrame(t).filter((col("a") * col("a")) > lit(9_000_000_000_000_000_000)).groupby().agg(
        col("v").sum().alias("s")).compute(device="cpu")
    want = _con(t).execute("select sum(v) from t where a * a > 9000000000000000000").fetchone()[0]
    assert got.column("s").to_pylist() == [want] == [1.0]


def _sum_table(n, groups=2):
    return pa.table({"a": A(np.full(n, 1 << 62, dtype=np.int64)),
                     "g": A((np.arange(n) % groups).astype(np.int64))})


def test_a_global_sum_past_int64_raises(mode):
    t = _sum_table(2)                                # 2 * 2^62 = 2^63
    assert _con(t).execute("select sum(a) from t").fetchone()[0] == 1 << 63      # DuckDB: HUGEINT
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(t).groupby().agg(col("a").sum().alias("s")).compute(device="cpu")


def test_a_grouped_sum_past_int64_raises(mode):
    t = _sum_table(8)                                # each group: 4 * 2^62
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(t).groupby("g").agg(col("a").sum().alias("s")).compute(device="cpu")


def test_a_sum_that_reaches_exactly_the_int64_limits_is_exact(mode):
    lo = np.array([-(1 << 62), -(1 << 62)], dtype=np.int64)               # -2^63 fits
    hi = np.array([(1 << 62) - 1, 1 << 62], dtype=np.int64)               # 2^63 - 1 fits
    t = pa.table({"lo": A(lo), "hi": A(hi), "g": A(np.zeros(2, dtype=np.int64))})
    con = _con(t)
    for c, v in (("lo", -(1 << 63)), ("hi", (1 << 63) - 1)):
        assert con.execute(f"select sum({c}) from t").fetchone()[0] == v
        got = LazyFrame(t).groupby().agg(col(c).sum().alias("s")).compute(device="cpu")
        assert got.column("s").to_pylist() == [v]
        got = LazyFrame(t).groupby("g").agg(col(c).sum().alias("s")).compute(device="cpu")
        assert got.column("s").to_pylist() == [v]


def test_a_sum_whose_partial_sums_cancel_is_not_an_overflow(mode):
    a = np.array([1 << 62, 1 << 62, -(1 << 62), -(1 << 62), 5], dtype=np.int64)
    t = pa.table({"a": A(a), "g": A(np.zeros(5, dtype=np.int64))})
    for lf in (LazyFrame(t).groupby().agg(col("a").sum().alias("s")),
               LazyFrame(t).groupby("g").agg(col("a").sum().alias("s"))):
        assert lf.compute(device="cpu").column("s").to_pylist() == [5]


def test_normal_sized_integer_queries_still_run_natively(monkeypatch):
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    n = 5000
    t = pa.table({"p": A(np.arange(1, n + 1, dtype=np.int64)), "g": A((np.arange(n) % 3).astype(np.int64))})
    got = LazyFrame(t).filter((col("p") * col("p")) > lit(50)).groupby("g").agg(
        (col("p") * col("p")).sum().alias("s")).sort("g").compute(device="cpu")
    assert nx.last_path() == "native"
    want = _con(t).execute("select g, sum(p*p) from t where p*p > 50 group by g order by g").fetchall()
    assert got.column("s").to_pylist() == [float(x[1]) for x in want]


def test_normal_sized_integer_queries_do_no_checked_arithmetic_work(monkeypatch):
    """The native path answers them: no Arrow arithmetic, no exact-sum pass (no new data pass)."""
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    import mxframe.custom_ops as co
    calls = []
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setattr(co, "_checked_arith", lambda *a: calls.append("arith"))
    monkeypatch.setattr(co, "_int_sum", lambda *a: calls.append("sum"))
    t = pa.table({"p": A(np.arange(1, 5001, dtype=np.int64)), "q": A(np.arange(5000, dtype=np.int32))})
    LazyFrame(t).filter((col("p") * col("q")) > lit(50)).groupby().agg(
        (col("p") + col("q")).sum().alias("s")).compute(device="cpu")
    assert nx.last_path() == "native" and calls == []


def test_the_gpu_session_does_not_wrap_either():
    from mxframe import gpu_stream
    if not gpu_stream.cuda_available():
        pytest.skip("no CUDA GPU")
    t = _t()
    with pytest.raises(OverflowError, match="Overflow in multiplication of INT64"):
        LazyFrame(t).filter((col("a") * col("a") * col("a")) > lit(0)).groupby().agg(
            col("v").sum().alias("s")).compute(device="gpu")
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(_sum_table(2)).groupby().agg(col("a").sum().alias("s")).compute(device="gpu")
    with pytest.raises(OverflowError, match="sum of INT64"):
        LazyFrame(_sum_table(8)).groupby("g").agg(col("a").sum().alias("s")).compute(device="gpu")
