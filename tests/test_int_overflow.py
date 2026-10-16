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


# -- shapes where the pipelines decline for an unrelated reason (an ordered string predicate) --------

def _ts():
    t = _t()
    return pa.table({"a": t.column("a"), "g": t.column("g"), "v": t.column("v"),
                     "s": pa.array(["a", "b", "c", "d"] * (len(t) // 4))})


@pytest.mark.parametrize("agg,sql", [
    (lambda: (col("a") * col("a") * col("a")).max(), "max(a*a*a)"),
    (lambda: (col("a") * col("a") * lit(1 << 40)).min(), "min(a*a*1099511627776)"),
    (lambda: (col("a") * col("a") * col("a")).sum(), "sum(a*a*a)"),
    (lambda: (col("a") * col("a") * col("a")).mean(), "avg(a*a*a)"),
], ids=["max_cube", "min_shift", "sum_cube", "mean_cube"])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "global"])
def test_an_uncompilable_predicate_does_not_hide_an_overflowing_aggregate(mode, agg, sql, grouped):
    t = _ts()
    lf = LazyFrame(t).filter(col("s") > lit("a"))
    lf = lf.groupby("g").agg(agg().alias("x")) if grouped else lf.groupby().agg(agg().alias("x"))
    q = f"select {'g, ' if grouped else ''}{sql} from t where s > 'a'{' group by g' if grouped else ''}"
    _both_raise(t, q, lf, "Overflow in multiplication of INT64")


def test_an_overflowing_aggregate_after_a_join_raises(mode):
    t = _ts()
    d = pa.table({"g": pa.array(np.arange(4, dtype=np.int64)), "name": pa.array(["w", "x", "y", "z"])})
    con = duckdb.connect()
    con.register("t", t)
    con.register("d", d)
    with pytest.raises(duckdb.OutOfRangeException, match="Overflow in multiplication"):
        con.execute("select name, max(a*a*a) from t join d using(g) group by name").fetchall()
    with pytest.raises(OverflowError, match="Overflow in multiplication of INT64"):
        LazyFrame(t).join(LazyFrame(d), on="g").groupby("name").agg(
            (col("a") * col("a") * col("a")).max().alias("x")).compute(device="cpu")


# -- group keys and projected columns ---------------------------------------------------------------

def _cube():
    return col("a") * col("a") * col("a")


def _dim():
    return pa.table({"g": pa.array(np.arange(4, dtype=np.int64)), "name": pa.array(["w", "x", "y", "z"])})


MSG = "Overflow in multiplication of INT64"


def test_a_group_key_past_int64_raises(mode):
    t = _t()
    _both_raise(t, "select a*a*a k, sum(v) from t group by k",
                LazyFrame(t).groupby(_cube().alias("k")).agg(col("v").sum().alias("x")), MSG)


@pytest.mark.parametrize("how", ["with_columns", "select"])
def test_a_projected_column_past_int64_raises(mode, how):
    t = _t()
    lf = LazyFrame(t)
    lf = lf.with_columns(_cube().alias("y")) if how == "with_columns" else lf.select(_cube().alias("y"))
    _both_raise(t, "select a*a*a y from t", lf, MSG)


def test_a_projection_then_sort_past_int64_raises(mode):
    t = _t()
    _both_raise(t, "select *, a*a*a x from t order by x limit 2",
                LazyFrame(t).with_columns(_cube().alias("x")).sort("x").limit(2), MSG)


def test_a_group_key_and_a_projection_after_a_join_raise(mode):
    t, d = _t(), _dim()
    con = duckdb.connect()
    con.register("t", t)
    con.register("d", d)
    for sql in ("select a*a*a k, sum(v) from t join d using(g) group by k",
                "select name, a*a*a y from t join d using(g)"):
        with pytest.raises(duckdb.OutOfRangeException, match=MSG):
            con.execute(sql).fetchall()
    joined = LazyFrame(t).join(LazyFrame(d), on="g")
    with pytest.raises(OverflowError, match=MSG):
        joined.groupby(_cube().alias("k")).agg(col("v").sum().alias("x")).compute(device="cpu")
    with pytest.raises(OverflowError, match=MSG):
        joined.with_columns(_cube().alias("y")).compute(device="cpu")


def test_small_integer_keys_and_projections_are_exact(mode):
    t = pa.table({"a": A(np.arange(1, 9, dtype=np.int64)), "g": A(np.arange(8, dtype=np.int64) % 2)})
    got = LazyFrame(t).groupby((col("a") * col("a")).alias("k")).agg(col("g").sum().alias("x")).sort("k").compute(device="cpu")
    want = _con(t).execute("select a*a k, sum(g) from t group by k order by k").fetchall()
    assert list(zip(got.column("k").to_pylist(), got.column("x").to_pylist())) == want
    got = LazyFrame(t).with_columns((col("a") * col("a") * col("a")).alias("y")).compute(device="cpu")
    assert got.column("y").to_pylist() == [x ** 3 for x in range(1, 9)]


# -- SUM with nulls ---------------------------------------------------------------------------------

MAXI = (1 << 63) - 1


def _nt(values):
    return pa.table({"a": pa.array(values, type=pa.int64()), "g": pa.array([0] * len(values), type=pa.int64())})


def _sums(t, grouped):
    lf = LazyFrame(t)
    lf = lf.groupby("g") if grouped else lf.groupby()
    return lf.agg(col("a").sum().alias("s")).compute(device="cpu").column("s").to_pylist()


@pytest.mark.parametrize("grouped", [False, True], ids=["global", "grouped"])
@pytest.mark.parametrize("values,want", [
    ([1 << 62, None, -5], (1 << 62) - 5),
    ([None, None], None),
    ([MAXI, 1, None, -1], MAXI),
    ([1 << 62, None, -(1 << 62), 7], 7),
])
def test_sums_with_nulls_equal_duckdb(mode, grouped, values, want):
    t = _nt(values)
    assert _con(t).execute("select sum(a) from t").fetchone()[0] == want
    assert _sums(t, grouped) == [want]


@pytest.mark.parametrize("grouped", [False, True], ids=["global", "grouped"])
def test_a_sum_with_nulls_past_int64_raises(mode, grouped):
    t = _nt([1 << 62, None, 1 << 62])
    with pytest.raises(OverflowError, match="sum of INT64"):
        _sums(t, grouped)
