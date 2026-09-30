"""F-016: operators above an aggregate, two-level groupby, empty global aggregates (vs DuckDB)."""
import math

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import conftest  # noqa: F401
from mxframe import LazyFrame, col, lit
from mxframe.lazy_frame import Scan


def _table():
    rng = np.random.default_rng(3)
    n = 200
    return pa.table({"k": pa.array(rng.integers(0, 6, n).astype(np.int64)),
                     "j": pa.array(rng.integers(0, 3, n).astype(np.int64)),
                     "v": pa.array(rng.normal(size=n).round(3))})


def _duck(sql, t):
    con = duckdb.connect()
    con.register("t", t)
    return con.execute(sql).fetch_arrow_table()


def _same(got, ref):
    assert got.num_rows == ref.num_rows, (got.num_rows, ref.num_rows)
    assert set(got.column_names) == set(ref.column_names), (got.column_names, ref.column_names)
    names = sorted(ref.column_names)
    g = sorted(zip(*[got.column(n).to_pylist() for n in names]), key=repr)
    r = sorted(zip(*[ref.column(n).to_pylist() for n in names]), key=repr)
    for a, b in zip(g, r):
        for x, y in zip(a, b):
            if x is None or y is None:
                assert x is None and y is None, (a, b)
            elif isinstance(x, float) or isinstance(y, float):
                assert math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6), (a, b)
            else:
                assert x == y, (a, b)


@pytest.fixture(params=["cpu", "auto"])
def device(request):
    return request.param


def _lf():
    return LazyFrame(Scan(_table()))


def test_with_columns_after_aggregate(device):
    got = (_lf().groupby("k").agg(col("v").count().alias("n"), col("v").sum().alias("s"))
           .with_columns((col("n") / lit(2)).alias("half")).compute(device=device))
    ref = _duck("select k, count(v) n, sum(v) s, count(v)/2 half from t group by k", _table())
    _same(got, ref)


def test_with_columns_replaces_aggregate_column(device):
    got = (_lf().groupby("k").agg(col("v").count().alias("n"))
           .with_columns((col("n") / lit(2)).alias("n")).compute(device=device))
    ref = _duck("select k, count(v)/2 n from t group by k", _table())
    _same(got, ref)


def test_select_after_aggregate(device):
    got = (_lf().groupby("k").agg(col("v").sum().alias("s"))
           .select(col("k"), (col("s") * lit(2)).alias("s2")).compute(device=device))
    ref = _duck("select k, sum(v)*2 s2 from t group by k", _table())
    _same(got, ref)


def test_filter_after_with_columns_after_aggregate(device):
    got = (_lf().groupby("k").agg(col("v").count().alias("n"))
           .with_columns((col("n") * lit(2)).alias("m")).filter(col("m") > lit(60)).compute(device=device))
    ref = _duck("select k, n, m from (select k, count(v) n, count(v)*2 m from t group by k) where m > 60", _table())
    _same(got, ref)


def test_global_aggregate_then_with_columns(device):
    got = (_lf().groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n"))
           .with_columns((col("s") / lit(7.0)).alias("x")).compute(device=device))
    ref = _duck("select sum(v) s, count(v) n, sum(v)/7.0 x from t", _table())
    _same(got, ref)


def test_two_level_groupby(device):
    got = (_lf().groupby("k", "j").agg(col("v").sum().alias("s"))
           .groupby("k").agg(col("s").max().alias("mx"), col("s").count().alias("c")).compute(device=device))
    ref = _duck("select k, max(s) mx, count(s) c from (select k, j, sum(v) s from t group by k, j) group by k", _table())
    _same(got, ref)


def test_groupby_on_an_aggregate_column(device):
    got = (_lf().groupby("k").agg(col("v").count().alias("n"))
           .groupby("n").agg(col("k").count().alias("ng")).compute(device=device))
    ref = _duck("select n, count(k) ng from (select k, count(v) n from t group by k) group by n", _table())
    _same(got, ref)


def test_global_aggregate_over_grouped_aggregate(device):
    got = (_lf().groupby("k").agg(col("v").sum().alias("s"))
           .groupby().agg(col("s").max().alias("mx"), col("s").sum().alias("tot")).compute(device=device))
    ref = _duck("select max(s) mx, sum(s) tot from (select k, sum(v) s from t group by k)", _table())
    _same(got, ref)


def test_sort_limit_after_with_columns_after_aggregate(device):
    got = (_lf().groupby("k").agg(col("v").sum().alias("s"))
           .with_columns((col("s") * lit(-1.0)).alias("neg")).sort("neg").limit(3).compute(device=device))
    ref = _duck("select k, s, neg from (select k, sum(v) s, -sum(v) neg from t group by k) order by neg limit 3", _table())
    _same(got, ref)
    assert got.column("neg").to_pylist() == sorted(got.column("neg").to_pylist())


def test_join_on_aggregate_result_then_project(device):
    a = _lf().groupby("k").agg(col("v").sum().alias("s"))
    b = _lf().groupby("k").agg(col("v").count().alias("n"))
    got = (a.join(b, on="k").with_columns((col("s") / col("n")).alias("avg")).compute(device=device))
    ref = _duck("select k, s, n, s/n avg from (select k, sum(v) s from t group by k) join "
                "(select k, count(v) n from t group by k) using (k)", _table())
    _same(got, ref)


@pytest.mark.parametrize("path", ["fused", "unfused"])
def test_empty_global_aggregates_are_null(device, path, monkeypatch):
    if path == "unfused":
        monkeypatch.setenv("MXFRAME_FUSED_PIPELINE", "0")
    lf = _lf().filter(col("v") > lit(1e9))
    got = lf.groupby().agg(col("v").sum().alias("s"), col("v").min().alias("lo"), col("v").max().alias("hi"),
                           col("v").mean().alias("m"), col("v").count().alias("n")).compute(device=device)
    ref = _duck("select sum(v) s, min(v) lo, max(v) hi, avg(v) m, count(v) n from t where v > 1e9", _table())
    _same(got, ref)
    assert got.column("n").to_pylist() == [0]
    assert got.column("s").to_pylist() == [None]


def test_empty_global_sum_of_product_is_null(device):
    got = (_lf().filter(col("v") > lit(1e9)).groupby().agg((col("v") * col("j")).sum().alias("s"))
           .compute(device=device))
    assert got.column("s").to_pylist() == [None]


def test_empty_table_global_aggregates(device):
    t = _table().slice(0, 0)
    got = (LazyFrame(Scan(t)).groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n"))
           .compute(device=device))
    assert got.column("s").to_pylist() == [None] and got.column("n").to_pylist() == [0]


# --- found by the fuzz (tests/test_f016_fuzz.py) -------------------------------------------------

def test_filter_over_project_under_aggregate_with_nulls(device):
    t = pa.table({"k": pa.array([1, 1, 2, None, 2], type=pa.int64()),
                  "v": pa.array([1.5, None, 2.5, 3.5, 4.5])})
    got = (LazyFrame(Scan(t)).with_columns((col("v") * lit(2)).alias("w")).filter(col("w") > lit(3.0))
           .groupby("k").agg(col("w").sum().alias("s"), col("v").count().alias("n")).compute(device=device))
    ref = _duck("select k, sum(w) s, count(v) n from (select k, v, v*2 w from t) where w > 3.0 group by k", t)
    _same(got, ref)


def test_integer_column_with_integer_literal_stays_exact_integer(device):
    big = (1 << 24) + 1
    t = pa.table({"a": pa.array([big, 5], type=pa.int64())})
    got = LazyFrame(Scan(t)).with_columns((col("a") + lit(2)).alias("b"), (col("a") / lit(2)).alias("h")).compute(device=device)
    assert got.column("b").to_pylist() == [big + 2, 7]
    assert pa.types.is_integer(got.schema.field("b").type)
    assert got.column("h").to_pylist() == [big / 2, 2.5]


def test_float64_column_keeps_float64_literal(device):
    t = pa.table({"a": pa.array([1.0, 2.0])})
    got = LazyFrame(Scan(t)).with_columns((col("a") + lit(0.1)).alias("b")).compute(device=device)
    assert got.column("b").to_pylist() == [1.0 + 0.1, 2.0 + 0.1]


def test_filter_on_a_global_aggregate_is_applied(device):
    got = (_lf().filter(col("v") > lit(1e9)).groupby().agg(col("v").min().alias("m"))
           .filter(col("m") == lit(0)).compute(device=device))
    assert got.num_rows == 0
    got = (_lf().groupby().agg(col("v").count().alias("n")).filter(col("n") > lit(5)).compute(device=device))
    assert got.column("n").to_pylist() == [200]


def test_sort_treats_negative_zero_as_zero(device):
    t = pa.table({"a": pa.array([0.0, -0.0, 0.0, -0.0]), "b": pa.array([4, 3, 2, 1], type=pa.int64())})
    got = LazyFrame(Scan(t)).sort("a", "b").compute(device=device)
    assert got.column("b").to_pylist() == [1, 2, 3, 4]
