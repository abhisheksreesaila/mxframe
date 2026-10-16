"""F-059: ORDER BY ties when a Project sits between the Sort and the group-by.

`groupby(...).agg(...).with_columns(y=...).sort(y)` must break ties by the group keys, like
`ORDER BY y, keys` in DuckDB, on both MXFRAME_NATIVE settings. A Project that drops a key leaves
nothing to break ties on (no tie-break, no crash)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A


def _table(n=300, seed=11):
    r = np.random.default_rng(seed)
    names = np.array(["delta", "alpha", "echo", "charlie", "bravo", "foxtrot"])
    sd = pa.array(names[r.integers(0, len(names), n)].tolist()).dictionary_encode()
    return pa.table({"s": sd,
                     "k": A(r.integers(0, 8, n).astype(np.int64)),
                     "g": A(r.integers(0, 40, n).astype(np.int64)),    # 40 integer groups, shuffled
                     "v": A(r.integers(0, 3, n).astype(np.int64))})   # tiny sums -> many ties


def _duck(sql, t):
    plain = t.cast(pa.schema([f.with_type(f.type.value_type) if pa.types.is_dictionary(f.type) else f
                              for f in t.schema]))
    con = duckdb.connect()
    con.register("t", plain)
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _rows(res, cols):
    d = res.select(cols).to_pydict()
    return list(zip(*[d[c] for c in cols]))


def _agg(lf):
    return lf.groupby("s", "k").agg(col("v").sum().alias("x"))


QUERIES = {
    "with_columns_desc_limit": (
        lambda lf: _agg(lf).with_columns((col("x") * lit(2)).alias("y")).sort("y", descending=[True]).limit(10),
        "SELECT s, k, x, 2*x y FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k) ORDER BY y DESC, s, k LIMIT 10",
        ["s", "k", "x", "y"]),
    "with_columns_asc_all_rows": (
        lambda lf: _agg(lf).with_columns((col("x") + lit(1)).alias("y")).sort("y"),
        "SELECT s, k, x, x+1 y FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k) ORDER BY y, s, k",
        ["s", "k", "x", "y"]),
    "select_keeps_keys": (
        lambda lf: _agg(lf).select(col("s"), col("k"), (col("x") * lit(3)).alias("y"))
        .sort("y", descending=[True]).limit(9),
        "SELECT s, k, 3*x y FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k) ORDER BY y DESC, s, k LIMIT 9",
        ["s", "k", "y"]),
    "two_projects": (
        lambda lf: _agg(lf).with_columns((col("x") * lit(2)).alias("y")).with_columns((col("y") + lit(1)).alias("z"))
        .sort("z", descending=[True]).limit(8),
        "SELECT s, k, z FROM (SELECT s, k, 2*x+1 z FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k)) "
        "ORDER BY z DESC, s, k LIMIT 8",
        ["s", "k", "z"]),
    "one_int_key_with_columns": (
        lambda lf: lf.groupby("g").agg(col("v").count().alias("c")).with_columns((col("c") * lit(2)).alias("y"))
        .sort("y", descending=[True]),
        "SELECT g, 2*c y FROM (SELECT g, count(v) c FROM t GROUP BY g) ORDER BY y DESC, g",
        ["g", "y"]),
    "with_columns_then_filter": (
        lambda lf: _agg(lf).with_columns((col("x") * lit(2)).alias("y")).filter(col("y") > lit(0))
        .sort("y", descending=[True]).limit(10),
        "SELECT s, k, y FROM (SELECT s, k, 2*x y FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k)) "
        "WHERE y > 0 ORDER BY y DESC, s, k LIMIT 10",
        ["s", "k", "y"]),
    "filter_then_with_columns": (
        lambda lf: _agg(lf).filter(col("x") > lit(0)).with_columns((col("x") * lit(2)).alias("y"))
        .sort("y", descending=[True]).limit(10),
        "SELECT s, k, 2*x y FROM (SELECT s, k, sum(v) x FROM t GROUP BY s, k) "
        "WHERE x > 0 ORDER BY y DESC, s, k LIMIT 10",
        ["s", "k", "y"]),
}


@pytest.mark.parametrize("name", list(QUERIES))
@pytest.mark.parametrize("native", ["1", "0"])
def test_ties_follow_group_key_order_through_project(name, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    build, sql, cols = QUERIES[name]
    t = _table()
    got = _rows(build(LazyFrame(t)).compute(device="cpu"), cols)
    want = _duck(sql, t)
    assert len(want) > len({r[-1] for r in want})        # the data does have ties on the sort column
    assert got == want


@pytest.mark.parametrize("native", ["1", "0"])
def test_project_that_drops_a_key_still_sorts(native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    t = _table()
    res = _agg(LazyFrame(t)).select(col("s"), (col("x") * lit(2)).alias("y")) \
        .sort("y", descending=[True]).limit(5).compute(device="cpu")
    rows = _rows(res, ["y", "s"])
    assert res.column_names == ["s", "y"]
    assert len(rows) == 5 and rows == sorted(rows, key=lambda r: (-r[0], r[1]))
