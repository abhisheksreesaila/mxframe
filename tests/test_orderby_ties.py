"""F-045: ORDER BY ties after a group-by come out in group-key order on every path.

The native executor breaks ties by the group keys, ascending (strings by bytes); the Python /
fused / kernel fallbacks used to keep whatever order their hash table produced. Each query runs
natively (when it can), with MXFRAME_NATIVE=0, and in DuckDB with the keys as the last ORDER BY
columns: the row orders must be identical."""
import datetime as dt

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A


def _table(n=4000, seed=7):
    r = np.random.default_rng(seed)
    names = np.array(["delta", "alpha", "echo", "charlie", "bravo", "foxtrot"])
    days = (r.integers(0, 6, n) + 19000).astype(np.int32)
    s = names[r.integers(0, len(names), n)]
    # dictionary strings in first-appearance order (not sorted), so code order != byte order
    sd = pa.array(s.tolist()).dictionary_encode()
    return pa.table({
        "d": pa.array(days, pa.date32()),
        "s": sd,
        "k": A(r.integers(0, 5, n).astype(np.int64)),
        "v": A(r.integers(0, 3, n).astype(np.int64)),      # tiny sums -> many ties
    })


def _duck(sql, t):
    plain = t.cast(pa.schema([f.with_type(f.type.value_type) if pa.types.is_dictionary(f.type) else f
                              for f in t.schema]))
    con = duckdb.connect()
    con.register("t", plain)
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _rows(res, cols):
    d = res.select(cols).to_pydict()
    return list(zip(*[d[c] for c in cols]))


QUERIES = {
    "date_str_keys_by_count": (
        lambda lf: lf.groupby("d", "s").agg(col("v").count().alias("c")).sort("c", descending=[True]),
        "SELECT d, s, count(v) c FROM t GROUP BY d, s ORDER BY c DESC, d, s", ["d", "s", "c"]),
    "date_str_keys_by_sum_limit": (
        lambda lf: lf.groupby("d", "s").agg(col("v").sum().alias("x")).sort("x", descending=[False]).limit(7),
        "SELECT d, s, sum(v) x FROM t GROUP BY d, s ORDER BY x, d, s LIMIT 7", ["d", "s", "x"]),
    "str_int_keys_desc_sort": (
        lambda lf: lf.groupby("s", "k").agg(col("v").sum().alias("x")).sort("x", descending=[True]).limit(9),
        "SELECT s, k, sum(v) x FROM t GROUP BY s, k ORDER BY x DESC, s, k LIMIT 9", ["s", "k", "x"]),
    "filtered_str_key": (
        lambda lf: lf.filter(col("k") < lit(4)).groupby("s").agg(col("v").sum().alias("x"))
        .sort("x", descending=[True]),
        "SELECT s, sum(v) x FROM t WHERE k < 4 GROUP BY s ORDER BY x DESC, s", ["s", "x"]),
}


@pytest.mark.parametrize("name", list(QUERIES))
@pytest.mark.parametrize("native", ["1", "0"])
def test_ties_follow_group_key_order(name, native, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    build, sql, cols = QUERIES[name]
    t = _table()
    got = _rows(build(LazyFrame(t)).compute(device="cpu"), cols)
    want = _duck(sql, t)
    norm = lambda rows: [tuple(x.isoformat() if isinstance(x, dt.date) else x for x in r) for r in rows]
    assert norm(got) == norm(want)
