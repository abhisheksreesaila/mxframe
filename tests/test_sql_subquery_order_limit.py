"""F-085: ORDER BY + LIMIT inside an IN / NOT IN / EXISTS subquery picks the rows the outer query
sees. The subquery's ORDER BY used to be dropped while its LIMIT stayed, so
`WHERE i IN (SELECT i FROM t ORDER BY i DESC LIMIT 5)` returned 0..4 where DuckDB returns 45..49.
Every shape is checked against DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx


def _t():
    r = np.random.default_rng(5)
    n = 50
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "c": pa.array(r.integers(0, 10, n), pa.int64()),
        "k": pa.array(np.arange(n) % 4, pa.int64()),
    })


def _u():
    return pa.table({"i": pa.array([1, 7, 45, 46, 48, 60], pa.int64()), "x": pa.array([1, 2, 3, 4, 5, 6], pa.int64())})


QUERIES = [
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i DESC LIMIT 5) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i LIMIT 5) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY c, i LIMIT 5) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY c DESC, i DESC LIMIT 5) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i AS j FROM t ORDER BY c, i LIMIT 3) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t WHERE k = 1 ORDER BY i DESC LIMIT 4) ORDER BY i",
    "SELECT i FROM t WHERE i NOT IN (SELECT i FROM t ORDER BY i DESC LIMIT 45) ORDER BY i",
    "SELECT i FROM t WHERE i NOT IN (SELECT i FROM t ORDER BY c, i LIMIT 40) ORDER BY i",
    "SELECT i, x FROM u WHERE i IN (SELECT i FROM t ORDER BY i DESC LIMIT 5) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i DESC) ORDER BY i LIMIT 3",
]


@pytest.mark.parametrize("q", QUERIES)
def test_subquery_order_by_limit_matches_duckdb(q):
    t, u = _t(), _u()
    exp = duckdb.sql(q).fetch_arrow_table()
    got = mx.sql(q, t=t, u=u).compute(device="cpu")
    assert got.to_pydict() == exp.to_pydict()


def test_the_reported_query():
    t = _t()
    got = mx.sql("SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i DESC LIMIT 5) ORDER BY i", t=t).compute(device="cpu")
    assert got.column("i").to_pylist() == [45, 46, 47, 48, 49]


@pytest.mark.parametrize("q", [
    "SELECT i FROM t WHERE EXISTS (SELECT 1 FROM t AS s ORDER BY s.i LIMIT 1) ORDER BY i LIMIT 2",
    "SELECT i FROM t WHERE EXISTS (SELECT s.i FROM t AS s WHERE s.i = t.i ORDER BY s.c LIMIT 1) ORDER BY i LIMIT 2",
    "SELECT i FROM t WHERE c > (SELECT max(c) FROM t ORDER BY 1 LIMIT 1) ORDER BY i",
])
def test_other_subquery_forms_with_limit_match_duckdb_or_raise(q):
    t = _t()
    exp = duckdb.sql(q).fetch_arrow_table()
    try:
        got = mx.sql(q, t=t).compute(device="cpu")
    except (NotImplementedError, ValueError):
        return                                     # declined loudly: never a wrong answer
    assert got.to_pydict() == exp.to_pydict()
