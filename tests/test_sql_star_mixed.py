"""F-088: `*` and `t.*` next to other SELECT items expand in place, in order. `SELECT *, a+1 AS x` used
to return only `x`; with ORDER BY it raised (F-083). Every shape is checked against DuckDB, ties broken
by the unique key `i`. A repeated output name is renamed `a_1` like DuckDB's default connection does."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import gpu_stream


def _t(n=400):
    r = np.random.default_rng(9)
    b = r.integers(0, 12, n)
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "a": pa.array((b * 7 % 5).tolist(), pa.int64()),
        "b": pa.array([None if x == 11 else int(x) for x in b], pa.int64()),
        "c": pa.array(r.integers(0, 50, n), pa.int64()),
        "k": pa.array(np.array(list("xyz"))[r.integers(0, 3, n)].tolist(), pa.string()),
        "v": pa.array(r.integers(1, 100, n), pa.int64()),
    })


def _w():
    return pa.table({"k": pa.array(["x", "y", "z"], pa.string()), "x": pa.array([10, 20, 30], pa.int64()),
                     "m": pa.array([1.5, 2.5, 3.5], pa.float64())})


def _duck(q, t):
    w = _w()                                                    # noqa: F841 (found by DuckDB by name)
    exp = duckdb.sql(q).arrow()
    return exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp


def _compare(q, device="cpu"):
    t = _t()
    exp = _duck(q, t)
    got = mx.sql(q, t=t, w=_w()).compute(device=device)
    assert got.column_names == exp.column_names, q
    assert got.to_pydict() == exp.to_pydict(), q


PLAIN = [
    "SELECT *, a + 1 AS x FROM t ORDER BY i",
    "SELECT a + 1 AS x, * FROM t ORDER BY i",
    "SELECT i, *, c * 2 AS d FROM t ORDER BY i",                        # i twice: the second is i_1
    "SELECT *, 1 AS one FROM t ORDER BY i LIMIT 5",
    "SELECT *, a + 1 AS x FROM t",                                      # no ORDER BY: scan order
    "SELECT *, a + 1 AS x FROM t WHERE c > 10 ORDER BY c, i LIMIT 7",
    "SELECT *, a + 1 AS x FROM t ORDER BY c, i LIMIT 5 OFFSET 2",
    "SELECT *, a + 1 AS x FROM t WHERE c > 10 ORDER BY c DESC, i",
    "SELECT *, k AS kk FROM t ORDER BY v DESC, i LIMIT 6",
    "SELECT t.*, v * 2 AS vv FROM t ORDER BY i LIMIT 9",
    "SELECT t.* FROM t ORDER BY c, i LIMIT 9",
    "SELECT v * 2 AS vv, t.* FROM t ORDER BY i LIMIT 9",
    "SELECT DISTINCT *, a AS aa FROM t ORDER BY i LIMIT 5",
]

# ORDER BY next to the star: an alias, a position, a column, an expression the select list has not got
ORDERED = [
    "SELECT *, a + 1 AS x FROM t ORDER BY x DESC, i",
    "SELECT *, a + 1 AS x FROM t ORDER BY 7 DESC, 1",                   # the star counts for its 6 columns
    "SELECT a + 1 AS x, * FROM t ORDER BY 1 DESC, 2",
    "SELECT *, a + 1 AS x FROM t ORDER BY b + c, i",                    # a hidden key
    "SELECT *, a + 1 AS x FROM t ORDER BY b + c DESC, i LIMIT 8",
    "SELECT *, a + 1 AS x FROM t ORDER BY a + 1, i",                    # the select list's own expression
    "SELECT a, *, v FROM t ORDER BY b, i",
    "SELECT t.*, c + v AS cv FROM t ORDER BY cv DESC, i LIMIT 5 OFFSET 1",
]

# a name in both the star and an alias: DuckDB names both columns, and ORDER BY a takes the alias
CLASHES = [
    "SELECT *, a + 1 AS a FROM t ORDER BY i",
    "SELECT *, a + 1 AS a FROM t ORDER BY a DESC, i",
    "SELECT *, b AS a FROM t ORDER BY a DESC, i",
    "SELECT *, i AS b FROM t ORDER BY b DESC",
    "SELECT a + 1 AS a, * FROM t ORDER BY a DESC, i",
    "SELECT *, a + 1 AS a FROM t ORDER BY t.a DESC, i",                  # t.a is the table's column
    "SELECT *, a + 1 AS a FROM t ORDER BY a + 1 DESC, i",
    "SELECT *, c AS i FROM t ORDER BY i, 1",
]

JOINED = [
    "SELECT t.*, w.x FROM t JOIN w ON t.k = w.k ORDER BY t.i",
    "SELECT w.x, t.* FROM t JOIN w ON t.k = w.k ORDER BY t.i LIMIT 12",
    "SELECT w.*, t.i FROM t JOIN w ON t.k = w.k ORDER BY t.i LIMIT 12",
    "SELECT t.*, w.x * 2 AS x2 FROM t JOIN w ON t.k = w.k WHERE t.c > 20 ORDER BY w.m DESC, t.i LIMIT 7",
    "SELECT t.* FROM t JOIN w ON t.k = w.k ORDER BY w.x, t.i LIMIT 7",   # a hidden key from the other table
    "SELECT t.*, w.m FROM t, w WHERE t.k = w.k ORDER BY t.i LIMIT 4",
]

NESTED = [
    "SELECT *, x * 2 AS y FROM (SELECT i, a + 1 AS x FROM t) ORDER BY i LIMIT 6",
    "WITH s AS (SELECT i, c FROM t WHERE c > 5) SELECT *, c + 1 AS d FROM s ORDER BY d DESC, i LIMIT 6",
    "SELECT *, c + 1 AS z FROM t WHERE i IN (SELECT i FROM t ORDER BY c, i LIMIT 5) ORDER BY i",
    "SELECT * FROM (SELECT *, c + 1 AS z FROM t ORDER BY z, i LIMIT 5) ORDER BY i DESC",
    "SELECT i FROM (SELECT *, c + 1 AS z FROM t) WHERE z > 30 ORDER BY z, i LIMIT 5",
    "SELECT sum(z) AS s FROM (SELECT *, c + 1 AS z FROM t)",
]


@pytest.mark.parametrize("q", PLAIN + ORDERED + CLASHES + JOINED + NESTED)
def test_star_among_other_items_matches_duckdb(q):
    _compare(q)


def test_star_expands_in_place_in_select_order():
    got = mx.sql("SELECT a + 1 AS x, *, v AS last FROM t ORDER BY i LIMIT 2", t=_t()).compute(device="cpu")
    assert got.column_names == ["x", "i", "a", "b", "c", "k", "v", "last"]
    assert got.column("x").to_pylist() == [got.column("a")[0].as_py() + 1, got.column("a")[1].as_py() + 1]


def test_star_with_a_hidden_key_hides_it():
    got = mx.sql("SELECT *, a + 1 AS x FROM t ORDER BY b + c, i", t=_t()).compute(device="cpu")
    assert got.column_names == ["i", "a", "b", "c", "k", "v", "x"]


@pytest.mark.parametrize("q", [
    "SELECT * EXCLUDE (a), 1 AS z FROM t ORDER BY i",
    "SELECT * REPLACE (a + 1 AS a), 1 AS z FROM t ORDER BY i",
])
def test_star_modifiers_next_to_other_items_raise(q):
    with pytest.raises(NotImplementedError, match="EXCLUDE|REPLACE|star"):
        mx.sql(q, t=_t()).compute(device="cpu")


def test_unknown_table_star_names_the_table():
    with pytest.raises(ValueError, match="'q'"):
        mx.sql("SELECT q.*, 1 AS z FROM t", t=_t()).compute(device="cpu")


@pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")
@pytest.mark.parametrize("q", [
    "SELECT *, a + 1 AS x FROM t ORDER BY i",
    "SELECT *, a + 1 AS x FROM t ORDER BY b + c, i LIMIT 8",
    "SELECT t.*, w.x FROM t JOIN w ON t.k = w.k ORDER BY t.i LIMIT 12",
])
def test_star_among_other_items_on_gpu(q):
    _compare(q, device="gpu")
