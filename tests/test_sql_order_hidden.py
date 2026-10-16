"""F-083: ORDER BY a column (or expression, or aggregate) the SELECT list leaves out. Every shape is
checked against DuckDB, with and without LIMIT, with ties in the sort key and NULLs. SELECT DISTINCT
with ORDER BY keys outside the select list raises (DuckDB 1.5 answers with an arbitrary key per group)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe import gpu_stream


def _t():
    r = np.random.default_rng(3)
    n = 400
    b = r.integers(0, 12, n)                     # many ties in b
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "a": pa.array((b * 7 % 5).tolist(), pa.int64()),      # a is a function of b: tie order cannot show
        "b": pa.array([None if x == 11 else int(x) for x in b], pa.int64()),
        "c": pa.array(r.integers(0, 50, n), pa.int64()),
        "k": pa.array(np.array(list("xyz"))[r.integers(0, 3, n)].tolist(), pa.string()),
        "v": pa.array(r.integers(1, 100, n), pa.int64()),
    })


def _compare(q, device="cpu"):
    t = _t()
    exp = duckdb.sql(q).arrow()
    if isinstance(exp, pa.RecordBatchReader):
        exp = exp.read_all()
    got = mx.sql(q, t=t).compute(device=device)
    assert got.column_names == exp.column_names
    assert got.to_pydict() == exp.to_pydict()


# the order is fully determined: ties are broken by a unique column, or `a` is a function of `b`
PLAIN = [
    "SELECT a FROM t ORDER BY b",
    "SELECT a FROM t ORDER BY b DESC",
    "SELECT i FROM t ORDER BY b, i",
    "SELECT i FROM t ORDER BY b DESC, c, i",
    "SELECT i, a FROM t WHERE c > 10 ORDER BY c DESC, i",
    "SELECT i FROM t ORDER BY b + c, i",
    "SELECT i FROM t ORDER BY c * 2 - b DESC, i",
    "SELECT i AS id FROM t ORDER BY c, id",
    "SELECT a FROM t ORDER BY b, a",
    "SELECT * FROM t ORDER BY b, i",
]
LIMITED = [q + " LIMIT 7" for q in PLAIN]

GROUPED = [
    "SELECT k FROM t GROUP BY k ORDER BY sum(v) DESC",
    "SELECT k FROM t GROUP BY k ORDER BY sum(v) DESC, k",
    "SELECT k FROM t GROUP BY k ORDER BY max(c) - min(c), k",
    "SELECT k, count(*) AS n FROM t GROUP BY k ORDER BY sum(v), k",
    "SELECT a FROM t GROUP BY a ORDER BY avg(v), a",
]
GROUPED_LIMITED = [q + " LIMIT 2" for q in GROUPED]


@pytest.mark.parametrize("q", PLAIN + LIMITED + GROUPED + GROUPED_LIMITED)
def test_hidden_sort_key_matches_duckdb(q):
    _compare(q)


def test_hidden_key_is_not_in_the_result():
    got = mx.sql("SELECT a FROM t ORDER BY b, i", t=_t()).compute(device="cpu")
    assert got.column_names == ["a"]


@pytest.mark.parametrize("q", [
    "SELECT DISTINCT a FROM t ORDER BY b",
    "SELECT DISTINCT a FROM t ORDER BY a + b",
])
def test_distinct_order_by_unselected_raises(q):
    # DuckDB 1.5 accepts these and sorts by an arbitrary b of each group: a result that depends on
    # which row it saw, so here it is an error (the SQL standard's rule)
    with pytest.raises(ValueError, match="DISTINCT"):
        mx.sql(q, t=_t()).compute(device="cpu")


@pytest.mark.parametrize("q", [
    "SELECT DISTINCT a FROM t ORDER BY a",
    "SELECT DISTINCT a, b FROM t ORDER BY b, a LIMIT 5",
    "SELECT DISTINCT a AS x FROM t ORDER BY x DESC",
])
def test_distinct_order_by_selected_still_works(q):
    _compare(q)


@pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")
@pytest.mark.parametrize("q", [
    "SELECT a FROM t ORDER BY b",
    "SELECT i FROM t ORDER BY b + c, i LIMIT 7",
    "SELECT k FROM t GROUP BY k ORDER BY sum(v) DESC",
])
def test_hidden_sort_key_on_gpu(q):
    _compare(q, device="gpu")
