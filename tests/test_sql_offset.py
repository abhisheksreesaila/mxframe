"""F-087: SQL OFFSET. Every shape is checked against DuckDB, ties broken by the unique key `i` (or the
order is fully determined by the sort keys). OFFSET used to be dropped: `LIMIT 3 OFFSET 2` returned the
first three rows."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import gpu_stream


def _t(n=400):
    r = np.random.default_rng(5)
    b = r.integers(0, 12, n)
    return pa.table({
        "i": pa.array(np.arange(n), pa.int64()),
        "a": pa.array((b * 7 % 5).tolist(), pa.int64()),          # a is a function of b
        "b": pa.array([None if x == 11 else int(x) for x in b], pa.int64()),
        "c": pa.array(r.integers(0, 50, n), pa.int64()),
        "k": pa.array(np.array(list("xyz"))[r.integers(0, 3, n)].tolist(), pa.string()),
        "v": pa.array(r.integers(1, 100, n), pa.int64()),
    })


def _w():
    return pa.table({"k": pa.array(["x", "y", "z"], pa.string()), "x": pa.array([10, 20, 30], pa.int64())})


def _duck(q, t):
    """DuckDB's answer (the default connection renames a repeated output name to x_1, as omxcore does)."""
    w = _w()                                                    # noqa: F841 (found by DuckDB by name)
    exp = duckdb.sql(q).arrow()
    return exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp


def _compare(q, device="cpu", n=400):
    t = _t(n)
    exp = _duck(q, t)
    got = mx.sql(q, t=t, w=_w()).compute(device=device)
    assert got.column_names == exp.column_names, q
    assert got.to_pydict() == exp.to_pydict(), q


TOP = [
    "SELECT i FROM t ORDER BY c, i LIMIT 3 OFFSET 2",
    "SELECT i FROM t ORDER BY c DESC, i LIMIT 7 OFFSET 0",
    "SELECT i, c FROM t ORDER BY c, i LIMIT 1 OFFSET 399",
    "SELECT i FROM t ORDER BY c, i LIMIT 50 OFFSET 390",            # the window runs past the end
    "SELECT i FROM t ORDER BY c, i LIMIT 5 OFFSET 1000",            # OFFSET past the end: no rows
    "SELECT i FROM t ORDER BY c, i LIMIT 0 OFFSET 3",
    "SELECT i FROM t ORDER BY c, i OFFSET 390",                     # OFFSET without LIMIT
    "SELECT i FROM t ORDER BY c DESC, i OFFSET 1000",
    "SELECT i FROM t ORDER BY i OFFSET 0",
    "SELECT i FROM t LIMIT 5 OFFSET 3",                             # no ORDER BY: scan order
    "SELECT * FROM t OFFSET 395",
    "SELECT * FROM t ORDER BY c, i LIMIT 4 OFFSET 10",
    "SELECT i FROM t WHERE c > 10 ORDER BY c DESC, i LIMIT 5 OFFSET 4",
    "SELECT i FROM t WHERE c > 10 ORDER BY c DESC, i OFFSET 350",
    "SELECT i AS id FROM t ORDER BY 1 DESC LIMIT 3 OFFSET 2",
    # hidden sort keys (F-083)
    "SELECT a FROM t ORDER BY b, i LIMIT 5 OFFSET 3",
    "SELECT i FROM t ORDER BY b + c, i LIMIT 6 OFFSET 20",
    "SELECT a FROM t ORDER BY b, i OFFSET 395",
    # group by
    "SELECT k, sum(v) AS s FROM t GROUP BY k ORDER BY s DESC, k LIMIT 2 OFFSET 1",
    "SELECT k, sum(v) AS s FROM t GROUP BY k ORDER BY s DESC, k OFFSET 2",
    "SELECT k FROM t GROUP BY k ORDER BY sum(v) DESC, k LIMIT 2 OFFSET 1",       # hidden aggregate key
    "SELECT a, count(*) AS n FROM t GROUP BY a ORDER BY n DESC, a LIMIT 3 OFFSET 1",
    "SELECT DISTINCT a FROM t ORDER BY a LIMIT 2 OFFSET 1",
    # a join
    "SELECT t.i, w.x FROM t JOIN w ON t.k = w.k ORDER BY t.i LIMIT 4 OFFSET 6",
]

NESTED = [
    # FROM subqueries
    "SELECT i FROM (SELECT i, c FROM t ORDER BY c, i LIMIT 10 OFFSET 5) ORDER BY i DESC",
    "SELECT sum(v) AS s FROM (SELECT v FROM t ORDER BY i LIMIT 10 OFFSET 5)",
    "SELECT k, count(*) AS n FROM (SELECT k, i FROM t ORDER BY i OFFSET 300) GROUP BY k ORDER BY k",
    "SELECT i FROM (SELECT i FROM t ORDER BY i LIMIT 20 OFFSET 10) WHERE i > 15 ORDER BY i",
    "SELECT i FROM (SELECT i, c FROM t ORDER BY c, i LIMIT 30 OFFSET 5) ORDER BY c, i LIMIT 4 OFFSET 2",
    "SELECT * FROM (SELECT i FROM t ORDER BY i LIMIT 10 OFFSET 395)",
    "SELECT i FROM (SELECT i FROM t ORDER BY i LIMIT 10 OFFSET 5) AS s JOIN w ON s.i = w.x ORDER BY i",
    # CTEs
    "WITH s AS (SELECT i, c FROM t ORDER BY c, i LIMIT 10 OFFSET 5) SELECT sum(i) AS x FROM s",
    "WITH s AS (SELECT i FROM t ORDER BY c, i LIMIT 10 OFFSET 5) SELECT i FROM s ORDER BY i",
    "WITH s AS (SELECT i FROM t ORDER BY i OFFSET 390) SELECT count(*) AS n FROM s",
    # IN / NOT IN subqueries
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY c, i LIMIT 5 OFFSET 2) ORDER BY i",
    "SELECT i FROM t WHERE i NOT IN (SELECT i FROM t ORDER BY c, i LIMIT 5 OFFSET 2) ORDER BY i LIMIT 8",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i DESC OFFSET 395) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY i LIMIT 5 OFFSET 1000) ORDER BY i",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t WHERE c > 10 ORDER BY c DESC, i LIMIT 5 OFFSET 3) ORDER BY i",
    # the hidden-key form inside a subquery: the sort column is not the selected one
    "SELECT i FROM t WHERE c IN (SELECT a FROM t ORDER BY b, i LIMIT 5 OFFSET 3) ORDER BY i LIMIT 9",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY b, c, i LIMIT 5 OFFSET 3) ORDER BY i",
]


@pytest.mark.parametrize("q", TOP + NESTED)
def test_offset_matches_duckdb(q):
    _compare(q)


@pytest.mark.parametrize("q", TOP[:3] + [TOP[6], TOP[12], TOP[15], TOP[18]] + NESTED[:2] + NESTED[7:8] + NESTED[10:11])
def test_offset_on_a_larger_table(q):
    _compare(q, n=6000)


def test_dataframe_limit_takes_an_offset():
    f = mx.sql("SELECT i, c FROM t ORDER BY c, i", t=_t())
    full = f.compute(device="cpu").to_pydict()
    got = f.limit(5, offset=3).compute(device="cpu").to_pydict()
    assert got == {k: v[3:8] for k, v in full.items()}
    assert f.limit(5, offset=1000).compute(device="cpu").num_rows == 0
    assert f.limit(5).compute(device="cpu").to_pydict() == {k: v[:5] for k, v in full.items()}


def test_dataframe_offset_below_a_sort():
    # an OFFSET window decides which rows the sort above it sees
    f = mx.sql("SELECT i, c FROM t", t=_t()).limit(10, offset=5).sort("c", "i")
    exp = _duck("SELECT i, c FROM (SELECT i, c FROM t LIMIT 10 OFFSET 5) ORDER BY c, i", _t())
    assert f.compute(device="cpu").to_pydict() == exp.to_pydict()


def _topn_calls(q, monkeypatch, n):
    """The k every top-n pass was asked for: the sort kernel's `topk` and the native heap's k."""
    from omxcore import native_exec
    from omxcore.custom_ops import CustomOpsCompiler
    seen = {"sort": [], "heap": []}
    orig_sort = CustomOpsCompiler._apply_sort_custom

    def sort_spy(self, table, node, topk=None):
        seen["sort"].append(topk)
        return orig_sort(self, table, node, topk=topk)
    monkeypatch.setattr(CustomOpsCompiler, "_apply_sort_custom", sort_spy)
    orig_heap = native_exec.try_topk

    def heap_spy(table, by, desc, k, *a, **kw):
        seen["heap"].append(k)
        return orig_heap(table, by, desc, k, *a, **kw)
    monkeypatch.setattr(native_exec, "try_topk", heap_spy)
    got = mx.sql(q, t=_t(n)).compute(device="cpu")
    return got, seen


def test_top_n_keeps_limit_plus_offset_rows_then_slices(monkeypatch):
    q = "SELECT i FROM t ORDER BY c, i LIMIT 10 OFFSET 25"
    got, seen = _topn_calls(q, monkeypatch, 6000)
    assert seen["sort"] == [35]                      # the sort still sees Sort + Limit, with k = 35
    assert seen["heap"] == [35]                      # and the native per-worker heap ran for 35 rows
    assert got.to_pydict() == _duck(q, _t(6000)).to_pydict()


def test_top_n_with_a_hidden_key_and_offset(monkeypatch):
    q = "SELECT a FROM t ORDER BY b, i LIMIT 10 OFFSET 25"
    got, seen = _topn_calls(q, monkeypatch, 6000)
    assert seen["sort"] == [35] and seen["heap"] == [35]
    assert got.to_pydict() == _duck(q, _t(6000)).to_pydict()
    assert got.column_names == ["a"]


def test_filtered_top_n_keeps_the_native_path_with_an_offset(monkeypatch):
    from omxcore import native_exec
    ks = []
    orig = native_exec.try_filter_topk

    def spy(comp, table, preds, by, desc, k):
        ks.append(k)
        return orig(comp, table, preds, by, desc, k)
    monkeypatch.setattr(native_exec, "try_filter_topk", spy)
    q = "SELECT i FROM t WHERE c > 10 ORDER BY c DESC, i LIMIT 10 OFFSET 25"
    got = mx.sql(q, t=_t(6000)).compute(device="cpu")
    assert ks == [35]
    assert got.to_pydict() == _duck(q, _t(6000)).to_pydict()


@pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")
@pytest.mark.parametrize("q", [
    "SELECT i FROM t ORDER BY c, i LIMIT 3 OFFSET 2",
    "SELECT i FROM t ORDER BY c, i OFFSET 390",
    "SELECT a FROM t ORDER BY b, i LIMIT 5 OFFSET 3",
    "SELECT i FROM t WHERE i IN (SELECT i FROM t ORDER BY c, i LIMIT 5 OFFSET 2) ORDER BY i",
    "SELECT i FROM (SELECT i, c FROM t ORDER BY c, i LIMIT 10 OFFSET 5) ORDER BY i DESC",
])
def test_offset_on_gpu(q):
    _compare(q, device="gpu")
