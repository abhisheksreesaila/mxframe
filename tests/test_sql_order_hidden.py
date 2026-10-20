"""F-083: ORDER BY a column (or expression, or aggregate) the SELECT list leaves out. Every shape is
checked against DuckDB, with and without LIMIT, with ties in the sort key and NULLs. SELECT DISTINCT
with ORDER BY keys outside the select list raises (DuckDB 1.5 answers with an arbitrary key per group)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import gpu_stream


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


# ---- review fixes -------------------------------------------------------------------------------

def _post_op_types(q, monkeypatch, **tables):
    """The Sort / Limit nodes the compiler peeled off the top of the plan, per call."""
    from omxcore.compiler import GraphCompiler
    seen = []
    orig = GraphCompiler._extract_post_ops.__func__

    def spy(cls, plan):
        core, ops = orig(cls, plan)
        seen.append([type(o).__name__ for o in ops])
        return core, ops
    monkeypatch.setattr(GraphCompiler, "_extract_post_ops", classmethod(spy))
    mx.sql(q, **tables).compute(device="cpu")
    return seen


def test_hidden_key_keeps_the_top_n_post_ops(monkeypatch):
    t = _t()
    hidden = _post_op_types("SELECT a FROM t ORDER BY b LIMIT 7", monkeypatch, t=t)
    selected = _post_op_types("SELECT a, b FROM t ORDER BY b LIMIT 7", monkeypatch, t=t)
    assert selected[0] == ["Sort", "Limit"]
    assert hidden[0] == selected[0]


def test_hidden_key_name_never_collides():
    u = pa.table({"__ord0": pa.array([1, 2, 3], pa.int64()), "b": pa.array([3, 1, 2], pa.int64())})
    for q in ["SELECT __ord0 FROM u ORDER BY b", "SELECT __ord0 FROM u ORDER BY b + 1",
              "SELECT b AS __ord0 FROM u ORDER BY __ord0 + 1", "SELECT 1 AS __ord0 FROM u ORDER BY b"]:
        exp = duckdb.sql(q).fetch_arrow_table()
        got = mx.sql(q, u=u).compute(device="cpu")
        assert got.to_pydict() == exp.to_pydict(), q
    t = _t().append_column("__ord0", pa.array(np.arange(400)[::-1], pa.int64()))
    q = "SELECT a AS __ord1 FROM t ORDER BY b + c, i"
    assert mx.sql(q, t=t).compute(device="cpu").to_pydict() == duckdb.sql(q).fetch_arrow_table().to_pydict()


def test_dataframe_sort_by_dropped_column_names_it():
    f = mx.sql("SELECT * FROM t", t=_t())
    assert f.sort("b").select("a").compute(device="cpu").column_names == ["a"]      # sort before select
    with pytest.raises(ValueError, match="'b'.*sort before select"):
        f.select("a").sort("b")
