"""F-033 (M3): SELECT DISTINCT through the keyed group-by (design `partitioned-hash-join.md` 3.4).

`DISTINCT k1..kn` is `GROUP BY k1..kn` with a hidden `MIN(row id)`, groups sorted by it (first appearance,
as `mx_distinct` and the Python path give). Directly addressed keys (dictionary codes, small integer
ranges) run in ONE `mx_execute` call fused with the filter: no filtered table, no `mx_gather`. Over a star
join the same sink runs in the fused (v3) call with no user aggregate. Wide key spaces keep `mx_distinct`.
Every result is compared with DuckDB and with the old path; the path asserts count the native calls."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _dict_col(values, n_chunks=1, seed=0):
    """A dictionary column cut into chunks that each carry their own dictionary, in their own order."""
    r = np.random.default_rng(seed)
    cuts = np.linspace(0, len(values), n_chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vals = list(values[a:b])
        d = sorted(set(vals), key=lambda _: r.random())
        idx = {v: i for i, v in enumerate(d)}
        parts.append(pa.DictionaryArray.from_arrays(A(np.array([idx[v] for v in vals], np.int32)), pa.array(d)))
    return pa.chunked_array(parts)


def _table(n=150_000, seed=5, chunks=1):
    r = np.random.default_rng(seed)
    words = [f"w{i % 37}" for i in range(n)]
    t = pa.table({
        "a": A(r.integers(0, 40, n).astype(np.int32)),
        "b": A(r.integers(-5, 300, n).astype(np.int64)),
        "s": _dict_col([words[i] for i in r.integers(0, n, n)], chunks, seed),
        "u": A(np.arange(n, dtype=np.int64)[::-1].copy()),            # every row distinct
        "v": A(r.random(n)),
    })
    return t


@pytest.fixture
def calls(monkeypatch):
    seen = {"execute": 0, "distinct": 0, "filter": 0, "gather": 0}
    for key, name in (("execute", "_call"), ("distinct", "_call_distinct"), ("filter", "_call_filter"),
                      ("gather", "_call_gather")):
        real = getattr(nx, name)
        monkeypatch.setattr(nx, name, lambda *a, _r=real, _k=key: (seen.__setitem__(_k, seen[_k] + 1), _r(*a))[1])
    return seen


def _old(plan, t, monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    try:
        return plan(t).compute(device="cpu")
    finally:
        monkeypatch.setenv("MXFRAME_NATIVE", "1")


def _rows(t: pa.Table):
    return [tuple(c) for c in zip(*[t.column(i).to_pylist() for i in range(t.num_columns)])]


def _duck(t: pa.Table, sql: str):
    con = duckdb.connect()
    con.register("t", pa.table({n: t.column(n).cast(pa.string()) if pa.types.is_dictionary(t.column(n).type) else t.column(n)
                                for n in t.column_names}))
    return _rows(con.execute(sql).arrow().read_all())


@pytest.mark.parametrize("chunks", [1, 4])
@pytest.mark.parametrize("keys", [["a"], ["s"], ["a", "s"], ["b", "a"]])
def test_filtered_distinct_is_one_native_call_in_first_appearance_order(calls, monkeypatch, keys, chunks):
    t = _table(chunks=chunks)
    plan = lambda t: LazyFrame(t).filter(col("v") < lit(0.6)).select(*keys).distinct()
    got = plan(t).compute(device="cpu")
    assert (calls["execute"], calls["distinct"], calls["filter"], calls["gather"]) == (1, 0, 0, 0), (calls, nx.last_decline())
    old = _old(plan, t, monkeypatch)
    assert got.schema == old.schema
    assert _rows(got) == _rows(old)                                  # same rows, same (first appearance) order
    assert sorted(_rows(got), key=str) == sorted(_duck(t, f"SELECT DISTINCT {', '.join(keys)} FROM t WHERE v < 0.6"), key=str)


def test_distinct_without_a_filter_and_with_limit(calls, monkeypatch):
    t = _table()
    plan = lambda t: LazyFrame(t).select("a", "s").distinct().limit(25)
    got = plan(t).compute(device="cpu")
    assert (calls["execute"], calls["distinct"], calls["gather"]) == (1, 0, 0), (calls, nx.last_decline())
    assert _rows(got) == _rows(_old(plan, t, monkeypatch))
    assert got.num_rows == 25


def test_distinct_then_sort_runs_the_sort_on_the_few_groups(calls, monkeypatch):
    t = _table()
    plan = lambda t: LazyFrame(t).filter(col("a") > lit(3)).select("a", "b").distinct().sort("b", "a", descending=[True, False]).limit(15)
    got = plan(t).compute(device="cpu")
    assert calls["execute"] == 1 and calls["distinct"] == 0
    assert _rows(got) == _duck(t, "SELECT DISTINCT a, b FROM t WHERE a > 3 ORDER BY b DESC, a LIMIT 15")


@pytest.mark.parametrize("keys", [["u"], ["b", "a", "s"]])      # 150k groups; a key space of 305 * 40 * 37 = 451k
def test_wide_key_space_keeps_mx_distinct(calls, monkeypatch, keys):
    """The hash-partitioned `mx_distinct` answers, not the group-by (which declines the key space)."""
    t = _table()
    plan = lambda t: LazyFrame(t).filter(col("v") < lit(0.9)).select(*keys).distinct()
    got = plan(t).compute(device="cpu")
    assert calls["distinct"] == 1, (calls, nx.last_decline())
    assert _rows(got) == _rows(_old(plan, t, monkeypatch))


@pytest.mark.parametrize("make", [
    lambda: pa.table({"x": pa.array([1.5, 1.5, 2.0] * 1000)}),                       # float column
    lambda: pa.table({"x": pa.array([1, None, 1] * 1000, pa.int64())}),              # nulls
    lambda: pa.table({"x": pa.array(["p", "q", "p"] * 1000)}),                       # plain strings: mx_distinct
])
def test_other_columns_keep_the_old_answer(calls, monkeypatch, make):
    t = make()
    plan = lambda t: LazyFrame(t).select("x").distinct()
    got = plan(t).compute(device="cpu")
    assert calls["execute"] == 0
    old = _old(plan, t, monkeypatch)
    assert got.schema == old.schema and _rows(got) == _rows(old)


def test_empty_after_the_filter(calls, monkeypatch):
    t = _table()
    plan = lambda t: LazyFrame(t).filter(col("v") > lit(5.0)).select("a", "s").distinct()
    got = plan(t).compute(device="cpu")
    assert got.num_rows == 0 and got.column_names == ["a", "s"]


def _star(n=120_000, seed=9):
    r = np.random.default_rng(seed)
    fact = pa.table({
        "k": A(r.integers(-20, 5200, n).astype(np.int64)),
        "g": A(r.integers(0, 6, n).astype(np.int32)),
        "v": A(np.round(r.random(n) * 100, 2)),
    })
    m = 5000
    dim = pa.table({
        "dk": A(np.arange(m, dtype=np.int64)),
        "cls": A(r.integers(0, 9, m).astype(np.int32)),
        "nm": _dict_col([f"n{i}" for i in r.integers(0, 7, m)], 1, seed + 1),
    })
    return fact, dim


def test_distinct_over_a_star_join_is_one_fused_call(calls, monkeypatch):
    fact, dim = _star()
    plan = lambda t: (LazyFrame(t).filter(col("v") > lit(20.0)).join(LazyFrame(dim), left_on="k", right_on="dk")
                      .select("g", "cls", "nm").distinct())
    got = plan(fact).compute(device="cpu")
    assert (calls["execute"], calls["filter"], calls["distinct"], calls["gather"]) == (1, 0, 0, 0), (calls, nx.last_decline())
    con = duckdb.connect()
    con.register("f", fact)
    con.register("d", pa.table({"dk": dim.column("dk"), "cls": dim.column("cls"), "nm": dim.column("nm").cast(pa.string())}))
    want = _rows(con.execute("SELECT DISTINCT g, cls, nm FROM f JOIN d ON k = dk WHERE v > 20 ").arrow().read_all())
    assert sorted(_rows(got), key=str) == sorted(want, key=str)
    old = _old(plan, fact, monkeypatch)
    assert _rows(got) == _rows(old)                                  # the join's output order: the fact table's rows
