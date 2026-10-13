"""F-032 (M2h): DISTINCT in one native call (`mx_distinct`): every row's tuple is hashed in parallel,
a per-block cache drops rows repeating an earlier row of their block, the rest go to 64 partitions
in row order, and each partition keeps the first row of every tuple. The kept rows, in row order,
are the distinct rows in first-appearance order, exactly what the Python path returns.
Also: plain-string group keys are interned in parallel (`_build_row_strings`)."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

from test_native_exec_m1 import _both, _same, native  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _table(n=200_000, seed=3, chunks=1, distinct_strings=500):
    r = np.random.default_rng(seed)
    words = np.array([f"w{i:04d}-{'x' * (i % 13)}" for i in range(distinct_strings)], dtype=object)
    t = pa.table({
        "a": A(r.integers(0, 40, n).astype(np.int32)),
        "b": A(r.integers(0, 3000, n).astype(np.int64)),
        "s": pa.array(words[r.integers(0, distinct_strings, n)].tolist(), pa.string()),
        "u": A(np.arange(n, dtype=np.int64)),                     # every row distinct
        "v": A(r.random(n)),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.Table.from_batches([b for a, z in zip(cuts[:-1], cuts[1:]) for b in t.slice(a, z - a).to_batches()])
    return t


@pytest.fixture
def distinct_calls(monkeypatch):
    calls = []
    real = nx._call_distinct

    def counting(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(nx, "_call_distinct", counting)
    return calls


@pytest.mark.parametrize("chunks", [1, 5])
@pytest.mark.parametrize("cols", [["a"], ["s"], ["a", "s"], ["a", "b", "s"], ["b", "u"], ["s", "a", "b", "u"]])
def test_distinct_matches_the_python_path_in_first_appearance_order(native, monkeypatch, distinct_calls, cols, chunks):
    t = _table(chunks=chunks)
    got, old, _, _ = _both(lambda t: LazyFrame(t).select(*cols).distinct(), t, monkeypatch)
    assert distinct_calls, nx.last_decline()
    assert got.schema == old.schema
    assert got.num_rows == old.num_rows
    for c in cols:                                       # same rows, same order
        assert got.column(c).to_pylist() == old.column(c).to_pylist(), c


def test_all_rows_equal_and_all_rows_distinct(native, monkeypatch, distinct_calls):
    n = 150_000
    same = pa.table({"k": A(np.full(n, 7, np.int64)), "s": pa.array(["x"] * n)})
    got, old, _, _ = _both(lambda t: LazyFrame(t).distinct(), same, monkeypatch)
    assert got.num_rows == 1 and got.to_pylist() == old.to_pylist()
    uniq = pa.table({"k": A(np.arange(n, dtype=np.int32)[::-1].copy())})
    got, old, _, _ = _both(lambda t: LazyFrame(t).distinct(), uniq, monkeypatch)
    assert got.column("k").to_pylist() == old.column("k").to_pylist()
    assert len(distinct_calls) == 2


def test_hash_collisions_across_strings_of_different_lengths(native, monkeypatch, distinct_calls):
    """Equal hashes alone never merge rows: values are compared, and "" vs a missing tail differ."""
    vals = ["", "a", "aa", "a\x00", "\x00", "ab", "ba", "", "a"] * 20_000
    t = pa.table({"s": pa.array(vals), "k": A(np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 0, 0], np.int32), 20_000))})
    got, old, _, _ = _both(lambda t: LazyFrame(t).distinct(), t, monkeypatch)
    assert got.to_pylist() == old.to_pylist()
    assert got.num_rows == 7


@pytest.mark.parametrize("make", [
    lambda: pa.table({"x": pa.array([1.5, 1.5, 2.0] * 1000)}),                       # float column
    lambda: pa.table({"x": pa.array([1, None, 1] * 1000, pa.int64())}),              # nulls
    lambda: pa.table({"x": pa.array((np.arange(3000) % 7).astype("datetime64[D]"))}),   # date32
])
def test_other_columns_decline_with_the_same_answer(native, monkeypatch, distinct_calls, make):
    t = make()
    got, old, _, _ = _both(lambda t: LazyFrame(t).distinct(), t, monkeypatch)
    assert not distinct_calls
    assert got.schema == old.schema and got.to_pylist() == old.to_pylist()


def test_distinct_after_a_join_is_native(native, monkeypatch, distinct_calls):
    """Q16's shape: a star join's joined rows (strings decoded by the join), then DISTINCT."""
    t = _table(n=120_000)
    dim = pa.table({"b": A(np.arange(3000, dtype=np.int64)), "tag": pa.array([f"T{i % 17}" for i in range(3000)])})

    def plan(t):
        return (LazyFrame(t).filter(col("a") < lit(30)).join(LazyFrame(dim), on="b")
                .select("tag", "a", "s").distinct())

    got, old, _, _ = _both(plan, t, monkeypatch)
    assert distinct_calls
    _same(got, old)


@pytest.mark.parametrize("chunks", [1, 4])
def test_many_plain_string_keys_across_blocks_and_partitions(native, monkeypatch, chunks):
    """Plain-string group keys over several 64k-row blocks: interned in parallel (partition-major
    entries), results still in key (byte) order."""
    t = _table(n=300_000, chunks=chunks, distinct_strings=5000)
    for plan, ordered in [
        (lambda t: LazyFrame(t).groupby("s").agg(col("v").sum().alias("x"), col("v").count().alias("n")), False),
        (lambda t: LazyFrame(t).filter(col("a") < lit(10)).groupby("s", "a").agg(col("v").max().alias("x"))
            .sort("x", descending=[True]).limit(50), True),
    ]:
        got, old, path, decline = _both(plan, t, monkeypatch)
        assert path == "native", decline
        _same(got, old, ordered=ordered)
        if not ordered:                       # no ORDER BY: native groups come out in key order
            s = got.column("s").to_pylist()
            assert s == sorted(s, key=lambda v: v.encode())
