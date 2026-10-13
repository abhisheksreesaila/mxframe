"""F-032 (M2g): a group-by with directly addressed keys numbers the groups per 1024-row block, for
the rows that pass the predicate only (no group-id array over the whole table, keys of filtered-out
rows never read), and aggregates into per-worker partials. Before, more than 4096 groups wrote a
group id for every row first. Same answers as the Python path for every key kind and group count."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

from test_native_exec_m1 import _both, _same, native  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _table(n=300_000, seed=7, chunks=1):
    r = np.random.default_rng(seed)
    names = np.array([f"name{i:05d}" for i in range(3000)], dtype=object)
    t = pa.table({
        "k": A(r.integers(1, 100_001, n).astype(np.int64)),          # ~100k groups: past the old 4096
        "j": A(r.integers(-50, 50, n).astype(np.int32)),
        "m": A(r.integers(0, 8, n).astype(np.int32)),                # with k: 800k combinations, direct
        "f": pa.DictionaryArray.from_arrays(A(r.integers(0, 4, n).astype(np.int32)), A(["A", "N", "O", "R"])),
        "s": pa.array(names[r.integers(0, 3000, n)].tolist(), pa.string()),
        "d": A(r.integers(0, 1000, n).astype(np.int32)),
        "v": A(r.random(n) * 100),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        parts = []
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            p = t.slice(a, b - a)
            if i % 2:          # a chunk with its own dictionary (other order, extra entry)
                f = p.column("f").combine_chunks()
                vals = f.dictionary.to_pylist()
                order = ["R", "Z", "O", "N", "A"]
                remap = np.array([order.index(v) for v in vals], dtype=np.int32)
                idx = remap[f.indices.to_numpy(zero_copy_only=False)]
                p = p.set_column(p.schema.get_field_index("f"), "f",
                                 pa.DictionaryArray.from_arrays(A(idx), A(order)))
            parts.append(p)
        t = pa.Table.from_batches([b for p in parts for b in p.to_batches()], schema=parts[0].schema)
    return t


PLANS = {
    "many groups, selective filter": (lambda t: LazyFrame(t).filter(col("d") < lit(40))
        .groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n")), False),
    "many groups, no filter": (lambda t: LazyFrame(t).groupby("k")
        .agg(col("v").min().alias("lo"), col("v").max().alias("hi"), col("v").mean().alias("m")), False),
    "composite int keys, filter": (lambda t: LazyFrame(t).filter((col("d") >= lit(100)) & (col("d") < lit(400)))
        .groupby("m", "k").agg(col("v").sum().alias("s")), False),
    "composite past the direct limit (compacted), filter": (lambda t: LazyFrame(t).filter(col("d") < lit(400))
        .groupby("j", "k").agg(col("v").sum().alias("s")), False),
    "dictionary x int key, filter": (lambda t: LazyFrame(t).filter(col("d") < lit(500))
        .groupby("f", "j").agg(col("v").sum().alias("s"), col("v").count().alias("n")), False),
    "string key, filter": (lambda t: LazyFrame(t).filter(col("d") < lit(300))
        .groupby("s").agg(col("v").sum().alias("s2")), False),
    "filter keeps nothing": (lambda t: LazyFrame(t).filter(col("d") < lit(0))
        .groupby("k").agg(col("v").sum().alias("s")), False),
    "filter keeps everything": (lambda t: LazyFrame(t).filter(col("d") < lit(5000))
        .groupby("j").agg(col("v").sum().alias("s")), False),
    "many groups, top 10": (lambda t: LazyFrame(t).filter(col("d") < lit(600)).groupby("k")
        .agg(col("v").sum().alias("s")).sort("s", descending=[True]).limit(10), True),
}


@pytest.mark.parametrize("chunks", [1, 4])
@pytest.mark.parametrize("name", list(PLANS))
def test_keyed_group_by_matches_the_python_path(native, monkeypatch, name, chunks):
    plan, ordered = PLANS[name]
    t = _table(chunks=chunks)
    got, old, path, decline = _both(plan, t, monkeypatch)
    assert path == "native", decline
    _same(got, old, ordered=ordered)


def test_keyed_group_by_rows_come_out_in_key_order(native, monkeypatch):
    t = _table(chunks=3)
    got = LazyFrame(t).filter(col("d") < lit(50)).groupby("k").agg(col("v").sum().alias("s")).compute(device="cpu")
    assert nx.last_path() == "native"
    k = got.column("k").to_numpy()
    assert (np.diff(k) > 0).all()


# Key ranges far wider than the input: mx_execute compacts the groups present instead of addressing
# the whole range (D >= 16384 and D > 2 x rows). The executor does not report which grouping mode
# ran, so the branch is reached by shape: k spans 100k values over 20k rows, (m, k) 800k.
SPARSE = {
    "one wide key, filter": (lambda t: LazyFrame(t).filter(col("d") < lit(400))
        .groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n")), False),
    "composite wide keys": (lambda t: LazyFrame(t).groupby("m", "k")
        .agg(col("v").min().alias("lo"), col("v").max().alias("hi")), False),
    "dictionary x wide key, filter": (lambda t: LazyFrame(t).filter(col("d") < lit(700))
        .groupby("f", "k").agg(col("v").mean().alias("m")), False),
    "wide key, top 5": (lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"))
        .sort("s", descending=[True]).limit(5), True),
    "wide key, HAVING": (lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"))
        .filter(col("s") > lit(60.0)), False),
    "wide key, filter keeps nothing": (lambda t: LazyFrame(t).filter(col("d") < lit(0))
        .groupby("k").agg(col("v").sum().alias("s")), False),
}


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("name", list(SPARSE))
def test_sparse_key_range_matches_the_python_path(native, monkeypatch, name, chunks):
    # 20k rows over a 100k-wide key (TPC-H Q11's shape: a few thousand part keys out of 200k)
    plan, ordered = SPARSE[name]
    t = _table(n=20_000, chunks=chunks)
    got, old, path, decline = _both(plan, t, monkeypatch)
    assert path == "native", decline
    _same(got, old, ordered=ordered)


def test_sparse_key_range_rows_in_key_order(native, monkeypatch):
    t = _table(n=20_000, chunks=2)
    got = LazyFrame(t).groupby("m", "k").agg(col("v").sum().alias("s")).compute(device="cpu")
    assert nx.last_path() == "native"
    m, k = got.column("m").to_numpy(), got.column("k").to_numpy()
    assert (np.lexsort((k, m)) == np.arange(len(k))).all()


# F-043: group-bys with millions of groups. Too many groups for a partial table per worker: sums and
# counts are added by all workers into the output atomically; compacted key ranges are encoded per
# block (no group id per row). Inputs of at least 1M rows, keys unsorted and sorted.
def _big(n=1_500_000, groups=600_000, sort=False, seed=11, keys="uniform"):
    r = np.random.default_rng(seed)
    k = r.integers(0, groups * 3, n).astype(np.int64)          # range 3x the groups: compacted
    if keys == "zipf":                                           # a few groups take most rows
        k = np.minimum(r.zipf(1.3, n), groups * 3 - 1).astype(np.int64)
    elif keys == "one hot":
        k = np.where(r.random(n) < 0.95, 7, k).astype(np.int64)
    if sort:
        k = np.sort(k)
    return pa.table({
        "k": A(k),
        "m": A(r.integers(0, 3, n).astype(np.int32)),
        "v": A(r.integers(1, 50, n).astype(np.float64)),
        "d": A(r.integers(0, 100, n).astype(np.int32)),
    })


BIG = {
    "sum and count": (lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"),
                                                               col("v").count().alias("n")), False),
    "mean, filter": (lambda t: LazyFrame(t).filter(col("d") < lit(70)).groupby("k")
        .agg(col("v").mean().alias("a")), False),
    "two keys, sum": (lambda t: LazyFrame(t).groupby("m", "k").agg(col("v").sum().alias("s")), False),
    "sum, HAVING": (lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"))
        .filter(col("s") > lit(60.0)), False),
    "min / max": (lambda t: LazyFrame(t).groupby("k")
        .agg(col("v").min().alias("lo"), col("v").max().alias("hi")), False),
}


@pytest.mark.parametrize("keys", ["uniform", "sorted", "zipf", "one hot"])
@pytest.mark.parametrize("name", list(BIG))
def test_millions_of_groups_match_the_python_path(native, monkeypatch, name, keys):
    # 8 workers: the partial-table caps leave one, so the shared output (atomics behind a per-worker
    # cache of groups) is used on any machine
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    plan, ordered = BIG[name]
    t = _big(sort=keys == "sorted", keys="uniform" if keys == "sorted" else keys)
    got, old, path, decline = _both(plan, t, monkeypatch)
    assert path == "native", decline
    _same(got, old, ordered=ordered)


def test_millions_of_groups_with_nan_values(native, monkeypatch):
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    t = _big()
    v = t.column("v").to_numpy().copy()
    v[::97] = np.nan
    t = t.set_column(t.schema.get_field_index("v"), "v", A(v))
    plan = lambda t: LazyFrame(t).groupby("k").agg(col("v").sum().alias("s"), col("v").min().alias("lo"),
                                                   col("v").max().alias("hi"))
    got, old, path, decline = _both(plan, t, monkeypatch)
    assert path == "native", decline
    _same(got, old, ordered=False)


# F-043: with an ORDER BY in the same call, hashed / string-keyed groups are no longer sorted into key
# order first; the keys become the sort's last tie-break columns. Ties must still come out in key order.
@pytest.mark.parametrize("having", [False, True])
@pytest.mark.parametrize("limit", [None, 7, 50])
@pytest.mark.parametrize("desc", [True, False])
def test_order_by_ties_fall_back_to_key_order(native, monkeypatch, limit, desc, having):
    r = np.random.default_rng(21)
    n = 60_000
    names = np.array([f"n{i:03d}" for i in range(40)], dtype=object)
    t = pa.table({
        "k": A((r.integers(0, 3000, n) * 1_000_003_917).astype(np.int64)),   # far apart: hash path
        "s": pa.array(names[r.integers(0, 40, n)].tolist(), pa.string()),
        "v": A(r.integers(0, 3, n).astype(np.float64)),                        # few values: many ties
    })

    q = LazyFrame(t).groupby("s", "k").agg(col("v").max().alias("m"))
    if having:
        q = q.filter(col("m") > lit(0.0))
    q = q.sort("m", descending=[desc])
    got = (q.limit(limit) if limit else q).compute(device="cpu")
    assert nx.last_path() == "native", nx.last_decline()
    best = {}
    for s, k, v in zip(t.column("s").to_pylist(), t.column("k").to_pylist(), t.column("v").to_pylist()):
        best[(s, k)] = max(best.get((s, k), v), v)
    if having:
        best = {g: m for g, m in best.items() if m > 0.0}
    exp = sorted(best.items(), key=lambda kv: ((-kv[1] if desc else kv[1]), kv[0][0], kv[0][1]))
    exp = exp[:limit] if limit else exp
    rows = list(zip(got.column("s").to_pylist(), got.column("k").to_pylist(), got.column("m").to_pylist()))
    assert rows == [(s, k, m) for (s, k), m in exp]
