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
