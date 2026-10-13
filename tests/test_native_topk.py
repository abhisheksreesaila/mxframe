"""F-044: ORDER BY plain numeric columns LIMIT k runs as one native pass of per-worker heaps
(`mx_topk`), and a WHERE without an aggregate runs through the native filter. Same rows and order
as a stable sort, for every key type and direction mix, ties, k past the end, chunked input;
NaN and nulls take the old path."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _t(n=300_000, seed=1, chunks=1):
    r = np.random.default_rng(seed)
    t = pa.table({
        "i32": A(r.integers(-50, 50, n).astype(np.int32)),
        "i64": A(r.integers(-10**12, 10**12, n).astype(np.int64)),
        "f32": A(r.integers(0, 20, n).astype(np.float32) / 4),        # many ties
        "f64": A(r.random(n)),
        "d": pa.array(r.integers(8000, 11000, n).astype(np.int32), pa.date32()),
        "s": pa.array([f"s{x}" for x in r.integers(0, 9, n)], pa.string()),
        "row": A(np.arange(n, dtype=np.int64)),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.Table.from_batches([b for x, y in zip(cuts[:-1], cuts[1:]) for b in t.slice(x, y - x).to_batches()])
    return t


def _expected(t, by, desc, k):
    def raw(c):
        a = t.column(c)
        if a.type == pa.date32():
            return a.cast(pa.int32()).to_numpy()
        if pa.types.is_timestamp(a.type):
            return a.cast(pa.int64()).to_numpy()
        return a.to_numpy()
    cols = {c: raw(c) for c in by}
    keys = [(-cols[c] if d else cols[c]) for c, d in zip(reversed(by), reversed(desc))]
    order = np.lexsort(keys)                       # stable: ties keep row order
    return t.column("row").to_numpy()[order][:k].tolist()


CASES = [
    (["f32"], [False]), (["f32"], [True]), (["i32", "f64"], [True, False]), (["d", "i64"], [False, True]),
    (["f32", "i32", "row"], [True, True, False]), (["i64"], [False]),
]


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("k", [1, 10, 500])
@pytest.mark.parametrize("by,desc", CASES)
def test_topk_matches_a_stable_sort(by, desc, k, chunks):
    t = _t(chunks=chunks)
    idx = nx.try_topk(t, by, desc, k)
    assert idx is not None
    assert t.column("row").to_numpy()[idx].tolist() == _expected(t, by, desc, k)


def test_k_past_the_end_and_tiny_tables():
    t = _t(n=7)
    assert len(nx.try_topk(t, ["f32"], [False], 50)) == 7
    assert t.column("row").to_numpy()[nx.try_topk(t, ["f64"], [True], 50)].tolist() == _expected(t, ["f64"], [True], 50)


def test_nan_and_nulls_and_strings_decline():
    t = _t(n=1000)
    f = t.column("f64").to_numpy().copy()
    f[3] = np.nan
    assert nx.try_topk(t.set_column(3, "f64", A(f)), ["f64"], [False], 5) is None
    assert nx.try_topk(t.set_column(0, "i32", pa.array([None] + [1] * 999, pa.int32())), ["i32"], [False], 5) is None
    assert nx.try_topk(t, ["s"], [False], 5) is None


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_limit_through_the_api(desc):
    t = _t()
    got = (LazyFrame(t).filter(col("i32") > lit(0)).select("row", "f32", "i64")
           .sort("f32", "row", descending=[desc, False]).limit(25).compute(device="cpu"))
    m = t.column("i32").to_numpy() > 0
    sub = t.filter(pa.array(m))
    assert got.column("row").to_pylist() == _expected(sub, ["f32", "row"], [desc, False], 25)
    assert got.column_names == ["row", "f32", "i64"]


def test_filter_only_plan_keeps_types_and_rows(monkeypatch):
    t = _t()
    t = t.set_column(t.schema.get_field_index("s"), "s", pa.compute.dictionary_encode(t.column("s")))
    got = LazyFrame(t).filter(col("f64") < lit(0.01)).compute(device="cpu")
    m = t.column("f64").to_numpy() < 0.01
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = LazyFrame(t).filter(col("f64") < lit(0.01)).compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    assert got.schema == old.schema                          # same output types as without the executor
    assert got.equals(old)
    assert got.column("row").to_pylist() == t.column("row").to_numpy()[m].tolist()
    got = LazyFrame(t).filter(col("f64") < lit(0.01)).select("row", "d").compute(device="cpu")
    assert got.column("row").to_pylist() == t.column("row").to_numpy()[m].tolist()
    assert got.schema.field("d").type == pa.date32()


def test_filter_only_plan_runs_natively_and_keeps_metadata(monkeypatch):
    t = _t().drop_columns(["s"]).replace_schema_metadata({b"k": b"v"})
    got = LazyFrame(t).filter(col("f64") < lit(0.01)).compute(device="cpu")
    assert nx.last_filter()[0] == "native_filter", nx.last_filter()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = LazyFrame(t).filter(col("f64") < lit(0.01)).compute(device="cpu")
    assert got.equals(old, check_metadata=True)


def test_int64_past_2_53_offsets_and_timestamps():
    r = np.random.default_rng(3)
    n = 50_000
    big = (np.int64(2) ** 60) + r.integers(-5, 5, n).astype(np.int64)        # differ below float64 precision
    t = pa.table({"b": A(big), "ts": pa.array(r.integers(0, 10**15, n), pa.timestamp("us")),
                  "row": A(np.arange(n, dtype=np.int64))}).slice(123, 40_000)
    for by, desc in ((["b"], [False]), (["b"], [True]), (["ts", "row"], [True, False])):
        idx = nx.try_topk(t, by, desc, 30)
        assert t.column("row").to_numpy()[idx].tolist() == _expected(t, by, desc, 30)
