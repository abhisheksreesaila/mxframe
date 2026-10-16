"""F-032 (M2l): a group-by on an integer expression of the row (TPC-H's order year, a CASE WHEN sum
over a date) runs in the native executor: one min / max pass measures the key's range, then the
program computes the key per row and the keyed path groups by it directly. Same values as the Python
path (which returns such keys as float64; the native path returns int64)."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit, when
from mxframe.np_arrow import array as A

from test_native_exec_m1 import native  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _table(n=200_000, seed=4, chunks=1):
    r = np.random.default_rng(seed)
    t = pa.table({
        "d": A(r.integers(8000, 10600, n).astype(np.int32)),          # days since epoch (as the TPC-H data)
        "g": A(r.integers(0, 5, n).astype(np.int32)),
        "x": A(r.integers(-50, 50, n).astype(np.int64)),
        "v": A(r.random(n) * 100),
        "f": pa.DictionaryArray.from_arrays(A(r.integers(0, 3, n).astype(np.int32)), A(["a", "b", "c"])),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.Table.from_batches([b for a, z in zip(cuts[:-1], cuts[1:]) for b in t.slice(a, z - a).to_batches()])
    return t


def _year(c):
    e = when(col(c) >= lit(8401), lit(1), lit(0))           # 1993-01-01 .. : year as a CASE WHEN sum
    for days in (8766, 9131, 9496, 9862, 10227):
        e = e + when(col(c) >= lit(days), lit(1), lit(0))
    return e + lit(1992)


def _run(plan, t, monkeypatch):
    log = []
    real = nx._decline
    monkeypatch.setattr(nx, "_decline", lambda r: (log.append(r), real(r))[1])
    got = plan(t).compute(device="cpu")
    path = nx.last_path()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = plan(t).compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return got, old, path, log


def _same_values(got, old, ordered=True):
    assert got.column_names == old.column_names and got.num_rows == old.num_rows
    if not ordered:
        keys = [c for c in got.column_names if not pa.types.is_floating(old.schema.field(c).type) or c in ("y", "x2")]
        old = old.cast(got.schema)
        got = got.sort_by([(k, "ascending") for k in keys])
        old = old.sort_by([(k, "ascending") for k in keys])
    for c in got.column_names:
        g, o = got.column(c).to_pylist(), old.column(c).to_pylist()
        if pa.types.is_string(got.schema.field(c).type):
            assert g == o, c
        else:
            np.testing.assert_allclose(np.array(g, dtype=float), np.array(o, dtype=float), rtol=1e-12, err_msg=c)


PLANS = {
    "year key, sorted": (lambda t: LazyFrame(t).with_columns(_year("d").alias("y"))
                         .groupby("y").agg(col("v").sum().alias("s"), col("v").count().alias("n")).sort("y"), True),
    "dictionary key and year key, filter": (lambda t: LazyFrame(t).filter(col("g") < lit(3))
                         .with_columns(_year("d").alias("y")).groupby("f", "y").agg(col("v").max().alias("m"))
                         .sort("f", "y", descending=[False, True]), True),
    "arithmetic key": (lambda t: LazyFrame(t).with_columns((col("x") * lit(3) + col("g")).alias("x2"))
                       .groupby("x2").agg(col("v").sum().alias("s")).sort("x2"), True),
}


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("name", list(PLANS))
def test_integer_expression_keys_run_natively(native, monkeypatch, name, chunks):
    plan, ordered = PLANS[name]
    got, old, path, log = _run(plan, _table(chunks=chunks), monkeypatch)
    assert path == "native", log
    assert not any("derived" in r for r in log), log
    _same_values(got, old, ordered)
    key = "y" if "y" in got.column_names else "x2"
    assert got.schema.field(key).type == pa.int64()


@pytest.mark.parametrize("plan", [
    lambda t: LazyFrame(t).with_columns((col("v") * lit(2.0)).alias("y")).groupby("y").agg(col("g").count().alias("n")),
], ids=["float expression"])
def test_other_expression_keys_decline_with_the_same_answer(native, monkeypatch, plan):
    got, old, path, log = _run(plan, _table(n=20_000), monkeypatch)
    assert any("derived group key" in r for r in log), log
    _same_values(got, old, ordered=False)


def test_an_expression_key_range_too_wide_for_direct_addressing_is_hashed(native, monkeypatch):
    """x * 100000 spans 10 million codes: past MAX_GROUPS the executor groups by the key's values in the hashed
    group-by (F-069; it used to decline "not a small integer range")."""
    plan = lambda t: LazyFrame(t).with_columns((col("x") * lit(100_000)).alias("y")).groupby("y").agg(col("v").sum().alias("s"))
    got, old, path, log = _run(plan, _table(n=20_000), monkeypatch)
    assert path == "native" and not any("derived" in r for r in log), log
    _same_values(got, old, ordered=False)


def test_a_filter_that_keeps_nothing(native, monkeypatch):
    plan = lambda t: (LazyFrame(t).filter(col("g") > lit(10)).with_columns(_year("d").alias("y"))
                      .groupby("y").agg(col("v").sum().alias("s")))
    got, old, path, log = _run(plan, _table(n=20_000), monkeypatch)
    assert got.num_rows == old.num_rows == 0


def test_expression_keys_too_wide_together_are_hashed_in_the_executor(native, monkeypatch):
    """Two expression keys ~3000 wide each: 9M combinations are past direct addressing and the compacted
    path takes only column keys (M2l review: the executor declined, E_BAD_KEY); the hashed group-by takes them (F-069)."""
    r = np.random.default_rng(8)
    n = 30_000
    t = pa.table({"a": A(r.integers(0, 3000, n).astype(np.int64)), "b": A(r.integers(0, 3000, n).astype(np.int64)),
                  "v": A(r.random(n))})
    plan = lambda t: (LazyFrame(t).with_columns((col("a") + lit(1)).alias("a1"), (col("b") * lit(2)).alias("b2"))
                      .groupby("a1", "b2").agg(col("v").sum().alias("s")))
    got, old, path, log = _run(plan, t, monkeypatch)
    assert path == "native", log
    assert not any("declined" in r for r in log), log
    _same_values(got, old, ordered=False)
