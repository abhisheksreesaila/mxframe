"""F-023: ``select(a, b).distinct().groupby(a).agg(col(b).count())`` (TPC-H Q21's suppliers per
order) runs as one native pass when ``a`` is sorted: equal keys form runs and the distinct b of each
run are counted, with no distinct table and no second group-by. Unsorted keys, nulls and other
shapes take the existing path. Same answers as plain Python and as the path without the executor."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


@pytest.fixture
def calls(monkeypatch):
    """Native count-distinct calls that answered (rc 0)."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    log = []
    real = nx._call_count_distinct

    def counting(*a):
        rc = real(*a)
        log.append(rc)
        return rc

    monkeypatch.setattr(nx, "_call_count_distinct", counting)
    return log


def _table(n=200_000, seed=3, sort=True, chunks=1, key_type=np.int64, run=7, strings=False):
    r = np.random.default_rng(seed)
    a = np.repeat(np.arange(1, n // run + 2) * 4, run)[:n] if sort else r.integers(0, n // run, n)
    b = r.integers(0, 5, n).astype(np.int64)
    t = pa.table({
        "a": A(a.astype(key_type)),
        "b": pa.array([f"s{x}" for x in b], pa.string()) if strings else A(b),
        "d": A(r.integers(0, 100, n).astype(np.int32)),
    })
    if chunks > 1:
        cuts = np.linspace(0, n, chunks + 1).astype(int)
        t = pa.Table.from_batches([bb for x, y in zip(cuts[:-1], cuts[1:]) for bb in t.slice(x, y - x).to_batches()])
    return t


def _expected(t, pred=lambda d: True):
    seen = {}
    for a, b, d in zip(t.column("a").to_pylist(), t.column("b").to_pylist(), t.column("d").to_pylist()):
        if pred(d):
            seen.setdefault(a, set()).add(b)
    return {k: len(v) for k, v in sorted(seen.items())}


def _q(t, filt=True, count_star=False):
    lf = LazyFrame(t)
    if filt:
        lf = lf.filter(col("d") < lit(60))
    agg = (lit(1).count() if count_star else col("b").count()).alias("n")
    return lf.select("a", "b").distinct().groupby("a").agg(agg)


def _check(res, exp):
    assert res.column_names == ["a", "n"]
    got = dict(zip(res.column("a").to_pylist(), res.column("n").to_pylist()))
    assert got == exp
    assert res.column("a").to_pylist() == sorted(exp)              # key order


@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("strings", [False, True])
def test_sorted_key_counts_natively(calls, chunks, strings):
    t = _table(chunks=chunks, strings=strings)
    res = _q(t).compute(device="cpu")
    _check(res, _expected(t, lambda d: d < 60))
    assert calls == [0]


def test_no_filter_and_int32_key(calls):
    t = _table(key_type=np.int32)
    res = _q(t, filt=False).compute(device="cpu")
    assert res.schema.field("a").type == pa.int32()
    _check(res, _expected(t))
    assert calls == [0]


def test_count_star_is_the_same(calls):
    t = _table()
    res = _q(t, count_star=True).compute(device="cpu")
    _check(res, _expected(t, lambda d: d < 60))
    assert calls == [0]


@pytest.mark.parametrize("run", [1, 33, 500, 300_000])          # single rows, past the pairwise limit, one run
def test_run_lengths(calls, run):
    t = _table(n=100_000, run=run)
    res = _q(t, filt=False).compute(device="cpu")
    _check(res, _expected(t))
    assert calls == [0]


def test_unsorted_key_takes_the_existing_path(calls):
    t = _table(sort=False)
    res = _q(t).compute(device="cpu")
    _check(res, _expected(t, lambda d: d < 60))
    assert calls == [nx.E_UNSORTED]


def test_nulls_take_the_existing_path(calls):
    t = _table(n=1000)
    b = t.column("b").to_pylist()
    b[5] = None
    t = t.set_column(1, "b", pa.array(b, pa.int64()))
    res = _q(t, filt=False).compute(device="cpu")
    exp = {}
    for a, v in zip(t.column("a").to_pylist(), b):               # COUNT(b) skips the null
        exp.setdefault(a, set())
        if v is not None:
            exp[a].add(v)
    assert dict(zip(res.column("a").to_pylist(), res.column("n").to_pylist())) == {k: len(v) for k, v in exp.items()}
    assert calls == []


def test_same_as_the_path_without_the_executor(calls, monkeypatch):
    t = _table(chunks=2)
    got = _q(t).compute(device="cpu")
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    old = _q(t).compute(device="cpu")
    assert got.column("a").to_pylist() == old.column("a").to_pylist()
    assert got.column("n").to_pylist() == old.column("n").to_pylist()


def test_q21_shape_having_and_join(calls):
    t = _table()
    single = (_q(t).filter(col("n") == lit(1)).select("a"))
    keys = LazyFrame(pa.table({"a": A(np.arange(0, 200_000, 2).astype(np.int64)), "w": A(np.ones(100_000))}))
    res = keys.join(single, on="a").groupby().agg(col("w").sum().alias("s")).compute(device="cpu")
    exp = _expected(t, lambda d: d < 60)
    assert res.column("s").to_pylist() == [float(sum(1 for k, v in exp.items() if v == 1 and k % 2 == 0))]
    assert calls == [0]


def test_other_shapes_are_left_alone(calls):
    t = _table(n=10_000)
    # two aggregates, a key that is not one of the distinct columns' pair, a subset distinct
    LazyFrame(t).select("a", "b").distinct().groupby("a").agg(col("b").count().alias("n"),
                                                              col("b").max().alias("m")).compute(device="cpu")
    LazyFrame(t).select("a", "b", "d").distinct().groupby("a").agg(col("b").count().alias("n")).compute(device="cpu")
    assert calls == []


def test_negative_int32_keys(calls):
    r = np.random.default_rng(9)
    a = np.sort(r.integers(-50_000, 50_000, 150_000)).astype(np.int32)
    t = pa.table({"a": A(a), "b": A(r.integers(0, 4, 150_000).astype(np.int64)),
                  "d": A(np.zeros(150_000, np.int32))})
    res = _q(t, filt=False).compute(device="cpu")
    _check(res, _expected(t))
    assert calls == [0]


def test_a_dip_at_a_block_boundary_is_unsorted(calls):
    # the kernel splits 200k rows into blocks of n // nb rows: put a single descent exactly there
    t = _table(n=200_000)
    a = t.column("a").to_numpy().copy()
    for cut in (65_536, 100_000, 50_000):
        a2 = a.copy()
        a2[cut] = a2[cut - 1] - 1
        t2 = t.set_column(0, "a", A(a2))
        calls.clear()
        res = _q(t2, filt=False).compute(device="cpu")
        _check(res, _expected(t2))
        assert calls == [nx.E_UNSORTED]


def test_sliced_input(calls):
    t = _table(n=50_000).slice(1_234, 30_000)
    res = _q(t, filt=False).compute(device="cpu")
    _check(res, _expected(t))
    assert calls == [0]


def test_empty_input(calls):
    t = _table(n=1_000)
    res = _q(t).filter(col("n") > lit(10**9)).compute(device="cpu")
    assert res.num_rows == 0
    res = (LazyFrame(t).filter(col("d") < lit(0)).select("a", "b").distinct()
           .groupby("a").agg(col("b").count().alias("n")).compute(device="cpu"))
    assert res.num_rows == 0


def _gpu_usable():
    try:
        import mxframe.custom_ops as co
        return co._max_accelerator() is not None
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU")
def test_gpu_keeps_the_old_plan(calls):
    t = _table(n=10_000)
    res = _q(t).compute(device="gpu")
    _check(res, _expected(t, lambda d: d < 60))
    assert calls == []
