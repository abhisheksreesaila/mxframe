"""F-040: a group-by on a renamed column (``select(col("s").alias("b")).groupby("b")``) returns the
key under its alias, on every execution path (native executor, fused pipeline, the kernel path), and
with the right values. It used to come back under the source column's name."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col
from mxframe.np_arrow import array as A

PATHS = {
    "default": {},
    "no-native": {"MXFRAME_NATIVE": "0"},
    "no-native-no-fused": {"MXFRAME_NATIVE": "0", "MXFRAME_FUSED_PIPELINE": "0"},
}


@pytest.fixture(params=list(PATHS))
def path(request, monkeypatch):
    for k, v in PATHS[request.param].items():
        monkeypatch.setenv(k, v)
    return request.param


@pytest.fixture
def served(monkeypatch):
    """Which engine path answered: "native", "fused", or neither (the kernel path)."""
    import mxframe.custom_ops as co
    log = []
    real_native, real_fused = co._native_exec.try_execute, co.try_fused_aggregate

    def native(*a, **k):
        r = real_native(*a, **k)
        if r is not None:
            log.append("native")
        return r

    def fused(*a, **k):
        r = real_fused(*a, **k)
        if r is not None:
            log.append("fused")
        return r

    monkeypatch.setattr(co._native_exec, "try_execute", native)
    monkeypatch.setattr(co, "try_fused_aggregate", fused)
    return log


def _table(n=50_000, seed=7):
    r = np.random.default_rng(seed)
    return pa.table({
        "s": A(np.array(["a", "b", "c", "d"])[r.integers(0, 4, n)]),
        "k": A(r.integers(0, 6, n).astype(np.int64)),
        "v": A(r.integers(0, 100, n).astype(np.int64)),
    })


def _expected(t, key, val):
    out = {}
    for kk, vv in zip(t.column(key).to_pylist(), t.column(val).to_pylist()):
        out[kk] = out.get(kk, 0) + vv
    return out


def _as_dict(res, key, val):
    return dict(zip(res.column(key).to_pylist(), res.column(val).to_pylist()))


def test_string_key_alias(path):
    t = _table()
    res = (LazyFrame(t).select(col("s").alias("b"), col("v"))
           .groupby("b").agg(col("v").sum().alias("x")).compute())
    assert res.column_names == ["b", "x"]
    assert _as_dict(res, "b", "x") == pytest.approx(_expected(t, "s", "v"))


@pytest.mark.parametrize("key,alias", [("s", "b"), ("k", "kk")])
def test_each_path_serves_the_renamed_key(path, served, key, alias):
    # the case must really run on the path it is named for, not fall back to another one
    t = _table()
    res = (LazyFrame(t).select(col(key).alias(alias), col("v"))
           .groupby(alias).agg(col("v").sum().alias("x")).compute(device="cpu"))
    assert res.column_names == [alias, "x"]
    assert _as_dict(res, alias, "x") == pytest.approx(_expected(t, key, "v"))
    want = {"default": "native" if key == "k" else "fused", "no-native": "fused",
            "no-native-no-fused": None}[path]
    assert (served[0] if served else None) == want


def _gpu_ok():
    try:
        import mxframe.custom_ops as co
        return co._max_accelerator() is not None
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="no GPU")
def test_renamed_key_on_gpu():
    t = _table()
    res = (LazyFrame(t).select(col("s").alias("b"), col("v"))
           .groupby("b").agg(col("v").sum().alias("x")).compute(device="gpu"))
    assert res.column_names == ["b", "x"]
    assert _as_dict(res, "b", "x") == pytest.approx(_expected(t, "s", "v"))


def test_int_key_alias(path):
    t = _table()
    res = (LazyFrame(t).select(col("k").alias("kk"), col("v"))
           .groupby("kk").agg(col("v").sum().alias("x")).compute())
    assert res.column_names == ["kk", "x"]
    assert _as_dict(res, "kk", "x") == pytest.approx(_expected(t, "k", "v"))


def test_alias_only_column_counted(path):
    t = _table()
    res = (LazyFrame(t).select(col("s").alias("b"))
           .groupby("b").agg(col("b").count().alias("n")).compute())
    assert res.column_names == ["b", "n"]
    exp = {}
    for s in t.column("s").to_pylist():
        exp[s] = exp.get(s, 0) + 1
    assert _as_dict(res, "b", "n") == exp


def test_alias_shadows_another_source_column(path):
    # the key is named "v" but holds s; the value column v is renamed to w
    t = _table()
    res = (LazyFrame(t).select(col("s").alias("v"), col("v").alias("w"))
           .groupby("v").agg(col("w").sum().alias("x")).compute())
    assert res.column_names == ["v", "x"]
    assert _as_dict(res, "v", "x") == pytest.approx(_expected(t, "s", "v"))


def test_alias_key_sorted(path):
    t = _table()
    res = (LazyFrame(t).select(col("s").alias("b"), col("v"))
           .groupby("b").agg(col("v").sum().alias("x")).sort("b").compute())
    assert res.column("b").to_pylist() == ["a", "b", "c", "d"]
    exp = _expected(t, "s", "v")
    assert res.column("x").to_pylist() == pytest.approx([exp[k] for k in "abcd"])


def test_alias_and_source_both_keys(path):
    t = _table()
    res = (LazyFrame(t).select(col("s"), col("s").alias("b"), col("v"))
           .groupby("s", "b").agg(col("v").sum().alias("x")).compute())
    assert res.column_names == ["s", "b", "x"]
    assert res.column("s").to_pylist() == res.column("b").to_pylist()
    assert dict(zip(res.column("s").to_pylist(), res.column("x").to_pylist())) == pytest.approx(_expected(t, "s", "v"))
