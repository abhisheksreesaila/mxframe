"""F-008: the dispatch in compile_and_run serves supported plans through the fused pipeline,
builds the same table the existing paths build, and declines everything else."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import custom_ops
from mxframe import pipeline_kernels as pk
from mxframe.lazy_expr import col, lit, when
from mxframe.lazy_frame import LazyFrame, Scan


def _gpu_ok():
    if not pk.gpu_available():
        return False
    try:
        pk._get_cuda()
        return True
    except Exception:
        return False


DEVICES = ["cpu"] + (["gpu"] if _gpu_ok() else [])
RNG = np.random.default_rng(11)


def make_table(n=30_000):
    return pa.table({
        "k": pa.array(RNG.choice(["a", "b", "c", "d"], n)),
        "j": pa.array(RNG.integers(0, 3, n).astype(np.int32)),
        "x": pa.array((RNG.random(n) * 100).astype(np.float32)),
        "y": pa.array(RNG.random(n).astype(np.float32)),
        "n": pa.array(RNG.integers(1, 50, n).astype(np.int64)),
    })


@pytest.fixture
def spy(monkeypatch):
    calls = {"served": 0, "declined": 0}
    real = custom_ops.try_fused_aggregate

    def wrapper(*a, **k):
        out = real(*a, **k)
        calls["served" if out is not None else "declined"] += 1
        return out

    monkeypatch.setattr(custom_ops, "try_fused_aggregate", wrapper)
    return calls


def plans(t):
    lf = LazyFrame(Scan(t))
    return {
        "global": lf.filter(col("x") > lit(20.0)).groupby().agg(
            (col("x") * col("y")).sum().alias("s"), col("x").min().alias("lo"), col("x").max().alias("hi"),
            col("y").mean().alias("m"), col("n").count().alias("c")),
        "grouped": lf.filter(col("y") < lit(0.9)).groupby("k").agg(
            (col("x") * (lit(1.0) - col("y"))).sum().alias("s"), col("x").mean().alias("m"),
            col("n").count().alias("c")),
        "two_keys": lf.groupby("k", "j").agg(col("x").sum().alias("s"), col("y").max().alias("hi")),
        "having_sort_limit": lf.groupby("k").agg(col("x").sum().alias("s"), col("n").count().alias("c"))
            .filter(col("c") > lit(100.0)).sort("k").limit(3),
        # (grouping by a single int32 key hangs the existing GPU group-encode kernel, so group by the string key)
        "case_when_string": lf.groupby("k").agg(
            when(col("j") == lit(1), col("x"), lit(0.0)).sum().alias("j1_x"),
            when(col("k").isin(["b", "c"]), col("y"), lit(0.0)).sum().alias("bc_y"),
            when(col("k").startswith("a"), lit(1.0), lit(0.0)).sum().alias("is_a")),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["global", "grouped", "two_keys", "having_sort_limit", "case_when_string"])
def test_served_and_same_table_shape_as_existing_path(device, name, spy, monkeypatch):
    t = make_table()
    lf = plans(t)[name]
    fused = lf.compute(device=device)
    assert spy["served"] == 1, name
    monkeypatch.setenv("MXFRAME_FUSED_PIPELINE", "0")
    existing = plans(t)[name].compute(device=device)
    assert fused.schema == existing.schema, (fused.schema, existing.schema)
    assert fused.num_rows == existing.num_rows
    for c in fused.column_names:
        a, b = fused.column(c), existing.column(c)
        if pa.types.is_floating(a.type):
            # the existing paths accumulate in float32; the fused kernel is the more exact one
            np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=2e-3)
        else:
            assert a.to_pylist() == b.to_pylist()


@pytest.mark.parametrize("device", DEVICES)
def test_unsupported_expression_falls_back_never_wrong(device, spy, monkeypatch):
    t = make_table()
    def build():
        return LazyFrame(Scan(t)).groupby("k").agg((col("n") / (col("n") - col("n") + lit(2))).sum().alias("s"))
    # integer / integer has floor semantics in Arrow: the fused compiler must decline it
    got = build().compute(device=device)
    assert spy["declined"] == 1 and spy["served"] == 0
    monkeypatch.setenv("MXFRAME_FUSED_PIPELINE", "0")
    want = build().compute(device=device)
    assert got.equals(want)


@pytest.mark.parametrize("device", DEVICES)
def test_nulls_fall_back(device, spy, monkeypatch):
    t = make_table(5000)
    x = t.column("x").to_pylist()
    x[7] = None
    t = t.set_column(t.schema.get_field_index("x"), "x", pa.array(x, type=pa.float32()))
    def build():
        return LazyFrame(Scan(t)).groupby("k").agg(col("y").sum().alias("s"), col("x").count().alias("c"))
    got = build().compute(device=device)
    monkeypatch.setenv("MXFRAME_FUSED_PIPELINE", "0")
    want = build().compute(device=device)
    assert got.column("c").to_pylist() == want.column("c").to_pylist()
    np.testing.assert_allclose(got.column("s").to_numpy(), want.column("s").to_numpy(), rtol=2e-3)


def test_switch_off_env(spy, monkeypatch):
    monkeypatch.setenv("MXFRAME_FUSED_PIPELINE", "0")
    plans(make_table())["grouped"].compute(device="cpu")
    assert spy["served"] == 0
