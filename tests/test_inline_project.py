"""F-004 review: with_columns(...) feeding a grouped aggregate is inlined; the inlined result
must equal the un-inlined one (keys, dtypes, values) on the edge cases."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.custom_ops as co
from mxframe import LazyFrame, col, lit
from mxframe.aot_kernels import AOT_AVAILABLE
from mxframe.lazy_frame import Scan

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")


def _gpu_usable() -> bool:
    try:
        return co._max_accelerator() is not None
    except Exception:
        return False


DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))]


def _t(n=5000):
    r = np.random.default_rng(0)
    return pa.table({"k1": pa.array(r.integers(0, 3, n).astype(np.int32)),
                     "k2": pa.array(r.integers(0, 2, n).astype(np.int32)),
                     "price": pa.array((r.random(n) * 1000).astype(np.float32)),
                     "disc": pa.array((r.integers(0, 11, n) / 100).astype(np.float32))})


def _L():
    return LazyFrame(Scan(_t()))


CASES = {
    "shadowed key": lambda: _L().with_columns((col("k1") + lit(10)).alias("k1")).groupby("k1")
        .agg(col("price").sum().alias("p")).sort("k1"),
    "shadowed agg argument": lambda: _L().with_columns((col("price") * lit(2.0)).alias("price"))
        .groupby("k1").agg(col("price").sum().alias("p")).sort("k1"),
    "computed int key": lambda: _L().with_columns((col("k1") * lit(2) + col("k2")).alias("g"))
        .groupby("g").agg(col("price").sum().alias("p")).sort("g"),
    "having on computed column": lambda: _L().with_columns((col("price") * lit(2.0)).alias("v"))
        .groupby("k1").agg(col("v").sum().alias("sv")).filter(col("sv") > lit(1.66e6)).sort("k1"),
    "chained with_columns": lambda: _L().with_columns((col("price") * lit(2.0)).alias("v"))
        .with_columns((col("v") + lit(1.0)).alias("w")).groupby("k1")
        .agg(col("w").sum().alias("p")).sort("k1"),
}


def _numpy_reference(name):
    t = _t()
    k1, k2 = t.column("k1").to_numpy(), t.column("k2").to_numpy()
    price = t.column("price").to_numpy().astype(np.float64)
    if name == "shadowed key":
        return {"k1": [10, 11, 12], "p": [price[k1 == g].sum() for g in range(3)]}
    if name == "shadowed agg argument":
        return {"k1": [0, 1, 2], "p": [2 * price[k1 == g].sum() for g in range(3)]}
    if name == "computed int key":
        g = k1 * 2 + k2
        keys = sorted(set(g.tolist()))
        return {"g": keys, "p": [price[g == x].sum() for x in keys]}
    if name == "chained with_columns":
        return {"k1": [0, 1, 2], "p": [(2 * price[k1 == g] + 1).sum() for g in range(3)]}
    return None


@pytest.mark.parametrize("name", list(CASES))
def test_inlined_result_is_correct(name):
    device = "cpu"
    out = CASES[name]().compute(device=device)
    ref = _numpy_reference(name)
    if ref is None:                              # having: just check it filtered on the computed column
        assert out.num_rows > 0 and all(v > 1.66e6 for v in out.column("sv").to_pylist())
        return
    for k, want in ref.items():
        got = out.column(k).to_pylist()
        assert got == pytest.approx(want, rel=1e-4), (name, k)


def test_computed_integer_key_stays_integer():
    out = CASES["computed int key"]().compute(device="cpu")
    assert pa.types.is_integer(out.schema.field("g").type)
    out = CASES["shadowed key"]().compute(device="cpu")
    assert pa.types.is_integer(out.schema.field("k1").type)


# Multi-key on purpose: a single int key on the GPU goes through group_encode_i32, which spins
# on this box (F-012).  The key evaluation under test is device independent.
@pytest.mark.parametrize("device", DEVICES)
def test_shadowed_key_with_a_second_key(device):
    out = (_L().with_columns((col("k1") + lit(10)).alias("k1")).groupby("k1", "k2")
           .agg(col("price").sum().alias("p")).sort("k1", "k2").compute(device=device))
    assert sorted(set(out.column("k1").to_pylist())) == [10, 11, 12]
    assert pa.types.is_integer(out.schema.field("k1").type)
    t = _t()
    total = t.column("price").to_numpy().astype(np.float64).sum()
    assert sum(out.column("p").to_pylist()) == pytest.approx(total, rel=1e-3)
