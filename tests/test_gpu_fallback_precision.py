"""F-022: a grouped aggregate the fused pipeline declines must give the same digits on the GPU as
on the CPU. The old `gpu_aot` fallback rounded float64 inputs to float32 and summed in float32."""
import numpy as np
import pyarrow as pa
import pytest

import mxframe.custom_ops as co
from mxframe import LazyFrame, col, lit
from mxframe.aot_kernels import AOT_AVAILABLE, GPU_AOT_AVAILABLE
from mxframe.lazy_expr import when
from mxframe.lazy_frame import Scan

pytestmark = pytest.mark.skipif(not (AOT_AVAILABLE and GPU_AOT_AVAILABLE),
                                reason="needs the CPU and GPU AOT kernels")


def _t(n=50_000):
    r = np.random.default_rng(22)
    return pa.table({
        "k": pa.array(r.integers(0, 7, n).astype(np.int32)),
        # large values with small fractional parts: float32 keeps ~7 digits, so a float32
        # detour changes the sums in the 1e-4 range
        "v": pa.array(1e6 + np.round(r.normal(size=n), 4)),
        "w": pa.array(np.round(r.normal(scale=10, size=n), 2)),
    })


CASES = {
    "sum": lambda t: LazyFrame(Scan(t)).groupby("k").agg(col("v").sum().alias("s")),
    "mean min max": lambda t: LazyFrame(Scan(t)).groupby("k").agg(
        col("v").mean().alias("m"), col("w").min().alias("lo"), col("w").max().alias("hi")),
    "filtered": lambda t: LazyFrame(Scan(t)).filter(col("w") > lit(0.0)).groupby("k")
        .agg(col("v").sum().alias("s")),
    "expression": lambda t: LazyFrame(Scan(t)).groupby("k")
        .agg((col("v") * lit(1.5) - col("w")).sum().alias("s")),
    "case_when": lambda t: LazyFrame(Scan(t)).groupby("k").agg(
        when(col("w") > lit(0.0), col("v"), lit(0.0)).sum().alias("s")),
}


def _reference(name, t):
    k = t.column("k").to_numpy()
    v = t.column("v").to_numpy()
    w = t.column("w").to_numpy()
    groups = range(7)
    if name == "sum":
        return {"s": [v[k == g].sum() for g in groups]}
    if name == "mean min max":
        return {"m": [v[k == g].mean() for g in groups], "lo": [w[k == g].min() for g in groups],
                "hi": [w[k == g].max() for g in groups]}
    if name == "filtered":
        return {"s": [v[(k == g) & (w > 0)].sum() for g in groups]}
    if name == "expression":
        return {"s": [(v * 1.5 - w)[k == g].sum() for g in groups]}
    if name == "case_when":
        return {"s": [np.where(w > 0, v, 0.0)[k == g].sum() for g in groups]}
    raise KeyError(name)


@pytest.fixture
def no_fused(monkeypatch):
    """Make the fused pipeline decline every plan, so the paths behind it run."""
    monkeypatch.setattr(co, "try_fused_aggregate", None)


@pytest.mark.parametrize("name", list(CASES))
def test_declined_plan_keeps_float64_digits_on_gpu(no_fused, name):
    t = _t()
    seen = []
    real = co.CustomOpsCompiler._execute_cpu_aot

    def spy(self, *a, **kw):
        res = real(self, *a, **kw)
        seen.append((self._session_device, kw.get("_aot_override"), res is not None))
        return res

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(co.CustomOpsCompiler, "_execute_cpu_aot", spy)
        out = CASES[name](t).sort("k").compute(device="gpu").to_pydict()
    # answered by the float64 CPU kernels inside the GPU session, not by the float32 GPU kernels
    assert ("gpu", None, True) in seen
    assert out["k"] == list(range(7))
    for c, want in _reference(name, t).items():
        np.testing.assert_allclose(out[c], want, rtol=1e-12, err_msg=f"{name}: {c}")


@pytest.mark.parametrize("name", list(CASES))
def test_declined_plan_same_digits_on_cpu_and_gpu(no_fused, name):
    t = _t()
    cpu = CASES[name](t).sort("k").compute(device="cpu").to_pydict()
    gpu = CASES[name](t).sort("k").compute(device="gpu").to_pydict()
    assert cpu == gpu
