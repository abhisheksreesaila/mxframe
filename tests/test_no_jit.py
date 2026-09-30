"""F-004: the query paths the TPC-H workload hits never compile a MAX Graph.

A first run in a fresh process must not pay seconds of JIT (`InferenceSession.load`), nor
create a MAX `InferenceSession` at all.  Both are counted by monkeypatching; the queries
run on `cpu` and (when a GPU is present) on `gpu`.  `device="cpu"` must additionally never
touch CUDA.
"""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, lit, when
from mxframe.lazy_frame import Scan
import mxframe.custom_ops as custom_ops
from mxframe.aot_kernels import AOT_AVAILABLE, GPU_AOT_AVAILABLE

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")


def _gpu_usable() -> bool:
    if not GPU_AOT_AVAILABLE:
        return False
    try:
        return custom_ops._max_accelerator() is not None
    except Exception:
        return False


DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))]


@pytest.fixture
def jit_counter(monkeypatch):
    """Counts MAX Graph compiles and InferenceSession creations; starts from cold caches."""
    from max import engine
    calls = {"load": 0, "session": 0}
    real_load = engine.InferenceSession.load
    real_init = engine.InferenceSession.__init__

    def load(self, *a, **k):
        calls["load"] += 1
        return real_load(self, *a, **k)

    def init(self, *a, **k):
        calls["session"] += 1
        return real_init(self, *a, **k)

    monkeypatch.setattr(engine.InferenceSession, "load", load)
    monkeypatch.setattr(engine.InferenceSession, "__init__", init)
    custom_ops.clear_cache()
    return calls


def _fact(n=60_000, seed=0):
    r = np.random.default_rng(seed)
    return pa.table({
        "k1": pa.array(r.integers(0, 3, n).astype(np.int32)),
        "k2": pa.array(r.integers(0, 2, n).astype(np.int32)),
        "fk": pa.array(r.integers(0, 500, n).astype(np.int32)),
        "flag": pa.array(np.array(["A", "N", "R"])[r.integers(0, 3, n)]),
        "price": pa.array((r.random(n) * 1000).astype(np.float32)),
        "disc": pa.array((r.integers(0, 11, n) / 100).astype(np.float32)),
        "ship": pa.array(r.integers(8000, 10600, n).astype(np.int32)),
    })


def _dim():
    return pa.table({"pk": pa.array(np.arange(500, dtype=np.int32)),
                     "region": pa.array((np.arange(500) % 5).astype(np.int32)),
                     "tier": pa.array((np.arange(500) % 3).astype(np.int32))})


def _shapes():
    f, d = _fact(), _dim()
    return {
        "grouped multi-key": lambda dev: (
            LazyFrame(Scan(f)).filter(col("ship") <= lit(10471)).groupby("k1", "k2")
            .agg(col("price").sum().alias("s"), col("price").mean().alias("m"),
                 (col("price") * (lit(1.0) - col("disc"))).sum().alias("rev"),
                 col("price").count().alias("c"))
            .sort("k1", "k2").compute(device=dev)),
        "grouped string key": lambda dev: (
            LazyFrame(Scan(f)).groupby("flag")
            .agg(col("price").sum().alias("s"), col("price").max().alias("hi"),
                 col("price").min().alias("lo"))
            .sort("flag").compute(device=dev)),
        "join then group": lambda dev: (
            LazyFrame(Scan(f)).join(LazyFrame(Scan(d)), left_on="fk", right_on="pk")
            .groupby("region", "tier").agg(col("price").sum().alias("s")).sort("region", "tier").compute(device=dev)),
        "computed column then group": lambda dev: (
            LazyFrame(Scan(f))
            .with_columns((col("price") * (lit(1.0) - col("disc"))).alias("vol"),
                          when(col("k1") == lit(1), col("price"), lit(0.0)).alias("k1_price"))
            .groupby("k1", "k2")
            .agg(col("vol").sum().alias("total"), col("k1_price").sum().alias("k1_total"))
            .sort("k1", "k2").compute(device=dev)),
        "filter selects nothing, one key": lambda dev: (
            LazyFrame(Scan(f)).filter(col("ship") > lit(1_000_000)).groupby("k1")
            .agg(col("price").sum().alias("s")).compute(device=dev)),
        "filter selects nothing, two keys": lambda dev: (
            LazyFrame(Scan(f)).filter(col("ship") > lit(1_000_000)).groupby("k1", "k2")
            .agg(col("price").sum().alias("s"), col("price").count().alias("c")).compute(device=dev)),
        "sort + limit": lambda dev: (
            LazyFrame(Scan(f)).sort("price", descending=True).limit(10).compute(device=dev)),
        "distinct": lambda dev: (
            LazyFrame(Scan(f)).distinct("k1", "k2").compute(device=dev)),
        "global masked agg": lambda dev: (
            LazyFrame(Scan(f))
            .filter((col("ship") >= lit(9000)) & (col("ship") < lit(9800)) & (col("disc") >= lit(0.045)))
            .groupby().agg((col("price") * col("disc")).sum().alias("rev"),
                           col("price").min().alias("lo")).compute(device=dev)),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", list(_shapes()))
def test_query_shape_compiles_no_graph(device, shape, jit_counter):
    out = _shapes()[shape](device)
    assert out.num_rows > 0 or shape.startswith("filter selects nothing")
    assert jit_counter == {"load": 0, "session": 0}, f"{shape} on {device} hit the MAX Graph JIT: {jit_counter}"


def test_cpu_device_never_initialises_the_gpu(monkeypatch):
    """device='cpu' must not create a CUDA context, load the GPU library or ask MAX for a GPU."""
    import mxframe.aot_kernels as ak
    monkeypatch.setattr(ak, "_cuda_driver", None)
    monkeypatch.setattr(ak, "_aot_gpu_singleton", None)
    touched = []
    monkeypatch.setattr(ak, "_CUDADriver", lambda *a, **k: touched.append("cuda") or (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(ak, "AOTKernelsGPU", lambda *a, **k: touched.append("aot_gpu") or (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(custom_ops, "_max_accelerator", lambda: touched.append("accelerator"))
    from max import driver
    monkeypatch.setattr(driver, "Accelerator", lambda *a, **k: touched.append("Accelerator"))
    _shapes()["grouped multi-key"]("cpu")
    _shapes()["join then group"]("cpu")
    assert touched == []


def test_cpu_compiler_creates_no_gpu_state():
    c = custom_ops.CustomOpsCompiler(device="cpu")
    assert c._aot_gpu is None and c._session_obj is None and not c._has_gpu
