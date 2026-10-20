"""F-004: GPU start-up is lazy and shared.

- a small `auto` query never waits for the GPU warm-up
- a big one waits for it and reuses the warm-up's kernel library (no second init)
- the CUDA context is bound on whichever thread calls the kernels
"""
import threading
import time

import numpy as np
import pyarrow as pa
import pytest

import omxcore.auto_mode as auto_mode
import omxcore.custom_ops as custom_ops
from omxcore import LazyFrame, col, lit, when
from omxcore.aot_kernels import AOT_AVAILABLE, GPU_AOT_AVAILABLE, get_aot_gpu
from omxcore.lazy_frame import Scan


def _gpu_usable() -> bool:
    try:
        return GPU_AOT_AVAILABLE and custom_ops._max_accelerator() is not None
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU")
needs_cpu_aot = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")


def _table(n, seed=0):
    r = np.random.default_rng(seed)
    return pa.table({
        "k1": pa.array(r.integers(0, 3, n).astype(np.int32)),
        "k2": pa.array(r.integers(0, 2, n).astype(np.int32)),
        "price": pa.array((r.random(n) * 1000).astype(np.float32)),
        "disc": pa.array((r.integers(0, 11, n) / 100).astype(np.float32)),
    })


def _plan(t):
    return (LazyFrame(Scan(t)).groupby("k1", "k2")
            .agg(col("price").sum().alias("s")).sort("k1", "k2").plan)


@needs_cpu_aot
def test_small_auto_query_does_not_wait_for_the_gpu(monkeypatch):
    slow = auto_mode.GpuWarmup(lambda: time.sleep(5) or object(), expected_ms=5000)
    monkeypatch.setattr(auto_mode, "get_gpu_warmup", lambda calib=None: slow.start())
    monkeypatch.setattr(custom_ops, "GPU_AOT_AVAILABLE", True)
    t = _table(2_000)
    c = custom_ops.CustomOpsCompiler(device="auto")
    t0 = time.perf_counter()
    out = c.compile_and_run(_plan(t))
    assert time.perf_counter() - t0 < 2.0
    assert c._session_device == "cpu" and not slow.ready
    assert out.num_rows == 6


@needs_gpu
def test_big_auto_query_waits_for_and_reuses_the_warmup():
    t = _table(60_000)
    c = custom_ops.CustomOpsCompiler(device="auto")
    out = c.compile_and_run(_plan(t))
    warm = auto_mode.get_gpu_warmup()
    assert warm.ready
    assert c._session_device == "gpu"
    assert c._aot_gpu is warm.result is get_aot_gpu()      # one library, initialised once
    assert out.num_rows == 6


@needs_gpu
def test_kernels_run_on_a_thread_other_than_the_one_that_initialised_them():
    """The library is created on the warm-up thread; results must be right when called from
    the main thread and from a third one (context is per-thread; each binds it)."""
    warm = auto_mode.get_gpu_warmup()
    assert warm.wait()
    gpu = warm.result
    rng = np.random.default_rng(1)
    v = rng.random(50_000).astype(np.float32)
    mask = (rng.random(50_000) > 0.5).astype(np.int32)
    want = float(v[mask == 1].astype(np.float64).sum())
    assert gpu.masked_global_sum_f32(v, mask) == pytest.approx(want, rel=1e-4)
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("v", gpu.masked_global_sum_f32(v, mask)))
    th.start(); th.join()
    assert box["v"] == pytest.approx(want, rel=1e-4)


@needs_cpu_aot
def test_computed_column_feeding_a_grouped_aggregate_matches_numpy():
    """with_columns(...) below a group-by is inlined into the aggregate (F-004 removed the
    MAX Graph path for it); check the numbers, not just that nothing compiled."""
    t = _table(40_000, seed=3)
    out = (LazyFrame(Scan(t))
           .with_columns((col("price") * (lit(1.0) - col("disc"))).alias("vol"),
                         when(col("k1") == lit(1), col("price"), lit(0.0)).alias("k1p"))
           .groupby("k1", "k2")
           .agg(col("vol").sum().alias("vol"), col("k1p").sum().alias("k1p"))
           .sort("k1", "k2").compute(device="cpu"))
    k1, k2 = t.column("k1").to_numpy(), t.column("k2").to_numpy()
    price = t.column("price").to_numpy().astype(np.float64)
    disc = t.column("disc").to_numpy().astype(np.float64)
    for i in range(out.num_rows):
        a, b = out.column("k1")[i].as_py(), out.column("k2")[i].as_py()
        sel = (k1 == a) & (k2 == b)
        assert out.column("vol")[i].as_py() == pytest.approx((price * (1 - disc))[sel].sum(), rel=1e-5)
        assert out.column("k1p")[i].as_py() == pytest.approx(price[sel].sum() if a == 1 else 0.0, rel=1e-5)
