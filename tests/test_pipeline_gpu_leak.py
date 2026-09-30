"""F-008 review: the Router spawns threads per run; the GPU kernel's scratch buffers must not be
tied to a thread, or every short-lived thread leaks its share of device memory."""
import ctypes
import threading

import numpy as np
import pyarrow as pa
import pytest

from mxframe import pipeline_kernels as pk
from mxframe.lazy_expr import col, lit
from mxframe.pipeline_program import compile_pipeline


def _gpu_ok():
    if not pk.gpu_available():
        return False
    try:
        pk._get_cuda()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="no GPU")
def test_short_lived_threads_do_not_leak_device_memory():
    n, g = 500_000, 250_000
    rng = np.random.default_rng(0)
    t = pa.table({"x": rng.random(n), "y": rng.random(n)})
    aggs = [col("x").sum().alias("a"), col("y").sum().alias("b"), col("x").max().alias("c"),
            (col("x") * col("y")).sum().alias("d")]
    prog = compile_pipeline([col("x") > lit(0.1)], aggs, t)
    cc = prog.bind_chunks(t)
    gids = rng.integers(0, g, n).astype(np.int32)
    pk.run_pipeline_gpu(prog, cc, gids, g, 0, n)          # warm: context, program upload, pool
    cu = pk._get_cuda()

    def free_mib():
        f, tot = ctypes.c_size_t(), ctypes.c_size_t()
        cu.bind_thread()
        cu.lib.cuMemGetInfo_v2(ctypes.byref(f), ctypes.byref(tot))
        return f.value / 2 ** 20

    before = free_mib()
    for _ in range(20):
        th = threading.Thread(target=lambda: pk.run_pipeline_gpu(prog, cc, gids, g, 0, n))
        th.start()
        th.join()
    assert before - free_mib() < 64, "device memory grew with the number of threads"
