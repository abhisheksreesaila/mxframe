"""F-008 review: the Router spawns threads per run; the GPU kernel's scratch buffers must not be
tied to a thread, or every short-lived thread leaks its share of device memory."""
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
def test_short_lived_threads_do_not_leak_device_memory(monkeypatch):
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

    # Count this process's own live device bytes (malloc minus free), not cuMemGetInfo: that reads
    # the whole device, so another process allocating on the GPU failed this test (F-075).
    live, sizes = [0], {}
    malloc, free = cu.malloc, cu.free

    def counting_malloc(nbytes):
        ptr = malloc(nbytes)
        sizes[ptr] = max(int(nbytes), 8)
        live[0] += sizes[ptr]
        return ptr

    def counting_free(ptr):
        live[0] -= sizes.pop(ptr, 0)
        free(ptr)

    monkeypatch.setattr(cu, "malloc", counting_malloc)
    monkeypatch.setattr(cu, "free", counting_free)
    for ptr in list(cu._sizes) + [p for fl in cu._pool.values() for p, _ in fl]:
        sizes.setdefault(ptr, 0)       # warm-up buffers were allocated before counting: not counted

    before = live[0]
    for _ in range(20):
        th = threading.Thread(target=lambda: pk.run_pipeline_gpu(prog, cc, gids, g, 0, n))
        th.start()
        th.join()
    assert (live[0] - before) / 2 ** 20 < 1, "device memory grew with the number of threads"
