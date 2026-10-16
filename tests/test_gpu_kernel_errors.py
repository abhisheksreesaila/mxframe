"""F-055: every exported GPU function returns a status and its Python wrapper raises on failure.

They used to end in `except: pass`: a kernel that ran out of GPU memory printed `oom log` on stderr and
left garbage behind (a sort returned wrong indices, a join wrong rows) that every caller took for the
answer. The failure is injected at the library boundary (a library function that returns the status
the Mojo code returns), so the tests are deterministic on a shared GPU. The real thing is in
`test_real_out_of_memory_raises` (opt-in: it holds most of one GPU's memory).
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, col, gpu_stream
from mxframe import auto_mode as am
from mxframe.aot_kernels import _bind_gpu, get_aot_gpu

pytestmark = pytest.mark.skipif(not gpu_stream.cuda_available() or get_aot_gpu() is None, reason="no CUDA GPU")


class _FailingLib:
    def __init__(self, rc):
        self.rc = rc

    def __getattr__(self, name):
        return lambda *a: self.rc


def _fail(monkeypatch, g, kernel, rc):
    """Make exported GPU function `kernel` (bound under attribute or symbol name) return status rc."""
    fake = _bind_gpu(_FailingLib(rc), kernel, [])
    attr = {"sort_topk_i32_gpu": "_sort_topk_gpu", "masked_global_sum_f32_gpu": "_masked_sum",
            "join_count_i32_gpu": "_join_count_gpu", "group_sum_f32_gpu": "_group_sum"}.get(kernel)
    if attr:
        monkeypatch.setattr(g, attr, fake)
    else:
        monkeypatch.setitem(g.__dict__.setdefault("_sym_cache", {}), kernel, fake)


def _keys():
    return np.arange(1000, dtype=np.int32)[::-1].copy()


CALLS = {
    "sort_topk_i32_gpu": lambda g: g.sort_topk(_keys(), 5),
    "masked_global_sum_f32_gpu": lambda g: g.masked_global_sum_f32(np.ones(1000, np.float32), np.ones(1000, np.int32)),
    "scan_i32_gpu": lambda g: g._scan_i32(g._upload(np.zeros(1000, np.int32)), g._upload(np.ones(1000, np.int32)), 1000),
    "join_count_i32_gpu": lambda g: g.hash_join(_keys(), _keys()),
    "group_sum_f32_gpu": lambda g: g.group_sum_f32(np.ones(1000, np.float32), np.zeros(1000, np.int32), 1),
}


@pytest.mark.parametrize("kernel", sorted(CALLS))
@pytest.mark.parametrize("rc, exc, text", [(-2, MemoryError, "OUT_OF_MEMORY"), (-1, RuntimeError, "CUDA_ERROR")])
def test_failed_kernel_raises_instead_of_returning_garbage(monkeypatch, kernel, rc, exc, text):
    g = get_aot_gpu()
    CALLS[kernel](g)                                  # works when nothing fails
    _fail(monkeypatch, g, kernel, rc)
    with pytest.raises(exc, match=text):
        CALLS[kernel](g)


def test_hash_join_raises_when_its_sort_of_the_right_keys_fails(monkeypatch):
    g = get_aot_gpu()
    lk, rk = _keys(), _keys()
    li, ri = g.hash_join(lk, rk)
    assert (lk[li] == rk[ri]).all() and len(li) == 1000
    _fail(monkeypatch, g, "sort_topk_i32_gpu", -2)    # the `order` buffer of the join
    with pytest.raises(MemoryError):
        g.hash_join(lk, rk)


def test_real_oversized_sort_returns_out_of_memory_status():
    """The real library, no memory hog: asking for 2**34 rows makes the sort's scratch 128 GB, which fails
    before any kernel runs. The raw exported function returns -2 (this also catches a rewording of the
    exception text `_status` in kernels_aot_gpu.mojo matches), the wrapper raises MemoryError, and the
    process is still usable afterwards."""
    import ctypes
    g = get_aot_gpu()
    keys = g._upload(_keys())
    out = g._alloc(len(_keys()) * 4)
    try:
        raw = g._lib.sort_topk_i32_gpu
        raw.restype = ctypes.c_int64
        raw.argtypes = [ctypes.c_int64] * 4
        assert raw(out, keys, 2**34, 0) == -2
        with pytest.raises(MemoryError, match="OUT_OF_MEMORY"):
            g._sort_topk_gpu(g._ptr(out), g._ptr(keys), 2**34, 0)
    finally:
        g._free(keys, out)
    assert g.sort_topk(np.array([9, 8, 7], dtype=np.int32), 3).tolist() == [2, 1, 0]
    assert g.sort_topk(_keys(), 5).tolist() == [999, 998, 997, 996, 995]


def _force_gpu(monkeypatch):
    monkeypatch.setattr(am, "choose_device", lambda plan, **kw: am.AutoDecision("gpu", "forced by test", None))
    am.forget_gpu_failures()


@pytest.mark.parametrize("kernel, query", [
    ("sort_topk_i32_gpu", lambda t, u: LazyFrame(t).sort("k", descending=True).limit(10)),
    ("sort_topk_i32_gpu", lambda t, u: LazyFrame(t).sort("k")),
    ("join_count_i32_gpu", lambda t, u: LazyFrame(t).join(LazyFrame(u), left_on="k", right_on="id")
        .groupby("g").agg(col("v").sum().alias("s")).sort("g")),
])
def test_auto_reruns_on_the_cpu_when_a_gpu_kernel_fails(monkeypatch, kernel, query):
    rng = np.random.default_rng(1)
    n = 20_000
    t = pa.table({"k": rng.integers(0, 500, n).astype(np.int32), "v": rng.random(n, dtype=np.float32),
                  "g": rng.integers(0, 5, n).astype(np.int32)})
    u = pa.table({"id": np.arange(500, dtype=np.int32)})
    lf = query(t, u)
    want = lf.compute(device="cpu")
    g = get_aot_gpu()
    _force_gpu(monkeypatch)
    _fail(monkeypatch, g, kernel, -2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = lf.compute(device="auto")
    assert got.equals(want)
    assert lf.last_auto_decision.device == "cpu" and lf.last_auto_decision.fallback
    assert any("fell back to CPU" in str(x.message) for x in w)
    with pytest.raises(MemoryError):
        lf.compute(device="gpu")
    am.forget_gpu_failures()


HOG = r"""
import ctypes, sys, time
leave_mb, secs = int(sys.argv[1]), float(sys.argv[2])
c = ctypes.CDLL("libcuda.so.1"); c.cuInit(0)
dev = ctypes.c_int(); c.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); c.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
free, tot = ctypes.c_size_t(), ctypes.c_size_t(); c.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(tot))
got, want = 0, free.value - leave_mb * 2**20
while got + 2**28 <= want:
    p = ctypes.c_uint64()
    if c.cuMemAlloc_v2(ctypes.byref(p), 2**28): break
    got += 2**28
print("held", got >> 20, flush=True)
time.sleep(secs)
"""

SORT = r"""
import numpy as np, pyarrow as pa
from mxframe import LazyFrame
rng = np.random.default_rng(9)
n = 30_000_000       # more scratch than the GPU has left
t = pa.table({"k": rng.integers(0, 1000, n).astype(np.int32), "j": rng.integers(0, 7, n).astype(np.int32),
              "v": np.arange(n, dtype=np.int32)})
want = np.lexsort((np.arange(n), -t.column("j").to_numpy(), -t.column("k").to_numpy()))[:10]
try:
    got = LazyFrame(t).sort("k", "j", descending=True).limit(10).compute(device="gpu").column("v").to_numpy()
except (MemoryError, RuntimeError) as e:
    print("RAISED", type(e).__name__); raise SystemExit(0)
assert got.tolist() == want.tolist(), (got, want)
print("CORRECT")
"""


@pytest.mark.skipif(not os.environ.get("MXFRAME_TEST_GPU_HOG"), reason="holds most of one GPU's memory: set MXFRAME_TEST_GPU_HOG=<free MB to leave, e.g. 400>")
def test_real_out_of_memory_raises(tmp_path):
    """With the GPU's memory held by another process the sort raises MemoryError (it is never wrong)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               PYTHONPATH=os.pathsep.join([sys.path[0], os.environ.get("PYTHONPATH", "")]))
    hog = subprocess.Popen([sys.executable, "-c", HOG, os.environ["MXFRAME_TEST_GPU_HOG"], "120"], env=env,
                           stdout=subprocess.PIPE, text=True)
    try:
        hog.stdout.readline()
        r = subprocess.run([sys.executable, "-c", SORT], env=env, capture_output=True, text=True, timeout=300)
    finally:
        hog.kill()
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "RAISED MemoryError", r.stdout
