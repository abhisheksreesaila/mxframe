"""F-055: a GPU sort that ran out of memory returns a silently wrong order (the kernel only prints
an `oom log` on stderr), which made test_sort_end_to_end flaky on a shared GPU. The engine checks the
GPU's top-k result against the keys and redoes the sort on the CPU when it is wrong."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import LazyFrame, gpu_stream
from mxframe.sort_check import is_sorted_topk


def test_is_sorted_topk_accepts_the_stable_top_k_and_rejects_everything_else():
    keys = np.array([5, 1, 3, 1, 9, 3], dtype=np.int32)
    assert is_sorted_topk(keys, np.array([1, 3, 2], dtype=np.int32), 3)       # stable: ties by row
    assert is_sorted_topk(keys, np.argsort(keys, kind="stable").astype(np.int32), 6)
    assert not is_sorted_topk(keys, np.array([3, 1, 2], dtype=np.int32), 3)   # tie out of row order
    assert not is_sorted_topk(keys, np.array([0, 0, 0], dtype=np.int32), 3)   # all zeros (swallowed OOM)
    assert not is_sorted_topk(keys, np.array([1, 3, 6], dtype=np.int32), 3)   # row 6 is out of range
    assert not is_sorted_topk(keys, np.array([1, 3, 0], dtype=np.int32), 3)   # 5 is not among the 3 smallest
    assert not is_sorted_topk(keys, np.array([1, 3], dtype=np.int32), 3)      # too short


@pytest.mark.skipif(not gpu_stream.cuda_available(), reason="no CUDA GPU")
@pytest.mark.parametrize("limit", [10, None])
def test_gpu_sort_with_a_failed_kernel_still_returns_the_right_rows(monkeypatch, limit):
    from mxframe.aot_kernels import AOTKernelsGPU
    rng = np.random.default_rng(3)
    n = 50_000
    t = pa.table({"k": rng.integers(0, 1000, n).astype(np.int32), "v": np.arange(n, dtype=np.int32)})
    monkeypatch.setattr(AOTKernelsGPU, "sort_topk",
                        lambda self, keys, k, descending=False, device_out=False: np.zeros(k, dtype=np.int32))
    lf = LazyFrame(t).sort("k", descending=True)
    if limit:
        lf = lf.limit(limit)
    got = lf.compute(device="gpu").column("v").to_numpy()
    want = np.argsort(-t.column("k").to_numpy().astype(np.int64), kind="stable")[:limit or n]
    assert got.tolist() == want.tolist()
