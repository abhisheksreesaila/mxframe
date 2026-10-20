"""F-014: the engine's CPU sort uses the parallel radix sort (same result as a stable argsort)."""
import numpy as np
import pyarrow as pa
import pytest

import conftest  # noqa: F401
from omxcore import LazyFrame, col
from omxcore.lazy_frame import Scan


def _t(n=300_000, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table({"a": rng.integers(0, 50, n).astype(np.int32), "b": rng.normal(size=n),
                     "c": rng.integers(-5, 5, n).astype(np.int64), "row": np.arange(n, dtype=np.int64)})


@pytest.mark.parametrize("by,desc", [(["a"], False), (["a"], True), (["a", "c"], [False, True]), (["b"], False)])
def test_sort_matches_stable_numpy(by, desc):
    t = _t()
    got = LazyFrame(Scan(t)).sort(*by, descending=desc).compute(device="cpu")
    keys = [t[c].to_numpy() for c in by]
    flags = desc if isinstance(desc, list) else [desc] * len(by)
    order = np.lexsort([(-k if d else k) for k, d in reversed(list(zip(keys, flags)))])
    np.testing.assert_array_equal(got["row"].to_numpy(), t["row"].to_numpy()[order])


def test_engine_calls_radix_sort(monkeypatch):
    from omxcore import aot_kernels
    calls = []
    real = aot_kernels.AOTKernels.sort_indices

    def spy(self, keys, *a, **k):
        calls.append(len(keys))
        return real(self, keys, *a, **k)

    monkeypatch.setattr(aot_kernels.AOTKernels, "sort_indices", spy)
    LazyFrame(Scan(_t())).sort("a").compute(device="cpu")
    assert calls and calls[0] == 300_000


def test_bad_radix_result_falls_back_to_argsort(monkeypatch):
    from omxcore import aot_kernels
    monkeypatch.setattr(aot_kernels.AOTKernels, "sort_indices",
                        lambda self, keys, *a, **k: np.arange(len(keys), dtype=np.int32)[::-1].copy())
    t = _t(100_000)
    with pytest.warns(RuntimeWarning, match="radix sort"):
        got = LazyFrame(Scan(t)).sort("a").compute(device="cpu")
    order = np.argsort(t["a"].to_numpy(), kind="stable")
    np.testing.assert_array_equal(got["row"].to_numpy(), t["row"].to_numpy()[order])
