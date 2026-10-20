"""F-012: CPU sort_indices is a stable O(n log n)-or-better parallel radix sort;
unique_mask / prefix_sum_count / filter_gather are parallel and match NumPy.

The old sort_indices was an in-place rotate merge sort, O(n^2): 1e4 random rows hung.
"""
import time

import numpy as np
import pytest

from omxcore.aot_kernels import AOT_AVAILABLE, AOTKernels, get_cpu_threads, set_cpu_threads

pytestmark = pytest.mark.skipif(not AOT_AVAILABLE, reason="libmxkernels_aot.so not built")

THREADS = [1, 2, 3, 24]


@pytest.fixture(scope="module")
def aot():
    return AOTKernels()


@pytest.fixture(autouse=True)
def _restore_threads():
    before = get_cpu_threads()
    yield
    set_cpu_threads(before)


def want_order(keys, descending):
    """Stable argsort; descending keeps equal keys in row order too. NaN last, -0.0 == +0.0."""
    k = keys.astype(np.float64) if keys.dtype.kind == "f" else keys.astype(np.int64)
    if k.dtype.kind == "f":
        nan = np.isnan(k)
        k = np.where(nan, 0.0, k) + 0.0                      # -0.0 -> +0.0
        rank = k if not descending else -k
        return np.lexsort((np.arange(len(k)), rank, nan)).astype(np.int32)
    return np.argsort(-k if descending else k, kind="stable").astype(np.int32)


def keys_of(dtype, n, rng, span):
    if np.dtype(dtype).kind == "f":
        return ((rng.random(n) - 0.5) * span).astype(dtype)
    return rng.integers(-span, span, n).astype(dtype)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 100, 65_537, 300_001])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("span", [4, 10**6])           # many ties / mostly distinct
def test_sort_indices_is_stable_and_matches_numpy(aot, dtype, n, descending, span):
    keys = keys_of(dtype, n, np.random.default_rng(n + span), span)
    for t in THREADS:
        got = aot.sort_indices(keys, descending, threads=t)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, want_order(keys, descending))


def test_sort_extremes_nan_and_signed_zero(aot):
    i32 = np.array([2**31 - 1, -2**31, 0, -1, 1, -2**31, 2**31 - 1], np.int32)
    np.testing.assert_array_equal(aot.sort_indices(i32), np.argsort(i32, kind="stable"))
    i64 = np.array([2**63 - 1, -2**63, 0, -1, 1, -2**63, 2**63 - 1], np.int64)
    np.testing.assert_array_equal(aot.sort_indices(i64), np.argsort(i64, kind="stable"))
    f = np.array([np.nan, 1.0, -0.0, 0.0, -np.inf, np.inf, -1.0, np.nan, 0.0, -0.0], np.float32)
    for desc in (False, True):
        np.testing.assert_array_equal(aot.sort_indices(f, desc), want_order(f, desc))
        np.testing.assert_array_equal(aot.sort_indices(f.astype(np.float64), desc), want_order(f, desc))
    # -0.0 and +0.0 are equal keys, so they stay in row order
    z = np.array([0.0, -0.0, 0.0, -0.0], np.float64)
    np.testing.assert_array_equal(aot.sort_indices(z), [0, 1, 2, 3])


def test_sort_accepts_other_dtypes_and_strided_input(aot):
    rng = np.random.default_rng(3)
    for dtype in (np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.bool_, np.float16):
        k = rng.integers(0, 2, 1000).astype(dtype) if dtype is np.bool_ else rng.integers(0, 100, 1000).astype(dtype)
        np.testing.assert_array_equal(aot.sort_indices(k), np.argsort(k, kind="stable"))
    base = rng.integers(-50, 50, 3000).astype(np.int32)
    view = base[::3]
    np.testing.assert_array_equal(aot.sort_indices(view), np.argsort(view, kind="stable"))
    with pytest.raises(TypeError):
        aot.sort_indices(np.array(["a", "b"]))


def test_sort_is_n_log_n_not_quadratic(aot):
    """1e4 random rows hung the old kernel; a million must take well under a second."""
    rng = np.random.default_rng(0)
    for n in (10_000, 1_000_000):
        keys = rng.integers(-2**31, 2**31 - 1, n, dtype=np.int64).astype(np.int32)
        t = time.perf_counter()
        got = aot.sort_indices(keys)
        dt = time.perf_counter() - t
        assert dt < 2.0, f"sorting {n} rows took {dt:.2f}s"
        np.testing.assert_array_equal(got, np.argsort(keys, kind="stable"))


def test_sort_does_not_mutate_keys_and_is_deterministic(aot):
    keys = np.random.default_rng(1).integers(0, 50, 200_000).astype(np.int32)
    before = keys.copy()
    a = aot.sort_indices(keys, threads=1)
    b = aot.sort_indices(keys, threads=24)
    np.testing.assert_array_equal(keys, before)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 65_537, 300_001])
def test_unique_mask_and_prefix_sum_and_filter_gather(aot, n):
    rng = np.random.default_rng(n)
    sorted_keys = np.sort(rng.integers(0, 50, n).astype(np.int32))
    want_um = np.r_[1, sorted_keys[1:] != sorted_keys[:-1]].astype(np.int32)[:n]
    mask = (rng.random(n) > 0.6).astype(np.int32)
    want_offs = np.r_[0, np.cumsum(mask)].astype(np.int32)
    f32 = rng.random(n, dtype=np.float32)
    i32 = rng.integers(-10**6, 10**6, n).astype(np.int32)
    i64 = rng.integers(-10**12, 10**12, n).astype(np.int64)
    for t in THREADS:
        np.testing.assert_array_equal(aot.unique_mask(sorted_keys, threads=t), want_um)
        offs = aot.prefix_sum_count(mask, threads=t)
        np.testing.assert_array_equal(offs, want_offs)
        np.testing.assert_array_equal(aot.filter_gather_f32(f32, mask, offs, threads=t), f32[mask == 1])
        np.testing.assert_array_equal(aot.filter_gather_i32(i32, mask, offs, threads=t), i32[mask == 1])
        np.testing.assert_array_equal(aot.filter_gather_i64(i64, mask, offs, threads=t), i64[mask == 1])


def test_filter_gather_rejects_mismatched_lengths(aot):
    v = np.arange(10, dtype=np.float32)
    with pytest.raises(ValueError):
        aot.filter_gather_f32(v, np.ones(9, np.int32), np.arange(11, dtype=np.int32))
    with pytest.raises(ValueError):
        aot.filter_gather_f32(v, np.ones(10, np.int32), np.arange(5, dtype=np.int32))
