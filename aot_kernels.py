"""
aot_kernels.py — ctypes bindings for libmxkernels_aot.so

Drop-in CPU dispatch layer.  No MAX Engine session, no JIT, no cold starts.
The .so is compiled once at build time; works for any row count N.

Usage (internal to custom_ops.py):
    from .aot_kernels import AOTKernels, AOT_AVAILABLE
    if AOT_AVAILABLE:
        _aot = AOTKernels()
        _aot.group_sum_f32(out, values, labels, n_groups)
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── Locate the shared library ─────────────────────────────────────────────────
def _find_lib() -> Optional[Path]:
    # 1. Env override (useful for development)
    env = os.environ.get("MXFRAME_AOT_LIB")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2. Next to this file (installed package)
    here = Path(__file__).parent
    candidate = here / "libmxkernels_aot.so"
    if candidate.exists():
        return candidate

    # 3. kernels_aot/ subdirectory of this file's directory (dev layout)
    dev = here / "kernels_aot" / "libmxkernels_aot.so"
    if dev.exists():
        return dev

    # 4. kernels_aot/ sibling of this file's parent (alternate layout)
    alt = here.parent / "kernels_aot" / "libmxkernels_aot.so"
    if alt.exists():
        return alt

    return None


_LIB_PATH = _find_lib()
AOT_AVAILABLE: bool = _LIB_PATH is not None

_c_int64 = ctypes.c_int64   # pointer-as-int (64-bit address)
_c_int   = ctypes.c_int64   # n_rows / n_groups etc. (use int64 for Mojo Int = Int64 on x86-64)


def _ptr(arr: np.ndarray) -> int:
    """Return the ctypes data pointer as a Python int."""
    return arr.ctypes.data


class AOTKernels:
    """Thin ctypes wrapper around libmxkernels_aot.so.

    All methods accept numpy arrays and dispatch directly to Mojo-compiled
    functions.  No MAX Graph session is needed.
    """

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        path = lib_path or _LIB_PATH
        if path is None:
            raise RuntimeError(
                "libmxkernels_aot.so not found. "
                "Run `mojo build --emit shared-lib kernels_aot/kernels_aot.mojo "
                "-o kernels_aot/libmxkernels_aot.so` to build it."
            )
        self._lib = ctypes.CDLL(str(path))
        self._bind()

    def _fn(self, name: str, argtypes: list) -> ctypes.CFUNCTYPE:
        fn = getattr(self._lib, name)
        fn.argtypes = argtypes
        fn.restype  = None
        return fn

    def _bind(self) -> None:
        I = _c_int64   # pointer-as-int or size
        # ── group aggs ────────────────────────────────────────────────────────
        self._group_sum_f32     = self._fn("group_sum_f32",     [I,I,I,I,I])
        self._group_sum_i64     = self._fn("group_sum_i64",     [I,I,I,I,I])
        self._group_min_f32     = self._fn("group_min_f32",     [I,I,I,I,I])
        self._group_max_f32     = self._fn("group_max_f32",     [I,I,I,I,I])
        self._group_mean_f32    = self._fn("group_mean_f32",    [I,I,I,I,I,I])
        self._group_count_f32   = self._fn("group_count_f32",   [I,I,I,I])
        self._group_composite   = self._fn("group_composite",   [I,I,I,I,I,I,I])
        # ── masked global aggs ───────────────────────────────────────────────
        self._masked_global_sum_f32         = self._fn("masked_global_sum_f32",         [I,I,I,I])
        self._masked_global_min_f32         = self._fn("masked_global_min_f32",         [I,I,I,I])
        self._masked_global_max_f32         = self._fn("masked_global_max_f32",         [I,I,I,I])
        self._masked_global_sum_product_f32 = self._fn("masked_global_sum_product_f32", [I,I,I,I,I])
        # ── gather ────────────────────────────────────────────────────────────
        self._gather_f32 = self._fn("gather_f32", [I,I,I,I])
        self._gather_i32 = self._fn("gather_i32", [I,I,I,I])
        self._gather_i64 = self._fn("gather_i64", [I,I,I,I])
        # ── sort / distinct ──────────────────────────────────────────────────
        self._sort_indices     = self._fn("sort_indices",     [I,I,I,I])
        self._unique_mask      = self._fn("unique_mask",      [I,I,I])
        # ── filter_gather ────────────────────────────────────────────────────
        self._prefix_sum_count = self._fn("prefix_sum_count", [I,I,I])
        self._filter_gather_f32 = self._fn("filter_gather_f32", [I,I,I,I,I])
        self._filter_gather_i32 = self._fn("filter_gather_i32", [I,I,I,I,I])
        self._filter_gather_i64 = self._fn("filter_gather_i64", [I,I,I,I,I])
        # ── joins ─────────────────────────────────────────────────────────────
        self._join_count        = self._fn("join_count",        [I,I,I,I,I,I])
        self._join_scatter      = self._fn("join_scatter",      [I,I,I,I,I,I,I,I])
        self._join_count_left   = self._fn("join_count_left",   [I,I,I,I,I,I])
        self._join_scatter_left = self._fn("join_scatter_left", [I,I,I,I,I,I,I,I])

    # ── Public API ────────────────────────────────────────────────────────────

    def group_sum_f32(self, values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out = np.zeros(n_groups, dtype=np.float32)
        self._group_sum_f32(_ptr(out), _ptr(values), _ptr(labels), len(values), n_groups)
        return out

    def group_sum_i64(self, values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out = np.zeros(n_groups, dtype=np.int64)
        self._group_sum_i64(_ptr(out), _ptr(values), _ptr(labels), len(values), n_groups)
        return out

    def group_min_f32(self, values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out = np.empty(n_groups, dtype=np.float32)
        self._group_min_f32(_ptr(out), _ptr(values), _ptr(labels), len(values), n_groups)
        return out

    def group_max_f32(self, values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out = np.empty(n_groups, dtype=np.float32)
        self._group_max_f32(_ptr(out), _ptr(values), _ptr(labels), len(values), n_groups)
        return out

    def group_mean_f32(self, values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out   = np.empty(n_groups, dtype=np.float32)
        count = np.empty(n_groups, dtype=np.int32)
        self._group_mean_f32(_ptr(out), _ptr(count), _ptr(values), _ptr(labels), len(values), n_groups)
        return out

    def group_count_f32(self, labels: np.ndarray, n_groups: int) -> np.ndarray:
        out = np.empty(n_groups, dtype=np.float32)
        self._group_count_f32(_ptr(out), _ptr(labels), len(labels), n_groups)
        return out

    def group_composite(self, k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, k3: np.ndarray,
                        strides: np.ndarray) -> np.ndarray:
        n = len(k0)
        out = np.empty(n, dtype=np.int64)
        self._group_composite(_ptr(out), _ptr(k0), _ptr(k1), _ptr(k2), _ptr(k3), _ptr(strides), n)
        return out

    def masked_global_sum_f32(self, values: np.ndarray, mask: np.ndarray) -> float:
        out = np.zeros(1, dtype=np.float32)
        self._masked_global_sum_f32(_ptr(out), _ptr(values), _ptr(mask), len(values))
        return float(out[0])

    def masked_global_min_f32(self, values: np.ndarray, mask: np.ndarray) -> float:
        out = np.zeros(1, dtype=np.float32)
        self._masked_global_min_f32(_ptr(out), _ptr(values), _ptr(mask), len(values))
        return float(out[0])

    def masked_global_max_f32(self, values: np.ndarray, mask: np.ndarray) -> float:
        out = np.zeros(1, dtype=np.float32)
        self._masked_global_max_f32(_ptr(out), _ptr(values), _ptr(mask), len(values))
        return float(out[0])

    def masked_global_sum_product_f32(self, values_a: np.ndarray, values_b: np.ndarray,
                                      mask: np.ndarray) -> float:
        out = np.zeros(1, dtype=np.float32)
        self._masked_global_sum_product_f32(_ptr(out), _ptr(values_a), _ptr(values_b),
                                            _ptr(mask), len(values_a))
        return float(out[0])

    def gather_f32(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        out = np.empty(len(indices), dtype=np.float32)
        self._gather_f32(_ptr(out), _ptr(values), _ptr(indices), len(indices))
        return out

    def gather_i32(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        out = np.empty(len(indices), dtype=np.int32)
        self._gather_i32(_ptr(out), _ptr(values), _ptr(indices), len(indices))
        return out

    def gather_i64(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        out = np.empty(len(indices), dtype=np.int64)
        self._gather_i64(_ptr(out), _ptr(values), _ptr(indices), len(indices))
        return out

    def sort_indices(self, keys: np.ndarray, descending: bool = False) -> np.ndarray:
        n = len(keys)
        out = np.empty(n, dtype=np.int32)
        self._sort_indices(_ptr(out), _ptr(keys), n, int(descending))
        return out

    def unique_mask(self, sorted_keys: np.ndarray) -> np.ndarray:
        n = len(sorted_keys)
        out = np.empty(n, dtype=np.int32)
        self._unique_mask(_ptr(out), _ptr(sorted_keys), n)
        return out

    def prefix_sum_count(self, mask: np.ndarray) -> np.ndarray:
        n = len(mask)
        offsets = np.empty(n + 1, dtype=np.int32)
        self._prefix_sum_count(_ptr(offsets), _ptr(mask), n)
        return offsets

    def filter_gather_f32(self, values: np.ndarray, mask: np.ndarray,
                          offsets: np.ndarray) -> np.ndarray:
        count = int(offsets[-1])
        out   = np.empty(count, dtype=np.float32)
        self._filter_gather_f32(_ptr(out), _ptr(values), _ptr(mask), _ptr(offsets), len(values))
        return out

    def filter_gather_i32(self, values: np.ndarray, mask: np.ndarray,
                          offsets: np.ndarray) -> np.ndarray:
        count = int(offsets[-1])
        out   = np.empty(count, dtype=np.int32)
        self._filter_gather_i32(_ptr(out), _ptr(values), _ptr(mask), _ptr(offsets), len(values))
        return out

    def filter_gather_i64(self, values: np.ndarray, mask: np.ndarray,
                          offsets: np.ndarray) -> np.ndarray:
        count = int(offsets[-1])
        out   = np.empty(count, dtype=np.int64)
        self._filter_gather_i64(_ptr(out), _ptr(values), _ptr(mask), _ptr(offsets), len(values))
        return out

    def join_count(self, left_keys: np.ndarray, right_keys: np.ndarray) -> np.ndarray:
        n_left  = len(left_keys)
        max_key = int(max(left_keys.max(), right_keys.max())) if n_left > 0 and len(right_keys) > 0 else 0
        out     = np.zeros(n_left, dtype=np.int32)
        self._join_count(_ptr(out), _ptr(left_keys), _ptr(right_keys),
                         n_left, len(right_keys), max_key)
        return out

    def join_scatter(self, left_keys: np.ndarray, right_keys: np.ndarray,
                     match_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Emit (left_idx, right_idx) pairs. match_counts[i] = # right matches for left row i."""
        n_left  = len(left_keys)
        n_right = len(right_keys)
        total   = int(match_counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        offsets = np.zeros(n_left, dtype=np.int32)
        if n_left > 1:
            np.cumsum(match_counts[:-1], out=offsets[1:])
        max_key = int(max(left_keys.max(), right_keys.max())) if n_left > 0 and n_right > 0 else 0
        lo      = np.empty(total, dtype=np.int32)
        ro      = np.empty(total, dtype=np.int32)
        self._join_scatter(_ptr(lo), _ptr(ro), _ptr(left_keys), _ptr(right_keys),
                           _ptr(offsets), n_left, n_right, max_key)
        return lo, ro

    def join_count_left(self, left_keys: np.ndarray, right_keys: np.ndarray) -> np.ndarray:
        n_left  = len(left_keys)
        max_key = int(max(left_keys.max(), right_keys.max())) if n_left > 0 and len(right_keys) > 0 else 0
        out     = np.zeros(n_left, dtype=np.int32)
        self._join_count_left(_ptr(out), _ptr(left_keys), _ptr(right_keys),
                              n_left, len(right_keys), max_key)
        return out

    def join_scatter_left(self, left_keys: np.ndarray, right_keys: np.ndarray,
                          match_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Emit left-outer (left_idx, right_idx) pairs. match_counts from join_count_left."""
        n_left  = len(left_keys)
        n_right = len(right_keys)
        total   = int(match_counts.sum())  # includes 1 for each unmatched left row
        if total == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        offsets = np.zeros(n_left, dtype=np.int32)
        if n_left > 1:
            np.cumsum(match_counts[:-1], out=offsets[1:])
        max_key = int(max(left_keys.max(), right_keys.max())) if n_left > 0 and n_right > 0 else 0
        lo      = np.empty(total, dtype=np.int32)
        ro      = np.empty(total, dtype=np.int32)
        self._join_scatter_left(_ptr(lo), _ptr(ro), _ptr(left_keys), _ptr(right_keys),
                                _ptr(offsets), n_left, n_right, max_key)
        return lo, ro


# ─────────────────────────────────────────────────────────────────────────────
# GPU AOT layer  (CUDA driver API path — no CuPy required)
# Uses Mojo's DeviceContext which shares the primary CUDA context.
# ─────────────────────────────────────────────────────────────────────────────

def _find_gpu_lib() -> Optional[Path]:
    env = os.environ.get("MXFRAME_AOT_GPU_LIB")
    if env:
        p = Path(env)
        if p.exists():
            return p
    here = Path(__file__).parent
    for candidate in [
        here / "libmxkernels_aot_gpu.so",
        here / "kernels_aot" / "libmxkernels_aot_gpu.so",
        here.parent / "kernels_aot" / "libmxkernels_aot_gpu.so",
    ]:
        if candidate.exists():
            return candidate
    return None


def _bind_gpu(lib, name: str, argtypes: list):
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = argtypes
    return fn


_P64 = ctypes.c_int64   # GPU device pointer as int64
_I64 = ctypes.c_int64   # plain integer sizes
_F32 = ctypes.c_float

_gpu_lib_path = _find_gpu_lib()
GPU_AOT_AVAILABLE = _gpu_lib_path is not None


class _CUDADriver:
    """Minimal CUDA driver API wrapper for H2D/D2H without CuPy dependency."""

    libcuda_names = ["libcuda.so.1", "libcuda.so", "cuda"]

    def __init__(self):
        lib = None
        for name in self.libcuda_names:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if lib is None:
            raise RuntimeError("libcuda.so not found")
        self._lib = lib

        # cuInit
        lib.cuInit.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]
        err = lib.cuInit(0)
        if err not in (0, 3):   # 3 = CUDA_ERROR_NO_DEVICE (ok in that case)
            raise RuntimeError(f"cuInit failed: {err}")

        # cuDeviceGet / cuDevicePrimaryCtxRetain / cuCtxSetCurrent
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_uint64]

        dev = ctypes.c_int(0)
        lib.cuDeviceGet(ctypes.byref(dev), 0)
        ctx = ctypes.c_uint64(0)
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        lib.cuCtxSetCurrent(ctx)
        self._ctx = ctx

        # cuMemAlloc / cuMemFree / memcpy
        lib.cuMemAlloc_v2.restype = ctypes.c_int
        lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.restype = ctypes.c_int
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
        lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuCtxSynchronize.restype = ctypes.c_int
        lib.cuCtxSynchronize.argtypes = []

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64(0)
        err = self._lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        if err != 0:
            raise RuntimeError(f"cuMemAlloc failed: {err}")
        return int(ptr.value)

    def free(self, ptr: int):
        self._lib.cuMemFree_v2(ctypes.c_uint64(ptr))

    def h2d(self, dev_ptr: int, arr: np.ndarray):
        hp = arr.ctypes.data_as(ctypes.c_void_p)
        self._lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), hp,
                                    ctypes.c_size_t(arr.nbytes))

    def d2h(self, arr: np.ndarray, dev_ptr: int):
        hp = arr.ctypes.data_as(ctypes.c_void_p)
        self._lib.cuMemcpyDtoH_v2(hp, ctypes.c_uint64(dev_ptr),
                                    ctypes.c_size_t(arr.nbytes))

    def sync(self):
        self._lib.cuCtxSynchronize()


# lazy singleton — initialized once per process
_cuda_driver: Optional["_CUDADriver"] = None


def _get_cuda_driver() -> Optional["_CUDADriver"]:
    global _cuda_driver
    if _cuda_driver is None:
        try:
            _cuda_driver = _CUDADriver()
        except Exception:
            return None
    return _cuda_driver


class AOTKernelsGPU:
    """ctypes bindings for libmxkernels_aot_gpu.so.

    Uses CUDA driver API for H2D/D2H without requiring CuPy.
    Mojo's DeviceContext() shares the primary CUDA context so
    device pointers allocated here are visible inside the Mojo kernels.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        p = lib_path or _gpu_lib_path
        if p is None:
            raise FileNotFoundError("libmxkernels_aot_gpu.so not found")
        self._lib = ctypes.CDLL(str(p))
        self._cu = _get_cuda_driver()
        if self._cu is None:
            raise RuntimeError("CUDA driver unavailable")
        # Device-buffer cache: avoids re-uploading the same numpy arrays on hot runs.
        # Key: (data_ptr, shape, dtype_str) → (device pointer, host owner).
        # Retaining the owner prevents NumPy from recycling a temporary array's
        # address while its old device buffer is still cached.
        # Evicts oldest entry when size exceeds _BUF_CACHE_MAX.
        self._buf_cache: dict = {}
        self._BUF_CACHE_MAX = 128
        self._bind()

    # ── symbol binding ──────────────────────────────────────────────────────

    def _bind(self):
        L = self._lib
        _5  = [_P64, _P64, _P64, _I64, _I64]
        _4  = [_P64, _P64, _P64, _I64]
        _4c = [_P64, _P64, _I64, _I64]                # group_count (no val)
        _mgs = [_P64, _P64, _P64, _I64]               # masked_global_sum
        _mgsp = [_P64, _P64, _P64, _P64, _I64]        # sum_product
        _range5 = [
            _P64, _P64, _P64, _P64, _P64, _P64,
            _I64, _I64, _I64, _F32, _F32, _F32,
        ]
        _semi_group = [_P64] * 8 + [_I64] * 6
        _key_counts = [_P64, _P64, _P64, _P64, _I64, _I64, _I64]
        _single_late = [_P64] * 11 + [_I64] * 4
        _fg  = [_P64, _P64, _P64, _P64, _I64]         # filter_gather
        _um  = [_P64, _P64, _I64]                      # unique_mask
        _utf8_sw = [_P64, _P64, _P64, _P64, _P64, _I64, _I64]
        _utf8_isin = [_P64, _P64, _P64, _P64, _P64, _P64, _I64, _I64]

        self._group_sum      = _bind_gpu(L, "group_sum_f32_gpu",             _5)
        self._group_min      = _bind_gpu(L, "group_min_f32_gpu",             _5)
        self._group_max      = _bind_gpu(L, "group_max_f32_gpu",             _5)
        self._group_count    = _bind_gpu(L, "group_count_f32_gpu",           _4c)
        self._masked_sum     = _bind_gpu(L, "masked_global_sum_f32_gpu",     _mgs)
        self._masked_min     = _bind_gpu(L, "masked_global_min_f32_gpu",     _mgs)
        self._masked_max     = _bind_gpu(L, "masked_global_max_f32_gpu",     _mgs)
        self._masked_sumprod = _bind_gpu(L, "masked_global_sum_product_f32_gpu", _mgsp)
        self._range5_sumprod = _bind_gpu(L, "range5_sum_product_f32_gpu", _range5)
        self._semi_range_group_count = _bind_gpu(
            L, "semi_range_group_count_i32_gpu", _semi_group)
        self._key_match_counts = _bind_gpu(
            L, "key_match_counts_i32_gpu", _key_counts)
        self._single_late_supplier_counts = _bind_gpu(
            L, "single_late_supplier_counts_i32_gpu", _single_late)
        # group_encode_i32_gpu: keys, out_ids, out_ng, htab, htid, n, cap
        _7  = [_P64, _P64, _P64, _P64, _P64, _I64, _I64]
        self._group_encode_i32 = _bind_gpu(L, "group_encode_i32_gpu", _7)
        self._gather_f32   = _bind_gpu(L, "gather_f32_gpu",  _4)
        self._gather_i32   = _bind_gpu(L, "gather_i32_gpu",  _4)
        self._gather_i64   = _bind_gpu(L, "gather_i64_gpu",  _4)
        self._fg_f32 = _bind_gpu(L, "filter_gather_f32_gpu", _fg)
        self._fg_i32 = _bind_gpu(L, "filter_gather_i32_gpu", _fg)
        self._fg_i64 = _bind_gpu(L, "filter_gather_i64_gpu", _fg)
        self._unique_mask  = _bind_gpu(L, "unique_mask_gpu", _um)
        self._utf8_startswith = _bind_gpu(
            L, "utf8_startswith_mask_gpu", _utf8_sw)
        self._utf8_contains = _bind_gpu(
            L, "utf8_contains_mask_gpu", _utf8_sw)
        self._utf8_equal = _bind_gpu(
            L, "utf8_equal_mask_gpu", _utf8_sw)
        self._utf8_isin = _bind_gpu(
            L, "utf8_isin_mask_gpu", _utf8_isin)
        # join kernels (Phase 4 — device-resident join pipeline)
        # join_count: (counts, table, left, right, n_left, n_right, table_size)
        _jc  = [_P64, _P64, _P64, _P64, _I64, _I64, _I64]
        # join_scatter: (left_out, right_out, left, offsets, right_order, right_starts, right_count, n_left, table_size)
        _js  = [_P64, _P64, _P64, _P64, _P64, _P64, _P64, _I64, _I64]
        self._join_count_gpu   = _bind_gpu(L, "join_count_i32_gpu",   _jc)
        self._join_scatter_gpu = _bind_gpu(L, "join_scatter_i32_gpu", _js)
        # sort (Phase 5 — GPU bitonic sort for ORDER BY ... LIMIT)
        # sort_topk: (out_idx, keys, n, desc_flag)
        _st  = [_P64, _P64, _I64, _I64]
        self._sort_topk_gpu = _bind_gpu(L, "sort_topk_i32_gpu", _st)

    # ── GPU memory helpers ──────────────────────────────────────────────────

    def _upload(self, arr: np.ndarray) -> int:
        """Copy numpy array to GPU, return device pointer (int)."""
        dev = self._cu.malloc(arr.nbytes)
        self._cu.h2d(dev, arr)
        return dev

    def _cached_upload(
        self,
        arr: np.ndarray,
        *,
        cache_key=None,
        owner=None,
    ) -> int:
        """Upload arr to GPU with caching by data pointer.

        On hot benchmark runs the same numpy arrays are reused (they live in
        _INPUT_PREP_CACHE), so this avoids redundant PCIe copies.  The cache
        is keyed by (data_ptr, shape, dtype) which is stable for the same
        Python array object.  Output buffers (written by kernels) must NOT
        be cached here — only read-only input arrays.
        """
        key = cache_key or (arr.ctypes.data, arr.shape, arr.dtype.str)
        entry = self._buf_cache.get(key)
        if entry is not None:
            return entry[0]
        ptr = self._upload(arr)
        if len(self._buf_cache) >= self._BUF_CACHE_MAX:
            # Evict the oldest entry (dict insertion order) and free GPU memory.
            old_key, (old_ptr, _) = next(iter(self._buf_cache.items()))
            del self._buf_cache[old_key]
            self._cu.free(old_ptr)
        self._buf_cache[key] = (ptr, owner if owner is not None else arr)
        return ptr

    def clear_buf_cache(self):
        """Free all cached device buffers and clear the cache."""
        for ptr, _ in self._buf_cache.values():
            self._cu.free(ptr)
        self._buf_cache.clear()

    def _download(self, dev_ptr: int, dtype, n: int) -> np.ndarray:
        out = np.empty(n, dtype=dtype)
        self._cu.d2h(out, dev_ptr)
        return out

    def _free(self, *ptrs: int):
        for p in ptrs:
            if p:
                self._cu.free(p)

    def _ptr(self, dev_ptr: int) -> ctypes.c_int64:
        return ctypes.c_int64(dev_ptr)

    # ── grouped aggs ────────────────────────────────────────────────────────

    @staticmethod
    def _c(arr: np.ndarray, dtype) -> np.ndarray:
        """Return arr as contiguous array of given dtype, zero-copy if possible."""
        if arr.dtype == dtype and arr.flags['C_CONTIGUOUS']:
            return arr
        return np.ascontiguousarray(arr, dtype=dtype)

    def group_sum_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v  = self._cached_upload(self._c(val,    np.float32))
        la = self._cached_upload(self._c(labels, np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_sum(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._cu.free(out)
        return res

    def group_min_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v  = self._cached_upload(self._c(val,    np.float32))
        la = self._cached_upload(self._c(labels, np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_min(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._cu.free(out)
        return res

    def group_max_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v  = self._cached_upload(self._c(val,    np.float32))
        la = self._cached_upload(self._c(labels, np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_max(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._cu.free(out)
        return res

    def group_count_f32(self, labels: np.ndarray, n_groups: int) -> np.ndarray:
        la = self._cached_upload(self._c(labels, np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_count(self._ptr(out), self._ptr(la), len(labels), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._cu.free(out)
        return res

    def group_mean_f32(self, val: np.ndarray, labels: np.ndarray,
                       n_groups: int) -> np.ndarray:
        """Grouped mean: computes sum and count on GPU, divides on CPU.

        The divide is over n_groups values (typically small) so the CPU cost
        is negligible. Re-uses existing sum/count kernels without needing a
        new Mojo kernel.
        """
        v  = self._cached_upload(self._c(val,    np.float32))
        la = self._cached_upload(self._c(labels, np.int32))
        sum_buf = self._cu.malloc(n_groups * 4)
        cnt_buf = self._cu.malloc(n_groups * 4)
        self._group_sum(self._ptr(sum_buf), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._group_count(self._ptr(cnt_buf), self._ptr(la), len(labels), n_groups)
        self._cu.sync()
        sums = self._download(sum_buf, np.float32, n_groups)
        cnts = self._download(cnt_buf, np.float32, n_groups)
        self._cu.free(sum_buf)
        self._cu.free(cnt_buf)
        mask = cnts > 0.0
        return np.where(mask, sums / np.where(mask, cnts, 1.0), 0.0).astype(np.float32)

    @staticmethod
    def _utf8_buffers(arr: "pa.Array") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expose a sliced Arrow UTF-8 array as offsets, data, and validity."""
        import pyarrow as pa

        if not pa.types.is_string(arr.type):
            raise TypeError(f"Expected Arrow string array, got {arr.type}")

        n = len(arr)
        validity_buffer, offsets_buffer, data_buffer = arr.buffers()
        raw_offsets = np.frombuffer(offsets_buffer, dtype=np.int32)
        offsets = np.ascontiguousarray(
            raw_offsets[arr.offset:arr.offset + n + 1] - raw_offsets[arr.offset],
            dtype=np.int32,
        )
        data_start = int(raw_offsets[arr.offset])
        data_end = int(raw_offsets[arr.offset + n])
        data = np.frombuffer(data_buffer, dtype=np.uint8)[data_start:data_end]
        data = np.ascontiguousarray(data, dtype=np.uint8)
        valid = np.ones(n, dtype=np.uint8)
        if validity_buffer is not None:
            valid = np.ascontiguousarray(
                arr.is_valid().to_numpy(zero_copy_only=False), dtype=np.uint8)
        return offsets, data, valid

    def _utf8_match_mask(self, arr: "pa.Array", pattern: str, kernel) -> np.ndarray:
        n = len(arr)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        offsets, data, valid = self._utf8_buffers(arr)
        pattern_bytes = np.frombuffer(pattern.encode("utf-8"), dtype=np.uint8).copy()
        if len(pattern_bytes) == 0:
            return valid.astype(np.int32, copy=False)

        offsets_dev = self._upload(offsets)
        data_dev = self._upload(data)
        valid_dev = self._upload(valid)
        pattern_dev = self._upload(pattern_bytes)
        out_dev = self._cu.malloc(n * 4)
        try:
            kernel(
                self._ptr(out_dev), self._ptr(offsets_dev), self._ptr(data_dev),
                self._ptr(valid_dev), self._ptr(pattern_dev), n, len(pattern_bytes),
            )
            self._cu.sync()
            return self._download(out_dev, np.int32, n)
        finally:
            self._free(offsets_dev, data_dev, valid_dev, pattern_dev, out_dev)

    def utf8_startswith_mask(self, arr: "pa.Array", prefix: str) -> np.ndarray:
        """Evaluate an Arrow UTF-8 starts-with predicate on the GPU."""
        return self._utf8_match_mask(arr, prefix, self._utf8_startswith)

    def utf8_contains_mask(self, arr: "pa.Array", pattern: str) -> np.ndarray:
        """Evaluate an Arrow UTF-8 substring predicate on the GPU."""
        return self._utf8_match_mask(arr, pattern, self._utf8_contains)

    def utf8_equal_mask(self, arr: "pa.Array", value: str) -> np.ndarray:
        """Evaluate exact Arrow UTF-8 scalar equality on the GPU."""
        n = len(arr)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        offsets, data, valid = self._utf8_buffers(arr)
        pattern = np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()
        pattern_storage = pattern if len(pattern) else np.zeros(1, dtype=np.uint8)
        devices = [self._upload(value) for value in (offsets, data, valid, pattern_storage)]
        out = self._cu.malloc(n * 4)
        try:
            self._utf8_equal(
                self._ptr(out), *(self._ptr(ptr) for ptr in devices), n, len(pattern))
            self._cu.sync()
            return self._download(out, np.int32, n)
        finally:
            self._free(*devices, out)

    def utf8_isin_mask(self, arr: "pa.Array", values: list[str]) -> np.ndarray:
        """Evaluate membership in packed UTF-8 literals on the GPU."""
        n = len(arr)
        if n == 0 or not values:
            return np.zeros(n, dtype=np.int32)
        unique_values = list(dict.fromkeys(values))
        encoded = [value.encode("utf-8") for value in unique_values]
        pattern_offsets = np.zeros(len(encoded) + 1, dtype=np.int32)
        np.cumsum([len(value) for value in encoded], out=pattern_offsets[1:])
        joined = b"".join(encoded)
        pattern_data = np.frombuffer(joined, dtype=np.uint8).copy()
        if len(pattern_data) == 0:
            pattern_data = np.zeros(1, dtype=np.uint8)
        offsets, data, valid = self._utf8_buffers(arr)
        devices = [self._upload(value) for value in (
            offsets, data, valid, pattern_offsets, pattern_data)]
        out = self._cu.malloc(n * 4)
        try:
            self._utf8_isin(
                self._ptr(out), *(self._ptr(ptr) for ptr in devices),
                n, len(unique_values))
            self._cu.sync()
            return self._download(out, np.int32, n)
        finally:
            self._free(*devices, out)

    def group_encode_i32(self, keys: np.ndarray, cap: int) -> tuple:
        """GPU open-addressing hash-table encode of int32 keys.

        Returns (dense_group_ids: np.int32[N], n_groups: int).
        Kernel initialises its own hash tables (htab, htid) from scratch;
        no pre-initialised arrays required from the caller.

        cap must be a power-of-two >= 2 * expected_distinct_values.
        Keys must not equal INT32_MIN (-2147483648) (empty-slot sentinel).
        """
        k    = self._cached_upload(self._c(keys, np.int32))
        out_ids = self._cu.malloc(len(keys) * 4)
        out_ng  = self._cu.malloc(4)
        htab    = self._cu.malloc(cap * 4)   # scratch: key table
        htid    = self._cu.malloc(cap * 4)   # scratch: id table
        self._group_encode_i32(
            self._ptr(k), self._ptr(out_ids), self._ptr(out_ng),
            self._ptr(htab), self._ptr(htid),
            len(keys), cap,
        )
        self._cu.sync()
        dense_ids = self._download(out_ids, np.int32, len(keys))
        n_groups  = int(self._download(out_ng, np.int32, 1)[0])
        self._free(out_ids, out_ng, htab, htid)
        return dense_ids, n_groups

    # ── masked global aggs ──────────────────────────────────────────────────

    def masked_global_sum_f32(self, val: np.ndarray, mask: np.ndarray) -> float:
        v = self._cached_upload(self._c(val,  np.float32))
        m = self._cached_upload(self._c(mask, np.int32))
        out = self._cu.malloc(4)
        self._masked_sum(self._ptr(out), self._ptr(v), self._ptr(m), len(val))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._cu.free(out)
        return float(res)

    def masked_global_min_f32(self, val: np.ndarray, mask: np.ndarray) -> float:
        v = self._cached_upload(self._c(val,  np.float32))
        m = self._cached_upload(self._c(mask, np.int32))
        out = self._cu.malloc(4)
        self._masked_min(self._ptr(out), self._ptr(v), self._ptr(m), len(val))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._cu.free(out)
        return float(res)

    def masked_global_max_f32(self, val: np.ndarray, mask: np.ndarray) -> float:
        v = self._cached_upload(self._c(val,  np.float32))
        m = self._cached_upload(self._c(mask, np.int32))
        out = self._cu.malloc(4)
        self._masked_max(self._ptr(out), self._ptr(v), self._ptr(m), len(val))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._cu.free(out)
        return float(res)

    def masked_global_sum_product_f32(self, a: np.ndarray, b: np.ndarray,
                                       mask: np.ndarray) -> float:
        av = self._cached_upload(self._c(a,    np.float32))
        bv = self._cached_upload(self._c(b,    np.float32))
        m  = self._cached_upload(self._c(mask, np.int32))
        out = self._cu.malloc(4)
        self._masked_sumprod(self._ptr(out), self._ptr(av), self._ptr(bv),
                              self._ptr(m), len(a))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._cu.free(out)
        return float(res)

    def range5_sum_product_f32(
        self,
        a: np.ndarray,
        b: np.ndarray,
        int_col: np.ndarray,
        float_col: np.ndarray,
        upper_col: np.ndarray,
        int_lo: int,
        int_hi: int,
        float_lo: float,
        float_hi: float,
        upper_hi: float,
    ) -> float:
        """Fuse five range comparisons with sum(a * b) in one GPU pass."""
        arrays = [a, b, float_col, upper_col]
        av, bv, fv, uv = [self._cached_upload(self._c(v, np.float32)) for v in arrays]
        iv = self._cached_upload(self._c(int_col, np.int32))
        out = self._cu.malloc(4)
        self._range5_sumprod(
            self._ptr(out), self._ptr(av), self._ptr(bv), self._ptr(iv),
            self._ptr(fv), self._ptr(uv), len(a), int_lo, int_hi,
            float_lo, float_hi, upper_hi,
        )
        self._cu.sync()
        result = float(self._download(out, np.float32, 1)[0])
        self._cu.free(out)
        return result

    def semi_range_group_count_i32(
        self,
        right_keys: np.ndarray,
        right_a: np.ndarray,
        right_b: np.ndarray,
        left_keys: np.ndarray,
        left_range: np.ndarray,
        left_groups: np.ndarray,
        n_groups: int,
        left_lo: int,
        left_hi: int,
    ) -> np.ndarray:
        """GPU semi-join plus range filter and grouped row count."""
        inputs = [right_keys, right_a, right_b, left_keys, left_range, left_groups]
        devices = [self._cached_upload(self._c(value, np.int32)) for value in inputs]
        table_size = int(max(np.max(right_keys), np.max(left_keys))) + 1
        marks = self._cu.malloc(table_size * 4)
        out = self._cu.malloc(n_groups * 4)
        self._semi_range_group_count(
            self._ptr(out), self._ptr(marks), *(self._ptr(ptr) for ptr in devices),
            len(right_keys), len(left_keys), table_size, n_groups, left_lo, left_hi,
        )
        self._cu.sync()
        result = self._download(out, np.int32, n_groups)
        self._free(marks, out)
        return result

    def key_match_counts_i32(
        self, left_keys: np.ndarray, right_keys: np.ndarray
    ) -> np.ndarray:
        """Return the number of right-key matches for every left key."""
        left = self._c(left_keys, np.int32)
        right = self._c(right_keys, np.int32)
        table_size = int(max(np.max(left), np.max(right))) + 1
        left_dev = self._cached_upload(left)
        right_dev = self._cached_upload(right)
        counts = self._cu.malloc(table_size * 4)
        out = self._cu.malloc(len(left) * 4)
        self._key_match_counts(
            self._ptr(out), self._ptr(counts), self._ptr(left_dev),
            self._ptr(right_dev), len(left), len(right), table_size,
        )
        self._cu.sync()
        result = self._download(out, np.int32, len(left))
        self._free(counts, out)
        return result

    def single_late_supplier_counts_i32(
        self,
        order_keys: np.ndarray,
        supplier_keys: np.ndarray,
        commit: np.ndarray,
        receipt: np.ndarray,
        finished_orders: np.ndarray,
        target_suppliers: np.ndarray,
    ) -> np.ndarray:
        """Count Q21 single-late-order waits per target supplier on GPU."""
        inputs = [order_keys, supplier_keys, commit, receipt,
                  finished_orders, target_suppliers]
        devices = [self._cached_upload(self._c(value, np.int32)) for value in inputs]
        n_rows = len(order_keys)
        n_orders = len(finished_orders)
        n_suppliers = len(target_suppliers)
        cap = 1
        while cap < max(2, n_rows * 2):
            cap <<= 1
        out = self._cu.malloc(n_suppliers * 4)
        order_counts = self._cu.malloc(n_orders * 4)
        states = self._cu.malloc(cap * 4)
        hash_orders = self._cu.malloc(cap * 4)
        hash_suppliers = self._cu.malloc(cap * 4)
        self._single_late_supplier_counts(
            self._ptr(out), self._ptr(order_counts), self._ptr(states),
            self._ptr(hash_orders), self._ptr(hash_suppliers),
            *(self._ptr(ptr) for ptr in devices),
            n_rows, n_orders, n_suppliers, cap,
        )
        self._cu.sync()
        result = self._download(out, np.int32, n_suppliers)
        self._free(out, order_counts, states, hash_orders, hash_suppliers)
        return result

    # ── gather ───────────────────────────────────────────────────────────────

    def gather_f32(self, src: np.ndarray, idx: np.ndarray) -> np.ndarray:
        s = self._upload(src.astype(np.float32))
        i = self._upload(idx.astype(np.int32))
        out = self._cu.malloc(len(idx) * 4)
        self._gather_f32(self._ptr(out), self._ptr(s), self._ptr(i), len(idx))
        self._cu.sync()
        res = self._download(out, np.float32, len(idx))
        self._free(s, i, out)
        return res

    def gather_i32(self, src: np.ndarray, idx: np.ndarray) -> np.ndarray:
        s = self._upload(src.astype(np.int32))
        i = self._upload(idx.astype(np.int32))
        out = self._cu.malloc(len(idx) * 4)
        self._gather_i32(self._ptr(out), self._ptr(s), self._ptr(i), len(idx))
        self._cu.sync()
        res = self._download(out, np.int32, len(idx))
        self._free(s, i, out)
        return res

    def gather_i64(self, src: np.ndarray, idx: np.ndarray) -> np.ndarray:
        s = self._upload(src.astype(np.int64))
        i = self._upload(idx.astype(np.int32))
        out = self._cu.malloc(len(idx) * 8)
        self._gather_i64(self._ptr(out), self._ptr(s), self._ptr(i), len(idx))
        self._cu.sync()
        res = self._download(out, np.int64, len(idx))
        self._free(s, i, out)
        return res

    # ── Phase 4: device-resident join pipeline ───────────────────────────────

    def hash_join(
        self, left_keys: np.ndarray, right_keys: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray]":
        """GPU two-pass hash join using AOT kernels (no MAX Graph JIT).

        Returns (left_idx, right_idx) as int32 numpy arrays.
        Key arrays are uploaded with _cached_upload so repeated calls with the
        same arrays avoid redundant PCIe copies.
        """
        n_left  = len(left_keys)
        n_right = len(right_keys)
        if n_left == 0 or n_right == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        lk = self._c(left_keys,  np.int32)
        rk = self._c(right_keys, np.int32)
        table_size = int(max(lk.max(), rk.max())) + 1 if len(lk) and len(rk) else 1

        lk_dev = self._cached_upload(lk)
        rk_dev = self._cached_upload(rk)

        # Work buffer: per-key right count table (pre-zeroed)
        table_np   = np.zeros(table_size, dtype=np.int32)
        table_dev  = self._upload(table_np)
        counts_dev = self._cu.malloc(n_left * 4)

        # Phase 1 — count
        self._join_count_gpu(
            self._ptr(counts_dev), self._ptr(table_dev),
            self._ptr(lk_dev),     self._ptr(rk_dev),
            n_left, n_right, table_size,
        )
        self._cu.sync()
        match_counts = self._download(counts_dev, np.int32, n_left)
        self._cu.free(counts_dev)
        self._cu.free(table_dev)

        total = int(match_counts.sum())
        if total == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        # Python bridge: offsets + right auxiliary arrays (small; stays on CPU)
        offsets = np.zeros(n_left, dtype=np.int32)
        np.cumsum(match_counts[:-1], out=offsets[1:])

        right_count  = np.bincount(rk, minlength=table_size).astype(np.int32)
        right_order  = np.argsort(rk, kind='stable').astype(np.int32)
        right_starts = np.zeros(table_size, dtype=np.int32)
        np.cumsum(right_count[:-1], out=right_starts[1:])

        offsets_dev      = self._upload(offsets)
        right_order_dev  = self._upload(right_order)
        right_starts_dev = self._upload(right_starts)
        right_count_dev  = self._upload(right_count)
        left_out_dev     = self._cu.malloc(total * 4)
        right_out_dev    = self._cu.malloc(total * 4)

        # Phase 2 — scatter
        self._join_scatter_gpu(
            self._ptr(left_out_dev),    self._ptr(right_out_dev),
            self._ptr(lk_dev),          self._ptr(offsets_dev),
            self._ptr(right_order_dev), self._ptr(right_starts_dev),
            self._ptr(right_count_dev),
            n_left, table_size,
        )
        self._cu.sync()

        left_idx  = self._download(left_out_dev,  np.int32, total)
        right_idx = self._download(right_out_dev, np.int32, total)
        self._free(offsets_dev, right_order_dev, right_starts_dev,
                   right_count_dev, left_out_dev, right_out_dev)
        return left_idx, right_idx

    def gather_table(
        self,
        table: "pa.Table",
        idx: np.ndarray,
    ) -> "pa.Table":
        """Gather all columns of *table* by *idx* using AOT GPU kernels.

        float32, int32, int64 columns are gathered on GPU with _cached_upload
        for source data (stable input) and a single index upload shared across
        all columns.  Other types (string, date, …) fall back to PyArrow CPU.

        Returns a new pa.Table with the same schema.
        """
        import pyarrow as pa
        n = len(idx)
        idx32 = np.ascontiguousarray(idx.astype(np.int32))
        idx_dev = self._upload(idx32)   # single H2D for index array

        arrays = []
        names  = []
        for col_name in table.column_names:
            col = table.column(col_name)
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            src_len = len(col)
            value_buffer = col.buffers()[1]
            buffer_address = value_buffer.address if value_buffer is not None else 0

            def cached_source(src_np, target_dtype):
                key = (
                    "arrow",
                    buffer_address,
                    col.offset,
                    src_len,
                    str(col.type),
                    np.dtype(target_dtype).str,
                )
                return self._cached_upload(src_np, cache_key=key, owner=col)

            if pa.types.is_float32(col.type):
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.float32))
                s = cached_source(src_np, np.float32)
                out = self._cu.malloc(n * 4)
                self._gather_f32(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.float32, n)))
                self._cu.free(out)
            elif pa.types.is_float64(col.type):
                # downcast float64→float32 for GPU, restore type after
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.float32))
                s = cached_source(src_np, np.float32)
                out = self._cu.malloc(n * 4)
                self._gather_f32(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(
                    self._download(out, np.float32, n).astype(np.float64)))
                self._cu.free(out)
            elif pa.types.is_integer(col.type) and col.type.bit_width == 64:
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.int64))
                s = cached_source(src_np, np.int64)
                out = self._cu.malloc(n * 8)
                self._gather_i64(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.int64, n)))
                self._cu.free(out)
            elif pa.types.is_integer(col.type):
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.int32))
                s = cached_source(src_np, np.int32)
                out = self._cu.malloc(n * 4)
                self._gather_i32(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.int32, n)))
                self._cu.free(out)
            else:
                # Non-numeric fallback: PyArrow CPU take
                arrays.append(col.take(pa.array(idx32.astype(np.int64))))
            names.append(col_name)

        self._cu.free(idx_dev)
        return pa.Table.from_arrays(arrays, names=names)

    def sort_topk(
        self,
        keys: np.ndarray,
        k: int,
        descending: bool = False,
    ) -> np.ndarray:
        """GPU bitonic sort → return top-k sorted row indices as int32 numpy array.

        Runs a full GPU bitonic sort of the key array, then downloads the first k
        indices.  For large N (>100K), the GPU sort is significantly faster than
        CPU NumPy argsort.

        Args:
            keys:       int32 sort-key array of length N
            k:          number of top elements to return (1 ≤ k ≤ N)
            descending: if True, return indices of the k *largest* keys

        Returns:
            int32 numpy array of length k: row indices in sorted order.
        """
        keys_i32 = self._c(keys, np.int32)
        n = len(keys_i32)
        k = min(k, n)
        keys_dev = self._cached_upload(keys_i32)
        out_dev   = self._cu.malloc(n * 4)
        self._sort_topk_gpu(
            self._ptr(out_dev), self._ptr(keys_dev),
            n, int(descending),
        )
        self._cu.sync()
        # Download only the first k indices to minimize D2H bandwidth.
        # After a full ascending sort the smallest k values are at [0..k-1].
        top_k = self._download(out_dev, np.int32, k)
        self._cu.free(out_dev)
        return top_k

    def filter_gather_f32(self, src: np.ndarray, mask: np.ndarray,
                           offsets: np.ndarray, n_out: int) -> np.ndarray:
        s = self._upload(src.astype(np.float32))
        m = self._upload(mask.astype(np.int32))
        o = self._upload(offsets.astype(np.int32))
        out = self._cu.malloc(n_out * 4)
        self._fg_f32(self._ptr(out), self._ptr(s), self._ptr(m),
                      self._ptr(o), len(src))
        self._cu.sync()
        res = self._download(out, np.float32, n_out)
        self._free(s, m, o, out)
        return res

    def filter_gather_i32(self, src: np.ndarray, mask: np.ndarray,
                           offsets: np.ndarray, n_out: int) -> np.ndarray:
        s = self._upload(src.astype(np.int32))
        m = self._upload(mask.astype(np.int32))
        o = self._upload(offsets.astype(np.int32))
        out = self._cu.malloc(n_out * 4)
        self._fg_i32(self._ptr(out), self._ptr(s), self._ptr(m),
                      self._ptr(o), len(src))
        self._cu.sync()
        res = self._download(out, np.int32, n_out)
        self._free(s, m, o, out)
        return res

    def filter_gather_i64(self, src: np.ndarray, mask: np.ndarray,
                           offsets: np.ndarray, n_out: int) -> np.ndarray:
        s = self._upload(src.astype(np.int64))
        m = self._upload(mask.astype(np.int32))
        o = self._upload(offsets.astype(np.int32))
        out = self._cu.malloc(n_out * 8)
        self._fg_i64(self._ptr(out), self._ptr(s), self._ptr(m),
                      self._ptr(o), len(src))
        self._cu.sync()
        res = self._download(out, np.int64, n_out)
        self._free(s, m, o, out)
        return res

    def unique_mask(self, keys: np.ndarray) -> np.ndarray:
        k = self._upload(keys.astype(np.int32))
        out = self._cu.malloc(len(keys) * 4)
        self._unique_mask(self._ptr(out), self._ptr(k), len(keys))
        self._cu.sync()
        res = self._download(out, np.int32, len(keys))
        self._free(k, out)
        return res
