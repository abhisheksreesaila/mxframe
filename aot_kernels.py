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
        self._bind()

    # ── symbol binding ──────────────────────────────────────────────────────

    def _bind(self):
        L = self._lib
        _5  = [_P64, _P64, _P64, _I64, _I64]
        _4  = [_P64, _P64, _P64, _I64]
        _4c = [_P64, _P64, _I64, _I64]                # group_count (no val)
        _mgs = [_P64, _P64, _P64, _I64]               # masked_global_sum
        _mgsp = [_P64, _P64, _P64, _P64, _I64]        # sum_product
        _fg  = [_P64, _P64, _P64, _P64, _I64]         # filter_gather
        _um  = [_P64, _P64, _I64]                      # unique_mask

        self._group_sum    = _bind_gpu(L, "group_sum_f32_gpu",    _5)
        self._group_min    = _bind_gpu(L, "group_min_f32_gpu",    _5)
        self._group_max    = _bind_gpu(L, "group_max_f32_gpu",    _5)
        self._group_count  = _bind_gpu(L, "group_count_f32_gpu",  _4c)
        self._masked_sum   = _bind_gpu(L, "masked_global_sum_f32_gpu",         _mgs)
        self._masked_sumprod = _bind_gpu(L, "masked_global_sum_product_f32_gpu", _mgsp)
        self._gather_f32   = _bind_gpu(L, "gather_f32_gpu",  _4)
        self._gather_i32   = _bind_gpu(L, "gather_i32_gpu",  _4)
        self._gather_i64   = _bind_gpu(L, "gather_i64_gpu",  _4)
        self._fg_f32 = _bind_gpu(L, "filter_gather_f32_gpu", _fg)
        self._fg_i32 = _bind_gpu(L, "filter_gather_i32_gpu", _fg)
        self._fg_i64 = _bind_gpu(L, "filter_gather_i64_gpu", _fg)
        self._unique_mask  = _bind_gpu(L, "unique_mask_gpu", _um)

    # ── GPU memory helpers ──────────────────────────────────────────────────

    def _upload(self, arr: np.ndarray) -> int:
        """Copy numpy array to GPU, return device pointer (int)."""
        dev = self._cu.malloc(arr.nbytes)
        self._cu.h2d(dev, arr)
        return dev

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

    def group_sum_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v = self._upload(val.astype(np.float32))
        la = self._upload(labels.astype(np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_sum(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._free(v, la, out)
        return res

    def group_min_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v = self._upload(val.astype(np.float32))
        la = self._upload(labels.astype(np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_min(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._free(v, la, out)
        return res

    def group_max_f32(self, val: np.ndarray, labels: np.ndarray,
                      n_groups: int) -> np.ndarray:
        v = self._upload(val.astype(np.float32))
        la = self._upload(labels.astype(np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_max(self._ptr(out), self._ptr(v), self._ptr(la),
                        len(val), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._free(v, la, out)
        return res

    def group_count_f32(self, labels: np.ndarray, n_groups: int) -> np.ndarray:
        la = self._upload(labels.astype(np.int32))
        out = self._cu.malloc(n_groups * 4)
        self._group_count(self._ptr(out), self._ptr(la), len(labels), n_groups)
        self._cu.sync()
        res = self._download(out, np.float32, n_groups)
        self._free(la, out)
        return res

    # ── masked global aggs ──────────────────────────────────────────────────

    def masked_global_sum_f32(self, val: np.ndarray, mask: np.ndarray) -> float:
        v = self._upload(val.astype(np.float32))
        m = self._upload(mask.astype(np.int32))
        out = self._cu.malloc(4)
        self._masked_sum(self._ptr(out), self._ptr(v), self._ptr(m), len(val))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._free(v, m, out)
        return float(res)

    def masked_global_sum_product_f32(self, a: np.ndarray, b: np.ndarray,
                                       mask: np.ndarray) -> float:
        av = self._upload(a.astype(np.float32))
        bv = self._upload(b.astype(np.float32))
        m  = self._upload(mask.astype(np.int32))
        out = self._cu.malloc(4)
        self._masked_sumprod(self._ptr(out), self._ptr(av), self._ptr(bv),
                              self._ptr(m), len(a))
        self._cu.sync()
        res = self._download(out, np.float32, 1)[0]
        self._free(av, bv, m, out)
        return float(res)

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
