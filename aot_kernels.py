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
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .np_arrow import take_rows

# The Mojo device-context memory manager reserves ~90% of the GPU's free memory the first time
# a kernel creates a buffer (the recursive scan does, at >= 1M rows), and this module's own
# cuMemAlloc calls (uploads, join outputs, gathers) then fail with out-of-memory at TPC-H SF10,
# as would any other process on the GPU.  A quarter is plenty for the kernels' scratch (the
# sort's is 8 bytes per row).  Set MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT to override.
os.environ.setdefault("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "25")

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

# Must match ABI_VERSION in kernels_aot/kernels_aot.mojo.  Bumped when the
# exported signatures change (F-002 added a trailing `threads` argument; ABI 3 added it to
# sort_indices / unique_mask / prefix_sum_count / filter_gather and added sort_indices_{i64,f32,f64}).
_EXPECTED_ABI = 3


def _ptr(arr: np.ndarray) -> int:
    """Return the ctypes data pointer as a Python int."""
    return arr.ctypes.data


def _check_len(name: str, arr: np.ndarray, n: int) -> None:
    """Kernels trust n; a shorter buffer would be read out of bounds."""
    if len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} elements, expected {n}")


def _arr(a: np.ndarray, dtype) -> np.ndarray:
    """C-contiguous view of `a` as `dtype` (no copy when it already is)."""
    return np.ascontiguousarray(a, dtype=dtype)


# ── CPU thread budget ─────────────────────────────────────────────────────────
# Every CPU kernel takes a `threads` argument: the max number of worker tasks it
# may use.  The default comes from env MXFRAME_CPU_THREADS (else all cores the
# process may run on) and can be changed at runtime with set_cpu_threads(); a
# co-processing scheduler that keeps some cores busy feeding the GPU lowers it
# for the CPU workers, or passes `threads=` per call.  There is no state inside
# the native library, so several Python threads may call kernels concurrently
# (ctypes releases the GIL), each with its own budget.

def _all_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:            # non-Linux
        return max(1, os.cpu_count() or 1)


def _threads_from_env() -> int:
    raw = os.environ.get("MXFRAME_CPU_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass
    return _all_cores()


_cpu_threads: int = _threads_from_env()
_PIN_LOCK = threading.Lock()
_RUNTIME_PINNED = False      # Mojo worker pool pinned once per process (see AOTKernels.__init__)


def get_cpu_threads() -> int:
    """Current default worker budget for CPU kernels."""
    return _cpu_threads


def set_cpu_threads(n: Optional[int]) -> int:
    """Set the default worker budget for CPU kernels; ``None`` resets it to
    ``MXFRAME_CPU_THREADS`` / all cores.  Returns the new value."""
    global _cpu_threads
    if n is None:
        _cpu_threads = _threads_from_env()
    else:
        n = int(n)
        if n < 1:
            raise ValueError("cpu threads must be >= 1")
        _cpu_threads = n
    return _cpu_threads


class AOTKernels:
    """Thin ctypes wrapper around libmxkernels_aot.so.

    All methods accept numpy arrays and dispatch directly to Mojo-compiled
    functions.  No MAX Graph session is needed.  Sum/mean accumulate in
    float64 and return float64; every method takes an optional ``threads=``
    (default: :func:`get_cpu_threads`).
    """

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        path = lib_path or _LIB_PATH
        if path is None:
            raise RuntimeError(
                "libmxkernels_aot.so not found. "
                "Run `mojo build -I kernels_aot --emit shared-lib kernels_aot/kernels_aot.mojo "
                "-o kernels_aot/libmxkernels_aot.so` to build it."
            )
        self._lib = ctypes.CDLL(str(path))   # CDLL (not PyDLL): the GIL is released per call
        try:
            abi = self._lib.mx_abi_version
        except AttributeError:
            abi = None
        if abi is not None:
            abi.restype = ctypes.c_int64
        if abi is None or abi() != _EXPECTED_ABI:
            raise RuntimeError(
                f"{path} is stale (expected kernel ABI {_EXPECTED_ABI}); rebuild it with "
                "`mojo build -I kernels_aot --emit shared-lib kernels_aot/kernels_aot.mojo "
                "-o kernels_aot/libmxkernels_aot.so`."
            )
        # Pin the Mojo worker pool for the life of the process: without a live
        # reference every parallel launch builds and tears down the pool (~1.7 ms).
        global _RUNTIME_PINNED
        self._lib.mx_init_runtime.restype = ctypes.c_int64
        with _PIN_LOCK:
            if not _RUNTIME_PINNED:
                self._lib.mx_init_runtime()
                _RUNTIME_PINNED = True
        self._lib.mx_num_physical_cores.restype = ctypes.c_int64
        self._bind()

    @property
    def physical_cores(self) -> int:
        return int(self._lib.mx_num_physical_cores())

    def _fn(self, name: str, argtypes: list) -> ctypes.CFUNCTYPE:
        fn = getattr(self._lib, name)
        fn.argtypes = argtypes
        fn.restype  = None
        return fn

    @staticmethod
    def _t(threads: Optional[int]) -> int:
        return _cpu_threads if threads is None else max(1, int(threads))

    def _bind(self) -> None:
        I = _c_int64   # pointer-as-int or size
        # ── group aggs: (out, values, labels, n_rows, n_groups, threads) ─────
        for name in ("group_sum_f32", "group_sum_f64", "group_sum_i64",
                     "group_min_f32", "group_min_f64", "group_max_f32", "group_max_f64",
                     "group_mean_f32", "group_mean_f64"):
            setattr(self, "_" + name, self._fn(name, [I]*6))
        self._group_count     = self._fn("group_count",     [I]*5)
        self._group_composite = self._fn("group_composite", [I]*8)
        # ── masked global aggs: (out, values, mask, n, threads) ──────────────
        for name in ("masked_global_sum_f32", "masked_global_sum_f64",
                     "masked_global_min_f32", "masked_global_min_f64",
                     "masked_global_max_f32", "masked_global_max_f64"):
            setattr(self, "_" + name, self._fn(name, [I]*5))
        self._masked_global_sum_product_f32 = self._fn("masked_global_sum_product_f32", [I]*6)
        self._masked_global_sum_product_f64 = self._fn("masked_global_sum_product_f64", [I]*6)
        # ── gather ────────────────────────────────────────────────────────────
        self._gather_f32 = self._fn("gather_f32", [I]*5)
        self._gather_i32 = self._fn("gather_i32", [I]*5)
        self._gather_i64 = self._fn("gather_i64", [I]*5)
        # ── sort / distinct / filter_gather: (…, n, threads) ──────────────────
        self._sort_indices = {
            "i32": self._fn("sort_indices",     [I]*5), "i64": self._fn("sort_indices_i64", [I]*5),
            "f32": self._fn("sort_indices_f32", [I]*5), "f64": self._fn("sort_indices_f64", [I]*5),
        }
        self._unique_mask      = self._fn("unique_mask",      [I]*4)
        self._prefix_sum_count = self._fn("prefix_sum_count", [I]*4)
        self._filter_gather_f32 = self._fn("filter_gather_f32", [I]*6)
        self._filter_gather_i32 = self._fn("filter_gather_i32", [I]*6)
        self._filter_gather_i64 = self._fn("filter_gather_i64", [I]*6)
        # ── joins ─────────────────────────────────────────────────────────────
        self._join_count        = self._fn("join_count",        [I]*7)
        self._join_scatter      = self._fn("join_scatter",      [I]*9)
        self._join_count_left   = self._fn("join_count_left",   [I]*7)
        self._join_scatter_left = self._fn("join_scatter_left", [I]*9)

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _vdt(values: np.ndarray) -> tuple:
        """('f32'|'f64', array): float32 stays float32, everything else -> float64."""
        if values.dtype == np.float32:
            return "f32", _arr(values, np.float32)
        return "f64", _arr(values, np.float64)

    def _group(self, op: str, values: np.ndarray, labels: np.ndarray, n_groups: int,
               threads: Optional[int]) -> np.ndarray:
        kind, v = self._vdt(values)
        lab = _arr(labels, np.int32)
        _check_len("labels", lab, len(v))
        out = np.empty(n_groups, dtype=np.float64)
        getattr(self, f"_group_{op}_{kind}")(_ptr(out), _ptr(v), _ptr(lab), len(v), n_groups,
                                             self._t(threads))
        return out

    def group_sum(self, values, labels, n_groups, threads=None) -> np.ndarray:
        """Per-group sum, float64 accumulation; float64 result (float32/float64 input)."""
        return self._group("sum", values, labels, n_groups, threads)

    def group_min(self, values, labels, n_groups, threads=None) -> np.ndarray:
        return self._group("min", values, labels, n_groups, threads)

    def group_max(self, values, labels, n_groups, threads=None) -> np.ndarray:
        return self._group("max", values, labels, n_groups, threads)

    def group_mean(self, values, labels, n_groups, threads=None) -> np.ndarray:
        """Per-group mean (sum/count in float64); empty groups stay 0."""
        return self._group("mean", values, labels, n_groups, threads)

    def group_count(self, labels, n_groups, threads=None) -> np.ndarray:
        """Per-group row count as float64 (exact to 2**53)."""
        lab = _arr(labels, np.int32)
        out = np.empty(n_groups, dtype=np.float64)
        self._group_count(_ptr(out), _ptr(lab), len(lab), n_groups, self._t(threads))
        return out

    # float32-named entry points kept for callers written against the old API
    # (now float64-accumulating and float64-returning).
    def group_sum_f32(self, values, labels, n_groups, threads=None):  return self.group_sum(values, labels, n_groups, threads)
    def group_min_f32(self, values, labels, n_groups, threads=None):  return self.group_min(values, labels, n_groups, threads)
    def group_max_f32(self, values, labels, n_groups, threads=None):  return self.group_max(values, labels, n_groups, threads)
    def group_mean_f32(self, values, labels, n_groups, threads=None): return self.group_mean(values, labels, n_groups, threads)
    def group_count_f32(self, labels, n_groups, threads=None):        return self.group_count(labels, n_groups, threads)

    def group_sum_i64(self, values: np.ndarray, labels: np.ndarray, n_groups: int,
                      threads: Optional[int] = None) -> np.ndarray:
        v = _arr(values, np.int64)
        lab = _arr(labels, np.int32)
        out = np.zeros(n_groups, dtype=np.int64)
        self._group_sum_i64(_ptr(out), _ptr(v), _ptr(lab), len(v), n_groups, self._t(threads))
        return out

    def group_composite(self, k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, k3: np.ndarray,
                        strides: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        n = len(k0)
        k0, k1, k2, k3 = (_arr(k, np.int32) for k in (k0, k1, k2, k3))
        strides = _arr(strides, np.int64)
        out = np.empty(n, dtype=np.int64)
        self._group_composite(_ptr(out), _ptr(k0), _ptr(k1), _ptr(k2), _ptr(k3), _ptr(strides), n,
                              self._t(threads))
        return out

    def _masked(self, op: str, values: np.ndarray, mask: np.ndarray, threads) -> float:
        kind, v = self._vdt(values)
        m = _arr(mask, np.int32)
        _check_len("mask", m, len(v))
        out = np.zeros(1, dtype=np.float64)
        getattr(self, f"_masked_global_{op}_{kind}")(_ptr(out), _ptr(v), _ptr(m), len(v), self._t(threads))
        return float(out[0])

    def masked_global_sum(self, values, mask, threads=None) -> float:
        """sum(values where mask != 0), float64 accumulation, SIMD + multi-core."""
        return self._masked("sum", values, mask, threads)

    def masked_global_min(self, values, mask, threads=None) -> float:
        return self._masked("min", values, mask, threads)

    def masked_global_max(self, values, mask, threads=None) -> float:
        return self._masked("max", values, mask, threads)

    def masked_global_sum_product(self, values_a, values_b, mask, threads=None) -> float:
        """sum(a * b where mask != 0); product and sum in float64."""
        kind, a = self._vdt(values_a)
        b = _arr(values_b, np.float32 if kind == "f32" else np.float64)
        m = _arr(mask, np.int32)
        _check_len("values_b", b, len(a))
        _check_len("mask", m, len(a))
        out = np.zeros(1, dtype=np.float64)
        getattr(self, f"_masked_global_sum_product_{kind}")(
            _ptr(out), _ptr(a), _ptr(b), _ptr(m), len(a), self._t(threads))
        return float(out[0])

    def masked_global_sum_f32(self, values, mask, threads=None):  return self.masked_global_sum(values, mask, threads)
    def masked_global_min_f32(self, values, mask, threads=None):  return self.masked_global_min(values, mask, threads)
    def masked_global_max_f32(self, values, mask, threads=None):  return self.masked_global_max(values, mask, threads)
    def masked_global_sum_product_f32(self, values_a, values_b, mask, threads=None):
        return self.masked_global_sum_product(values_a, values_b, mask, threads)

    def gather_f32(self, values: np.ndarray, indices: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        values, indices = _arr(values, np.float32), _arr(indices, np.int32)
        out = np.empty(len(indices), dtype=np.float32)
        self._gather_f32(_ptr(out), _ptr(values), _ptr(indices), len(indices), self._t(threads))
        return out

    def gather_i32(self, values: np.ndarray, indices: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        values, indices = _arr(values, np.int32), _arr(indices, np.int32)
        out = np.empty(len(indices), dtype=np.int32)
        self._gather_i32(_ptr(out), _ptr(values), _ptr(indices), len(indices), self._t(threads))
        return out

    def gather_i64(self, values: np.ndarray, indices: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        values, indices = _arr(values, np.int64), _arr(indices, np.int32)
        out = np.empty(len(indices), dtype=np.int64)
        self._gather_i64(_ptr(out), _ptr(values), _ptr(indices), len(indices), self._t(threads))
        return out

    @staticmethod
    def _sort_kind(keys: np.ndarray) -> tuple:
        """('i32'|'i64'|'f32'|'f64', keys as that dtype): the narrowest sort that loses nothing."""
        dt = keys.dtype
        if dt.kind == "f":
            return ("f32", _arr(keys, np.float32)) if dt.itemsize <= 4 else ("f64", _arr(keys, np.float64))
        if dt.kind in "biu":
            if dt.kind == "u" and dt.itemsize == 8 and len(keys) and int(keys.max()) >= 2**63:
                raise ValueError("uint64 keys above 2**63 cannot be sorted")
            narrow = dt.itemsize < 4 or (dt.kind != "u" and dt.itemsize == 4) or dt.kind == "b"
            return ("i32", _arr(keys, np.int32)) if narrow else ("i64", _arr(keys, np.int64))
        raise TypeError(f"cannot sort keys of dtype {dt}")

    def sort_indices(self, keys: np.ndarray, descending: bool = False,
                     threads: Optional[int] = None) -> np.ndarray:
        """Stable argsort (int32 row indices) of int32/int64/float32/float64 keys.

        Equal keys keep their row order in both directions (identical to
        ``np.argsort(keys, kind="stable")``, or of the negated keys for
        ``descending``); NaN sorts last either way.  Parallel LSD radix sort.
        """
        kind, k = self._sort_kind(np.asarray(keys))
        out = np.empty(len(k), dtype=np.int32)
        self._sort_indices[kind](_ptr(out), _ptr(k), len(k), int(descending), self._t(threads))
        return out

    def unique_mask(self, sorted_keys: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        k = _arr(sorted_keys, np.int32)
        out = np.empty(len(k), dtype=np.int32)
        self._unique_mask(_ptr(out), _ptr(k), len(k), self._t(threads))
        return out

    def prefix_sum_count(self, mask: np.ndarray, threads: Optional[int] = None) -> np.ndarray:
        """Exclusive prefix sum of `mask` in an int32 array of length n + 1 (last = total)."""
        m = _arr(mask, np.int32)
        offsets = np.empty(len(m) + 1, dtype=np.int32)
        self._prefix_sum_count(_ptr(offsets), _ptr(m), len(m), self._t(threads))
        return offsets

    def _filter_gather(self, fn, dtype, values, mask, offsets, threads) -> np.ndarray:
        v, m, o = _arr(values, dtype), _arr(mask, np.int32), _arr(offsets, np.int32)
        _check_len("mask", m, len(v))
        if len(o) not in (len(v), len(v) + 1):
            raise ValueError(f"offsets has {len(o)} elements, expected {len(v)} or {len(v) + 1}")
        count = int(o[-1]) if len(o) == len(v) + 1 else (int(o[-1]) + int(m[-1]) if len(v) else 0)
        if count < 0 or count > len(v):
            raise ValueError("offsets do not describe a selection of these rows")
        out = np.empty(count, dtype=dtype)
        fn(_ptr(out), _ptr(v), _ptr(m), _ptr(o), len(v), self._t(threads))
        return out

    def filter_gather_f32(self, values, mask, offsets, threads: Optional[int] = None) -> np.ndarray:
        return self._filter_gather(self._filter_gather_f32, np.float32, values, mask, offsets, threads)

    def filter_gather_i32(self, values, mask, offsets, threads: Optional[int] = None) -> np.ndarray:
        return self._filter_gather(self._filter_gather_i32, np.int32, values, mask, offsets, threads)

    def filter_gather_i64(self, values, mask, offsets, threads: Optional[int] = None) -> np.ndarray:
        return self._filter_gather(self._filter_gather_i64, np.int64, values, mask, offsets, threads)

    @staticmethod
    def _max_key(left_keys: np.ndarray, right_keys: np.ndarray) -> int:
        if len(left_keys) > 0 and len(right_keys) > 0:
            return int(max(left_keys.max(), right_keys.max()))
        return 0

    def join_count(self, left_keys: np.ndarray, right_keys: np.ndarray,
                   threads: Optional[int] = None) -> np.ndarray:
        left_keys, right_keys = _arr(left_keys, np.int32), _arr(right_keys, np.int32)
        n_left  = len(left_keys)
        out     = np.zeros(n_left, dtype=np.int32)
        self._join_count(_ptr(out), _ptr(left_keys), _ptr(right_keys),
                         n_left, len(right_keys), self._max_key(left_keys, right_keys),
                         self._t(threads))
        return out

    def join_scatter(self, left_keys: np.ndarray, right_keys: np.ndarray,
                     match_counts: np.ndarray, threads: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Emit (left_idx, right_idx) pairs. match_counts[i] = # right matches for left row i."""
        left_keys, right_keys = _arr(left_keys, np.int32), _arr(right_keys, np.int32)
        n_left  = len(left_keys)
        n_right = len(right_keys)
        total   = int(match_counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        offsets = np.zeros(n_left, dtype=np.int32)
        if n_left > 1:
            np.cumsum(match_counts[:-1], out=offsets[1:])
        lo      = np.empty(total, dtype=np.int32)
        ro      = np.empty(total, dtype=np.int32)
        self._join_scatter(_ptr(lo), _ptr(ro), _ptr(left_keys), _ptr(right_keys),
                           _ptr(offsets), n_left, n_right, self._max_key(left_keys, right_keys),
                           self._t(threads))
        return lo, ro

    def join_count_left(self, left_keys: np.ndarray, right_keys: np.ndarray,
                        threads: Optional[int] = None) -> np.ndarray:
        left_keys, right_keys = _arr(left_keys, np.int32), _arr(right_keys, np.int32)
        n_left  = len(left_keys)
        out     = np.zeros(n_left, dtype=np.int32)
        self._join_count_left(_ptr(out), _ptr(left_keys), _ptr(right_keys),
                              n_left, len(right_keys), self._max_key(left_keys, right_keys),
                              self._t(threads))
        return out

    def join_scatter_left(self, left_keys: np.ndarray, right_keys: np.ndarray,
                          match_counts: np.ndarray, threads: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Emit left-outer (left_idx, right_idx) pairs. match_counts from join_count_left."""
        left_keys, right_keys = _arr(left_keys, np.int32), _arr(right_keys, np.int32)
        n_left  = len(left_keys)
        n_right = len(right_keys)
        total   = int(match_counts.sum())  # includes 1 for each unmatched left row
        if total == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        offsets = np.zeros(n_left, dtype=np.int32)
        if n_left > 1:
            np.cumsum(match_counts[:-1], out=offsets[1:])
        lo      = np.empty(total, dtype=np.int32)
        ro      = np.empty(total, dtype=np.int32)
        self._join_scatter_left(_ptr(lo), _ptr(ro), _ptr(left_keys), _ptr(right_keys),
                                _ptr(offsets), n_left, n_right, self._max_key(left_keys, right_keys),
                                self._t(threads))
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
        # The current context is per thread.  The driver may be created on the warm-up
        # thread and used from the caller's thread, so every driver call re-binds
        # the (shared, primary) context once per thread.
        self._tls = threading.local()
        self._tls.bound = True

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

    def bind_thread(self) -> None:
        """Make the primary context current on the calling thread (once per thread).
        Mojo's DeviceContext() inside the AOT library retains the same primary context."""
        if not getattr(self._tls, "bound", False):
            self._lib.cuCtxSetCurrent(self._ctx)
            self._tls.bound = True

    def malloc(self, nbytes: int) -> int:
        self.bind_thread()
        ptr = ctypes.c_uint64(0)
        err = self._lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        if err != 0:
            raise RuntimeError(f"cuMemAlloc failed: {err}")
        return int(ptr.value)

    def free(self, ptr: int):
        self.bind_thread()
        self._lib.cuMemFree_v2(ctypes.c_uint64(ptr))

    def h2d(self, dev_ptr: int, arr: np.ndarray):
        self.bind_thread()
        hp = arr.ctypes.data_as(ctypes.c_void_p)
        self._lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), hp,
                                    ctypes.c_size_t(arr.nbytes))

    def d2h(self, arr: np.ndarray, dev_ptr: int):
        self.bind_thread()
        hp = arr.ctypes.data_as(ctypes.c_void_p)
        self._lib.cuMemcpyDtoH_v2(hp, ctypes.c_uint64(dev_ptr),
                                    ctypes.c_size_t(arr.nbytes))

    def sync(self):
        self.bind_thread()
        self._lib.cuCtxSynchronize()


# lazy singleton — initialized once per process
_cuda_driver: Optional["_CUDADriver"] = None


_gpu_init_lock = threading.RLock()


def _get_cuda_driver() -> Optional["_CUDADriver"]:
    global _cuda_driver
    with _gpu_init_lock:
        if _cuda_driver is None:
            try:
                _cuda_driver = _CUDADriver()
            except Exception:
                return None
        return _cuda_driver


_aot_gpu_singleton: Optional["AOTKernelsGPU"] = None
_aot_gpu_failed = False


def get_aot_gpu() -> Optional["AOTKernelsGPU"]:
    """The process-wide ``AOTKernelsGPU``, created on first call (thread-safe), or None.

    Nothing touches CUDA until this is called, so CPU-only use never creates a context.
    Creation happens once, on whichever thread asks first (the auto-mode warm-up thread
    or the caller's); a failure (no driver, no device) is remembered and returns None."""
    global _aot_gpu_singleton, _aot_gpu_failed
    if _aot_gpu_singleton is not None or _aot_gpu_failed or not GPU_AOT_AVAILABLE:
        return _aot_gpu_singleton
    with _gpu_init_lock:
        if _aot_gpu_singleton is None and not _aot_gpu_failed:
            try:
                _aot_gpu_singleton = AOTKernelsGPU()
            except Exception:
                _aot_gpu_failed = True
        return _aot_gpu_singleton


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
        dev = self._alloc(arr.nbytes)
        if arr.nbytes:
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
        if out.nbytes:
            self._cu.d2h(out, dev_ptr)
        return out

    def _sym(self, name: str, argtypes: list):
        """Bind (once) and return an exported kernel of the GPU library by name."""
        cache = self.__dict__.setdefault("_sym_cache", {})
        fn = cache.get(name)
        if fn is None:
            fn = cache[name] = _bind_gpu(self._lib, name, argtypes)
        return fn

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

    # ── device-resident inputs and outputs ──────────────────────────────────
    # Every wrapper below takes NumPy arrays (uploaded, and cached, as before) or a
    # `DeviceArray` (a device pointer + dtype + length, used in place with no copy),
    # and has a `device_out=` flag: True returns the result as a `DeviceArray` the
    # caller owns instead of downloading it.  The residency tracker and the Router
    # chain operators this way without host round trips.

    def _alloc(self, nbytes: int) -> int:
        """Device allocation; at least 16 bytes (cuMemAlloc rejects 0)."""
        return self._cu.malloc(max(int(nbytes), 16))

    def _dev_in(self, x, dtype, tmps: Optional[list] = None) -> tuple:
        """(device pointer, length) of a kernel input.

        A DeviceArray is used in place; its dtype must match (nothing is cast on
        the device).  A host array is converted to a contiguous `dtype` array and
        uploaded through the buffer cache, or, when `tmps` is a list, uploaded
        uncached with the pointer appended to `tmps` for the caller to free."""
        dtype = np.dtype(dtype)
        if isinstance(x, DeviceArray):
            if x.dtype != dtype:
                raise TypeError(f"device array holds {x.dtype}, the kernel needs {dtype}")
            return x.ptr, x.n
        arr = self._c(np.asarray(x), dtype)
        if tmps is None:
            return self._cached_upload(arr), len(arr)
        ptr = self._upload(arr)
        tmps.append(ptr)
        return ptr, len(arr)

    def _result(self, ptr: int, dtype, n: int, device_out: bool):
        """A kernel output: a DeviceArray the caller owns, or NumPy (buffer freed)."""
        if device_out:
            return DeviceArray(ptr, dtype, n, owns=True)
        try:
            return self._download(ptr, dtype, n)
        finally:
            self._cu.free(ptr)

    def _group_agg(self, fn, val, labels, n_groups: int, device_out: bool):
        n_groups = int(n_groups)
        if n_groups < 0:
            raise ValueError("n_groups must be >= 0")
        lab, n = self._dev_in(labels, np.int32)
        args = []
        if val is not None:
            v, nv = self._dev_in(val, np.float32)
            if nv != n:
                raise ValueError(f"values has {nv} elements, labels has {n}")
            args.append(v)
        out = self._alloc(n_groups * 4)
        fn(self._ptr(out), *(self._ptr(a) for a in args), self._ptr(lab), n, n_groups)
        self._cu.sync()
        return self._result(out, np.float32, n_groups, device_out)

    def group_sum_f32(self, val, labels, n_groups: int, device_out: bool = False):
        """Per-group sum of `val` (float32); labels outside [0, n_groups) are ignored."""
        return self._group_agg(self._group_sum, val, labels, n_groups, device_out)

    def group_min_f32(self, val, labels, n_groups: int, device_out: bool = False):
        """Per-group min; an empty group holds float32 max."""
        return self._group_agg(self._group_min, val, labels, n_groups, device_out)

    def group_max_f32(self, val, labels, n_groups: int, device_out: bool = False):
        """Per-group max; an empty group holds -float32 max."""
        return self._group_agg(self._group_max, val, labels, n_groups, device_out)

    def group_count_f32(self, labels, n_groups: int, device_out: bool = False):
        return self._group_agg(self._group_count, None, labels, n_groups, device_out)

    def group_mean_f32(self, val, labels, n_groups: int, device_out: bool = False):
        """Grouped mean (0 for an empty group): sum and count kernels, then a divide kernel."""
        n_groups = int(n_groups)
        sums = self.group_sum_f32(val, labels, n_groups, device_out=True)
        cnts = self.group_count_f32(labels, n_groups, device_out=True)
        out = self._alloc(n_groups * 4)
        self._sym("group_mean_finish_f32_gpu", [_P64, _P64, _P64, _I64])(
            self._ptr(out), self._ptr(sums.ptr), self._ptr(cnts.ptr), n_groups)
        self._cu.sync()
        sums.free()
        cnts.free()
        return self._result(out, np.float32, n_groups, device_out)

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

    def group_encode_i32(self, keys, cap: Optional[int] = None, device_out: bool = False) -> tuple:
        """GPU open-addressing hash-table encode of int32 keys.

        Returns (dense_group_ids: int32[N], n_groups: int).  Every int32 value is
        a legal key.  Ids are assigned in claim order, so the numbering (not the
        grouping) can differ between runs.  `cap` is the hash-table size: it must
        exceed the number of distinct keys; the default is the next power of two
        >= 2 * len(keys).  `device_out=True` returns the ids as a DeviceArray.
        """
        k, n = self._dev_in(keys, np.int32)
        if cap is None:
            cap = 1 << max(1, (2 * n - 1).bit_length()) if n else 2
        cap = int(cap)
        if cap <= n:
            raise ValueError(f"cap={cap} must exceed the number of distinct keys (<= {n})")
        out_ids = self._alloc(n * 4)
        out_ng = self._alloc(4)
        htab = self._alloc(cap * 4)   # scratch: keys
        htid = self._alloc(cap * 4)   # scratch: slot state / dense id
        self._group_encode_i32(
            self._ptr(k), self._ptr(out_ids), self._ptr(out_ng),
            self._ptr(htab), self._ptr(htid), n, cap,
        )
        self._cu.sync()
        n_groups = int(self._download(out_ng, np.int32, 1)[0])
        self._free(out_ng, htab, htid)
        return self._result(out_ids, np.int32, n, device_out), n_groups

    # ── masked global aggs ──────────────────────────────────────────────────

    def _masked(self, fn, arrays: list, mask, device_out: bool):
        ptrs, n = [], None
        for a in arrays:
            p, m = self._dev_in(a, np.float32)
            if n is not None and m != n:
                raise ValueError(f"operands have {n} and {m} elements")
            ptrs.append(p)
            n = m
        mk, nm = self._dev_in(mask, np.int32)
        if nm != n:
            raise ValueError(f"mask has {nm} elements, values have {n}")
        out = self._alloc(8)
        fn(self._ptr(out), *(self._ptr(p) for p in ptrs), self._ptr(mk), n)
        self._cu.sync()
        res = self._result(out, np.float64, 1, device_out)
        return res if device_out else float(res[0])

    def masked_global_sum_f32(self, val, mask, device_out: bool = False):
        return self._masked(self._masked_sum, [val], mask, device_out)

    def masked_global_min_f32(self, val, mask, device_out: bool = False):
        return self._masked(self._masked_min, [val], mask, device_out)

    def masked_global_max_f32(self, val, mask, device_out: bool = False):
        return self._masked(self._masked_max, [val], mask, device_out)

    def masked_global_sum_product_f32(self, a, b, mask, device_out: bool = False):
        return self._masked(self._masked_sumprod, [a, b], mask, device_out)

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
        out = self._cu.malloc(8)
        self._range5_sumprod(
            self._ptr(out), self._ptr(av), self._ptr(bv), self._ptr(iv),
            self._ptr(fv), self._ptr(uv), len(a), int_lo, int_hi,
            float_lo, float_hi, upper_hi,
        )
        self._cu.sync()
        result = float(self._download(out, np.float64, 1)[0])
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

    def _gather(self, fn, src, idx, dtype, device_out: bool):
        """out[i] = src[idx[i]].  A host `idx` is range-checked (a bad index would fault
        the kernel); a DeviceArray `idx` is trusted."""
        dtype = np.dtype(dtype)
        tmps: list = []
        try:
            s, ns = self._dev_in(src, dtype, tmps)
            if not isinstance(idx, DeviceArray):
                idx = np.asarray(idx)
                if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= ns):
                    raise IndexError(f"gather index out of range for a source of {ns} rows")
            i, n = self._dev_in(idx, np.int32, tmps)
            out = self._alloc(n * dtype.itemsize)
            if n:
                fn(self._ptr(out), self._ptr(s), self._ptr(i), n)
            self._cu.sync()
            return self._result(out, dtype, n, device_out)
        finally:
            self._free(*tmps)

    def gather_f32(self, src, idx, device_out: bool = False):
        return self._gather(self._gather_f32, src, idx, np.float32, device_out)

    def gather_i32(self, src, idx, device_out: bool = False):
        return self._gather(self._gather_i32, src, idx, np.int32, device_out)

    def gather_i64(self, src, idx, device_out: bool = False):
        return self._gather(self._gather_i64, src, idx, np.int64, device_out)

    # ── Phase 4: device-resident join pipeline ───────────────────────────────

    def hash_join(self, left_keys, right_keys, device_out: bool = False):
        """GPU two-pass hash join on int32 keys (inner join on equality).

        Returns (left_idx, right_idx), int32 row-index pairs: NumPy, or
        DeviceArrays the caller owns with ``device_out=True``.  Inputs may be
        NumPy or DeviceArrays; every step, including the prefix sums and the
        stable argsort of the right keys, runs on the device, so a device-resident
        join never touches the host except to read the two scalars (table size and
        result size) that size the buffers.  Rows pair up left-row-major, and among
        the matches of one left row in right-row order.
        """
        tmps: list = []
        bufs: list = []                                   # scratch freed on every exit, errors included
        try:
            lk, n_left = self._dev_in(left_keys, np.int32, tmps)
            rk, n_right = self._dev_in(right_keys, np.int32, tmps)
            if n_left == 0 or n_right == 0:
                return self._pair_result(self._empty_pair(), device_out)

            def scratch(nbytes: int) -> int:
                ptr = self._alloc(nbytes)
                bufs.append(ptr)
                return ptr

            # table_size = max key + 1 (keys must be >= 0, as before)
            mx = scratch(4)
            self._fill_i32(mx, 1, 0)
            self._sym("max_i32_gpu", [_P64, _P64, _I64])(self._ptr(mx), self._ptr(lk), n_left)
            self._sym("max_i32_gpu", [_P64, _P64, _I64])(self._ptr(mx), self._ptr(rk), n_right)
            table_size = int(self._download(mx, np.int32, 1)[0]) + 1

            table = scratch(table_size * 4)                # right rows per key
            self._fill_i32(table, table_size, 0)
            counts = scratch(n_left * 4)                   # matches per left row
            self._join_count_gpu(self._ptr(counts), self._ptr(table), self._ptr(lk), self._ptr(rk),
                                 n_left, n_right, table_size)
            offsets = scratch(n_left * 4)                  # exclusive prefix sum of counts
            total = self._scan_i32(offsets, counts, n_left)
            if total == 0:
                return self._pair_result(self._empty_pair(), device_out)
            starts = scratch(table_size * 4)               # exclusive prefix sum of table
            self._scan_i32(starts, table, table_size)
            order = scratch(n_right * 4)                   # stable argsort of right keys
            self._sort_topk_gpu(self._ptr(order), self._ptr(rk), n_right, 2)   # negative keys last
            left_out = self._alloc(total * 4)
            right_out = self._alloc(total * 4)
            pair = (DeviceArray(left_out, np.int32, total, owns=True),
                    DeviceArray(right_out, np.int32, total, owns=True))   # owned from here on
            self._join_scatter_gpu(
                self._ptr(left_out), self._ptr(right_out), self._ptr(lk), self._ptr(offsets),
                self._ptr(order), self._ptr(starts), self._ptr(table), n_left, table_size)
            self._cu.sync()
            return self._pair_result(pair, device_out)
        finally:
            self._free(*bufs, *tmps)

    def _empty_pair(self) -> tuple:
        return tuple(DeviceArray(self._alloc(4), np.int32, 0, owns=True) for _ in range(2))

    def _pair_result(self, pair, device_out: bool):
        if device_out:
            return pair
        out = tuple(d.to_numpy() for d in pair)
        for d in pair:
            d.free()
        return out

    def _fill_i32(self, ptr: int, n: int, value: int) -> None:
        if n > 0:
            self._sym("fill_i32_gpu", [_P64, _I64, _I64])(self._ptr(ptr), n, value)

    def _scan_i32(self, out: int, src: int, n: int) -> int:
        """Exclusive prefix sum of int32 `src` into `out`; returns the grand total."""
        if n <= 0:
            return 0
        tot = self._alloc(4)
        try:
            self._sym("scan_i32_gpu", [_P64, _P64, _P64, _I64])(
                self._ptr(out), self._ptr(src), self._ptr(tot), n)
            return int(self._download(tot, np.int32, 1)[0])
        finally:
            self._cu.free(tot)

    def gather_table(
        self,
        table: "pa.Table",
        idx: np.ndarray,
    ) -> "pa.Table":
        """Gather all columns of *table* by *idx* using AOT GPU kernels.

        float32, float64 (as raw 8-byte words), int32, int64 columns without nulls are gathered
        on GPU with _cached_upload for source data (stable input) and a single index upload
        shared across all columns.  Everything else (strings, nulls, other widths) falls back
        to PyArrow CPU take, which keeps types and nulls.

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

            is_signed_int = pa.types.is_signed_integer(col.type)
            if col.null_count:
                # a validity mask is not gathered by these kernels: Arrow's take keeps the nulls
                arrays.append(take_rows(col, idx32))
            elif pa.types.is_float32(col.type):
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.float32))
                s = cached_source(src_np, np.float32)
                out = self._cu.malloc(n * 4)
                self._gather_f32(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.float32, n)))
                self._cu.free(out)
            elif pa.types.is_float64(col.type):
                # exact: gather the 8-byte words as int64 (float32 would round every value)
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.float64)).view(np.int64)
                s = cached_source(src_np, np.int64)
                out = self._cu.malloc(n * 8)
                self._gather_i64(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.int64, n).view(np.float64)))
                self._cu.free(out)
            elif is_signed_int and col.type.bit_width == 64:
                src_np = np.ascontiguousarray(
                    col.to_numpy(zero_copy_only=False).astype(np.int64))
                s = cached_source(src_np, np.int64)
                out = self._cu.malloc(n * 8)
                self._gather_i64(self._ptr(out), self._ptr(s), self._ptr(idx_dev), n)
                self._cu.sync()
                arrays.append(pa.array(self._download(out, np.int64, n)))
                self._cu.free(out)
            elif is_signed_int and col.type.bit_width == 32:
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
                arrays.append(take_rows(col, idx32))
            names.append(col_name)

        self._cu.free(idx_dev)
        return pa.Table.from_arrays(arrays, names=names)

    def sort_topk(
        self,
        keys,
        k: int,
        descending: bool = False,
        device_out: bool = False,
    ):
        """GPU stable sort of int32 keys; returns the first k sorted row indices.

        Equal keys keep their row order in both directions (the result is exactly
        ``np.argsort(keys, kind="stable")[:k]``, or of ``-keys`` for descending).
        Any n works, not only powers of two.

        Args:
            keys:       int32 sort keys, length N (NumPy or DeviceArray)
            k:          number of indices to return (clamped to N)
            descending: sort the keys largest-first
            device_out: return a DeviceArray instead of NumPy

        Returns:
            int32 row indices, length k.
        """
        keys_dev, n = self._dev_in(keys, np.int32)
        k = max(0, min(int(k), n))
        out_dev = self._alloc(n * 4)
        if n:
            self._sort_topk_gpu(self._ptr(out_dev), self._ptr(keys_dev), n, int(descending))
        self._cu.sync()
        if device_out:
            return DeviceArray(out_dev, np.int32, k, owns=True)
        try:
            return self._download(out_dev, np.int32, k)
        finally:
            self._cu.free(out_dev)

    def _filter_gather(self, fn, src, mask, offsets, n_out: int, dtype, device_out: bool):
        dtype = np.dtype(dtype)
        n_out = int(n_out)
        tmps: list = []
        try:
            s, n = self._dev_in(src, dtype, tmps)
            m, nm = self._dev_in(mask, np.int32, tmps)
            o, no = self._dev_in(offsets, np.int32, tmps)
            if nm != n or no != n:
                raise ValueError(f"src, mask and offsets must have equal lengths ({n}, {nm}, {no})")
            out = self._alloc(n_out * dtype.itemsize)
            if n:
                fn(self._ptr(out), self._ptr(s), self._ptr(m), self._ptr(o), n)
            self._cu.sync()
            return self._result(out, dtype, n_out, device_out)
        finally:
            self._free(*tmps)

    def filter_gather_f32(self, src, mask, offsets, n_out: int, device_out: bool = False):
        return self._filter_gather(self._fg_f32, src, mask, offsets, n_out, np.float32, device_out)

    def filter_gather_i32(self, src, mask, offsets, n_out: int, device_out: bool = False):
        return self._filter_gather(self._fg_i32, src, mask, offsets, n_out, np.int32, device_out)

    def filter_gather_i64(self, src, mask, offsets, n_out: int, device_out: bool = False):
        return self._filter_gather(self._fg_i64, src, mask, offsets, n_out, np.int64, device_out)

    def unique_mask(self, keys, device_out: bool = False):
        """1 where a sorted key differs from the previous one (first of each run), else 0."""
        tmps: list = []
        try:
            k, n = self._dev_in(keys, np.int32, tmps)
            out = self._alloc(n * 4)
            if n:
                self._unique_mask(self._ptr(out), self._ptr(k), n)
            self._cu.sync()
            return self._result(out, np.int32, n, device_out)
        finally:
            self._free(*tmps)


_thread_state = threading.local()


def _bind_thread(cu) -> None:
    """Make the primary CUDA context current on the calling thread (once per thread).

    The context is per thread; the co-processing Router runs its GPU worker on a helper thread
    that never touched CUDA, where cuMemAlloc fails with "invalid context".  Applied to every
    _CUDADriver entry point below (kept out of the class body so it does not collide with
    F-004's own per-thread binding there; binding twice is harmless)."""
    ctx = getattr(cu, "_ctx", None)
    if ctx is None:
        return
    key = int(getattr(ctx, "value", ctx))
    if getattr(_thread_state, "ctx", None) != key:
        cu._lib.cuCtxSetCurrent(ctx)
        _thread_state.ctx = key


def _bound(method):
    def call(self, *args, **kwargs):
        _bind_thread(self)
        return method(self, *args, **kwargs)
    call.__name__ = method.__name__
    call.__doc__ = method.__doc__
    return call


for _name in ("malloc", "free", "h2d", "d2h", "sync"):
    setattr(_CUDADriver, _name, _bound(getattr(_CUDADriver, _name)))


class DeviceArray:
    """A device-resident 1-D array: device pointer, dtype and length.

    The GPU wrappers of :class:`AOTKernelsGPU` accept these wherever they accept
    NumPy arrays (used in place, nothing is copied) and return them when called
    with ``device_out=True``, so operators chain without host round trips.

    *Owning* arrays (what ``device_out=True`` returns) free their buffer on
    :meth:`free`, on leaving a ``with`` block, or when garbage collected.
    *Borrowed* arrays (:meth:`borrow`, for a pointer that belongs to the
    residency tracker or to a DeviceStager slot) never free anything.
    """

    __slots__ = ("ptr", "dtype", "n", "owns")

    def __init__(self, ptr: int, dtype, n: int, owns: bool = False):
        self.ptr = int(ptr)
        self.dtype = np.dtype(dtype)
        self.n = int(n)
        self.owns = bool(owns)

    @classmethod
    def borrow(cls, ptr: int, dtype, n: int) -> "DeviceArray":
        """Wrap a device pointer somebody else owns."""
        return cls(ptr, dtype, n, owns=False)

    def __len__(self) -> int:
        return self.n

    @property
    def nbytes(self) -> int:
        return self.n * self.dtype.itemsize

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self.n, dtype=self.dtype)
        if out.nbytes:
            _get_cuda_driver().d2h(out, self.ptr)
        return out

    def free(self) -> None:
        """Release the buffer if this array owns it (no-op for a borrowed one)."""
        if self.owns:
            if self.ptr:
                _get_cuda_driver().free(self.ptr)
            self.ptr, self.owns = 0, False

    def __enter__(self) -> "DeviceArray":
        return self

    def __exit__(self, *exc) -> None:
        self.free()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"DeviceArray(ptr=0x{self.ptr:x}, dtype={self.dtype}, n={self.n}, owns={self.owns})"
