"""ctypes bindings for the fused pipeline kernels (F-008).

``libmxpipeline_cpu.so`` and ``libmxpipeline_gpu.so`` are built ahead of time
from ``kernels_aot/pipeline_{cpu,gpu}.mojo`` (see ``scripts/build_pipeline.sh``);
nothing is compiled at run time. Both libraries are loaded lazily, and the GPU
library and the CUDA driver are touched only when a GPU function is called.

Entry points return :class:`~mxframe.pipeline_program.Partials` for a row range
``[start, end)`` so a Router can hand morsels to either device and merge the
results with :func:`merge_partials`.
"""

from __future__ import annotations

import ctypes
import os
import threading
import weakref
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .pipeline_program import (
    ChunkedColumns,
    Partials,
    PipelineProgram,
    dtype_code,
    empty_partials,
    finalize,
    merge_partials,
)

__all__ = [
    "cpu_available", "gpu_available", "default_threads", "run_pipeline_cpu",
    "run_pipeline_gpu", "DeviceColumns", "upload_columns", "merge_partials", "finalize",
]

_HERE = Path(__file__).parent
_lock = threading.Lock()
_cpu_lib = None
_gpu_lib = None

_I64 = ctypes.c_int64


def _find(name: str, env: str) -> Optional[Path]:
    override = os.environ.get(env)
    if override and Path(override).exists():
        return Path(override)
    for cand in (_HERE / "kernels_aot" / name, _HERE / name):
        if cand.exists():
            return cand
    return None


def cpu_available() -> bool:
    return _find("libmxpipeline_cpu.so", "MXFRAME_PIPELINE_CPU_LIB") is not None


def gpu_available() -> bool:
    return _find("libmxpipeline_gpu.so", "MXFRAME_PIPELINE_GPU_LIB") is not None


def default_threads() -> int:
    env = os.environ.get("MXFRAME_CPU_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return os.cpu_count() or 1


def _load_cpu():
    global _cpu_lib
    with _lock:
        if _cpu_lib is None:
            path = _find("libmxpipeline_cpu.so", "MXFRAME_PIPELINE_CPU_LIB")
            if path is None:
                raise FileNotFoundError("libmxpipeline_cpu.so not found")
            lib = ctypes.CDLL(str(path))
            fn = lib.mx_pipeline_cpu
            fn.restype = _I64
            fn.argtypes = [_I64] * 14
            _cpu_lib = lib
    return _cpu_lib


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _as_chunked(program: PipelineProgram, columns) -> ChunkedColumns:
    """Normalise ``columns`` (numpy arrays, or already chunked) to :class:`ChunkedColumns`."""
    if isinstance(columns, ChunkedColumns):
        cc = columns
    else:
        arrays = []
        for a in columns:
            if not isinstance(a, np.ndarray) or not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
            arrays.append(a)
        cc = ChunkedColumns.from_arrays(arrays)
        if not arrays:
            cc = ChunkedColumns([], [], 0)
    if cc.n_cols != len(program.columns):
        raise ValueError(f"program reads {len(program.columns)} columns, got {cc.n_cols}")
    for ch in cc.chunks[:1]:
        for a in ch:
            dtype_code(a)  # raises KeyError on an unsupported dtype
    return cc


def _check_gids(gids: Optional[np.ndarray], n_groups: int) -> Optional[np.ndarray]:
    if gids is not None:
        if gids.dtype != np.int32 or not gids.flags["C_CONTIGUOUS"]:
            gids = np.ascontiguousarray(gids, dtype=np.int32)
    elif n_groups != 1:
        raise ValueError("gids may only be omitted for a global aggregate (n_groups == 1)")
    return gids


def _column_dtypes(cc: ChunkedColumns) -> List[int]:
    if not cc.chunks:
        return [0] * cc.n_cols
    return [dtype_code(a) for a in cc.chunks[0]]


def run_pipeline_cpu(program: PipelineProgram, columns, gids: Optional[np.ndarray], n_groups: int,
                     start: int, end: int, threads: Optional[int] = None) -> Partials:
    """Run the program over rows ``[start, end)`` on the CPU (all cores by default).

    ``columns`` is a sequence of numpy arrays or a :class:`ChunkedColumns`
    (from ``program.bind_chunks(table)``, which avoids concatenating Arrow chunks).
    """
    lib = _load_cpu()
    cc = _as_chunked(program, columns)
    gids = _check_gids(gids, n_groups)
    n_rows = cc.n_rows if cc.n_cols else (len(gids) if gids is not None else end)
    if gids is not None:
        n_rows = min(n_rows, len(gids))
    if start < 0 or end > n_rows or start > end:
        raise ValueError(f"row range [{start}, {end}) outside 0..{n_rows}")
    n_slots = program.n_slots
    vals = np.empty((n_slots, n_groups), dtype=np.float64)
    cnt = np.empty(n_groups, dtype=np.int64)
    n_cols = cc.n_cols
    if cc.chunks:
        bounds = cc.bounds
        tbl = np.empty((len(bounds), 2 + n_cols), dtype=np.int64)
        for k, ((s, e), arrs) in enumerate(zip(bounds, cc.chunks)):
            tbl[k, 0], tbl[k, 1] = s, e
            for c, a in enumerate(arrs):
                tbl[k, 2 + c] = a.ctypes.data - s * a.itemsize
    else:  # no column data (e.g. count-only over gids): one dummy chunk spanning the rows
        tbl = np.array([[0, n_rows]], dtype=np.int64)
    col_dt = np.array(_column_dtypes(cc) or [0], dtype=np.int32)
    rc = lib.mx_pipeline_cpu(
        _addr(program.code), _addr(program.consts), _addr(program.set_bytes),
        _addr(program.set_meta), _addr(tbl), int(tbl.shape[0]), _addr(col_dt),
        _addr(gids) if gids is not None else 0, n_groups, start, end,
        _addr(vals), _addr(cnt), int(threads) if threads else default_threads(),
    )
    if rc != 0:
        raise RuntimeError(f"mx_pipeline_cpu failed with code {rc}")
    return Partials(vals=vals, cnt=cnt)


# ── GPU ─────────────────────────────────────────────────────────────────────
# The CUDA driver API on the device's primary context, the same context Mojo's
# DeviceContext() uses, so device pointers allocated here are visible in the
# kernels. Nothing below runs (and libcuda is not loaded) until a GPU function
# is called.

class _Cuda:
    """Minimal CUDA driver wrapper (memory + copies + SM count)."""

    def __init__(self):
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if lib is None:
            raise RuntimeError("libcuda.so not found")
        self.lib = lib
        u64, i32, sz, vp = ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
        lib.cuInit.argtypes = [ctypes.c_uint]
        err = lib.cuInit(0)
        if err != 0:
            raise RuntimeError(f"cuInit failed: {err}")
        dev = ctypes.c_int(0)
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(i32), i32]
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(u64), i32]
        lib.cuCtxSetCurrent.argtypes = [u64]
        if lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            raise RuntimeError("no CUDA device")
        ctx = u64(0)
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        self.ctx = ctx
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), sz]
        lib.cuMemFree_v2.argtypes = [u64]
        lib.cuMemcpyHtoD_v2.argtypes = [u64, vp, sz]
        lib.cuMemcpyDtoH_v2.argtypes = [vp, u64, sz]
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(i32), i32, i32]
        n = ctypes.c_int(0)
        lib.cuDeviceGetAttribute(ctypes.byref(n), 16, dev)  # MULTIPROCESSOR_COUNT
        self.sm_count = max(int(n.value), 1)
        self._tls = threading.local()

    def bind_thread(self) -> None:
        """Make the primary context current on the calling thread (Router workers)."""
        if not getattr(self._tls, "bound", False):
            self.lib.cuCtxSetCurrent(self.ctx)
            self._tls.bound = True

    def malloc(self, nbytes: int) -> int:
        self.bind_thread()
        ptr = ctypes.c_uint64(0)
        err = self.lib.cuMemAlloc_v2(ctypes.byref(ptr), max(int(nbytes), 8))
        if err != 0:
            raise MemoryError(f"cuMemAlloc({nbytes}) failed: {err}")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr:
            self.bind_thread()
            self.lib.cuMemFree_v2(ctypes.c_uint64(ptr))

    def h2d(self, dev_ptr: int, arr: np.ndarray) -> None:
        self.bind_thread()
        if arr.nbytes:
            err = self.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), ctypes.c_void_p(arr.ctypes.data),
                                           ctypes.c_size_t(arr.nbytes))
            if err != 0:
                raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

    def h2d_at(self, dev_ptr: int, arr: np.ndarray) -> None:
        self.h2d(dev_ptr, arr)

    def d2h(self, arr: np.ndarray, dev_ptr: int) -> None:
        self.bind_thread()
        if arr.nbytes:
            err = self.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(arr.ctypes.data), ctypes.c_uint64(dev_ptr),
                                           ctypes.c_size_t(arr.nbytes))
            if err != 0:
                raise RuntimeError(f"cuMemcpyDtoH failed: {err}")

    def sync(self) -> None:
        self.bind_thread()
        self.lib.cuCtxSynchronize()

    def upload(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        self.h2d(ptr, arr)
        return ptr


_cuda: Optional[_Cuda] = None


def _get_cuda() -> _Cuda:
    global _cuda
    with _lock:
        if _cuda is None:
            _cuda = _Cuda()
    return _cuda


def _load_gpu():
    global _gpu_lib
    with _lock:
        if _gpu_lib is None:
            path = _find("libmxpipeline_gpu.so", "MXFRAME_PIPELINE_GPU_LIB")
            if path is None:
                raise FileNotFoundError("libmxpipeline_gpu.so not found")
            lib = ctypes.CDLL(str(path))
            fn = lib.mx_pipeline_gpu
            fn.restype = _I64
            fn.argtypes = [_I64] * 15
            _gpu_lib = lib
    return _gpu_lib


class _DeviceProgram:
    """Bytecode, constants, sets and identities of one program, resident on the GPU."""

    def __init__(self, program: PipelineProgram, cu: _Cuda):
        self.cu = cu
        self.code = cu.upload(program.code)
        self.consts = cu.upload(program.consts)
        self.set_bytes = cu.upload(program.set_bytes)
        self.set_meta = cu.upload(program.set_meta)
        self.ident = cu.upload(program.identity if program.n_slots else np.zeros(1))
        self._ptrs = [self.code, self.consts, self.set_bytes, self.set_meta, self.ident]

    def free(self):
        for p in self._ptrs:
            self.cu.free(p)
        self._ptrs = []


def _device_program(program: PipelineProgram, cu: _Cuda) -> _DeviceProgram:
    dp = getattr(program, "_device_program", None)
    if dp is None:
        dp = _DeviceProgram(program, cu)
        program._device_program = dp
        weakref.finalize(program, dp.free)
    return dp


class DeviceColumns:
    """Columns (and group ids) resident on the GPU for rows ``[base_row, base_row + n_rows)``.

    Lets a Router upload a morsel itself (on its own copy stream) and hand the
    kernel already-resident pointers.
    """

    def __init__(self, ptrs: List[int], dtypes: List[int], gid_ptr: int, base_row: int, n_rows: int,
                 owned: Optional[List[int]] = None, cu: Optional[_Cuda] = None):
        self.ptrs, self.dtypes, self.gid_ptr = ptrs, dtypes, gid_ptr
        self.base_row, self.n_rows = base_row, n_rows
        self._owned = owned or []
        self._cu = cu

    def free(self) -> None:
        if self._cu is not None:
            for p in self._owned:
                self._cu.free(p)
        self._owned = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()


def upload_columns(program: PipelineProgram, columns, gids: Optional[np.ndarray],
                   start: int, end: int) -> DeviceColumns:
    """Copy rows ``[start, end)`` of each column (and ``gids``) to the GPU.

    Chunked columns are copied chunk by chunk straight into one device buffer
    per column: no host-side concatenation.
    """
    cu = _get_cuda()
    cc = _as_chunked(program, columns)
    if gids is not None:
        gids = _check_gids(gids, 2)
    n = end - start
    itemsizes = [a.itemsize for a in cc.chunks[0]] if cc.chunks else [8] * cc.n_cols
    owned: List[int] = []
    try:
        ptrs = [cu.malloc(n * isz) for isz in itemsizes]
        owned += ptrs
        for (s, e), arrs in zip(cc.bounds, cc.chunks):
            lo, hi = max(s, start), min(e, end)
            if hi <= lo:
                continue
            for c, a in enumerate(arrs):
                piece = np.ascontiguousarray(a[lo - s:hi - s])
                cu.h2d_at(ptrs[c] + (lo - start) * a.itemsize, piece)
        gp = 0
        if gids is not None:
            gp = cu.upload(gids[start:end])
            owned.append(gp)
        cu.sync()
    except BaseException:
        for p in owned:
            cu.free(p)
        raise
    return DeviceColumns(ptrs, _column_dtypes(cc), gp, start, n, owned, cu)


def run_pipeline_gpu(program: PipelineProgram, columns: Union[Sequence[np.ndarray], ChunkedColumns, DeviceColumns],
                     gids: Optional[np.ndarray], n_groups: int, start: int, end: int) -> Partials:
    """Run the program over rows ``[start, end)`` on the GPU.

    ``columns`` are host arrays / chunked columns (only the row range is copied
    over) or a :class:`DeviceColumns` already on the device.
    """
    lib = _load_gpu()
    cu = _get_cuda()
    dp = _device_program(program, cu)
    if start < 0 or start > end:
        raise ValueError(f"bad row range [{start}, {end})")
    n_slots = program.n_slots
    temps: List[int] = []
    dev_cols: Optional[DeviceColumns] = None
    try:
        if isinstance(columns, DeviceColumns):
            dc = columns
            if start < dc.base_row or end > dc.base_row + dc.n_rows:
                raise ValueError("row range outside the resident columns")
        else:
            cc = _as_chunked(program, columns)
            gids = _check_gids(gids, n_groups)
            n_rows = cc.n_rows if cc.n_cols else (len(gids) if gids is not None else end)
            if gids is not None:
                n_rows = min(n_rows, len(gids))
            if end > n_rows:
                raise ValueError(f"row range [{start}, {end}) outside 0..{n_rows}")
            dc = dev_cols = upload_columns(program, cc, gids, start, end)
        k_start, k_end = start - dc.base_row, end - dc.base_row
        col_ptrs = cu.upload(np.array(dc.ptrs or [0], dtype=np.int64))
        col_dt = cu.upload(np.array(dc.dtypes or [0], dtype=np.int32))
        out_vals = cu.malloc(max(n_slots * n_groups, 1) * 8)
        out_cnt = cu.malloc(n_groups * 8)
        temps += [col_ptrs, col_dt, out_vals, out_cnt]
        cu.sync()
        if n_groups != 1 and not dc.gid_ptr:
            raise ValueError("gids may only be omitted for a global aggregate (n_groups == 1)")
        rc = lib.mx_pipeline_gpu(
            dp.code, dp.consts, dp.set_bytes, dp.set_meta, col_ptrs, col_dt,
            dc.gid_ptr if n_groups > 1 else 0, n_groups, n_slots, k_start, k_end,
            out_vals, out_cnt, dp.ident, cu.sm_count * 6,
        )
        if rc != 0:
            raise RuntimeError(f"mx_pipeline_gpu failed with code {rc}")
        vals = np.empty((n_slots, n_groups), dtype=np.float64)
        cnt = np.empty(n_groups, dtype=np.int64)
        if n_slots:
            cu.d2h(vals, out_vals)
        cu.d2h(cnt, out_cnt)
        return Partials(vals=vals, cnt=cnt)
    finally:
        for p in temps:
            cu.free(p)
        if dev_cols is not None:
            dev_cols.free()
