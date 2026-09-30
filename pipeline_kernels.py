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
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .pipeline_program import (
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
            fn.argtypes = [_I64] * 13
            _cpu_lib = lib
    return _cpu_lib


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _columns_and_gids(program: PipelineProgram, columns: Sequence[np.ndarray],
                      gids: Optional[np.ndarray], n_groups: int):
    cols = []
    for a in columns:
        if not isinstance(a, np.ndarray) or not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        dtype_code(a)  # raises KeyError on an unsupported dtype
        cols.append(a)
    if len(cols) != len(program.columns):
        raise ValueError(f"program reads {len(program.columns)} columns, got {len(cols)}")
    if gids is not None:
        if gids.dtype != np.int32 or not gids.flags["C_CONTIGUOUS"]:
            gids = np.ascontiguousarray(gids, dtype=np.int32)
    elif n_groups != 1:
        raise ValueError("gids may only be omitted for a global aggregate (n_groups == 1)")
    return cols, gids


def run_pipeline_cpu(program: PipelineProgram, columns: Sequence[np.ndarray],
                     gids: Optional[np.ndarray], n_groups: int, start: int, end: int,
                     threads: Optional[int] = None) -> Partials:
    """Run the program over rows ``[start, end)`` on the CPU (all cores by default)."""
    lib = _load_cpu()
    cols, gids = _columns_and_gids(program, columns, gids, n_groups)
    n_rows = min((len(c) for c in cols), default=(len(gids) if gids is not None else end))
    if gids is not None:
        n_rows = min(n_rows, len(gids))
    if start < 0 or end > n_rows or start > end:
        raise ValueError(f"row range [{start}, {end}) outside 0..{n_rows}")
    n_slots = program.n_slots
    vals = np.empty((n_slots, n_groups), dtype=np.float64)
    cnt = np.empty(n_groups, dtype=np.int64)
    col_ptrs = np.array([_addr(c) for c in cols] or [0], dtype=np.int64)
    col_dt = np.array([dtype_code(c) for c in cols] or [0], dtype=np.int32)
    rc = lib.mx_pipeline_cpu(
        _addr(program.code), _addr(program.consts), _addr(program.set_bytes),
        _addr(program.set_meta), _addr(col_ptrs), _addr(col_dt),
        _addr(gids) if gids is not None else 0, n_groups, start, end,
        _addr(vals), _addr(cnt), int(threads) if threads else default_threads(),
    )
    if rc != 0:
        raise RuntimeError(f"mx_pipeline_cpu failed with code {rc}")
    return Partials(vals=vals, cnt=cnt)
