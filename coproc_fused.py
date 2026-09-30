"""CPU+GPU co-processing for fused aggregates under ``device="auto"``.

Measured on the reference box (RTX 3090 on PCIe 4.0 x8, Ryzen 5900X), TPC-H SF10 Q1
fused stage, 60M rows:

    CPU only (24 threads)                   57 ms
    GPU only, columns in host memory       353 ms   (the link, ~5 GB/s effective)
    GPU only, columns resident on the GPU   39 ms
    CPU 40% + GPU 60%, GPU share resident   25 ms   <- both devices at once

So for a first scan the CPU is the right device, and co-processing pays off once the
columns live on the GPU. This module does both halves:

- host data: return ``None`` (the caller runs the CPU kernel) and start a background
  upload of the query's columns and group ids, using the otherwise idle link, so the
  next query over the same table finds them resident;
- resident data: split the rows statically between one GPU call over its resident
  slice and the multi-core CPU kernel over the rest, at the fraction that balances
  the two (calibration ``pipeline`` fits, refined by every run via ``observe``).

A static split with one call per device beat a dynamic morsel queue here because
each GPU call has a few ms of fixed cost. Residency is tied to the host table's
lifetime (weakref), so freed memory can never alias a cached device copy.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

MIN_UPLOAD_BYTES = 64 << 20      # smaller inputs finish on the CPU before an upload would help
RESIDENT_MEM_FRACTION = 0.5      # share of free GPU memory the resident cache may use
MIN_SHARE = 0.05                 # a split giving a device less than this is not worth it
DEFAULT_GPU_ALPHA_MS = 4.5       # fixed cost of one fused GPU call (measured)


@dataclass
class Policy:
    gpu_fraction: Optional[float] = None
    min_upload_bytes: int = MIN_UPLOAD_BYTES
    background_upload: bool = True


_policy: contextvars.ContextVar = contextvars.ContextVar("mxframe_coproc_policy", default=None)


def current_policy() -> Optional[Policy]:
    return _policy.get()


@contextlib.contextmanager
def policy(**opts):
    """Enable co-processing for fused aggregates inside the block (``device="auto"`` does this)."""
    token = _policy.set(Policy(**opts))
    try:
        yield
    finally:
        _policy.reset(token)


def enabled_by_env() -> bool:
    return os.environ.get("MXFRAME_COPROC", "1") != "0"


def balanced_fraction(cpu_ms: float, gpu_ms: float, gpu_alpha: float = 0.0) -> float:
    """GPU share f with (1-f)*cpu == alpha + f*(gpu-alpha); clamped to 0 / 1 at the edges."""
    var = max(gpu_ms - gpu_alpha, 1e-9)
    f = (cpu_ms - gpu_alpha) / (cpu_ms + var)
    if f < MIN_SHARE:
        return 0.0
    if f > 1 - MIN_SHARE:
        return 1.0
    return float(f)


# ── residency cache ──────────────────────────────────────────────────────────

@dataclass
class _Entry:
    dc: Any            # pipeline_kernels.DeviceColumns
    nbytes: int
    gids: Optional[np.ndarray]   # strong ref: its address is part of the key


_lock = threading.Lock()
_cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
_bytes = 0
_budget: Optional[int] = None
_inflight: Dict[tuple, threading.Thread] = {}


def _key(cc, gids) -> tuple:
    cols = tuple(tuple((int(b.addr), int(b.itemsize)) for b in arrs) for arrs in cc.chunks)
    g = (int(gids.ctypes.data), len(gids)) if gids is not None else None
    return (tuple(tuple(b) for b in cc.bounds), cols, g)


def _input_bytes(cc, gids, n_rows) -> int:
    per_row = sum(b.itemsize for b in cc.chunks[0]) if cc.chunks else 0
    return n_rows * (per_row + (4 if gids is not None else 0))


def _budget_bytes() -> int:
    global _budget
    if _budget is None:
        try:
            from . import gpu_stream
            free, _ = gpu_stream.mem_info()
            _budget = int(free * RESIDENT_MEM_FRACTION)
        except Exception:
            _budget = 0
    return _budget


def resident_count() -> int:
    with _lock:
        return len(_cache)


def _evict(key) -> None:
    global _bytes
    with _lock:
        e = _cache.pop(key, None)
        if e is None:
            return
        _bytes -= e.nbytes
    try:
        e.dc.free()
    except Exception:
        pass


def clear_residency() -> None:
    wait_for_uploads(30)
    with _lock:
        keys = list(_cache)
    for k in keys:
        _evict(k)


def wait_for_uploads(timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        with _lock:
            threads = list(_inflight.values())
        if not threads:
            return True
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            with _lock:
                return not _inflight


def _lookup(key):
    with _lock:
        e = _cache.get(key)
        if e is not None:
            _cache.move_to_end(key)
        return e


def _start_upload(prog, cc, gids, n_rows, owner, nbytes) -> None:
    """Upload the query's columns + group ids in the background (deduplicated per key)."""
    key = _key(cc, gids)
    with _lock:
        if key in _cache or key in _inflight:
            return
        if nbytes > _budget_bytes():
            return

        def run():
            global _bytes
            from . import pipeline_kernels as pk
            try:
                dc = pk.upload_columns(prog, cc, gids, 0, n_rows)
            except Exception:
                with _lock:
                    _inflight.pop(key, None)
                return
            with _lock:
                _inflight.pop(key, None)
                while _cache and _bytes + nbytes > _budget_bytes():
                    old_key, old = _cache.popitem(last=False)
                    _bytes -= old.nbytes
                    old.dc.free()
                _cache[key] = _Entry(dc, nbytes, gids)
                _bytes += nbytes
            try:
                weakref.finalize(owner, _evict, key)
            except TypeError:
                pass

        t = threading.Thread(target=run, name="mxframe-resident-upload", daemon=True)
        _inflight[key] = t
    t.start()


# ── the split ────────────────────────────────────────────────────────────────

def _gpu_usable() -> bool:
    from . import auto_mode, pipeline_kernels as pk
    return enabled_by_env() and pk.gpu_available() and auto_mode._gpu_disabled is None


def try_split(prog, cc, gids, n_groups: int, n_rows: int, owner) -> Optional[Tuple[Any, str, dict]]:
    """Returns (partials, device label, info) when co-processing served the query, else None."""
    pol = current_policy()
    if pol is None or n_rows == 0 or not _gpu_usable():
        return None
    nbytes = _input_bytes(cc, gids, n_rows)
    entry = _lookup(_key(cc, gids))
    if entry is None:
        if pol.background_upload and nbytes >= pol.min_upload_bytes:
            _start_upload(prog, cc, gids, n_rows, owner, nbytes)
        return None

    from . import pipeline_kernels as pk
    from .calibration import get_calibration
    from .coproc import Worker, run_router
    from .pipeline_program import merge_partials

    cal = get_calibration()
    cpu_ms = cal.estimate("pipeline", "cpu", n_rows)
    gpu_ms = cal.estimate("pipeline", "gpu", n_rows)
    f = pol.gpu_fraction if pol.gpu_fraction is not None else balanced_fraction(cpu_ms, gpu_ms, DEFAULT_GPU_ALPHA_MS)
    k = int(round(n_rows * f))
    dc = entry.dc
    workers = []
    if k < n_rows:
        workers.append(Worker("cpu", "cpu", n_rows,
                              compute=lambda r: pk.run_pipeline_cpu(prog, cc, gids, n_groups, r[0], r[1])))
    if k > 0:
        workers.append(Worker("gpu", "gpu", n_rows,
                              compute=lambda r: pk.run_pipeline_gpu(prog, dc, None, n_groups, r[0], r[1])))
    fractions = {"gpu": k / n_rows} if len(workers) == 2 else None
    res = run_router(n_rows, workers, merge=lambda parts: merge_partials(prog, parts), static_fractions=fractions)
    for name, st in res.stats.items():
        if st.rows:
            try:
                cal.observe("pipeline", st.device, st.rows, st.busy_ms)
            except Exception:
                pass
    label = "split" if len(workers) == 2 else ("gpu_resident" if k else "cpu")
    info = {"resident": True, "gpu_fraction": round(k / n_rows, 3),
            "est_cpu_ms": round(cpu_ms, 1), "est_gpu_ms": round(gpu_ms, 1),
            "cpu_ms": round(res.stats["cpu"].busy_ms, 1) if "cpu" in res.stats else 0.0,
            "gpu_ms": round(res.stats["gpu"].busy_ms, 1) if "gpu" in res.stats else 0.0}
    return res.value, label, info
