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

Uploads that do not fit the budget (e.g. while a larger entry is pinned by a
running split) are skipped and retried after FAILED_RETRY_S (120 s).
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

# Queries running right now (device="auto"). Background uploads pause between chunk
# copies while this is non-zero, so work meant for the next query never slows this one.
_active = 0
_active_cv = threading.Condition()
YIELD_MAX_S = 2.0          # an upload never waits longer than this per copy


@contextlib.contextmanager
def query_running():
    global _active
    with _active_cv:
        _active += 1
    try:
        yield
    finally:
        with _active_cv:
            _active -= 1
            _active_cv.notify_all()


def _yield_to_queries() -> None:
    with _active_cv:
        if _active:
            _active_cv.wait_for(lambda: _active == 0, timeout=YIELD_MAX_S)
_deferred: contextvars.ContextVar = contextvars.ContextVar("mxframe_coproc_deferred", default=None)


class _Background:
    """GPU work queued while a query runs; started when the query has returned."""

    def __init__(self):
        self.threads = []
        self.callbacks = []

    def later(self, fn) -> None:
        self.callbacks.append(fn)

    def flush(self) -> None:
        threads, self.threads = self.threads, []
        callbacks, self.callbacks = self.callbacks, []
        for fn in callbacks:
            try:
                fn()
            except Exception:
                pass
        for t in threads:
            t.start()


@contextlib.contextmanager
def deferred_background():
    """Queue background uploads (and ``later`` callbacks) until the block exits.

    A CPU-bound query should not share its cores and memory bandwidth with GPU
    start-up or uploads meant for the *next* query.
    """
    bg = _Background()
    token = _deferred.set(bg)
    try:
        yield bg
    finally:
        _deferred.reset(token)
        bg.flush()


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
#
# Only programs whose every column chunk is a zero-copy Arrow buffer are made
# resident: Arrow buffers are immutable and the entry holds their owners, so a
# cached address can never be reused for other data while the entry exists.
# Converted columns (dictionary codes, bools, narrow ints) are fresh arrays on
# every query and are never uploaded. Entries in use by a split are pinned; an
# evicted entry is freed only when its last user releases it.

@dataclass
class _Entry:
    dc: Any                  # pipeline_kernels.DeviceColumns
    nbytes: int
    owners: list             # Arrow chunk owners + gids: keep every keyed address alive
    users: int = 0
    evicted: bool = False


_lock = threading.Lock()
_cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
_bytes = 0
_inflight_bytes = 0
_inflight: Dict[tuple, threading.Thread] = {}
_failed: Dict[tuple, float] = {}
_finalized: set = set()


def _stable_columns(cc) -> bool:
    import pyarrow as pa
    return all(isinstance(b.owner, pa.Array) for arrs in cc.chunks for b in arrs)


def _key(cc, gids) -> tuple:
    cols = tuple(tuple((int(b.addr), int(b.n), int(b.itemsize), int(b.dt)) for b in arrs) for arrs in cc.chunks)
    g = (int(gids.ctypes.data), len(gids)) if gids is not None else None
    return (tuple(tuple(b) for b in cc.bounds), cols, g)


def _input_bytes(cc, gids, n_rows) -> int:
    per_row = sum(b.itemsize for b in cc.chunks[0]) if cc.chunks else 0
    return n_rows * (per_row + (4 if gids is not None else 0))


def _budget_bytes() -> int:
    """Half of what the GPU could hold for us right now (refreshed on every upload)."""
    try:
        from . import gpu_stream
        free, _ = gpu_stream.mem_info()
    except Exception:
        return 0
    return int((free + _bytes) * RESIDENT_MEM_FRACTION)


def resident_count() -> int:
    with _lock:
        return len(_cache)


def _free_entry(e: _Entry) -> None:
    try:
        e.dc.free()
    except Exception:
        pass


def _evict_locked(key) -> Optional[_Entry]:
    """Remove from the cache; returns the entry if it can be freed now."""
    global _bytes
    e = _cache.pop(key, None)
    if e is None:
        return None
    _bytes -= e.nbytes
    e.evicted = True
    return e if e.users == 0 else None


def _evict(key) -> None:
    with _lock:
        e = _evict_locked(key)
        _finalized.discard(key)
    if e is not None:
        _free_entry(e)


def clear_residency() -> None:
    wait_for_uploads(30)
    with _lock:
        freeable = [e for e in (_evict_locked(k) for k in list(_cache)) if e is not None]
        _failed.clear()
        _finalized.clear()
    for e in freeable:
        _free_entry(e)


def wait_for_uploads(timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        with _lock:
            threads = list(_inflight.values())
        if not threads:
            return True
        for t in threads:
            if t.ident is None:            # queued behind a running query, not started yet
                time.sleep(0.005)
                continue
            t.join(max(0.0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            with _lock:
                return not _inflight


def _acquire(key) -> Optional[_Entry]:
    with _lock:
        e = _cache.get(key)
        if e is not None:
            _cache.move_to_end(key)
            e.users += 1
        return e


def _release(e: _Entry) -> None:
    with _lock:
        e.users -= 1
        free_now = e.evicted and e.users == 0
    if free_now:
        _free_entry(e)


def _start_upload(prog, cc, gids, n_rows, owner, nbytes) -> None:
    """Upload the query's columns + group ids in the background (deduplicated per key).

    The query thread only registers the upload: every CUDA call (memory info,
    eviction frees, the copy itself) happens on the upload thread, so a first
    query never waits for GPU initialisation.
    """
    key = _key(cc, gids)
    with _lock:
        if key in _cache or key in _inflight or _recently_failed(key):
            return
        owners = [b.owner for arrs in cc.chunks for b in arrs] + ([gids] if gids is not None else [])

        def run():
            global _bytes, _inflight_bytes
            from . import pipeline_kernels as pk

            def give_up(reserved: bool) -> None:
                global _inflight_bytes
                with _lock:
                    _inflight.pop(key, None)
                    if reserved:
                        _inflight_bytes -= nbytes
                    _failed[key] = time.monotonic()

            try:
                budget = _budget_bytes()
            except Exception:
                give_up(False)
                return
            with _lock:
                unpinned = sum(e.nbytes for e in _cache.values() if e.users == 0)
                if _bytes - unpinned + _inflight_bytes + nbytes > budget:
                    # A pinned (in-use) or recent entry holds the budget: skip; retried after FAILED_RETRY_S.
                    _inflight.pop(key, None)
                    _failed[key] = time.monotonic()
                    return
                freeable = []
                for old in list(_cache):
                    if _bytes + _inflight_bytes + nbytes <= budget:
                        break
                    e = _evict_locked(old)
                    if e is not None:
                        freeable.append(e)
                _inflight_bytes += nbytes
            for e in freeable:
                _free_entry(e)
            try:
                dc = pk.upload_columns(prog, cc, gids, 0, n_rows, before_copy=_yield_to_queries)
            except Exception:
                give_up(True)
                return
            with _lock:   # leave in-flight and enter the cache atomically: no duplicate upload can slip in
                _inflight.pop(key, None)
                _inflight_bytes -= nbytes
                _cache[key] = _Entry(dc, nbytes, owners)
                _bytes += nbytes
                _failed.pop(key, None)
                register = key not in _finalized
                _finalized.add(key)
            if register:
                try:
                    weakref.finalize(owner, _evict, key)
                except TypeError:
                    pass

        t = threading.Thread(target=run, name="mxframe-resident-upload", daemon=True)
        _inflight[key] = t
    bg = _deferred.get()
    if bg is not None:
        bg.threads.append(t)
    else:
        t.start()


FAILED_RETRY_S = 120.0   # a failed or rejected upload (e.g. shared-GPU memory pressure) is retried after this


def _recently_failed(key) -> bool:
    t = _failed.get(key)
    return t is not None and time.monotonic() - t < FAILED_RETRY_S


# ── the split ────────────────────────────────────────────────────────────────

def _gpu_usable() -> bool:
    from . import auto_mode, pipeline_kernels as pk
    return enabled_by_env() and pk.gpu_available() and auto_mode._gpu_disabled is None


def try_split(prog, cc, gids, n_groups: int, n_rows: int, owner,
              stable_gids: bool = True) -> Optional[Tuple[Any, str, dict]]:
    """Returns (partials, device label, info) when co-processing served the query, else None."""
    pol = current_policy()
    if pol is None or n_rows == 0 or not _gpu_usable() or not _stable_columns(cc):
        return None
    key = _key(cc, gids)
    entry = _acquire(key)
    if entry is None:
        nbytes = _input_bytes(cc, gids, n_rows)
        if pol.background_upload and stable_gids and nbytes >= pol.min_upload_bytes:
            _start_upload(prog, cc, gids, n_rows, owner, nbytes)
        return None
    try:
        return _split(prog, cc, gids, n_groups, n_rows, entry, key, pol)
    finally:
        _release(entry)


def _split(prog, cc, gids, n_groups, n_rows, entry, key, pol):
    from . import auto_mode
    from . import pipeline_kernels as pk
    from .calibration import get_calibration
    from .coproc import Worker, run_router
    from .pipeline_program import merge_partials

    cal = get_calibration()
    card = max(n_groups, 1)
    cpu_ms = cal.estimate("pipeline", "cpu", n_rows, card=card)
    gpu_ms = cal.estimate("pipeline", "gpu", n_rows, card=card)
    f = pol.gpu_fraction if pol.gpu_fraction is not None else balanced_fraction(cpu_ms, gpu_ms, DEFAULT_GPU_ALPHA_MS)
    k = int(round(n_rows * f))
    dc = entry.dc
    cpu = Worker("cpu", "cpu", n_rows,
                 compute=lambda r: pk.run_pipeline_cpu(prog, cc, gids, n_groups, r[0], r[1]))
    # Fallible: if the GPU share fails, the Router redoes its rows on the CPU worker.
    gpu = Worker("gpu", "gpu", n_rows, fallible=True,
                 compute=lambda r: pk.run_pipeline_gpu(prog, dc, None, n_groups, r[0], r[1]))
    if k <= 0:
        workers, fractions = [cpu], None
    elif k >= n_rows:
        workers, fractions = [gpu, cpu], {"gpu": 1.0}
    else:
        workers, fractions = [gpu, cpu], {"gpu": k / n_rows}
    res = run_router(n_rows, workers, merge=lambda parts: merge_partials(prog, parts), static_fractions=fractions)
    gstat = res.stats.get("gpu")
    if gstat is not None and gstat.failed:
        err = gstat.error
        if err is not None and auto_mode.is_sticky_gpu_error(err):
            auto_mode.disable_gpu(f"{type(err).__name__}: {err}")
        _evict(key)
        with _lock:
            _failed[key] = time.monotonic()      # e.g. GpuUnsupported: don't re-upload on every query
    else:
        for st in res.stats.values():
            if st.rows:
                try:
                    cal.observe("pipeline", st.device, st.rows, st.busy_ms, card=card)
                except Exception:
                    pass
    used = {n for n, st in res.stats.items() if st.rows}
    label = "split" if used == {"cpu", "gpu"} else ("gpu_resident" if used == {"gpu"} else "cpu")
    info = {"resident": True, "gpu_fraction": round(k / n_rows, 3),
            "est_cpu_ms": round(cpu_ms, 1), "est_gpu_ms": round(gpu_ms, 1),
            "cpu_ms": round(res.stats["cpu"].busy_ms, 1),
            "gpu_ms": round(gstat.busy_ms, 1) if gstat is not None else 0.0,
            "gpu_failed": bool(gstat is not None and gstat.failed)}
    return res.value, label, info
