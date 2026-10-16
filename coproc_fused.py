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
SPLIT_MARGIN = 0.10              # a split must beat the faster device alone by this much to be chosen
# CPU share slowdown while the GPU runs beside it (launch thread, result copies; a resident GPU share reads no
# host memory, so this is a small allowance, not measured: F-006's 25 ms split matches CPU 40% / GPU 60% unslowed)
CPU_CONTENTION = 1.05
SPLIT_MERGE_MS = 0.5             # thread hand-off and the merge of the partials, plus per group (below)
MERGE_BYTES_PER_MS = 4_000_000   # partials copied back and merged: ~4 GB/s


@dataclass
class Policy:
    gpu_fraction: Optional[float] = None
    min_upload_bytes: int = MIN_UPLOAD_BYTES
    background_upload: bool = True


_policy: contextvars.ContextVar = contextvars.ContextVar("mxframe_coproc_policy", default=None)

# Queries running right now (device="auto"). Background uploads pause between chunk
# copies while this is non-zero, so work meant for the next query never slows this one.
_active = 0
_last_end = 0.0            # monotonic time the last query finished
_active_cv = threading.Condition()
YIELD_MAX_S = 2.0          # an upload never waits longer than this per copy


@contextlib.contextmanager
def query_running():
    global _active, _last_end
    with _active_cv:
        _active += 1
        _active_cv.notify_all()
    try:
        yield
    finally:
        with _active_cv:
            _active -= 1
            _last_end = time.monotonic()
            _active_cv.notify_all()


def _yield_to_queries() -> None:
    """Called before every chunk copy of a background upload: wait for running queries, and
    abandon the upload once the process is exiting (F-063: exit waits at most one chunk)."""
    with _active_cv:
        if _active and not _closing:
            _active_cv.wait_for(lambda: _active == 0 or _closing, timeout=YIELD_MAX_S)
    if _closing:
        raise RuntimeError("process is exiting")
_deferred: contextvars.ContextVar = contextvars.ContextVar("mxframe_coproc_deferred", default=None)


_closing = False


def _start_thread(t: threading.Thread) -> None:
    """Start an upload thread unless the process is exiting (see ``close_background``)."""
    with _lock:
        try:
            if _closing:
                raise RuntimeError("process is exiting")
            t.start()
        except Exception:      # exiting: free the key so a later query retries
            for k in [k for k, v in _inflight.items() if v is t]:
                _inflight.pop(k, None)


def close_background(timeout: float) -> None:
    """At exit: start no more uploads and wait for the running ones (F-063: a daemon thread
    inside libcuda while `exit()` unloads the driver aborts or segfaults the process)."""
    global _closing
    with _lock:
        _closing = True
        running = [t for t in _inflight.values() if t.is_alive()]
    with _active_cv:
        _active_cv.notify_all()
    deadline = time.monotonic() + timeout
    for t in running:
        t.join(max(0.0, deadline - time.monotonic()))


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
        if not threads and not callbacks:
            return

        def launch():
            for fn in callbacks:
                try:
                    fn()
                except Exception:
                    pass
            for t in threads:
                _start_thread(t)

        if IDLE_BEFORE_BACKGROUND_S <= 0:
            launch()
            return
        threading.Thread(target=lambda: (_wait_until_idle(IDLE_BEFORE_BACKGROUND_S), launch()),
                         name="mxframe-background-launch", daemon=True).start()


# GPU start-up and uploads meant for later queries begin only after no auto query has run for
# this long: a script that chains computes (TPC-H Q17, Q11: two or three in a row) would
# otherwise run its next query next to CUDA initialisation (+10-25 ms per SF1 query, F-009).
IDLE_BEFORE_BACKGROUND_S = 0.1


def _wait_until_idle(quiet_s: float) -> None:
    """Return once no query is running and none has finished for ``quiet_s``."""
    with _active_cv:
        while True:
            _active_cv.wait_for(lambda: _active == 0)
            left = quiet_s - (time.monotonic() - _last_end)
            if left <= 0:
                return
            _active_cv.wait(timeout=left)


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


def stable_group_keys(group_by) -> bool:
    """Bare, unrenamed key columns give the same group ids on every query (so their columns can be
    resident); derived or renamed keys get fresh ids each time and are never uploaded."""
    return all(e.op == "col" and e._alias in (None, e.args[0]) for e in group_by)


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
    owners: list             # Arrow chunk owners + gids: keep every keyed address alive (none for a file's read)
    users: int = 0
    evicted: bool = False
    sig: tuple = ()          # (source, columns, group key names): what ``resident_for`` is asked


_lock = threading.Lock()
_cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
_bytes = 0
_inflight_bytes = 0
_inflight: Dict[tuple, threading.Thread] = {}
_failed: Dict[tuple, float] = {}
_finalized: set = set()


def stable_columns(cc) -> bool:
    import pyarrow as pa
    return all(isinstance(b.owner, pa.Array) for arrs in cc.chunks for b in arrs)


def gid_fingerprint(unique_key_cols, n_groups: int) -> bytes:
    """What group id ``i`` means: a hash of the group keys in id order. Two encodings of the same rows with the
    same fingerprint give every row the same id, so a device copy of one's ids is valid for the other's keys."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(int(n_groups).to_bytes(8, "little"))
    for name in sorted(unique_key_cols):
        h.update(name.encode() + b"\0")
        v = unique_key_cols[name]
        for ch in (v.chunks if hasattr(v, "chunks") else [v]):
            h.update(str(ch.type).encode() + len(ch).to_bytes(8, "little"))
            for b in ch.buffers():
                if b is not None:
                    h.update(memoryview(b))
            if hasattr(ch, "dictionary"):
                for b in ch.dictionary.buffers():
                    if b is not None:
                        h.update(memoryview(b))
    return h.digest()


def _key(cc, gids, source=None, columns=(), gid_fp=None) -> tuple:
    """What a device copy holds. A table read from a file is keyed by the file's identity, the columns and the
    group-id fingerprint, so the next compute, which reads the file into new buffers, finds it; any other
    table by the addresses of its (immutable, kept alive) Arrow buffers."""
    bounds = tuple(tuple(b) for b in cc.bounds)
    if source is not None:
        layout = tuple(tuple((int(b.n), int(b.itemsize), int(b.dt)) for b in arrs) for arrs in cc.chunks)
        return ("source", source, tuple(columns), bounds, layout, gid_fp, len(gids) if gids is not None else None)
    cols = tuple(tuple((int(b.addr), int(b.n), int(b.itemsize), int(b.dt)) for b in arrs) for arrs in cc.chunks)
    g = (int(gids.ctypes.data), len(gids)) if gids is not None else None
    return (bounds, cols, g)


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


def _start_upload(prog, cc, gids, n_rows, owner, nbytes, key, sig, by_source=False) -> None:
    """Upload the query's columns + group ids in the background (deduplicated per key).

    The query thread only registers the upload: every CUDA call (memory info,
    eviction frees, the copy itself) happens on the upload thread, so a first
    query never waits for GPU initialisation. An entry keyed by a file's identity holds no host
    buffers (a later read of the file brings its own) and lives until evicted or the file changes.
    """
    with _lock:
        if key in _cache or key in _inflight or _recently_failed(key):
            return
        owners = [] if by_source else [b.owner for arrs in cc.chunks for b in arrs] + ([gids] if gids is not None else [])

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
                _cache[key] = _Entry(dc, nbytes, owners, sig=sig)
                _bytes += nbytes
                _failed.pop(key, None)
                register = key not in _finalized
                _finalized.add(key)
            if register and not by_source:
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
        _start_thread(t)


FAILED_RETRY_S = 120.0   # a failed or rejected upload (e.g. shared-GPU memory pressure) is retried after this


def _recently_failed(key) -> bool:
    t = _failed.get(key)
    return t is not None and time.monotonic() - t < FAILED_RETRY_S


# ── the split ────────────────────────────────────────────────────────────────

def _gpu_usable() -> bool:
    from . import auto_mode, pipeline_kernels as pk
    return enabled_by_env() and pk.gpu_available() and auto_mode._gpu_disabled is None


def _source_of(owner):
    """The token of the file read behind ``owner``, or None (an in-memory table)."""
    from .io_parquet import table_source
    return table_source(owner)


def _evict_stale(source) -> None:
    """A file that changed (new inode, mtime or size) leaves its old device copies unreachable: free them now."""
    if source is None or source[0] != "parquet":
        return
    with _lock:
        stale = [k for k, e in _cache.items() if e.sig and e.sig[0][:2] == source[:2] and e.sig[0] != source]
        freeable = [e for e in (_evict_locked(k) for k in stale) if e is not None]
    for e in freeable:
        _free_entry(e)


def resident_for(source, columns, gkeys) -> bool:
    """Is there a device copy of (at least) the columns a query reads from this source, with its group ids?
    ``source`` is ``SplitInputs.source``; the answer decides whether auto may split the query (the exact
    key is checked again when it runs)."""
    columns, gkeys = frozenset(columns), tuple(gkeys)
    with _lock:
        return any(e.sig and e.sig[0] == source and e.sig[1] <= columns and e.sig[2] == gkeys
                   for e in _cache.values())


_prepared: Dict[tuple, float] = {}       # (source, columns, group keys) -> when a background prepare began


def begin_prepare(source, columns, gkeys) -> bool:
    """True once per ``FAILED_RETRY_S`` for a source: the caller builds its group ids and starts the upload.
    (Building them costs a pass over the rows; a query repeated before the upload lands must not redo it.)"""
    sig = (source, frozenset(columns), tuple(gkeys))
    with _lock:
        t = _prepared.get(sig)
        if t is not None and time.monotonic() - t < FAILED_RETRY_S:
            return False
        if any(e.sig == sig for e in _cache.values()):
            return False
        _prepared[sig] = time.monotonic()
    return True


def request_upload(prog, cc, gids, n_groups: int, n_rows: int, owner, *, gid_fp=None, gkeys=(),
                   min_upload_bytes: Optional[int] = None) -> None:
    """Start the background upload of a query's inputs (no-op for inputs not worth or not able to be kept)."""
    source = _source_of(owner)
    if source is not None and gids is not None and gid_fp is None:
        source = None                                   # ids with no fingerprint cannot be told apart
    if source is None and not stable_columns(cc):
        return
    nbytes = _input_bytes(cc, gids, n_rows)
    if nbytes < (MIN_UPLOAD_BYTES if min_upload_bytes is None else min_upload_bytes):
        return
    _evict_stale(source)
    key = _key(cc, gids, source, prog.columns, gid_fp)
    sig = ((source if source is not None else ("memory", id(owner))), frozenset(prog.columns), tuple(gkeys))
    _start_upload(prog, cc, gids, n_rows, owner, nbytes, key, sig, by_source=source is not None)


def try_split(prog, cc, gids, n_groups: int, n_rows: int, owner,
              stable_gids: bool = True, *, cpu_rows=None, gid_fp=None, gkeys=()
              ) -> Optional[Tuple[Any, str, dict]]:
    """Returns (partials, device label, info) when co-processing served the query, else None.

    ``cpu_rows(start, end)`` computes the CPU share's partials (the native executor, see
    ``pipeline_exec._cpu_rows``); the pipeline kernel when omitted."""
    pol = current_policy()
    if pol is None or n_rows == 0 or not _gpu_usable():
        return None
    source = _source_of(owner)
    if source is not None and gids is not None and gid_fp is None:
        source = None
    if source is None and not stable_columns(cc):
        return None
    key = _key(cc, gids, source, prog.columns, gid_fp)
    entry = _acquire(key)
    if entry is None:
        if pol.background_upload and stable_gids:
            request_upload(prog, cc, gids, n_groups, n_rows, owner, gid_fp=gid_fp, gkeys=gkeys,
                           min_upload_bytes=pol.min_upload_bytes)
        return None
    try:
        return _split(prog, cc, gids, n_groups, n_rows, entry, key, pol, cpu_rows)
    finally:
        _release(entry)


@dataclass
class SplitPlan:
    """What the cost model says about running a resident aggregate on both devices."""
    gpu_fraction: float
    split_ms: float
    cpu_alone_ms: float
    gpu_alone_ms: float        # compute + the per-call cost
    gpu_compute_ms: float
    merge_ms: float
    contended_cpu_ms: float
    wins: bool                 # beats the faster device alone by SPLIT_MARGIN

    def explain(self) -> str:
        gpu = (f"gpu alone {self.gpu_alone_ms:.1f}ms (compute {self.gpu_compute_ms:.1f} + "
               f"{DEFAULT_GPU_ALPHA_MS:g} per call)")
        if self.gpu_fraction >= 1.0:
            return (f"resident gpu alone est {self.gpu_alone_ms:.1f}ms (compute {self.gpu_compute_ms:.1f} + "
                    f"{DEFAULT_GPU_ALPHA_MS:g} per call) vs cpu alone {self.cpu_alone_ms:.1f}ms, "
                    f"needs {SPLIT_MARGIN * 100:.0f}% less")
        best = min(self.cpu_alone_ms, self.gpu_alone_ms)
        gain = 100.0 * (1 - self.split_ms / best) if best > 0 else 0.0
        return (f"est {self.split_ms:.1f}ms with gpu share {self.gpu_fraction:.2f} "
                f"(cpu alone {self.cpu_alone_ms:.1f}ms, x{CPU_CONTENTION:g} contention when sharing = "
                f"{self.contended_cpu_ms:.1f}ms; {gpu}; merge {self.merge_ms:.1f}ms) "
                f"vs best single device {best:.1f}ms: {gain:+.0f}%, needs {SPLIT_MARGIN * 100:.0f}%")


def plan_split(cpu_ms: float, gpu_ms: float, n_groups: int = 1, n_slots: int = 1,
               native_factor: float = 1.0) -> SplitPlan:
    """Price a split of resident data. ``cpu_ms`` / ``gpu_ms``: calibrated compute time of one device over all
    rows; every GPU call also pays DEFAULT_GPU_ALPHA_MS (F-006: launch, sync and the partials' copy back).
    Both devices run at once, so the CPU share runs a little slower than alone (CPU_CONTENTION) and the
    partials cost a merge. The split wins only if it beats the faster device alone by SPLIT_MARGIN: below
    that a tie goes to the device with no merge and no GPU call."""
    cpu_alone = cpu_ms * native_factor
    gpu_alone = gpu_ms + DEFAULT_GPU_ALPHA_MS
    merge_ms = SPLIT_MERGE_MS + n_groups * (n_slots + 1) * 8 / MERGE_BYTES_PER_MS
    cpu_c = cpu_alone * CPU_CONTENTION
    f = balanced_fraction(cpu_c, gpu_alone, DEFAULT_GPU_ALPHA_MS)
    if f <= 0:
        split_ms, wins = cpu_alone, False
    elif f >= 1:                      # the resident GPU alone is the plan: it must beat the CPU by the margin
        split_ms, wins = gpu_alone, gpu_alone < cpu_alone * (1 - SPLIT_MARGIN)
    else:
        split_ms = (1 - f) * cpu_c + merge_ms
        wins = split_ms < min(cpu_alone, gpu_alone) * (1 - SPLIT_MARGIN)
    return SplitPlan(f, split_ms, cpu_alone, gpu_alone, gpu_ms, merge_ms, cpu_c, wins)


def _split(prog, cc, gids, n_groups, n_rows, entry, key, pol, cpu_rows=None):
    from . import auto_mode
    from . import pipeline_kernels as pk
    from .calibration import get_calibration
    from .coproc import Worker, run_router
    from .pipeline_program import merge_partials

    cal = get_calibration()
    card = max(n_groups, 1)
    cpu_ms = cal.estimate("pipeline", "cpu", n_rows, card=card)
    gpu_ms = cal.estimate("pipeline", "gpu", n_rows, card=card)
    sp = plan_split(cpu_ms, gpu_ms, n_groups, prog.n_slots)
    if pol.gpu_fraction is not None:
        f = pol.gpu_fraction
    else:      # resident data, split only when it beats the faster device by the margin; else the faster one alone
        f = sp.gpu_fraction if sp.wins else (1.0 if sp.gpu_alone_ms < sp.cpu_alone_ms else 0.0)
    k = int(round(n_rows * f))
    dc = entry.dc
    from . import native_exec
    repro = native_exec.reproducible()          # the router's worker threads do not inherit the context variable
    cpu_fn = cpu_rows or (lambda a, b: pk.run_pipeline_cpu(prog, cc, gids, n_groups, a, b, reproducible=repro))
    cpu = Worker("cpu", "cpu", n_rows, compute=lambda r: cpu_fn(r[0], r[1]))
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
            "est_split_ms": round(sp.split_ms, 1), "wall_ms": round(res.wall_ms, 1),
            "cpu_ms": round(res.stats["cpu"].busy_ms, 1),
            "gpu_ms": round(gstat.busy_ms, 1) if gstat is not None else 0.0,
            "gpu_failed": bool(gstat is not None and gstat.failed)}
    return res.value, label, info
