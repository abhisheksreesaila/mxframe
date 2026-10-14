"""Operator placement for `device="auto"`.

The engine materializes each operator, so auto mode decides every operator's
device right before running it, from the *actual* input size, the calibrated
per-device cost, the copy cost of non-resident input, and whether the GPU has
finished initialising. Three outcomes per operator:

- ``cpu``   run it on the CPU kernels.
- ``gpu``   run it on the GPU kernels (copy non-resident input over the link).
- ``split`` a Router: CPU workers and the GPU each take part of the rows and the
  partial results are merged (only for decomposable operators).

Cost of an operator on device d (spec, "Planner + cost model"):
    T_d = alpha_d + n * c_d + nonresident_bytes / BW_h2d + needed_out_bytes / BW_d2h
plus, for the GPU, whatever is left of its start-up cost.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

# Only pick the GPU (or a split) when it wins by this margin: the CPU path has
# no memory-capacity risk and no start-up cost, so ties go to the CPU.
GPU_WIN_MARGIN = 0.10
# CPU throughput while the GPU is also pulling host memory over DMA; both share DRAM.
SPLIT_CPU_EFFICIENCY = 0.90
# Fixed cost of running a Router: thread hand-off plus merging partials.
SPLIT_OVERHEAD_MS = 0.5
# A split that gives either side less than this share is not worth the overhead.
MIN_SPLIT_SHARE = 0.05


class CalibrationLike(Protocol):
    def estimate(self, op: str, device: str, rows: int, **shape) -> float: ...
    def transfer_ms(self, nbytes: int, direction: str = "h2d", pinned: bool = False) -> float: ...


class GpuLike(Protocol):
    available: bool

    def remaining_init_ms(self) -> float: ...


@dataclass
class Placement:
    """The decision for one operator, kept for explain() and profile."""

    op: str
    device: str                      # "cpu" | "gpu" | "split"
    rows: int
    est: Dict[str, float]            # estimated ms per option considered
    bytes_h2d: int = 0
    gpu_fraction: float = 0.0
    reason: str = ""
    actual_ms: Optional[float] = None

    def line(self) -> str:
        ests = " ".join(f"{k}={v:.1f}ms" for k, v in self.est.items() if v != float("inf"))
        extra = f" f_gpu={self.gpu_fraction:.2f}" if self.device == "split" else ""
        actual = f" actual={self.actual_ms:.1f}ms" if self.actual_ms is not None else ""
        return (f"{self.op:<14} {self.device:<5} rows={self.rows:,} est {ests}{extra}"
                f" h2d={_fmt_bytes(self.bytes_h2d)}{actual}  reason: {self.reason}")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n}B"


class Placer:
    """Decides cpu / gpu / split per operator and records every decision."""

    def __init__(
        self,
        calib: CalibrationLike,
        gpu: GpuLike,
        *,
        prefer: Optional[str] = None,
        placement: Optional[Dict[str, str]] = None,
        gpu_fraction: Optional[float] = None,
    ):
        self.calib = calib
        self.gpu = gpu
        self.prefer = prefer
        self.overrides = dict(placement or {})
        self.fixed_fraction = gpu_fraction
        self.decisions: List[Placement] = []

    def place(
        self,
        op: str,
        rows: int,
        *,
        bytes_in: int,
        bytes_out: int = 0,
        resident: str = "host",
        splittable: bool = False,
        **shape,
    ) -> Placement:
        d = self._decide(op, rows, bytes_in, bytes_out, resident, splittable, shape)
        self.decisions.append(d)
        return d

    def _decide(self, op, rows, bytes_in, bytes_out, resident, splittable, shape) -> Placement:
        cpu_ms = self.calib.estimate(op, "cpu", rows, **shape)
        if not self.gpu.available:
            return Placement(op, "cpu", rows, {"cpu": cpu_ms}, reason="no gpu")

        copy_in = 0.0 if resident == "gpu" else self.calib.transfer_ms(bytes_in, "h2d")
        copy_out = self.calib.transfer_ms(bytes_out, "d2h") if bytes_out else 0.0
        gpu_compute = self.calib.estimate(op, "gpu", rows, **shape)
        init_ms = self.gpu.remaining_init_ms()
        gpu_ms = gpu_compute + copy_in + copy_out + init_ms
        est = {"cpu": cpu_ms, "gpu": gpu_ms}
        h2d_full = 0 if resident == "gpu" else int(bytes_in)

        forced = self.overrides.get(op)
        if forced == "cpu":
            return Placement(op, "cpu", rows, est, reason="placement override")
        if forced == "gpu":
            return Placement(op, "gpu", rows, est, bytes_h2d=h2d_full, reason="placement override")

        split = None
        if splittable and init_ms == 0.0:
            split = self._split(cpu_ms, gpu_compute, copy_in + copy_out)
            if split is not None:
                est["split"] = split[1]
        if split is not None and self.fixed_fraction is not None:
            f = self.fixed_fraction
            return Placement(op, "split", rows, est, bytes_h2d=int(h2d_full * f),
                             gpu_fraction=f, reason="gpu_fraction override")

        best_other = min(cpu_ms, gpu_ms)
        if split is not None and split[1] < best_other * (1 - GPU_WIN_MARGIN):
            f = split[0]
            return Placement(op, "split", rows, est, bytes_h2d=int(h2d_full * f),
                             gpu_fraction=f, reason="both devices comparable; router splits rows")

        margin = 0.0 if self.prefer == "gpu" else GPU_WIN_MARGIN
        if gpu_ms < cpu_ms * (1 - margin):
            reason = "input resident on gpu" if resident == "gpu" else "gpu faster incl. copy"
            return Placement(op, "gpu", rows, est, bytes_h2d=h2d_full, reason=reason)
        if init_ms > 0 and gpu_compute + copy_in + copy_out < cpu_ms * (1 - margin):
            reason = "gpu still initialising"
        elif copy_in > gpu_compute and copy_in > 0:
            reason = "copy over link costs more than cpu compute"
        else:
            reason = "cpu faster"
        return Placement(op, "cpu", rows, est, reason=reason)

    @staticmethod
    def _split(cpu_ms: float, gpu_compute_ms: float, copy_ms: float):
        """Best (gpu_fraction, est_ms) for running both devices at once, or None.

        CPU share takes (1-f) * cpu / efficiency; GPU share takes
        alpha + f * (variable compute + copy). Balance the two.
        """
        cpu_eff = cpu_ms / SPLIT_CPU_EFFICIENCY
        gpu_var = max(gpu_compute_ms, 0.0) + copy_ms
        if cpu_eff <= 0 or gpu_var <= 0:
            return None
        f = cpu_eff / (cpu_eff + gpu_var)
        if f < MIN_SPLIT_SHARE or f > 1 - MIN_SPLIT_SHARE:
            return None
        return f, (1 - f) * cpu_eff + SPLIT_OVERHEAD_MS

    def explain_lines(self) -> List[str]:
        return [d.line() for d in self.decisions]


class GpuWarmup:
    """Initialises the GPU in a background thread so CPU work never waits on it.

    While initialisation is running the GPU counts as available, with the
    remaining expected start-up time added to its cost; after a failure it is
    unavailable for the rest of the process.
    """

    def __init__(self, init_fn: Callable[[], object], expected_ms: float):
        self._init_fn = init_fn
        self.expected_ms = float(expected_ms)
        self._done = threading.Event()
        self._ok = False
        self._t0: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.result = None
        self.error: Optional[BaseException] = None

    def start(self) -> "GpuWarmup":
        with self._lock:
            if self._thread is None:
                self._t0 = time.perf_counter()
                self._thread = threading.Thread(target=self._run, name="mxframe-gpu-warmup", daemon=True)
                self._thread.start()
        return self

    def _run(self):
        try:
            self.result = self._init_fn()
            self._ok = True
        except BaseException as e:  # any failure means "no gpu", never a crash
            self.error = e
            self._ok = False
        finally:
            self._done.set()

    @property
    def available(self) -> bool:
        return (not self._done.is_set()) or self._ok

    @property
    def ready(self) -> bool:
        return self._done.is_set() and self._ok

    def remaining_init_ms(self) -> float:
        if self._done.is_set():
            return 0.0 if self._ok else float("inf")
        if self._t0 is None:
            return self.expected_ms
        elapsed = (time.perf_counter() - self._t0) * 1e3
        return max(self.expected_ms - elapsed, 1e-3)

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.start()
        self._done.wait(timeout)
        return self.ready


# ── Whole-query choice (Phase 1) ─────────────────────────────────────────────
#
# Walks the plan once, estimating each operator's rows and its cost on each
# path. Filters, projections and group-key encoding run on the host on either
# path, so they cost the same on both; joins, aggregates and sorts differ.

import logging
import warnings

_log = logging.getLogger("mxframe.auto")

FILTER_SELECTIVITY = 0.3      # spec default until statistics exist
JOIN_GPU_SAFETY = 1.15        # measured GPU join estimate runs 5-30% low under load (F-003 review)
PREFER_CPU_MARGIN = 0.30      # prefer="cpu": the GPU must win by 30%
DEFAULT_GROUPS = 1000


@dataclass
class OpCost:
    op: str
    rows: int
    cpu_ms: float
    gpu_ms: float
    h2d_bytes: int = 0
    note: str = ""

    def line(self) -> str:
        g = "n/a" if self.gpu_ms == float("inf") else f"{self.gpu_ms:.1f}ms"
        note = f"  ({self.note})" if self.note else ""
        return (f"  {self.op:<10} rows={self.rows:,} est cpu={self.cpu_ms:.1f}ms gpu={g}"
                f" h2d={_fmt_bytes(self.h2d_bytes)}{note}")


@dataclass
class PlanEstimate:
    lines: List[OpCost] = field(default_factory=list)

    @property
    def cpu_ms(self) -> float:
        return sum(l.cpu_ms for l in self.lines)

    @property
    def gpu_ms(self) -> float:
        # device="gpu" runs every operator of the query on its GPU path, so one
        # unusable GPU operator makes the whole GPU plan impossible.
        return sum(l.gpu_ms for l in self.lines)

    @property
    def h2d_bytes(self) -> int:
        return sum(l.h2d_bytes for l in self.lines)


@dataclass
class AutoDecision:
    device: str
    reason: str
    estimate: Optional[PlanEstimate]
    init_ms: float = 0.0
    fallback: bool = False
    actual_ms: Optional[float] = None
    gpu_available: bool = True
    coproc: Optional[dict] = None     # set when a fused aggregate co-processed on cpu+gpu

    def explain(self) -> str:
        head = f"auto -> {self.device}  reason: {self.reason}"
        if self.fallback:
            head += "  [fell back to cpu after a gpu error]"
        if self.estimate is None:
            return head
        e = self.estimate
        gpu_total = e.gpu_ms + self.init_ms
        g = "n/a" if (not self.gpu_available or gpu_total == float("inf")) else f"{gpu_total:.1f}ms"
        tail = f"  total est cpu={e.cpu_ms:.1f}ms gpu={g}"
        if self.init_ms and g != "n/a":
            tail += f" (incl. gpu start-up {self.init_ms:.0f}ms)"
        if self.gpu_available:
            tail += f" h2d={_fmt_bytes(e.h2d_bytes)}"
        if self.actual_ms is not None:
            tail += f" actual={self.actual_ms:.1f}ms"
        if self.coproc:
            c = self.coproc
            tail += (f"\n  co-processing: gpu_fraction={c.get('gpu_fraction')} cpu busy={c.get('cpu_ms')}ms "
                     f"gpu busy={c.get('gpu_ms')}ms (resident columns)")
        lines = [l.line() if self.gpu_available else l.line().split(" gpu=")[0] for l in e.lines]
        return "\n".join([head] + lines + [tail])


# GPU paths that currently return wrong results; auto never picks the GPU for a
# plan that would run them. Remove an entry once its fix lands (F-012 / F-011).
KNOWN_BROKEN_GPU = {
}

_AGG_OP = {"sum": "group_sum", "mean": "group_sum", "count": "group_count",
           "min": "group_min", "max": "group_max"}
_SAMPLE_ROWS = 65_536
_UNKNOWN_COL_BYTES = 8


def _scan_rows_and_schema(plan):
    from .lazy_frame import Scan, ParquetScan
    if isinstance(plan, Scan):
        return plan.table.num_rows, plan.table
    if isinstance(plan, ParquetScan):
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(plan.path).metadata.num_rows, None
        except Exception:
            return 0, None
    return None, None


def _expr_cols(expr, out):
    if getattr(expr, "op", None) == "col":
        out.add(expr.args[0])
    for a in getattr(expr, "args", ()):
        if hasattr(a, "op"):
            _expr_cols(a, out)
    return out


def _col_bytes(tables, names) -> int:
    """Bytes per row of the named columns (their real widths when known)."""
    total = 0
    for n in names:
        w = None
        for t in tables:
            if t is not None and n in t.column_names:
                typ = t.schema.field(n).type
                try:
                    w = typ.bit_width // 8
                except Exception:
                    w = None
                break
        total += w or _UNKNOWN_COL_BYTES
    return max(total, 1)


def _key_card(name, tables, rows) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc
    from .np_arrow import array as _npa
    for t in tables:
        if t is None or name not in t.column_names:
            continue
        column = t.column(name)
        if pa.types.is_dictionary(column.type):
            return max((len(ch.dictionary) for ch in column.chunks), default=1)
        if len(column) == 0:
            return 1
        # Strided, not a prefix: data is often sorted or clustered by the key.
        import numpy as np
        n = len(column)
        if n <= _SAMPLE_ROWS:
            idx = np.arange(n, dtype=np.int64)
        else:   # linspace(0, n - 1, S) in integers: strictly increasing, no np.unique needed
            idx = np.arange(_SAMPLE_ROWS, dtype=np.int64) * (n - 1) // (_SAMPLE_ROWS - 1)
        d = _sample_distinct(column, idx)
        if d < 0.5 * len(idx):             # saturated: the sample saw (nearly) every value
            return max(1, d)
        return max(1, int(rows * d / len(idx)))
    return DEFAULT_GROUPS


def _sample_distinct(column, idx) -> int:
    """Distinct non-null values among ``column[idx]``. Numeric chunks are read through NumPy: on the
    first query of a process this is on the user-felt clock, and the Arrow path cost ~20 ms there
    (``np.unique`` pulls in ``numpy.ma``; the first ``count_distinct`` initialises Arrow compute)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    from .np_arrow import array as _npa, to_numpy as _np_of
    typ = column.type
    if (pa.types.is_integer(typ) or pa.types.is_floating(typ)) and column.null_count == 0:
        chunks = column.chunks
        starts = np.cumsum([0] + [len(ch) for ch in chunks])
        cuts = np.searchsorted(idx, starts)          # idx is sorted: one slice of it per chunk
        parts = []
        for c, ch in enumerate(chunks):
            sel = idx[cuts[c]:cuts[c + 1]]
            if len(sel):
                parts.append(_np_of(ch, zero_copy_only=False)[sel - starts[c]])
        v = np.sort(np.concatenate(parts)) if parts else np.empty(0)
        if len(v) == 0:
            return 0
        if v.dtype.kind == "f":
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return 0
        return int(1 + np.count_nonzero(v[1:] != v[:-1]))
    return int(pc.count_distinct(column.take(_npa(idx))).as_py())


def _group_card(keys, tables, rows) -> int:
    card = 1
    for k in keys:
        name = k.args[0] if getattr(k, "op", None) == "col" else None
        card *= _key_card(name, tables, rows) if name else 64   # derived keys (e.g. year) have few values
    return int(max(1, min(card, max(rows, 1))))


def estimate_plan(plan, calib, gpu) -> PlanEstimate:
    from .lazy_frame import Filter, Project, Aggregate, Sort, Limit, Distinct, Tail, Join
    est = PlanEstimate()
    INF = float("inf")

    def h2d(nbytes):
        return calib.transfer_ms(nbytes, "h2d")

    def walk(p):
        """Returns (rows, source tables, rows before the filters since the last materialisation)."""
        rows, table = _scan_rows_and_schema(p)
        if rows is not None:
            return rows, [table], rows
        if isinstance(p, Filter):
            n, ts, src = walk(p.input)
            c = calib.estimate("filter_take", "cpu", n)
            est.lines.append(OpCost("filter", n, c, c, note="host on both paths"))
            return max(1, int(n * FILTER_SELECTIVITY)), ts, src
        if isinstance(p, Join):
            nl, tl, _ = walk(p.left)
            nr, tr, _ = walk(p.right)
            cpu = calib.estimate("join", "cpu", nl + nr)
            nbytes = nl * _col_bytes(tl, p.left_on) + nr * _col_bytes(tr, p.right_on)
            try:
                g = calib.estimate_gpu_join_total(nl, nr) * JOIN_GPU_SAFETY
            except Exception:
                g = calib.estimate("join", "gpu", nl + nr)
            out = max(nl, nr)          # key/foreign-key join: one row per row of the larger side
            est.lines.append(OpCost("join", out, cpu, g + h2d(nbytes), nbytes))
            return out, tl + tr, out
        if isinstance(p, Aggregate):
            n, ts, src = walk(p.input)
            cols = set()
            for a in p.aggs:
                _expr_cols(a, cols)
            ops = [_AGG_OP.get(a.op, "group_sum") for a in p.aggs] or ["group_count"]
            width = _col_bytes(ts, sorted(cols)) if cols else 4
            if p.group_by:
                card = _group_card(p.group_by, ts, n)
                enc = calib.estimate("group_encode", "cpu", n)
                cpu = enc + sum(calib.estimate(o, "cpu", n, card=card) for o in ops)
                nbytes = n * (4 + width)
                g = enc + sum(calib.estimate(o, "gpu", n, card=card) for o in ops) + h2d(nbytes) \
                    + calib.transfer_ms(card * 8 * len(ops), "d2h")
                est.lines.append(OpCost("group_agg", n, cpu, g, nbytes, note=f"~{card:,} groups"))
                return card, ts, card
            # Global aggregates run masked kernels over the unfiltered input.
            nbytes = src * (width + 4)
            cpu = len(ops) * calib.estimate("masked_sum", "cpu", src)
            if "global_agg" in KNOWN_BROKEN_GPU:
                est.lines.append(OpCost("global_agg", src, cpu, INF, note=KNOWN_BROKEN_GPU["global_agg"]))
                return 1, ts, 1
            g = len(ops) * calib.estimate("masked_sum", "gpu", src) + h2d(nbytes)
            est.lines.append(OpCost("global_agg", src, cpu, g, nbytes))
            return 1, ts, 1
        if isinstance(p, Sort):
            n, ts, src = walk(p.input)
            cpu = calib.estimate("sort_topk", "cpu", n)
            if "sort" in KNOWN_BROKEN_GPU:
                est.lines.append(OpCost("sort", n, cpu, INF, note=KNOWN_BROKEN_GPU["sort"]))
            else:
                nbytes = n * 4
                est.lines.append(OpCost("sort", n, cpu, calib.estimate("sort_topk", "gpu", n) + h2d(nbytes), nbytes))
            return n, ts, src
        if isinstance(p, (Project, Limit, Distinct, Tail)):
            n, ts, src = walk(p.input)
            if isinstance(p, (Limit, Tail)):
                n = min(n, p.n)
            return n, ts, src
        inner = getattr(p, "input", None)
        return walk(inner) if inner is not None else (0, [None], 0)

    walk(plan)
    return est


_gpu_disabled: Optional[str] = None


def disable_gpu(reason: str) -> None:
    """After a sticky device error (e.g. illegal address) the CUDA context is unusable."""
    global _gpu_disabled
    _gpu_disabled = reason


def is_sticky_gpu_error(err: BaseException) -> bool:
    text = f"{type(err).__name__}: {err}"
    if isinstance(err, MemoryError) or "OUT_OF_MEMORY" in text:
        return False
    return "CUDA_ERROR" in text or "cuda error" in text.lower()


_failed_shapes: set = set()
_warned_shapes: set = set()


def _shape_key(plan):
    try:
        return plan.signature()
    except Exception:
        return None


def remember_gpu_failure(plan) -> None:
    k = _shape_key(plan)
    if k is not None:
        _failed_shapes.add(k)


def forget_gpu_failures() -> None:
    global _gpu_disabled
    _failed_shapes.clear()
    _warned_shapes.clear()
    _gpu_disabled = None


def warn_fallback_once(plan, err: BaseException) -> None:
    k = _shape_key(plan)
    if k is not None and k in _warned_shapes:
        return
    if k is not None:
        _warned_shapes.add(k)
    warnings.warn(f"mxframe auto: GPU execution failed ({type(err).__name__}: {err}); "
                  f"fell back to CPU. This query shape will plan CPU from now on.",
                  RuntimeWarning, stacklevel=3)


def choose_device(plan, *, calib=None, gpu=None, prefer: Optional[str] = None,
                  placement: Optional[Dict[str, str]] = None, quick: bool = True) -> AutoDecision:
    """Pick cpu or gpu for the whole query (spec Phase 1)."""
    if quick and calib is None and gpu is None and not placement and prefer != "gpu":
        rows = _source_rows(plan)
        if rows < SMALL_QUERY_ROWS:
            return AutoDecision("cpu", f"small input ({rows:,} rows): gpu start-up and copies cannot pay off",
                                None)
    if calib is None:
        from .calibration import get_calibration
        calib = get_calibration()
    if gpu is None:
        gpu = get_gpu_warmup(calib, start=False) if getattr(calib, "has_gpu", False) else _NO_GPU
    est = estimate_plan(plan, calib, gpu)
    if not gpu.available:
        return AutoDecision("cpu", "no gpu", est, gpu_available=False)
    if _gpu_disabled is not None:
        return AutoDecision("cpu", f"gpu disabled after error: {_gpu_disabled}", est, gpu_available=False)
    if _shape_key(plan) in _failed_shapes:
        return AutoDecision("cpu", "this query shape failed on gpu before", est)
    init = gpu.remaining_init_ms()
    blocked = [l for l in est.lines if l.gpu_ms == float("inf")]
    if blocked:   # checked before overrides: never run a GPU path known to be wrong or missing
        why = blocked[0].note or f"{blocked[0].op} has no usable gpu kernel"
        return AutoDecision("cpu", f"gpu path unusable: {why}", est, init)
    wanted = set((placement or {}).values())
    if "gpu" in wanted:
        return AutoDecision("gpu", "placement override", est, init)
    if wanted == {"cpu"}:
        return AutoDecision("cpu", "placement override", est)
    cpu_ms, gpu_ms = est.cpu_ms, est.gpu_ms + init
    margin = {"gpu": 0.0, "cpu": PREFER_CPU_MARGIN}.get(prefer, GPU_WIN_MARGIN)
    if gpu_ms < cpu_ms * (1 - margin):
        return AutoDecision("gpu", "gpu faster incl. copies", est, init)
    if est.gpu_ms < cpu_ms * (1 - margin) and init > 0:
        return AutoDecision("cpu", "gpu faster but still starting up", est, init)
    return AutoDecision("cpu", "cpu faster (copies over the link cost more)" if est.h2d_bytes else "cpu faster",
                        est, init)


SMALL_QUERY_ROWS = 200_000   # below this, skip calibration (loading it initialises CUDA)


def _source_rows(plan) -> int:
    rows, _ = _scan_rows_and_schema(plan)
    if rows is not None:
        return rows
    kids = [getattr(plan, a) for a in ("input", "left", "right") if getattr(plan, a, None) is not None]
    return sum(_source_rows(k) for k in kids)


class _NoGpu:
    available = False
    ready = False

    def remaining_init_ms(self):
        return float("inf")


_NO_GPU = _NoGpu()
_warmup: Optional[GpuWarmup] = None
_warmup_lock = threading.Lock()


def _init_gpu():
    # The engine shares this one instance (and its CUDA driver / device-buffer cache).
    from .aot_kernels import get_aot_gpu
    gpu = get_aot_gpu()
    if gpu is None:
        raise RuntimeError("GPU AOT kernels unavailable")
    return gpu


def get_gpu_warmup(calib=None, start: bool = True) -> GpuWarmup:
    """Process-wide GPU warm-up: CUDA context + GPU AOT library, started in the background.

    ``start=False`` returns it unstarted (its remaining cost counts in full) so a
    CPU-bound query can start it after it has returned.
    """
    global _warmup
    with _warmup_lock:
        if _warmup is None:
            expected = 250.0
            if calib is not None:
                try:
                    expected = float(calib.gpu_init_cost_ms())
                except Exception:
                    pass
            _warmup = GpuWarmup(_init_gpu, expected_ms=expected if expected != float("inf") else 250.0)
        if start:
            _warmup.start()
        return _warmup
