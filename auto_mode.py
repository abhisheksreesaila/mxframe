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
        return sum(min(l.gpu_ms, l.cpu_ms) if l.gpu_ms == float("inf") else l.gpu_ms for l in self.lines)

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

    def explain(self) -> str:
        head = f"auto -> {self.device}  reason: {self.reason}"
        if self.fallback:
            head += "  [fell back to cpu after a gpu error]"
        if self.estimate is None:
            return head
        e = self.estimate
        gpu_total = e.gpu_ms + self.init_ms
        tail = f"  total est cpu={e.cpu_ms:.1f}ms gpu={gpu_total:.1f}ms"
        if self.init_ms:
            tail += f" (incl. gpu start-up {self.init_ms:.0f}ms)"
        tail += f" h2d={_fmt_bytes(e.h2d_bytes)}"
        if self.actual_ms is not None:
            tail += f" actual={self.actual_ms:.1f}ms"
        return "\n".join([head] + [l.line() for l in e.lines] + [tail])


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


def _group_card(keys, table, rows) -> int:
    """Estimated group count: dictionary sizes when known, else a guess."""
    card = 1
    for k in keys:
        name = k.args[0] if getattr(k, "op", None) == "col" else None
        c = None
        if table is not None and name in (table.column_names if table is not None else []):
            typ = table.schema.field(name).type
            import pyarrow as pa
            if pa.types.is_dictionary(typ):
                chunks = table.column(name).chunks
                c = max((len(ch.dictionary) for ch in chunks), default=1)
            elif pa.types.is_string(typ) or pa.types.is_large_string(typ):
                c = DEFAULT_GROUPS
        if c is None:
            c = max(1, rows // 4) if name else 64   # integer keys: many groups; derived keys (year): few
        card *= c
    return int(max(1, min(card, max(rows, 1))))


def estimate_plan(plan, calib, gpu) -> PlanEstimate:
    from .lazy_frame import Scan, ParquetScan, Filter, Project, Aggregate, Sort, Limit, Distinct, Tail, Join
    est = PlanEstimate()

    def h2d(nbytes):
        return calib.transfer_ms(nbytes, "h2d")

    def walk(p):
        """Returns (rows, a representative source table for schema lookups)."""
        rows, table = _scan_rows_and_schema(p)
        if rows is not None:
            return rows, table
        if isinstance(p, Filter):
            n, t = walk(p.input)
            c = calib.estimate("filter_take", "cpu", n)
            est.lines.append(OpCost("filter", n, c, c, note="host on both paths"))
            return max(1, int(n * FILTER_SELECTIVITY)), t
        if isinstance(p, Join):
            nl, tl = walk(p.left)
            nr, _ = walk(p.right)
            cpu = calib.estimate("join", "cpu", nl) + calib.estimate("join", "cpu", nr) * 0.2
            nbytes = 4 * (nl + nr)
            try:
                g = calib.estimate_gpu_join_total(nl, nr) * JOIN_GPU_SAFETY
            except Exception:
                g = calib.estimate("join", "gpu", nl + nr)
            est.lines.append(OpCost("join", nl, cpu, g + h2d(nbytes), nbytes))
            return nl, tl
        if isinstance(p, Aggregate):
            n, t = walk(p.input)
            cols = set()
            for a in p.aggs:
                _expr_cols(a, cols)
            if p.group_by:
                card = _group_card(p.group_by, t, n)
                enc = calib.estimate("group_encode", "cpu", n)
                k = max(1, len(p.aggs))
                cpu = enc + k * calib.estimate("group_sum", "cpu", n, card=card)
                nbytes = n * (4 + 4 * max(1, len(cols)))
                g = enc + k * calib.estimate("group_sum", "gpu", n, card=card) + h2d(nbytes) \
                    + calib.transfer_ms(card * 8 * k, "d2h")
                est.lines.append(OpCost("group_agg", n, cpu, g, nbytes, note=f"~{card:,} groups"))
                return card, t
            k = max(1, len(p.aggs))
            nbytes = n * 4 * max(1, len(cols))
            cpu = k * calib.estimate("masked_sum", "cpu", n)
            g = k * calib.estimate("masked_sum", "gpu", n) + h2d(nbytes)
            est.lines.append(OpCost("global_agg", n, cpu, g, nbytes))
            return 1, t
        if isinstance(p, Sort):
            n, t = walk(p.input)
            if n > 1000:
                nbytes = n * 4
                est.lines.append(OpCost("sort", n, calib.estimate("sort_topk", "cpu", n),
                                        calib.estimate("sort_topk", "gpu", n) + h2d(nbytes), nbytes))
            return n, t
        if isinstance(p, (Project, Limit, Distinct, Tail)):
            n, t = walk(p.input)
            if isinstance(p, (Limit, Tail)):
                n = min(n, p.n)
            return n, t
        inner = getattr(p, "input", None)
        return walk(inner) if inner is not None else (0, None)

    walk(plan)
    return est


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
    _failed_shapes.clear()
    _warned_shapes.clear()


def warn_fallback_once(plan, err: BaseException) -> None:
    k = _shape_key(plan)
    if k in _warned_shapes:
        return
    _warned_shapes.add(k)
    warnings.warn(f"mxframe auto: GPU execution failed ({type(err).__name__}: {err}); "
                  f"fell back to CPU. This query shape will plan CPU from now on.",
                  RuntimeWarning, stacklevel=3)


def choose_device(plan, *, calib=None, gpu=None, prefer: Optional[str] = None,
                  placement: Optional[Dict[str, str]] = None) -> AutoDecision:
    """Pick cpu or gpu for the whole query (spec Phase 1)."""
    if calib is None:
        from .calibration import get_calibration
        calib = get_calibration()
    if gpu is None:
        gpu = get_gpu_warmup(calib) if getattr(calib, "has_gpu", False) else _NO_GPU
    est = estimate_plan(plan, calib, gpu)
    if not gpu.available:
        return AutoDecision("cpu", "no gpu", est)
    if _shape_key(plan) in _failed_shapes:
        return AutoDecision("cpu", "this query shape failed on gpu before", est)
    wanted = set((placement or {}).values())
    if "gpu" in wanted:
        return AutoDecision("gpu", "placement override", est, gpu.remaining_init_ms())
    if wanted == {"cpu"}:
        return AutoDecision("cpu", "placement override", est)
    init = gpu.remaining_init_ms()
    cpu_ms, gpu_ms = est.cpu_ms, est.gpu_ms + init
    margin = {"gpu": 0.0, "cpu": PREFER_CPU_MARGIN}.get(prefer, GPU_WIN_MARGIN)
    if gpu_ms < cpu_ms * (1 - margin):
        return AutoDecision("gpu", "gpu faster incl. copies", est, init)
    if est.gpu_ms < cpu_ms * (1 - margin) and init > 0:
        return AutoDecision("cpu", "gpu faster but still starting up", est, init)
    return AutoDecision("cpu", "cpu faster (copies over the link cost more)" if est.h2d_bytes else "cpu faster",
                        est, init)


class _NoGpu:
    available = False
    ready = False

    def remaining_init_ms(self):
        return float("inf")


_NO_GPU = _NoGpu()
_warmup: Optional[GpuWarmup] = None
_warmup_lock = threading.Lock()


def _init_gpu():
    from .aot_kernels import AOTKernelsGPU
    return AOTKernelsGPU()


def get_gpu_warmup(calib=None) -> GpuWarmup:
    """Process-wide GPU warm-up: CUDA context + GPU AOT library, started in the background."""
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
            _warmup.start()
        return _warmup
