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
