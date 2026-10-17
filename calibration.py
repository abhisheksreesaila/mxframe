"""Calibration store for MXFrame auto mode (F-003).

Measures, persists and serves the constants the auto-mode cost model needs:

* host<->device link bandwidth (pageable and pinned) and per-transfer latency,
* per-call launch overhead per device,
* GPU init cost in a *fresh process* (CUDA context + GPU AOT library + first kernel),
* per-(op, device) linear fits ``time_ms = alpha + c * rows [+ g * cardinality]``.

Consumers (F-005 planner, F-006 co-processing) use::

    cal = get_calibration()
    cal.estimate("group_sum", "gpu", rows=6_000_000, card=1000)   # kernel time, ms
    cal.transfer_ms(nbytes, "h2d", pinned=False)                   # link time, ms
    cal.observe("group_sum", "gpu", rows, measured_ms)              # online refinement

Contract: ``estimate(op, device, rows, ...)`` is COMPUTE ONLY, with inputs already
resident on that device and the output left on that device.  It includes kernel
launch/sync and per-call context overhead but NO H2D/D2H copies and no host-side
pre/post-processing that is not inherent to the op; the placer adds every copy via
``transfer_ms``.  (GPU join has host steps between its two kernels; they are a
separate ``join_host`` CPU term.)  Group ops on the GPU are fitted per group-count
bucket (an op listed in ``GPU_MAX_CARD`` returns ``inf`` above its limit; the table is
empty since F-012 fixed the min/max kernels that faulted above 8192 groups).  With no GPU, ``estimate(..., "gpu", ...)`` returns ``inf`` so
``min()`` just works.  Planner op names (``group_agg``, ``join_probe``, ``sort``,
``filter``, ...) are aliased to the measured ops; ``pipeline`` is reserved for
F-008's fused kernel.

The set of benchmarked kernels lives in a registry of probe callables
(``PROBES``); when a kernel's signature or dtype changes, edit its one probe.
"""
from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import sys
import threading   # platform, subprocess, tempfile: imported where used (auto's first query pays for module imports)
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA_VERSION = 3      # v3: sort_topk@cpu times the radix sort (was PyArrow); pipeline fits are measured
                        # (v2: compute-only estimate contract, group-count fit keys, size exponent)
MAX_OBS_SAMPLES = 64        # bounded per-fit sample list (HyPE-style)
MIN_OBS = 3                 # samples before observations start to count
OBS_RAMP = 6                # observed weight ramps 0 -> 1 over this many samples past MIN_OBS
PERSIST_EVERY_N = 32        # observe(): save after this many new observations ...
PERSIST_EVERY_S = 60.0      # ... or this many seconds
DEFAULT_CARD = 1000         # assumed group cardinality when the caller gives none
BUCKET_EDGES = (64, 1024, 8192)          # group-count buckets: <=64, <=1024, <=8192, >8192
BUCKET_REP_CARDS = (8, 64, 100, 1000, 2048, 8192, 10_000, 200_000)   # two per bucket; interpolated on log(groups)
GPU_MAX_CARD: Dict[str, int] = {}     # op -> largest group count its GPU kernel handles (none now:
                                      # min/max faulted above 8192 groups until F-012)
OP_ALIASES = {"group_agg": "group_sum", "join_probe": "join",
              "sort": "sort_topk", "filter": "filter_take"}
HOST_ONLY_OPS = frozenset({"join_host"})     # CPU work; estimate(..., "gpu") is inf
_log = logging.getLogger(__name__)
_warned_ops: set = set()


def bucket_name(card: float) -> str:
    for e in BUCKET_EDGES:
        if card <= e:
            return f"<={e}"
    return f">{BUCKET_EDGES[-1]}"

DEVICES = ("cpu", "gpu")
OPS = (
    "pipeline", "join_host", "join_total", "group_sum", "group_count", "group_min", "group_max",
    "masked_sum", "join", "gather", "sort_topk", "group_encode", "filter_take",
    "native_global", "native_filter", "native_group", "native_join",
)

QUICK_SIZES = (10_000, 100_000, 1_000_000, 6_000_000, 30_000_000)
FULL_SIZES = (10_000, 100_000, 1_000_000, 10_000_000, 30_000_000, 100_000_000)
QUICK_LINK_MB = (1, 8, 64)
FULL_LINK_MB = (1, 16, 128, 1024)
QUICK_OP_BUDGET_S = 1.5     # skip a size when its projected time exceeds this (quick)
FULL_OP_BUDGET_S = 6.0


# --------------------------------------------------------------------------- #
# Paths / identity
# --------------------------------------------------------------------------- #
def calibration_path() -> Path:
    env = os.environ.get("MXFRAME_CALIBRATION")
    return Path(env).expanduser() if env else Path.home() / ".mxframe" / "calibration.json"


def _mxframe_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or platform.machine() or "cpu"


def _nvidia_driver_string() -> Optional[str]:
    try:
        txt = Path("/proc/driver/nvidia/version").read_text()
        for tok in txt.split():
            if tok.count(".") >= 1 and tok.replace(".", "").isdigit():
                return tok
    except Exception:
        pass
    return None


def _gpu_identity() -> Optional[Dict[str, Any]]:
    """(name, driver, memory) of GPU 0 as seen through CUDA_VISIBLE_DEVICES, or None.

    Reads /proc/driver/nvidia when it can (no CUDA initialisation, so loading the
    calibration never makes a CPU-bound first query wait ~40 ms on cuInit), else
    asks the CUDA driver. Tests monkeypatch this function to simulate machines
    with and without a GPU.
    """
    ident = _proc_gpu_identity()
    return ident if ident is not None else _cuda_gpu_identity()


def _proc_gpu_identity() -> Optional[Dict[str, Any]]:
    """NVIDIA model + driver from /proc when every visible GPU is the same model, else None.

    ``mem_bytes`` is 0 here; the stored calibration carries the measured value.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip() in ("", "-1"):
        return None
    try:
        models = []
        for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
            for line in info.read_text().splitlines():
                if line.startswith("Model:"):
                    models.append(line.split(":", 1)[1].strip())
        driver = _nvidia_driver_string()
    except Exception:
        return None
    if not models or driver is None or len(set(models)) != 1:
        return None
    return {"name": models[0], "driver": driver, "mem_bytes": 0}


def _cuda_gpu_identity() -> Optional[Dict[str, Any]]:
    for lib_name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(lib_name)
            break
        except OSError:
            lib = None
    if lib is None:
        return None
    try:
        if lib.cuInit(0) != 0:
            return None
        n = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value < 1:
            return None
        dev = ctypes.c_int(0)
        lib.cuDeviceGet(ctypes.byref(dev), 0)
        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName(name, 256, dev)
        mem = ctypes.c_size_t(0)
        lib.cuDeviceTotalMem_v2(ctypes.byref(mem), dev)
        ver = ctypes.c_int(0)
        lib.cuDriverGetVersion(ctypes.byref(ver))
        driver = _nvidia_driver_string() or f"cuda{ver.value}"
        return {"name": name.value.decode(errors="replace"), "driver": driver,
                "mem_bytes": int(mem.value)}
    except Exception:
        return None


def _detect_cpu_threads() -> int:
    """Worker budget of the CPU kernels (MXFRAME_CPU_THREADS / set_cpu_threads, default all cores)."""
    try:
        from . import aot_kernels as ak
        return max(1, int(ak.get_cpu_threads()))
    except Exception:
        v = os.environ.get("MXFRAME_CPU_THREADS", "")
        return int(v) if v.isdigit() and int(v) > 0 else max(1, os.cpu_count() or 1)


def machine_key(identity: Optional[Dict[str, Any]] = None, version: Optional[str] = None) -> str:
    name = identity["name"] if identity else f"cpu-only:{_cpu_model()}"
    driver = identity["driver"] if identity else "none"
    return f"{name}|{driver}|{version or _mxframe_version()}"


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #
def _nnls_rel(rows: List[Tuple[float, float, float]], use_card: bool) -> Tuple[float, float, float]:
    """Relative-error least squares for ms = a + c*n + g*card with a,c,g >= 0.

    rows: (n, card, ms).  Returns (alpha_ms, c_ms_per_row, g_ms_per_group).
    Non-negativity via active-set elimination (drop the most negative column).
    """
    import numpy as np
    ys = np.array([r[2] for r in rows], dtype=np.float64)
    w = 1.0 / np.maximum(ys, 1e-6)
    cols = {"a": np.ones(len(rows)),
            "c": np.array([r[0] for r in rows], dtype=np.float64) / 1e6,   # per million rows
            "g": np.array([r[1] for r in rows], dtype=np.float64) / 1e6}
    active = ["a", "c"] + (["g"] if use_card else [])
    coef = {"a": 0.0, "c": 0.0, "g": 0.0}
    while active:
        A = np.stack([cols[k] for k in active], axis=1) * w[:, None]
        sol, *_ = np.linalg.lstsq(A, ys * w, rcond=None)
        if (sol >= 0).all():
            coef.update({k: float(v) for k, v in zip(active, sol)})
            break
        active.remove(active[int(np.argmin(sol))])
    return coef["a"], coef["c"] / 1e6, coef["g"] / 1e6


@dataclass
class Fit:
    """time_ms = alpha + c*rows + g*card.  `prior_*` come from calibration/defaults;
    `obs` holds online observations [rows, card, ms] (bounded)."""
    alpha: float
    c: float
    g: float = 0.0
    obs: List[List[float]] = field(default_factory=list)
    n_points: int = 0            # points used by the calibration fit (0 = built-in default)
    n_max: float = 0.0           # largest measured rows; beyond it the fit extrapolates linearly
    p: float = 1.0               # size exponent: time = alpha + c * rows**p + g*card (p > 1: super-linear)
    _cur: Optional[Tuple[float, float, float]] = field(default=None, repr=False, compare=False)

    # -- prediction ------------------------------------------------------- #
    def params(self) -> Tuple[float, float, float]:
        if self._cur is None:
            self._cur = self._blend()
        return self._cur

    def _size_term(self, rows: float) -> float:
        p = self.p
        if p == 1.0:
            return rows
        if self.n_max and rows > self.n_max:      # slope at the last measured point, not the power law
            return self.n_max ** p + p * self.n_max ** (p - 1.0) * (rows - self.n_max)
        return rows ** p

    def predict(self, rows: float, card: Optional[float] = None) -> float:
        a, c, g = self.params()
        gterm = g * (DEFAULT_CARD if card is None else card) if g else 0.0
        return max(0.0, a + c * self._size_term(rows) + gterm)

    # -- online refinement ------------------------------------------------ #
    def observe(self, rows: float, ms: float, card: Optional[float] = None) -> None:
        if not (rows >= 0 and ms >= 0 and math.isfinite(ms)):
            return
        self.obs.append([float(rows), float(DEFAULT_CARD if card is None else card), float(ms)])
        if len(self.obs) > MAX_OBS_SAMPLES:
            del self.obs[: len(self.obs) - MAX_OBS_SAMPLES]
        self._cur = None

    def _blend(self) -> Tuple[float, float, float]:
        prior = (self.alpha, self.c, self.g)
        n = len(self.obs)
        if n < MIN_OBS:
            return prior
        w = min(1.0, (n - MIN_OBS + 1) / OBS_RAMP)      # >0.5 (outweighs prior) from MIN_OBS+3
        rows = [(o[0] ** self.p, o[1], o[2]) for o in self.obs]
        sizes = [o[0] for o in self.obs]
        if max(sizes) >= 2.0 * max(min(sizes), 1.0):      # enough spread in n: refit the line
            use_card = self.g > 0 and len({round(r[1]) for r in rows}) >= 3
            a, c, g = _nnls_rel(rows, use_card)     # no group term refit: it decays with the prior
        else:                                              # one size only: rescale the prior
            ratios = sorted(o[2] / max(self.predict_prior(o[0], o[1]), 1e-9) for o in self.obs)
            k = ratios[len(ratios) // 2]
            a, c, g = self.alpha * k, self.c * k, self.g * k
        return tuple((1 - w) * p + w * q for p, q in zip(prior, (a, c, g)))  # type: ignore[return-value]

    def predict_prior(self, rows: float, card: float) -> float:
        return self.alpha + self.c * self._size_term(rows) + self.g * card

    # -- (de)serialisation ------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "c": self.c, "g": self.g,
                "n_points": self.n_points, "obs": self.obs, "p": self.p,
                "n_max": self.n_max}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fit":
        f = cls(float(d["alpha"]), float(d["c"]), float(d.get("g", 0.0)),
                [[float(x) for x in o] for o in d.get("obs", [])][-MAX_OBS_SAMPLES:],
                int(d.get("n_points", 0)), p=float(d.get("p", 1.0)), n_max=float(d.get("n_max", 0.0)))
        if not (all(math.isfinite(v) and v >= 0 for v in (f.alpha, f.c, f.g)) and 0.5 <= f.p <= 3.0):
            raise ValueError("bad fit")
        return f


# --------------------------------------------------------------------------- #
# Calibration object
# --------------------------------------------------------------------------- #
_DIRS = ("h2d", "d2h")


@dataclass
class Calibration:
    key: str
    device_name: str
    driver: str
    mxframe_version: str
    has_gpu: bool
    cpu_threads: int
    cpu_cores: int
    pinned_available: bool
    link_gbps: Dict[str, float]              # "h2d_pageable", "h2d_pinned", "d2h_pageable", "d2h_pinned" (GB/s)
    link_latency_ms: Dict[str, float]        # same keys, fixed cost per transfer
    launch_overhead_ms: Dict[str, float]     # {"cpu":..., "gpu":...}
    gpu_init_ms: float                       # fresh process: context + GPU AOT lib + first kernel
    gpu_mem_bytes: int
    fits: Dict[str, Fit]                     # "op@device"
    source: str = "defaults"                 # "defaults" | "measured"
    quick: bool = True
    created: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)
    gpu_init_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)
    _dirty: int = field(default=0, repr=False, compare=False)
    _last_save: float = field(default_factory=time.monotonic, repr=False, compare=False)

    # -- queries ---------------------------------------------------------- #
    def _card_fits(self, base: str) -> List[Tuple[float, Fit]]:
        pre = f"{base}:g"
        return sorted((float(k[len(pre):]), f) for k, f in self.fits.items() if k.startswith(pre))

    def fit(self, op: str, device: str, card: Optional[float] = None) -> Optional[Fit]:
        """Fit for (op, device), routed by alias; for group ops the nearest measured
        group-count fit (same bucket preferred)."""
        op = OP_ALIASES.get(op, op)
        base = f"{op}@{device}"
        cf = self._card_fits(base)
        if cf:
            q = DEFAULT_CARD if card is None else card
            same = [t for t in cf if bucket_name(t[0]) == bucket_name(q)] or cf
            return min(same, key=lambda t: abs(math.log(t[0]) - math.log(max(q, 1))))[1]
        if base in self.fits:
            return self.fits[base]
        if (op, device) not in _warned_ops:
            _warned_ops.add((op, device))
            _log.debug("no calibration for op %r on %s; using generic fit", op, device)
        return self.fits.get(f"generic@{device}")

    def _predict(self, op: str, device: str, rows: float, card: Optional[float]) -> Optional[float]:
        """Predicted ms; group ops interpolate on log(groups) between the two measured
        group counts of the query's bucket (clamped at the ends)."""
        real = OP_ALIASES.get(op, op)
        cf = self._card_fits(f"{real}@{device}")
        if cf:
            q = DEFAULT_CARD if card is None else max(card, 1)
            same = [t for t in cf if bucket_name(t[0]) == bucket_name(q)] or cf
            lo = max((t for t in same if t[0] <= q), key=lambda t: t[0], default=None)
            hi = min((t for t in same if t[0] >= q), key=lambda t: t[0], default=None)
            if lo is None or hi is None or lo[0] == hi[0]:
                t = lo or hi
                return t[1].predict(rows, q)
            w = (math.log(q) - math.log(lo[0])) / (math.log(hi[0]) - math.log(lo[0]))
            return (1 - w) * lo[1].predict(rows, q) + w * hi[1].predict(rows, q)
        f = self.fit(op, device, card)
        return None if f is None else f.predict(rows, card)

    def estimate(self, op: str, device: str, rows: float, card: Optional[float] = None,
                 **shape: Any) -> float:
        """Compute-only ms (see module docstring).  ``inf`` if the device is unavailable
        or the op cannot run there.  ``join_total`` (GPU) needs ``right_rows=``; it is
        `estimate_gpu_join_total`.  Extra ``shape`` kwargs are ignored (forward compat)."""
        real = OP_ALIASES.get(op, op)
        if device == "gpu" and (not self.has_gpu or real in HOST_ONLY_OPS):
            return math.inf
        if real == "join_total":
            if device == "gpu":
                return self.estimate_gpu_join_total(rows, shape.get("right_rows", rows // 8 or 1))
            return self.estimate("join", "cpu", rows)
        if card is None:
            card = shape.get("cardinality")
        limit = GPU_MAX_CARD.get(real)
        if device == "gpu" and limit is not None and card is not None and card > limit:
            return math.inf
        v = self._predict(op, device, rows, card)
        if v is None:
            return math.inf if device == "gpu" else 0.0
        return v

    def estimate_gpu_join_total(self, left_rows: float, right_rows: float,
                                table_size: Optional[float] = None, pinned: bool = False,
                                out_rows: Optional[float] = None) -> float:
        """Price of the real current GPU hash join (`AOTKernelsGPU.hash_join`), key columns resident.

        Measured END TO END (`join_total@gpu`, fit alpha + a*L^p + b*R over left/right rows at
        ratios 1:8 and 1:1): both kernels, the host bincount/argsort/cumsum, counts D2H, aux
        uploads and the result D2H.  Excludes only the key upload (the placer prices it).
        The probe has ~1 match per left row; pass ``out_rows`` (expected join output rows)
        to swap the result-copy term (8 B per output row) for the real one.

        Without a measured fit (defaults) it falls back to a sum of parts: `join@gpu` +
        `join_host@cpu` + counts D2H + aux H2D (4*L + 4*R + 3*4*table bytes) + result D2H.
        """
        if not self.has_gpu:
            return math.inf
        f = self.fits.get("join_total@gpu")
        if f is not None and f.n_points >= 3:
            v = f.predict(left_rows, right_rows)
            if out_rows is not None:
                v += (self.transfer_ms(8 * out_rows, "d2h", pinned)
                      - self.transfer_ms(8 * left_rows, "d2h", pinned))
            return max(v, 0.0)
        tab = right_rows if table_size is None else table_size
        h2d_bytes = 4 * left_rows + 4 * right_rows + 3 * 4 * tab
        res = left_rows if out_rows is None else out_rows
        return (self.estimate("join", "gpu", left_rows)
                + self.estimate("join_host", "cpu", left_rows)
                + self.transfer_ms(4 * left_rows, "d2h", pinned)
                + self.transfer_ms(h2d_bytes, "h2d", pinned)
                + self.transfer_ms(8 * res, "d2h", pinned))

    def bandwidth_gbps(self, direction: str, pinned: bool = False) -> float:
        k = f"{direction}_{'pinned' if pinned and self.pinned_available else 'pageable'}"
        return self.link_gbps[k]

    def transfer_ms(self, nbytes: float, direction: str, pinned: bool = False) -> float:
        """Host<->device copy time.  ``pinned`` falls back to pageable if unavailable."""
        if direction not in _DIRS:
            raise ValueError(f"direction must be one of {_DIRS}")
        if nbytes <= 0:
            return 0.0
        k = f"{direction}_{'pinned' if pinned and self.pinned_available else 'pageable'}"
        return self.link_latency_ms[k] + nbytes / (self.link_gbps[k] * 1e9) * 1e3

    def gpu_init_cost_ms(self) -> float:
        return self.gpu_init_ms if self.has_gpu else math.inf

    # -- online refinement ------------------------------------------------ #
    def observe(self, op: str, device: str, rows: float, ms: float,
                card: Optional[float] = None, persist: bool = True) -> None:
        with self._lock:
            op = OP_ALIASES.get(op, op)
            k = f"{op}@{device}"
            cf = self._card_fits(k)
            if cf:                             # route to the nearest measured group count
                q = DEFAULT_CARD if card is None else card
                same = [t for t in cf if bucket_name(t[0]) == bucket_name(q)] or cf
                near = min(same, key=lambda t: abs(math.log(t[0]) - math.log(max(q, 1))))[0]
                k = f"{k}:g{int(near)}"
            f = self.fits.get(k)
            if f is None:                     # new op: start from the generic prior if any
                g = self.fits.get(f"generic@{device}")
                f = Fit(g.alpha, g.c, g.g) if g else Fit(0.05, 1e-6)
                self.fits[k] = f
            f.observe(rows, ms, card)
            self._dirty += 1
            due = (self._dirty >= PERSIST_EVERY_N
                   or time.monotonic() - self._last_save >= PERSIST_EVERY_S)
        if persist and due:
            self.save()

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "device_name": self.device_name, "driver": self.driver,
            "mxframe_version": self.mxframe_version, "has_gpu": self.has_gpu,
            "cpu_threads": self.cpu_threads, "cpu_cores": self.cpu_cores,
            "pinned_available": self.pinned_available,
            "link_gbps": self.link_gbps, "link_latency_ms": self.link_latency_ms,
            "launch_overhead_ms": self.launch_overhead_ms,
            "gpu_init_ms": self.gpu_init_ms, "gpu_mem_bytes": self.gpu_mem_bytes,
            "gpu_init_breakdown_ms": self.gpu_init_breakdown_ms,
            "fits": {k: f.to_dict() for k, f in self.fits.items()},
            "source": self.source, "quick": self.quick, "created": self.created,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Calibration":
        base = default_calibration(_identity_from_dict(d))
        fits = dict(base.fits)
        fits.update({k: Fit.from_dict(v) for k, v in d["fits"].items()})
        link = {**base.link_gbps, **{k: float(v) for k, v in d["link_gbps"].items()}}
        lat = {**base.link_latency_ms, **{k: float(v) for k, v in d["link_latency_ms"].items()}}
        if not all(math.isfinite(v) and v > 0 for v in link.values()):
            raise ValueError("bad bandwidth")
        return cls(
            key=str(d["key"]), device_name=str(d["device_name"]), driver=str(d["driver"]),
            mxframe_version=str(d["mxframe_version"]), has_gpu=bool(d["has_gpu"]),
            cpu_threads=int(d["cpu_threads"]), cpu_cores=int(d.get("cpu_cores", 1)),
            pinned_available=bool(d["pinned_available"]), link_gbps=link, link_latency_ms=lat,
            launch_overhead_ms={**base.launch_overhead_ms,
                                **{k: float(v) for k, v in d["launch_overhead_ms"].items()}},
            gpu_init_ms=float(d["gpu_init_ms"]), gpu_mem_bytes=int(d.get("gpu_mem_bytes", 0)),
            fits=fits, source=str(d.get("source", "measured")), quick=bool(d.get("quick", True)),
            created=float(d.get("created", 0.0)), errors=dict(d.get("errors", {})),
            gpu_init_breakdown_ms=dict(d.get("gpu_init_breakdown_ms", {})),
        )

    def save(self, path: Optional[Path] = None) -> bool:
        """Merge this entry into the JSON file (atomic replace).  Never raises."""
        p = Path(path) if path else calibration_path()
        with self._lock:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with _file_lock(p):          # read-merge-write must not interleave across processes
                    doc = _read_doc(p) or {"schema": SCHEMA_VERSION, "entries": {}}
                    doc["entries"][self.key] = self.to_dict()
                    import tempfile
                    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".calib-", suffix=".tmp")
                    with os.fdopen(fd, "w") as f:
                        json.dump(doc, f, indent=1)
                    os.replace(tmp, p)
                self._dirty = 0
                self._last_save = time.monotonic()
                return True
            except Exception:
                return False

    def summary(self) -> str:
        L = [f"MXFrame calibration [{self.source}, {'quick' if self.quick else 'full'}]  {self.key}",
             f"  cpu threads={self.cpu_threads} (cores={self.cpu_cores}), gpu={self.has_gpu}, "
             f"pinned={self.pinned_available}, gpu_mem={self.gpu_mem_bytes / 2**30:.1f} GiB"]
        if self.has_gpu:
            L.append("  link GB/s: " + ", ".join(f"{k}={v:.1f}" for k, v in self.link_gbps.items()))
            L.append("  link latency ms: " + ", ".join(f"{k}={v:.3f}" for k, v in self.link_latency_ms.items()))
            bd = ", ".join(f"{k}={v:.0f}" for k, v in self.gpu_init_breakdown_ms.items())
            L.append(f"  gpu init (fresh process): {self.gpu_init_ms:.0f} ms" + (f"  ({bd})" if bd else ""))
        L.append("  launch overhead ms: " + ", ".join(f"{k}={v:.4f}" for k, v in self.launch_overhead_ms.items()))
        for k in sorted(self.fits):
            a, c, g = self.fits[k].params()
            p_ = self.fits[k].p
            L.append(f"  {k:<22} alpha={a:8.4f} ms  c={c * 1e6:9.4f} ms/Mrow"
                     + (f"  p={p_:.2f}" if p_ != 1.0 else "")
                     + (f"  g={g * 1e6:.4f} ms/Mgroup" if g else "") + f"  (pts={self.fits[k].n_points})")
        for k, v in self.errors.items():
            L.append(f"  ! {k}: {v}")
        return "\n".join(L)


class _file_lock:
    """Advisory exclusive lock on ``<file>.lock`` (fcntl); a no-op where unavailable."""

    def __init__(self, path: Path):
        self._lock_path = str(path) + ".lock"
        self._fd = None

    def __enter__(self):
        try:
            import fcntl
            self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)          # closing releases the flock
            self._fd = None
        return False


def _identity_from_dict(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if d.get("has_gpu"):
        return {"name": d["device_name"], "driver": d["driver"], "mem_bytes": d.get("gpu_mem_bytes", 0)}
    return None


# --------------------------------------------------------------------------- #
# Built-in defaults: RTX 3090 over PCIe + modern desktop CPU, conservative
# --------------------------------------------------------------------------- #
# (alpha_ms, c_ms_per_Mrow, g_ms_per_Mgroup).  Taken from a quick calibration on the
# reference box (RTX 3090, 24-core desktop, single-threaded AOT CPU kernels), GPU
# rows padded ~1.2x so the un-calibrated planner leans to CPU; GPU alpha (allocs,
# sync, D2H) is ~20-40x the CPU alpha, so small inputs always go to the CPU.
_DEFAULT_FITS: Dict[str, Tuple[float, float, float]] = {
    # GPU group ops after F-012 (grid-stride kernels, relaxed atomics): 30M rows, resident
    # input, 8..1.5M groups measured 0.1-0.5 ms per Mrow on a shared GPU; rows padded ~1.2x.
    "group_sum@cpu": (0.005, 0.8, 2.5), "group_sum@gpu": (0.05, 0.3, 0.0),
    "group_count@cpu": (0.005, 0.8, 2.5), "group_count@gpu": (0.05, 0.3, 0.0),
    "group_min@cpu": (0.005, 0.9, 10.0), "group_min@gpu": (0.1, 0.3, 0.0),
    "group_max@cpu": (0.005, 0.9, 12.0), "group_max@gpu": (0.1, 0.3, 0.0),
    "masked_sum@cpu": (0.005, 0.3, 0.0), "masked_sum@gpu": (0.05, 0.02, 0.0),   # F-013: block-reduced GPU kernel
    "join@cpu": (0.02, 11.0, 0.0), "join@gpu": (0.05, 0.25, 0.0),
    "gather@cpu": (0.005, 0.8, 0.0), "gather@gpu": (0.03, 0.04, 0.0),
    "sort_topk@cpu": (0.1, 7.0, 0.0), "sort_topk@gpu": (0.1, 2.5, 0.0),   # CPU: parallel radix; GPU: stable tiled bitonic
    "group_encode@cpu": (0.02, 9.0, 0.0), "group_encode@gpu": (1.0, 4.0, 0.0),
    "filter_take@cpu": (0.02, 4.5, 0.0),
    # join_host: host steps between the two GPU join kernels (cumsum, bincount, argsort)
    "join_host@cpu": (0.01, 1.2, 0.0),
    # pipeline: F-008 fused filter+expr+agg kernel, Q1-shaped (5 columns, 5 sums, 4 groups),
    # resident input (F-013 probe): CPU ~0.85 ms/Mrow on 24 threads, GPU ~0.26 ms/Mrow.
    "pipeline@cpu": (0.1, 0.9, 0.0), "pipeline@gpu": (0.1, 0.3, 0.0),
    "generic@cpu": (0.05, 6.0, 0.0), "generic@gpu": (1.0, 6.0, 0.0),
    # native executor (F-058; what device="cpu" runs), measured warm on the reference box, 24 threads, float64:
    # end to end through compute(device="cpu") on 2M / 8M / 32M in-memory rows. Group-by is keyed by group count
    # and a join by build-side rows (``card=``): both get slower per row once their table leaves the cache.
    "native_global@cpu": (0.3, 0.34, 0.0),               # one sum
    "native_filter@cpu": (0.3, 0.68, 0.0),               # filter + one sum (the filter alone is the difference)
    "native_group@cpu:g8": (0.5, 0.55, 0.0), "native_group@cpu:g1000": (0.5, 0.72, 0.0),
    "native_group@cpu:g100000": (0.5, 1.0, 0.0), "native_group@cpu:g1000000": (0.5, 9.5, 0.0),
    "native_join@cpu:g250000": (0.5, 0.9, 0.0), "native_join@cpu:g2000000": (0.5, 1.9, 0.0),  # rows = both sides
    "native_join@cpu:g8000000": (0.5, 5.2, 0.0),
}
_DEFAULT_GPU_INIT_MS = 400.0


def default_calibration(identity: Optional[Dict[str, Any]] = None,
                        version: Optional[str] = None) -> Calibration:
    has_gpu = identity is not None
    fits = {}
    for k, (a, c, g) in _DEFAULT_FITS.items():
        if k.endswith("@gpu") and not has_gpu:
            continue
        fits[k] = Fit(a, c / 1e6, g / 1e6)
    ver = version or _mxframe_version()
    return Calibration(
        key=machine_key(identity, ver),
        device_name=identity["name"] if identity else "cpu-only",
        driver=identity["driver"] if identity else "none",
        mxframe_version=ver, has_gpu=has_gpu,
        cpu_threads=_detect_cpu_threads(), cpu_cores=os.cpu_count() or 1,
        pinned_available=has_gpu,
        link_gbps={"h2d_pageable": 9.0, "h2d_pinned": 11.0, "d2h_pageable": 7.0, "d2h_pinned": 11.0},
        link_latency_ms={"h2d_pageable": 0.05, "h2d_pinned": 0.03,
                         "d2h_pageable": 0.05, "d2h_pinned": 0.03},
        launch_overhead_ms={"cpu": 0.005, "gpu": 0.1},
        gpu_init_ms=_DEFAULT_GPU_INIT_MS if has_gpu else 0.0,
        gpu_mem_bytes=int(identity.get("mem_bytes", 0)) if identity else 0,
        fits=fits, source="defaults", quick=True, created=0.0,
    )


# --------------------------------------------------------------------------- #
# Load / cache
# --------------------------------------------------------------------------- #
def _read_doc(p: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(p.read_text())
        if (isinstance(doc, dict) and doc.get("schema") == SCHEMA_VERSION
                and isinstance(doc.get("entries"), dict)):
            return doc
    except Exception:
        pass
    return None


def load_calibration(identity: Optional[Dict[str, Any]] = None,
                     path: Optional[Path] = None) -> Optional[Calibration]:
    """Entry for this machine from disk, or None (missing/corrupt/stale/other machine)."""
    doc = _read_doc(Path(path) if path else calibration_path())
    if not doc:
        return None
    entry = doc["entries"].get(machine_key(identity))
    if not entry:
        return None
    try:
        cal = Calibration.from_dict(entry)
    except Exception:
        return None
    if cal.cpu_threads != _detect_cpu_threads():
        # CPU kernels changed parallelism since measurement: CPU fits are stale.
        base = default_calibration(identity)
        for k in [k for k in cal.fits if k.endswith("@cpu")]:
            cal.fits[k] = base.fits.get(k, cal.fits[k])
        cal.cpu_threads = base.cpu_threads
        cal.source = "measured(gpu-only)"
    return cal


_CACHE: Optional[Calibration] = None
_CACHE_LOCK = threading.Lock()


def get_calibration() -> Calibration:
    """Load-or-default, cached in process.  Never raises."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            try:
                ident = _gpu_identity()
                _CACHE = load_calibration(ident) or default_calibration(ident)
            except Exception:
                _CACHE = default_calibration(None)
        return _CACHE


def reset_calibration_cache() -> None:
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


def estimate(op: str, device: str, rows: float, **shape: Any) -> float:
    return get_calibration().estimate(op, device, rows, **shape)


def transfer_ms(nbytes: float, direction: str, pinned: bool = False) -> float:
    return get_calibration().transfer_ms(nbytes, direction, pinned)


def observe(op: str, device: str, rows: float, ms: float, **shape: Any) -> None:
    card = shape.get("card", shape.get("cardinality"))
    get_calibration().observe(op, device, rows, ms, card=card)


# --------------------------------------------------------------------------- #
# Probe registry
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    """One benchmarkable (op, device).

    ``make(ctx, n, card)`` builds inputs (untimed) and returns a zero-arg callable
    that runs the kernel once (timed).  Update the body when a kernel changes.
    ``h2d_bytes_per_row`` > 0 means the wrapper re-uploads inputs on every call
    (no device-buffer cache), so calibrate subtracts that copy from the GPU fit
    (the planner adds transfers separately).  ``max_rows`` caps the size grid.
    ``uses_card``: fit the cardinality term.  ``bucketed``: fit separately per
    group-count bucket (see BUCKET_EDGES) instead of a single line.

    GPU probes time the BARE kernel on device-resident buffers (ctx.dev / ctx.dev_out)
    followed by a sync: no host copies, no dtype conversion, no host pre/post steps.
    """
    op: str
    device: str
    make: Callable[["ProbeCtx", int, int], Callable[[], Any]]
    h2d_bytes_per_row: float = 0.0
    max_rows: Optional[int] = None
    uses_card: bool = False
    enabled: bool = True             # False: skipped (defaults kept), noted in Calibration.errors
    max_card: Optional[int] = None   # GPU kernel's group-count limit, if any (see GPU_MAX_CARD)
    bucketed: bool = False
    two_d: bool = False              # points are (left_rows, right_rows, ms); fit alpha + a*L^p + b*R
    grid: Optional[Callable[[Tuple[int, ...]], List[Tuple[int, int]]]] = None   # explicit (rows, card|R) grid
    budget_s: Optional[float] = None  # per-probe override of the size-skip budget
    stat: str = "median"              # "min" for host-heavy end-to-end probes (least contended run)


PROBES: Dict[Tuple[str, str], Probe] = {}


def probe(op: str, device: str, **kw):
    def deco(fn):
        PROBES[(op, device)] = Probe(op, device, fn, **kw)
        return fn
    return deco


class ProbeCtx:
    """Lazily-created kernels and shared synthetic data (slices of one big array)."""

    def __init__(self, max_rows: int, seed: int = 0):
        self.max_rows = max_rows
        self.seed = seed
        self._cpu = None
        self._gpu = None
        self._data: Dict[str, Any] = {}
        self._dev_ptrs: List[int] = []

    @property
    def cpu(self):
        if self._cpu is None:
            from . import aot_kernels as ak
            self._cpu = ak.AOTKernels()
        return self._cpu

    @property
    def gpu(self):
        if self._gpu is None:
            from . import aot_kernels as ak
            # the process-wide instance (one CUDA driver / buffer cache); probes free their own buffers
            self._gpu = ak.get_aot_gpu()
            if self._gpu is None:
                raise RuntimeError("GPU AOT kernels unavailable")
        return self._gpu

    def dev(self, arr) -> int:
        """Upload ``arr`` once (untimed setup) and return its device pointer."""
        ptr = self.gpu._upload(arr)
        self._dev_ptrs.append(ptr)
        return ptr

    def dev_out(self, nbytes: int) -> int:
        ptr = self.gpu._cu.malloc(int(nbytes))
        self._dev_ptrs.append(ptr)
        return ptr

    def release_gpu_buffers(self) -> None:
        for dc in self._data.pop("pipe_dcs", []):      # resident pipeline columns
            try:
                dc.free()
            except Exception:
                pass
        if self._gpu is not None:
            for ptr in self._dev_ptrs:
                try:
                    self._gpu._cu.free(ptr)
                except Exception:
                    pass
            self._dev_ptrs.clear()
            try:
                self._gpu.clear_buf_cache()
            except Exception:
                pass

    def f32(self, n: int):
        import numpy as np
        if "f32" not in self._data:
            self._data["f32"] = np.random.default_rng(self.seed).random(self.max_rows, dtype=np.float32)
        return self._data["f32"][:n]

    def labels(self, n: int, card: int):
        """int32 group ids in [0, card).  Cached per card."""
        import numpy as np
        k = f"lab{card}"
        if k not in self._data:
            self._data[k] = np.random.default_rng(self.seed + 1).integers(
                0, card, self.max_rows, dtype=np.int32)
        return self._data[k][:n]

    def mask(self, n: int):
        import numpy as np
        if "mask" not in self._data:
            self._data["mask"] = (np.random.default_rng(self.seed + 2).random(self.max_rows) < 0.3
                                  ).astype(np.int32)
        return self._data["mask"][:n]

    def perm(self, n: int):
        import numpy as np
        if "perm" not in self._data:
            self._data["perm"] = np.random.default_rng(self.seed + 3).integers(
                0, self.max_rows, self.max_rows, dtype=np.int32)
        return self._data["perm"][:n] % max(n, 1)

    def keys(self, n: int):
        """int32 sort / encode keys (wide range)."""
        import numpy as np
        if "keys" not in self._data:
            self._data["keys"] = np.random.default_rng(self.seed + 4).integers(
                0, 1 << 30, self.max_rows, dtype=np.int32)
        return self._data["keys"][:n]

    def arrow_table(self, n: int):
        import pyarrow as pa
        return pa.table({"v": self.f32(n), "k": self.labels(n, 1000)})


def _cpu_group(method: str):
    def make(ctx: ProbeCtx, n: int, card: int):
        v, lab = ctx.f32(n), ctx.labels(n, card)
        m = getattr(ctx.cpu, method)
        return lambda: m(v, lab, card)
    return make


def _gpu_group(attr: str):
    def make(ctx: ProbeCtx, n: int, card: int):
        g = ctx.gpu
        dv, dl, out = ctx.dev(ctx.f32(n)), ctx.dev(ctx.labels(n, card)), ctx.dev_out(card * 4)
        fn, cu = getattr(g, attr), g._cu

        def run():
            fn(g._ptr(out), g._ptr(dv), g._ptr(dl), n, card)
            cu.sync()
        return run
    return make


for _op, _m, _a in (("group_sum", "group_sum_f32", "_group_sum"), ("group_min", "group_min_f32", "_group_min"),
                    ("group_max", "group_max_f32", "_group_max")):
    PROBES[(_op, "cpu")] = Probe(_op, "cpu", _cpu_group(_m), uses_card=True)
    PROBES[(_op, "gpu")] = Probe(_op, "gpu", _gpu_group(_a), uses_card=True, bucketed=True,
                                 max_card=GPU_MAX_CARD.get(_op))


@probe("group_count", "cpu", uses_card=True)
def _p_gc_cpu(ctx, n, card):
    lab = ctx.labels(n, card)
    return lambda: ctx.cpu.group_count_f32(lab, card)


@probe("group_count", "gpu", uses_card=True, bucketed=True)
def _p_gc_gpu(ctx, n, card):
    g = ctx.gpu
    dl, out = ctx.dev(ctx.labels(n, card)), ctx.dev_out(card * 4)

    def run():
        g._group_count(g._ptr(out), g._ptr(dl), n, card)
        g._cu.sync()
    return run


@probe("masked_sum", "cpu")
def _p_ms_cpu(ctx, n, card):
    v, m = ctx.f32(n), ctx.mask(n)
    return lambda: ctx.cpu.masked_global_sum_f32(v, m)


@probe("masked_sum", "gpu")
def _p_ms_gpu(ctx, n, card):
    g = ctx.gpu
    dv, dm, out = ctx.dev(ctx.f32(n)), ctx.dev(ctx.mask(n)), ctx.dev_out(4)

    def run():
        g._masked_sum(g._ptr(out), g._ptr(dv), g._ptr(dm), n)
        g._cu.sync()
    return run


def _join_inputs(ctx, n):
    import numpy as np
    m = max(n // 8, 1)
    left = (ctx.labels(n, 1 << 30) % m).astype(np.int32)     # left keys in [0, m)
    right = np.arange(m, dtype=np.int32)                    # unique right keys
    return left, right


@probe("join", "cpu", max_rows=30_000_000)
def _p_join_cpu(ctx, n, card):
    left, right = _join_inputs(ctx, n)

    def run():
        cnt = ctx.cpu.join_count(left, right)
        return ctx.cpu.join_scatter(left, right, cnt)
    return run


@probe("join", "gpu", max_rows=30_000_000)
def _p_join_gpu(ctx, n, card):
    """Bare join kernels: count phase + scatter phase on resident buffers.  The host
    steps between them (cumsum/bincount/argsort, counts D2H) are the `join_host` term."""
    import numpy as np
    g = ctx.gpu
    left, right = _join_inputs(ctx, n)
    m = len(right)
    counts = np.bincount(right, minlength=m).astype(np.int32)[left]        # setup, untimed
    offsets = np.zeros(n, dtype=np.int32)
    np.cumsum(counts[:-1], out=offsets[1:])
    right_order = np.argsort(right, kind="stable").astype(np.int32)
    right_starts = np.zeros(m, dtype=np.int32)
    rc = np.bincount(right, minlength=m).astype(np.int32)
    np.cumsum(rc[:-1], out=right_starts[1:])
    total = int(counts.sum())
    lk, rk = ctx.dev(left), ctx.dev(right)
    table, cnt = ctx.dev(np.zeros(m, dtype=np.int32)), ctx.dev_out(n * 4)
    d_off, d_ord, d_st, d_rc = (ctx.dev(offsets), ctx.dev(right_order), ctx.dev(right_starts), ctx.dev(rc))
    lo, ro = ctx.dev_out(max(total, 1) * 4), ctx.dev_out(max(total, 1) * 4)
    P = g._ptr

    def run():
        # (table is not re-zeroed between calls: only timing matters; scatter inputs are fixed)
        g._join_count_gpu(P(cnt), P(table), P(lk), P(rk), n, m, m)
        g._join_scatter_gpu(P(lo), P(ro), P(lk), P(d_off), P(d_ord), P(d_st), P(d_rc), n, m)
        g._cu.sync()
    return run


def _join_total_grid(sizes):
    """(left_rows, right_rows): ratios 1:8 and 1:1; 1:1 stops early (it is ~4x slower)."""
    g = []
    for n in sizes:
        if n < 100_000 or n > 30_000_000:
            continue
        g.append((n, max(n // 8, 1)))
        if n <= 6_000_000:
            g.append((n, n))
    return g


@probe("join_total", "gpu", two_d=True, grid=_join_total_grid, budget_s=3.0, stat="min")
def _p_join_total_gpu(ctx, n, card):
    """END-TO-END `AOTKernelsGPU.hash_join` with resident key buffers (keys are uploaded once
    in the untimed warm-up and served from the device-buffer cache).  Includes everything the
    wrapper does: table upload, counts D2H, host bincount/argsort/cumsum, aux uploads,
    scatter, result D2H.  Excludes the key upload.  card = right rows; ~1 match per left row."""
    import numpy as np
    g = ctx.gpu
    right = np.arange(card, dtype=np.int32)
    left = (ctx.labels(n, 1 << 30) % max(card, 1)).astype(np.int32)
    return lambda: g.hash_join(left, right)


@probe("join_host", "cpu", max_rows=30_000_000)
def _p_join_host(ctx, n, card):
    """Host work the GPU join does between its two kernels (rows = left rows)."""
    import numpy as np
    left, right = _join_inputs(ctx, n)
    counts = np.ones(n, dtype=np.int32)

    def run():
        offsets = np.zeros(n, dtype=np.int32)
        np.cumsum(counts[:-1], out=offsets[1:])
        rc = np.bincount(right, minlength=len(right)).astype(np.int32)
        np.argsort(right, kind="stable")
        starts = np.zeros(len(right), dtype=np.int32)
        np.cumsum(rc[:-1], out=starts[1:])
    return run


# --- fused pipeline (F-008): Q1-shaped program ----------------------------------------
# filter (shipdate <= cutoff, ~95% of rows pass) + 5 float64 sums over 4 float32 columns
# (sum(qty), sum(price), sum(price*(1-disc)), sum(price*(1-disc)*(1+tax)), sum(disc)) into
# 4 groups (returnflag x linestatus).  coproc_fused balances the CPU/GPU row split from
# these two fits.
PIPELINE_GROUPS = 4


def _pipeline_case(ctx, n):
    """(program, [columns in program order], gids) for the first ``n`` rows; built once."""
    import numpy as np
    import pyarrow as pa
    from .lazy_expr import col, lit
    from .pipeline_program import compile_pipeline
    d = ctx._data
    if "pipe" not in d:
        f = ctx.f32(ctx.max_rows)
        cols = {
            "l_shipdate": (f * np.float32(2556)).astype(np.int32),
            "l_quantity": f * np.float32(50) + np.float32(1),
            "l_extendedprice": f * np.float32(1e5) + np.float32(900),
            "l_discount": f * np.float32(0.1),
            "l_tax": f[::-1] * np.float32(0.08),
        }
        disc_price = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
        aggs = [col("l_quantity").sum().alias("sum_qty"), col("l_extendedprice").sum().alias("sum_base"),
                disc_price.sum().alias("sum_disc_price"),
                (disc_price * (lit(1.0) + col("l_tax"))).sum().alias("sum_charge"),
                col("l_discount").sum().alias("sum_disc")]
        head = pa.table({k: pa.array(v[:64]) for k, v in cols.items()})
        prog = compile_pipeline([col("l_shipdate") <= lit(2436)], aggs, head)
        if prog is None:
            raise RuntimeError("Q1-shaped program did not compile")
        d["pipe"] = (prog, [cols[c] for c in prog.columns],
                     ctx.labels(ctx.max_rows, PIPELINE_GROUPS))
    prog, cols, gids = d["pipe"]
    return prog, [c[:n] for c in cols], gids[:n]


@probe("pipeline", "cpu")
def _p_pipe_cpu(ctx, n, card):
    from . import pipeline_kernels as pk
    prog, cols, gids = _pipeline_case(ctx, n)
    return lambda: pk.run_pipeline_cpu(prog, cols, gids, PIPELINE_GROUPS, 0, n)


@probe("pipeline", "gpu")
def _p_pipe_gpu(ctx, n, card):
    """Compute-only: columns and group ids are resident (uploaded here, untimed); the timed
    call is the kernel launch plus the tiny per-group partials D2H."""
    from . import pipeline_kernels as pk
    prog, cols, gids = _pipeline_case(ctx, n)
    dc = pk.upload_columns(prog, cols, gids, 0, n)
    ctx._data.setdefault("pipe_dcs", []).append(dc)
    return lambda: pk.run_pipeline_gpu(prog, dc, None, PIPELINE_GROUPS, 0, n)


@probe("gather", "cpu")
def _p_gather_cpu(ctx, n, card):
    v, idx = ctx.f32(n), ctx.perm(n)
    return lambda: ctx.cpu.gather_f32(v, idx)


@probe("gather", "gpu")
def _p_gather_gpu(ctx, n, card):
    g = ctx.gpu
    ds, di, out = ctx.dev(ctx.f32(n)), ctx.dev(ctx.perm(n)), ctx.dev_out(n * 4)

    def run():
        g._gather_f32(g._ptr(out), g._ptr(ds), g._ptr(di), n)
        g._cu.sync()
    return run


@probe("sort_topk", "cpu")
def _p_sort_cpu(ctx, n, card):
    # The CPU sort kernel: `AOTKernels.sort_indices`, the parallel stable LSD radix sort
    # (F-012), on the int32 composite key like the GPU probe below.
    keys = ctx.keys(n)
    return lambda: ctx.cpu.sort_indices(keys)


@probe("sort_topk", "gpu", max_rows=30_000_000)
def _p_sort_gpu(ctx, n, card):
    g = ctx.gpu
    dk, out = ctx.dev(ctx.keys(n)), ctx.dev_out(n * 4)

    def run():
        g._sort_topk_gpu(g._ptr(out), g._ptr(dk), n, 0)      # full bitonic sort, result stays on device
        g._cu.sync()
    return run


@probe("group_encode", "cpu")
def _p_enc_cpu(ctx, n, card):
    import pyarrow as pa
    arr = pa.array(ctx.labels(n, card))
    return lambda: arr.dictionary_encode()


@probe("group_encode", "gpu")
def _p_enc_gpu(ctx, n, card):
    k = ctx.labels(n, card)
    cap = 1 << max(1, (2 * n - 1).bit_length())    # engine rule: next pow2 >= 2*rows
    return lambda: ctx.gpu.group_encode_i32(k, cap)


@probe("filter_take", "cpu")
def _p_ft_cpu(ctx, n, card):
    import pyarrow as pa
    import pyarrow.compute as pc
    t = ctx.arrow_table(n)
    return lambda: t.filter(pc.less(t["v"], pa.scalar(0.3, pa.float32())))


# Native executor (F-058): what device="cpu" really runs. One plan per operator over in-memory Arrow
# tables, timed end to end through `compute(device="cpu")` (fused scan + operator + result). auto prices
# the CPU side of a plan with these; the kernel probes above price the GPU path and the co-processing split.
def _native_table(ctx, n, card=None):
    import pyarrow as pa
    cols = {"v": ctx.f32(n), "w": ctx.f32(n)[::-1].copy(), "m": ctx.mask(n)}
    if card is not None:
        cols["k"] = ctx.labels(n, card)
    return pa.table(cols)


def _native_run(lf):
    return lambda: lf.compute(device="cpu")


@probe("native_global", "cpu")
def _p_native_global(ctx, n, card):
    from .lazy_frame import LazyFrame
    from .lazy_expr import col
    return _native_run(LazyFrame(_native_table(ctx, n)).groupby().agg(col("v").sum().alias("s")))


@probe("native_filter", "cpu")
def _p_native_filter(ctx, n, card):
    """Filter + one sum: the filter's own price is this minus native_global."""
    from .lazy_frame import LazyFrame
    from .lazy_expr import col, lit
    return _native_run(LazyFrame(_native_table(ctx, n)).filter(col("w") < lit(0.3))
                       .groupby().agg(col("v").sum().alias("s")))


@probe("native_group", "cpu", uses_card=True, bucketed=True)
def _p_native_group(ctx, n, card):
    from .lazy_frame import LazyFrame
    from .lazy_expr import col
    return _native_run(LazyFrame(_native_table(ctx, n, card)).groupby("k").agg(col("v").sum().alias("s")))


@probe("native_join", "cpu", max_rows=30_000_000)
def _p_native_join(ctx, n, card):
    """Key / foreign-key join + one sum; `n` counts both sides (8/9 probe rows, 1/9 build rows)."""
    import numpy as np
    import pyarrow as pa
    from .lazy_frame import LazyFrame
    from .lazy_expr import col
    m = max(n // 9, 1)
    left = n - m
    fact = pa.table({"fk": (ctx.labels(left, 1 << 30) % m).astype(np.int64), "v": ctx.f32(left)})
    dim = pa.table({"pk": np.random.default_rng(ctx.seed + 5).permutation(m).astype(np.int64),
                    "w": ctx.f32(m)})
    return _native_run(LazyFrame(fact).join(LazyFrame(dim), left_on="fk", right_on="pk")
                       .groupby().agg((col("v") * col("w")).sum().alias("s")))


# --------------------------------------------------------------------------- #
# Measurement machinery
# --------------------------------------------------------------------------- #
def _time_call(fn: Callable[[], Any], timer: Callable[[], float], budget_s: float,
               stat: str = "median") -> float:
    """Median-of-few (or min, for host-heavy probes sensitive to machine noise) wall time in ms;
    the warm-up call is excluded."""
    fn()
    t = timer(); fn(); first = timer() - t
    samples = [first]
    reps = 2 if first > 0.05 else (3 if first > 0.01 else 7)
    if stat == "min":
        reps = max(reps, 3)
    for _ in range(reps - 1):
        t = timer(); fn(); samples.append(timer() - t)
    samples.sort()
    return (samples[0] if stat == "min" else samples[len(samples) // 2]) * 1e3


def _card_for(n: int) -> int:
    return max(2, min(1000, n // 4))


def _collect_points(p: Probe, ctx: Any, sizes: Tuple[int, ...], budget_s: float,
                    timer: Callable[[], float], h2d_gbps: float,
                    progress: Optional[Callable[[int], None]] = None) -> List[List[float]]:
    """Time one probe over the size grid -> [[rows, card, ms], ...].  May raise.

    ``progress(n)`` is called after each finished size (keeps the child watchdog fed)."""
    pts: List[List[float]] = []
    last: Dict[int, Tuple[int, float]] = {}          # per group count: (rows, ms) of last run
    sizes_ok = [n for n in sizes if not (p.max_rows and n > p.max_rows)]
    cap = p.max_card or (1 << 30)
    if p.bucketed:                        # every size at one representative card per bucket
        cards = [c for c in BUCKET_REP_CARDS if c <= cap]
        big = [n for n in sizes_ok if n >= 100_000] or sizes_ok
        low_card_sizes = big[:-1] if len(big) >= 4 else big    # top size is slow for mid group counts
        grid = [(n, c) for c in cards
                for n in (big if c >= 100_000 else low_card_sizes) if c <= max(n, 8)]
    elif p.grid is not None:
        grid = p.grid(tuple(sizes_ok))
        budget_s = p.budget_s or budget_s
    else:
        grid = [(n, _card_for(n)) for n in sizes_ok]
        if p.uses_card:                   # cardinality sweep at a mid size
            mid = min((n for n in sizes if n >= 1_000_000), default=sizes[-1])
            if not (p.max_rows and mid > p.max_rows):
                grid += [(mid, c) for c in (8, 200_000) if c <= cap]
    for n, card in grid:
        lk = round(card / n, 3) if p.two_d else card       # series key: same ratio / same group count
        prev = last.get(lk)
        if prev is not None and n > prev[0] and prev[1] / 1e3 * (n / prev[0]) * 4 > budget_s:
            continue                      # projected too slow for the budget: skip this size
        try:
            ms = _time_call(p.make(ctx, n, card), timer, budget_s, p.stat)
        except BaseException:
            if p.device == "gpu" and hasattr(ctx, "release_gpu_buffers"):
                ctx.release_gpu_buffers()      # a failed probe must not keep resident buffers
            raise
        if p.h2d_bytes_per_row and h2d_gbps > 0:      # wrapper re-uploads inputs: remove that copy
            ms = max(ms - p.h2d_bytes_per_row * n / (h2d_gbps * 1e9) * 1e3, ms * 0.05)
        pts.append([float(n), float(card), ms])
        if prev is None or n >= prev[0]:
            last[lk] = (n, ms)
        if p.device == "gpu" and hasattr(ctx, "release_gpu_buffers"):
            ctx.release_gpu_buffers()
        if progress:
            progress(n)
    return pts


def _fit_scaling(sub: List[List[float]]) -> Optional[Fit]:
    """alpha + c * rows**p, tuned for where the planner decides (large inputs).

    alpha comes from the smallest size; (c, p) from points with rows >= 1M, weighted by
    absolute error so the biggest sizes dominate.  p > 1 only when the data show it
    (cache misses, n log^2 n sorts); otherwise the fit stays linear.
    """
    import numpy as np
    by_n: Dict[float, List[float]] = {}
    for n, _, ms in sub:
        by_n.setdefault(float(n), []).append(float(ms))
    xs = sorted(by_n)
    if len(xs) < 3:
        return None
    ys = {n: sorted(v)[len(v) // 2] for n, v in by_n.items()}
    large = [n for n in xs if n >= 1_000_000]
    if len(large) < 2:
        large = xs[-3:]
    alpha = ys[xs[0]]
    for _ in range(2):                      # refine alpha: subtract the small-size slope share
        r = np.array([ys[n] - alpha for n in large])
        ok = r > 1e-9
        if ok.sum() < 2:
            break
        nl = np.array(large)[ok]
        slope, p = _power_fit(nl, r[ok])
        alpha = max(0.0, ys[xs[0]] - slope * xs[0] ** p)
    r = np.array([ys[n] - alpha for n in large])
    ok = r > 1e-9
    if ok.sum() < 2:
        c, p = max((ys[xs[-1]] - alpha) / xs[-1], 0.0), 1.0
    else:
        c, p = _power_fit(np.array(large)[ok], r[ok])
    return Fit(alpha, c, 0.0, n_points=len(sub), p=p, n_max=xs[-1])


def _power_fit(n, r) -> Tuple[float, float]:
    """r ~ c * n**p on the given (large) points; p in [1, 1.5], snapped to 1 below 1.12."""
    import numpy as np
    p = float(np.polyfit(np.log(n), np.log(r), 1, w=np.sqrt(n / n.max()))[0])   # biggest sizes dominate
    p = 1.0 if p < 1.12 else min(p, 1.5)
    c = float((n ** p * r).sum() / (n ** (2 * p)).sum())      # absolute-error weighting
    return c, p


def _fit_join2d(pts: List[List[float]]) -> Optional[Fit]:
    """time = alpha + a * L**p + b * R over (L=left rows, R=right rows) points.

    The right-row term rides in Fit.g (predict(L, card=R)).  p is chosen by grid search
    to minimise the worst relative error on the large points; coefficients >= 0."""
    import numpy as np
    if len({q[0] for q in pts}) < 3 or len({round(q[1] / q[0], 3) for q in pts}) < 2:
        return None
    L = np.array([q[0] for q in pts]); R = np.array([q[1] for q in pts]); y = np.array([q[2] for q in pts])
    big = L >= 1_000_000
    if big.sum() < 3:
        big = np.ones(len(L), dtype=bool)
    best = None
    for p in np.arange(1.0, 1.51, 0.05):
        cols = {"a": np.ones(len(L)), "c": L ** p / 1e6 ** p, "g": R / 1e6}
        active = ["a", "c", "g"]
        w = 1.0 / np.sqrt(np.maximum(y, 1e-3))
        coef = {"a": 0.0, "c": 0.0, "g": 0.0}
        while active:
            A = np.stack([cols[k] for k in active], axis=1) * w[:, None]
            sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
            if (sol >= 0).all():
                coef.update({k: float(v) for k, v in zip(active, sol)})
                break
            active.remove(active[int(np.argmin(sol))])
        pred = coef["a"] + coef["c"] * cols["c"] + coef["g"] * cols["g"]
        err = float(np.max(np.abs(pred[big] - y[big]) / y[big]))
        if best is None or err < best[0] - 1e-9:
            best = (err, float(p), coef)
    _, p, coef = best
    return Fit(coef["a"], coef["c"] / 1e6 ** p, coef["g"] / 1e6, n_points=len(pts), p=p, n_max=float(L.max()))


def _fit_points(pts: List[List[float]], uses_card: bool, bucketed: bool = False,
                two_d: bool = False) -> Tuple[Dict[str, Fit], Optional[str]]:
    """-> ({"": base fit, ":g<groups>": per-group-count fit, ...}, error).  {} on failure."""
    def line(sub, card_term):
        if len({q[0] for q in sub}) < 3:
            return None
        if card_term:                      # CPU group ops: cheap linear + per-group term
            a, c, g = _nnls_rel([tuple(q) for q in sub], len({q[1] for q in sub}) >= 3)
            return Fit(a, c, g, n_points=len(sub))
        return _fit_scaling(sub)

    if two_d:
        f = _fit_join2d(pts)
        return ({"": f}, None) if f else ({}, "too few (rows, ratio) points within time budget")
    if not bucketed:
        f = line(pts, uses_card)
        return ({"": f}, None) if f else ({}, "too few sizes within time budget")
    fits: Dict[str, Fit] = {}
    for card in sorted({q[1] for q in pts}):
        f = line([q for q in pts if q[1] == card], False)
        if f:
            fits[f":g{int(card)}"] = f
    if not fits:
        return {}, "too few sizes within time budget"
    near = min(fits, key=lambda k: abs(math.log(int(k[2:])) - math.log(DEFAULT_CARD)))
    fits[""] = fits[near]
    return fits, None


def _store_fits(cal: "Calibration", op: str, dev: str, fits: Dict[str, Fit]) -> None:
    for suffix, f in fits.items():
        cal.fits[f"{op}@{dev}{suffix}"] = f


# -- link + GPU init ------------------------------------------------------- #
def measure_link(sizes_mb: Tuple[int, ...], timer: Callable[[], float] = time.perf_counter
                 ) -> Dict[str, Any]:
    """H2D/D2H bandwidth + latency, pageable and pinned (CUDA driver API)."""
    import numpy as np
    from . import aot_kernels as ak
    cu = ak._get_cuda_driver()
    if cu is None:
        raise RuntimeError("CUDA driver unavailable")
    lib = cu._lib
    pinned_ok = False
    try:
        lib.cuMemAllocHost_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cuMemAllocHost_v2.restype = ctypes.c_int
        lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        lib.cuMemFreeHost.restype = ctypes.c_int
        pinned_ok = True
    except AttributeError:
        pass

    def alloc_pinned(nbytes):
        ptr = ctypes.c_void_p()
        if lib.cuMemAllocHost_v2(ctypes.byref(ptr), nbytes) != 0:
            return None, None
        buf = (ctypes.c_char * nbytes).from_address(ptr.value)
        return ptr, np.frombuffer(buf, dtype=np.uint8)

    dev = cu.malloc(max(sizes_mb) * 2**20)
    res: Dict[str, List[Tuple[float, float, float]]] = {
        f"{d}_{k}": [] for d in _DIRS for k in ("pageable", "pinned")}
    pinned_seen = False
    try:
        for mb in sizes_mb:
            nbytes = mb * 2**20
            page = np.empty(nbytes, dtype=np.uint8)
            page[:] = 1                          # touch pages so we don't time page faults
            host = {"pageable": (None, page)}
            if pinned_ok:
                ptr, arr = alloc_pinned(nbytes)
                if arr is not None:
                    arr[:] = 1
                    host["pinned"] = (ptr, arr)
                    pinned_seen = True
            for kind, (_, arr) in host.items():
                # D2H into a FRESH np.empty (what the engine does) is ~1.45x slower than into
                # pre-touched pages (page faults); measure pageable D2H that way.
                fresh = kind == "pageable"
                for d, fn in (("h2d", lambda a: cu.h2d(dev, a)),
                              ("d2h", (lambda a: cu.d2h(np.empty(a.nbytes, np.uint8), dev)) if fresh
                               else (lambda a: cu.d2h(a, dev)))):
                    fn(arr)                     # warm-up
                    ts = []
                    for _ in range(3):
                        t = timer(); fn(arr); ts.append(timer() - t)
                    res[f"{d}_{kind}"].append((float(nbytes), 0.0, sorted(ts)[1] * 1e3))
            for _, (ptr, _) in host.items():
                if ptr is not None:
                    lib.cuMemFreeHost(ptr)
    finally:
        cu.free(dev)
    gbps: Dict[str, float] = {}
    lat: Dict[str, float] = {}
    for k, pts in res.items():
        if not pts:
            continue
        a, c, _ = _nnls_rel(pts, False)         # ms = latency + bytes * c
        gbps[k] = 1.0 / (c * 1e6) if c > 0 else 1.0 / (min(ms / b for b, _, ms in pts) * 1e6)
        lat[k] = a
    return {"gbps": gbps, "latency_ms": lat, "pinned_available": pinned_seen}


_GPU_INIT_SCRIPT = r"""
import importlib.util, json, sys, time
import numpy as np
spec = importlib.util.spec_from_file_location("_mx_aot_probe", sys.argv[1])
m = importlib.util.module_from_spec(spec)
t0 = time.perf_counter()
spec.loader.exec_module(m)
t1 = time.perf_counter()
k = m.AOTKernelsGPU()
t2 = time.perf_counter()
k.masked_global_sum_f32(np.ones(64, np.float32), np.ones(64, np.int32))
t3 = time.perf_counter()
print(json.dumps({"import_ms": (t1-t0)*1e3, "context_ms": (t2-t1)*1e3, "first_kernel_ms": (t3-t2)*1e3}))
"""


def measure_gpu_init(timeout_s: float = 90.0) -> Dict[str, float]:
    """Fresh-process GPU init: CUDA ctx + GPU AOT library load + first kernel launch
    (Mojo DeviceContext creation).  Excludes interpreter/numpy start-up."""
    aot = Path(__file__).with_name("aot_kernels.py")
    import subprocess
    out = subprocess.run([sys.executable, "-c", _GPU_INIT_SCRIPT, str(aot)],
                         capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy())
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "gpu init failed")
    d = json.loads(out.stdout.strip().splitlines()[-1])
    d["total_ms"] = d["context_ms"] + d["first_kernel_ms"]
    return d


# -- GPU measurement: isolated in a child process ---------------------------- #
# A CUDA fault (e.g. illegal address in a kernel at an extreme shape) poisons the
# whole context, so GPU probes never run in the caller's process.  The child emits
# one JSON line per finished item; whatever it completed before dying is kept.
def _measure_gpu_local(emit: Callable[[Dict[str, Any]], None], registry: Dict[Tuple[str, str], Probe],
                       ctx: Any, sizes: Tuple[int, ...], budget_s: float,
                       link_sizes_mb: Tuple[int, ...], timer: Callable[[], float],
                       link: Optional[Callable[[Tuple[int, ...]], Dict[str, Any]]] = None) -> None:
    h2d = 0.0
    try:
        r = (link or (lambda mb: measure_link(mb, timer)))(link_sizes_mb)
        h2d = r["gbps"].get("h2d_pageable", 0.0)
        emit({"kind": "link", **r})
    except Exception as exc:
        emit({"kind": "error", "what": "link", "msg": f"{type(exc).__name__}: {exc}"})
    for (op, dev), p in registry.items():
        if dev != "gpu":
            continue
        if not p.enabled:
            emit({"kind": "error", "what": f"{op}@gpu", "msg": "probe disabled (kernel unusable); defaults kept"})
            continue
        try:
            emit({"kind": "points", "op": op, "points": _collect_points(
                p, ctx, sizes, budget_s, timer, h2d, progress=lambda n: emit({"kind": "progress", "n": n}))})
        except Exception as exc:
            emit({"kind": "error", "what": f"{op}@gpu", "msg": f"{type(exc).__name__}: {exc}"})
    p = registry.get(("masked_sum", "gpu"))
    if p is not None:
        try:
            emit({"kind": "launch", "ms": _time_call(p.make(ctx, 64, 8), timer, 0.05)})
        except Exception as exc:
            emit({"kind": "error", "what": "launch@gpu", "msg": f"{type(exc).__name__}: {exc}"})


def _gpu_worker_main() -> None:      # pragma: no cover - runs in the child process
    args = json.loads(sys.stdin.read())
    ctx = ProbeCtx(max(args["sizes"]))
    _measure_gpu_local(lambda d: print(json.dumps({**d, "t": time.time()}), flush=True), PROBES, ctx,
                       tuple(args["sizes"]), args["budget_s"], tuple(args["link_mb"]),
                       time.perf_counter)


def _run_gpu_subprocess(sizes: Tuple[int, ...], budget_s: float, link_mb: Tuple[int, ...],
                        item_timeout_s: Optional[float] = None, total_timeout_s: float = 1800.0,
                        child_cmd: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run the GPU probes in a child.  The watchdog kills it after ``item_timeout_s``
    without any output line (the child emits one per finished size, so slow 1e8-row
    steps in full mode do not trip it; a hung kernel does).  ``child_cmd`` is a test hook."""
    import queue
    import subprocess
    if item_timeout_s is None:
        item_timeout_s = max(45.0, 20.0 * budget_s)
    pkg = __package__ or "mxframe"
    code = f"import {pkg}.calibration as c; c._gpu_worker_main()"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    items: List[Dict[str, Any]] = []
    proc = subprocess.Popen(child_cmd or [sys.executable, "-c", code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    q: "queue.Queue[Optional[str]]" = queue.Queue()
    threading.Thread(target=lambda: ([q.put(l) for l in proc.stdout], q.put(None)), daemon=True).start()
    err_lines: List[str] = []
    threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True).start()
    proc.stdin.write(json.dumps({"sizes": list(sizes), "budget_s": budget_s, "link_mb": list(link_mb)}))
    proc.stdin.close()
    t_end = time.monotonic() + total_timeout_s
    problem = None
    while True:
        try:
            line = q.get(timeout=min(item_timeout_s, max(0.1, t_end - time.monotonic())))
        except queue.Empty:
            problem = "no progress (kernel hang?)" if time.monotonic() < t_end else "timeout"
            proc.kill()
            break
        if line is None:
            break
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("kind") != "progress":
            items.append(item)
    rc = proc.wait()
    if problem is None and rc != 0:
        tail = "".join(err_lines).strip().splitlines()
        problem = tail[-1] if tail else f"exit {rc}"
    if problem:
        items.append({"kind": "crash", "what": "gpu", "msg": f"GPU probe process stopped: {problem}"})
    return items


# --------------------------------------------------------------------------- #
# calibrate()
# --------------------------------------------------------------------------- #
def calibrate(full: bool = False, *, save: bool = True, verbose: bool = False,
              registry: Optional[Dict[Tuple[str, str], Probe]] = None,
              gpu_runner: Optional[Callable[[Tuple[int, ...], float, Tuple[int, ...]],
                                            List[Dict[str, Any]]]] = None,
              gpu_init: Optional[Callable[[], Dict[str, float]]] = None,
              identity: Any = "auto",
              timer: Callable[[], float] = time.perf_counter,
              ctx: Any = None, path: Optional[Path] = None) -> Calibration:
    """Benchmark this machine and return (and by default persist) a Calibration.

    Quick mode (default) targets < 30 s; ``full=True`` adds sizes up to 1e8 rows and
    a 1 GiB link test.  GPU probes run in a child process (a CUDA fault cannot take
    the caller down).  Never raises for missing GPUs or broken kernels: affected
    entries keep the built-in defaults and are listed in ``errors``.

    Keyword-only hooks (registry, gpu_runner, gpu_init, identity, timer, ctx, path)
    exist so tests can run deterministically with fake probes and timers.
    """
    global _CACHE
    ident = _gpu_identity() if isinstance(identity, str) else identity
    cal = default_calibration(ident)
    cal.source, cal.quick, cal.created = "measured", not full, time.time()
    sizes = FULL_SIZES if full else QUICK_SIZES
    link_mb = FULL_LINK_MB if full else QUICK_LINK_MB
    budget = FULL_OP_BUDGET_S if full else QUICK_OP_BUDGET_S
    registry = PROBES if registry is None else registry
    _t0 = time.perf_counter()
    log = (lambda s: print(f"[{time.perf_counter() - _t0:6.1f}s] {s}", file=sys.stderr, flush=True)) if verbose else (lambda s: None)
    ctx = ProbeCtx(max(sizes)) if ctx is None else ctx

    # CPU probes: in-process.
    for (op, dev), p in registry.items():
        if dev != "cpu":
            continue
        if not p.enabled:
            cal.errors[f"{op}@cpu"] = "probe disabled (kernel unusable); defaults kept"
            continue
        try:
            fits, err = _fit_points(_collect_points(p, ctx, sizes, budget, timer, 0.0),
                                    p.uses_card, p.bucketed, p.two_d)
        except Exception as exc:
            fits, err = {}, f"{type(exc).__name__}: {exc}"
        if fits:
            _store_fits(cal, op, "cpu", fits)
            fit = fits[""]
            log(f"{op}@cpu: alpha={fit.alpha:.4f} c={fit.c * 1e6:.3f} ms/Mrow")
        else:
            cal.errors[f"{op}@cpu"] = err or "unknown"
    p = registry.get(("masked_sum", "cpu"))
    if p is not None:
        try:
            cal.launch_overhead_ms["cpu"] = _time_call(p.make(ctx, 64, 8), timer, 0.05)
        except Exception as exc:
            cal.errors["launch@cpu"] = f"{type(exc).__name__}: {exc}"

    # GPU: fresh-process init cost, then link + kernel probes in a child process.
    if ident is not None:
        try:
            gi = (gpu_init or measure_gpu_init)()
            cal.gpu_init_ms = float(gi["total_ms"])
            cal.gpu_init_breakdown_ms = {k: float(v) for k, v in gi.items() if k != "total_ms"}
            log(f"gpu init: {cal.gpu_init_ms:.0f} ms")
        except Exception as exc:
            cal.errors["gpu_init"] = f"{type(exc).__name__}: {exc}"
        runner = gpu_runner or _run_gpu_subprocess
        for item in runner(sizes, budget, link_mb):
            kind = item.get("kind")
            if kind == "link":
                cal.link_gbps.update(item["gbps"]); cal.link_latency_ms.update(item["latency_ms"])
                cal.pinned_available = bool(item["pinned_available"])
                if not cal.pinned_available:      # pinned falls back to pageable numbers
                    for d in _DIRS:
                        cal.link_gbps[f"{d}_pinned"] = cal.link_gbps[f"{d}_pageable"]
                        cal.link_latency_ms[f"{d}_pinned"] = cal.link_latency_ms[f"{d}_pageable"]
                log(f"link GB/s: {cal.link_gbps}")
            elif kind == "points":
                op = item["op"]
                probe_ = registry.get((op, "gpu"))
                fits, err = _fit_points(item["points"], bool(probe_ and probe_.uses_card),
                                        bool(probe_ and probe_.bucketed), bool(probe_ and probe_.two_d))
                if fits:
                    _store_fits(cal, op, "gpu", fits)
                    fit = fits[""]
                    log(f"{op}@gpu: alpha={fit.alpha:.4f} c={fit.c * 1e6:.3f} ms/Mrow")
                else:
                    cal.errors[f"{op}@gpu"] = err or "unknown"
            elif kind == "launch":
                cal.launch_overhead_ms["gpu"] = float(item["ms"])
            elif kind in ("error", "crash"):
                cal.errors[item.get("what", "gpu")] = item["msg"]
    cal.cpu_threads = _detect_cpu_threads()

    if save:
        cal.save(path)
    with _CACHE_LOCK:
        if _CACHE is None or cal.key == _CACHE.key:
            _CACHE = cal
    return cal
