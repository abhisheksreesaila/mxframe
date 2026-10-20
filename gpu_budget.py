"""F-056 step 1: data bigger than GPU memory never fails.

Every GPU piece (a fused aggregate, a join's row pairs, a sort, a gather) runs through `run_piece`:

1. Pre-check: the piece's device memory is estimated (inputs + scratch + outputs, from the sizes the
   kernels allocate). Over 90% of the GPU's free memory (HeavyDB's default) -> the piece runs on the CPU.
2. Out of memory anyway (`MemoryError`, which every GPU kernel raises since F-055): an aggregate over
   row chunks is retried once with half-size chunks, then the piece is rerun on the CPU.
3. The shape that ran out of memory is remembered, so the next run goes straight to the CPU.

A piece never changes its inputs and a failed attempt's GPU output is dropped, so the CPU rerun gives
the answer the CPU gives. Each GPU is its own pool: the budget is that of the GPU the process runs on.
`MXFRAME_GPU_MEM_BUDGET` fakes a smaller GPU (a test knob): "512M" for every pool, "0:512M,1:2G" per pool.

`device="gpu"` and `"auto"` both fall back (a `RuntimeWarning` for gpu, silent for auto: the decision
is in provenance and `explain`); `strict=True` raises the `MemoryError` instead.
"""
import contextlib
import contextvars
import os
import warnings
from typing import Callable, List, Optional

PRECHECK_FRACTION = 0.9
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def _parse_size(text: str) -> int:
    t = text.strip().upper().removesuffix("B")
    unit = t[-1:] if t[-1:] in ("K", "M", "G") else ""
    return int(float(t[:len(t) - len(unit)]) * _UNITS[unit])


def budget_override(pool: int = 0) -> Optional[int]:
    """The fake budget of GPU `pool` from MXFRAME_GPU_MEM_BUDGET, or None."""
    raw = os.environ.get("MXFRAME_GPU_MEM_BUDGET", "").strip()
    if not raw:
        return None
    default = None
    for part in raw.split(","):
        if ":" in part:
            idx, size = part.split(":", 1)
            if int(idx) == pool:
                return _parse_size(size)
        else:
            default = _parse_size(part)
    return default


def pool_index() -> int:
    """The GPU this process runs on (the first visible one; a second GPU is a second process)."""
    return 0


def free_bytes(pool: Optional[int] = None) -> int:
    """Free memory of a GPU pool: what the driver reports, capped by the fake budget if one is set."""
    pool = pool_index() if pool is None else pool
    from . import gpu_stream
    free = gpu_stream.mem_info()[0]
    fake = budget_override(pool)
    return free if fake is None else min(free, fake)


class JoinOutputTooLarge(MemoryError):
    """A join output past what the kernels address (2^31 rows): a size limit on every device, not a GPU
    out-of-memory, so it is neither retried on the CPU (which has the same limit) nor remembered."""


class LeaveGpu(Exception):
    """Raised by `leave_gpu` for callers that already catch errors and continue on a CPU path."""


def leave_gpu():
    raise LeaveGpu("left the GPU")


class Run:
    """The GPU pieces of one `compute()`: what ran where, and why a piece left the GPU."""

    def __init__(self, strict: bool = False, device: str = "gpu"):
        self.strict, self.device = strict, device
        self.events: List[dict] = []

    @property
    def lines(self) -> List[str]:
        return [e["line"] for e in self.events]

    @property
    def oom_free(self) -> Optional[int]:
        """Free GPU bytes when a piece last ran out of memory, or None if none did."""
        free = [e["free"] for e in self.events if e["why"] == "out of memory"]
        return min(free) if free else None


_run: contextvars.ContextVar = contextvars.ContextVar("mxframe_gpu_run", default=None)
RETRY_FREE_FACTOR = 1.5       # a remembered shape gets another GPU try once this much more is free
_remembered: dict = {}        # (piece, shape) that ran out of GPU memory -> free bytes when it did
_warned: set = set()


def strict() -> bool:
    """True inside a `compute(strict=True)`: handlers that swallow errors must let a MemoryError through."""
    run = _run.get()
    return bool(run and run.strict)


def forget() -> None:
    _remembered.clear()
    _warned.clear()


@contextlib.contextmanager
def track(strict: bool = False, device: str = "gpu"):
    run = Run(strict, device)
    token = _run.set(run)
    try:
        yield run
    finally:
        _run.reset(token)


def run_piece(name: str, need: int, gpu: Callable, cpu: Callable, *, shape=None,
              retry: Optional[Callable] = None, left: str = "ran on CPU"):
    """Run one GPU piece with the memory safety net described at the top of this module.

    name:  what the piece is ("aggregate", "join", "sort", "gather"), shown in provenance and explain
    need:  estimated device bytes of `gpu()`
    gpu:   the GPU attempt; cpu: the exact CPU rerun; retry: an optional GPU attempt with half the working set
    shape: hashable key of the piece's size, for remembering a failure
    left:  how the line says the piece left the GPU (a caller that continues on a CPU path of its own
           says so instead of "ran on CPU")
    """
    run = _run.get() or Run()
    key = (name, shape)

    def leave(why: str, free: Optional[int], err: Optional[BaseException] = None):
        detail = {"precheck": f"needs about {fmt_bytes(need)} of GPU memory, over {int(PRECHECK_FRACTION * 100)}% "
                              f"of the {fmt_bytes(free or 0)} free",
                  "remembered": "this shape ran out of GPU memory before",
                  "out of memory": f"ran out of GPU memory (needed about {fmt_bytes(need)}): {err}"}[why]
        if run.strict:
            raise MemoryError(f"{name}: {detail}")
        line = f"{name}: {left}, {detail}"
        run.events.append({"piece": name, "action": "cpu", "why": why, "need": need, "free": free, "line": line})
        if run.device == "gpu" and key + (why,) not in _warned:
            _warned.add(key + (why,))
            warnings.warn(f"omxcore gpu: {line}", RuntimeWarning, stacklevel=4)
        return cpu()

    free = free_bytes()
    if key in _remembered:
        if free < RETRY_FREE_FACTOR * _remembered[key]:
            return leave("remembered", free)
        del _remembered[key]                    # clearly more is free now: try the GPU again
    if need > PRECHECK_FRACTION * free:
        return leave("precheck", free)
    try:
        return gpu()
    except JoinOutputTooLarge:
        raise
    except MemoryError as err:
        if retry is not None and not run.strict:
            try:
                out = retry()
                run.events.append({"piece": name, "action": "gpu_halved", "why": "out of memory", "need": need,
                                   "free": free, "line": f"{name}: ran out of GPU memory, retried on the GPU "
                                                         f"with half the rows per chunk"})
                return out
            except MemoryError as err2:
                err = err2
        _remembered[key] = free
        return leave("out of memory", free, err)


# --- what a piece needs on the device, from the buffers its kernels allocate -----------------------------

_SLACK = 1 << 20      # launch arguments, block partials, the allocator's rounding


def aggregate_bytes(row_bytes: int, rows: int, n_groups: int, n_slots: int) -> int:
    """One chunk of a fused aggregate: its columns and group ids, then the per-group outputs."""
    return rows * row_bytes + n_groups * (n_slots + 1) * 8 + _SLACK


def sort_bytes(rows: int, key_bytes: int = 4) -> int:
    """sort_topk: keys, the index output, and the sort's scratch (keys and indices again)."""
    return rows * (key_bytes + 4 + key_bytes + 4)


def join_bytes(n_left: int, n_right: int, table_size: int) -> int:
    """hash_join: both key columns, the per-key table and its prefix sum, per-left counts and offsets, the
    right order, and the two output index columns (sized as one match per left row: the real total is
    only known on the device, and an output that is larger runs out of memory and is caught below)."""
    return 4 * (n_left + n_right) + 8 * table_size + 8 * n_left + 4 * n_right + 8 * max(n_left, n_right) + _SLACK


def gather_bytes(table, rows: int) -> int:
    """gather_table: the index column, then each fixed-width column's source copy and its output."""
    def width(t):
        try:
            return t.bit_width // 8 if t.bit_width >= 32 else 0
        except ValueError:                               # strings etc. are gathered on the CPU
            return 0
    width = sum(width(f.type) for f in table.schema)
    return rows * 4 + (table.num_rows + rows) * width
