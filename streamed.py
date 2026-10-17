"""F-056 step 2: a query over a Parquet file bigger than memory runs in batches of row groups.

When the working set of a plan's `scan_parquet` source is bigger than the host budget, `try_stream` cuts the
file's row groups into batches that fit the budget and runs the plan once per batch with the streamed scan
reading just that batch (`io_parquet.streaming`). The plan's lowest `Aggregate` is run as a partial aggregate
(sum -> sum, count -> count, min / max -> min / max, mean -> sum + count); the partial tables are merged by
another aggregate and whatever sat above the aggregate runs on the merged table. Each batch is an ordinary
`compute(device)`, so filters, the join to the resident dimension tables, the native executor, the GPU
pipeline and step 1's budget safety net all apply per batch. Memory is one batch + the group state.

Design, limits and the step 3 list: docs/design/streamed-execution.md.
"""
from __future__ import annotations

import contextvars
import dataclasses
import os
import warnings
from collections import Counter
from typing import List, Optional

import pyarrow as pa

from . import gpu_budget
from ._lazy_pc import pc
from .gpu_budget import fmt_bytes
from .lazy_expr import Expr, col
from .lazy_frame import (Aggregate, Distinct, Join, LazyFrame, Limit, LogicalPlan, ParquetScan, Project,
                         Scan, Sort, Tail)

WORKING_FACTOR = 3            # in-memory path: the table, the filtered copy, group ids / gather buffers
RESIDENT_FACTOR = 1.5         # a dimension table stays resident: its columns and the join index built on them
HOST_FRACTION = 0.7           # of MemAvailable
REPRODUCIBLE_FRACTION = 0.5   # of MemTotal: does not move with what other processes hold
GPU_BATCH_DIVISOR = 1.3       # a GPU piece needs about this much per byte of batch (columns + group ids)
MIN_MERGE_BYTES = 8 << 20

_active: contextvars.ContextVar = contextvars.ContextVar("mxframe_streamed_active", default=False)
_AGGS = ("sum", "mean", "count", "min", "max")
_WINDOW = ("window", "rank", "dense_rank", "lag", "lead", "cum_sum", "row_number")


def active() -> bool:
    """True inside a streamed query's batch computes (no nested streaming, no background GPU uploads)."""
    return _active.get()


def enabled() -> bool:
    return os.environ.get("MXFRAME_STREAM", "1") not in ("0", "false", "off")


def _meminfo(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    raise OSError(key)


def host_budget(reproducible: bool = False) -> Optional[int]:
    """Bytes of host memory a query may use: `MXFRAME_HOST_MEM_BUDGET` ("64M", "2G": a test knob), else
    70% of MemAvailable; in reproducible mode 50% of MemTotal, so batch boundaries do not depend on what
    other processes hold at the moment. None where /proc/meminfo is not there (nothing streams)."""
    raw = os.environ.get("MXFRAME_HOST_MEM_BUDGET", "").strip()
    if raw:
        return gpu_budget._parse_size(raw)
    try:
        return int(REPRODUCIBLE_FRACTION * _meminfo("MemTotal")) if reproducible \
            else int(HOST_FRACTION * _meminfo("MemAvailable"))
    except OSError:
        return None


# --- looking at the plan ------------------------------------------------------------------------------------

def _has_op(e, ops) -> bool:
    if isinstance(e, Expr):
        return e.op in ops or any(_has_op(a, ops) for a in e.args)
    if isinstance(e, (list, tuple)):
        return any(_has_op(a, ops) for a in e)
    return False


def _leaves(plan):
    from .io_parquet import _leaves as leaves
    return list(leaves(plan))


def _chain(plan, leaf) -> Optional[List[LogicalPlan]]:
    """Nodes from `plan` down to `leaf` (by identity), or None."""
    if plan is leaf:
        return [plan]
    kids = [plan.left, plan.right] if isinstance(plan, Join) else [getattr(plan, "input", None)]
    for k in kids:
        if k is not None and (sub := _chain(k, leaf)) is not None:
            return [plan] + sub
    return None


def _replace(plan, target, repl):
    if plan is target:
        return repl
    if isinstance(plan, Join):
        return dataclasses.replace(plan, left=_replace(plan.left, target, repl), right=_replace(plan.right, target, repl))
    if getattr(plan, "input", None) is not None:
        return dataclasses.replace(plan, input=_replace(plan.input, target, repl))
    return plan


@dataclasses.dataclass
class _Leaf:
    node: ParquetScan
    path: str
    columns: tuple
    pushed: tuple
    groups: List[int]
    sizes: List[int]

    @property
    def bytes(self) -> int:
        return sum(self.sizes)


def _footprint(node: ParquetScan, read: ParquetScan) -> Optional[_Leaf]:
    """The row groups and decoded size a ParquetScan would read (`read`: the same leaf after
    `resolve_plan(read=False)` named its columns and pushed filters). None for a directory."""
    from . import io_parquet as iop
    if not os.path.isfile(node.path):
        return None
    cols = tuple(read.columns or node.schema.names)
    pushed = tuple(read.filters or ())
    groups = iop.kept_groups(node.path, node.schema, pushed)
    return _Leaf(node, os.path.abspath(node.path), cols, pushed, groups, iop.group_bytes(node.path, node.schema, cols, groups))


def _cheap_fits(plan, budget: int) -> bool:
    """True when even reading every column of every Parquet leaf would fit: the usual case, decided from the
    footers alone (one number per file)."""
    from . import io_parquet as iop
    total = 0
    for leaf in _leaves(plan):
        if isinstance(leaf, ParquetScan):
            if not os.path.isfile(leaf.path):
                return True                      # datasets (directories) are not streamed
            total += iop.decoded_upper_bound(leaf.path, leaf.schema)
        else:
            total += leaf.table.nbytes
    return total * WORKING_FACTOR <= budget


class _Decline(Exception):
    pass


@dataclasses.dataclass
class _Spec:
    agg: Aggregate
    key_names: List[str]
    partial: Aggregate                  # the per-batch plan
    merge_ops: List[tuple]              # (hidden column, merge op) in partial order
    finals: List[tuple]                 # per original aggregate: (kind, hidden columns, output name)
    types: dict = dataclasses.field(default_factory=dict)       # hidden column -> Arrow type
    int_sums: dict = dataclasses.field(default_factory=dict)    # integer sum's hidden column -> (min, max) hidden columns
    rows: str = "__rows"                                          # hidden count of the rows that reach the aggregate


def _spec(agg: Aggregate) -> _Spec:
    keys = []
    for k in agg.group_by:
        name = k._alias or (k.args[0] if k.op == "col" and isinstance(k.args[0], str) else None)
        if name is None:
            raise _Decline("a group key that is an unnamed expression")
        keys.append(name)
    partial, merge_ops, finals = [], [], []

    def add(op, e, name, merge):
        partial.append(Expr(op, *e.args, **e.kwargs).alias(name))
        merge_ops.append((name, merge))

    for i, e in enumerate(agg.aggs):
        name = e._alias or f"agg_{i}"
        if e.op not in _AGGS or len(e.args) != 1 or not isinstance(e.args[0], Expr) or _has_op(e.args[0], _AGGS + _WINDOW):
            raise _Decline(f"the aggregate {e.op}(...) cannot be split into per-batch partials")
        if e.op == "mean":
            add("sum", e, f"__p{i}s", "sum")
            add("count", e, f"__p{i}c", "sum")
            finals.append(("mean", (f"__p{i}s", f"__p{i}c"), name))
        else:
            add(e.op, e, f"__p{i}", {"sum": "sum", "count": "sum", "min": "min", "max": "max"}[e.op])
            finals.append((e.op, (f"__p{i}",), name))
    return _Spec(agg, keys, Aggregate(agg.input, list(agg.group_by), partial), merge_ops, finals)


def _argument_types(spec: _Spec) -> None:
    """Fix the types of the hidden columns, and of the result, from the engine itself.

    A zero-row dry run of the aggregate's arguments says which are integers; a dry run of the original
    aggregate gives the type the engine answers min / max with (float32 comes back as double). An integer sum
    is merged as int64 whatever the engine answered per batch, and its result type is decided at the end by
    the engine's own rule on the totals (`_State.result`), so it does not depend on how the rows were batched:
    the argument's min and max and the number of rows reaching the aggregate travel as hidden partials."""
    agg = spec.agg
    args = [Expr(e.args[0].op, *e.args[0].args, **e.args[0].kwargs).alias(f"__a{i}") for i, e in enumerate(agg.aggs)]
    got = LazyFrame(Project(agg.input, args)).schema
    zero = LazyFrame(agg).schema
    if got is None:
        raise _Decline("the aggregate's arguments cannot be typed")
    need_rows = False
    for i, (kind, hidden, name) in enumerate(spec.finals):
        t = got.field(f"__a{i}").type
        if pa.types.is_dictionary(t):
            t = t.value_type
        if kind in ("sum", "mean"):
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise _Decline(f"{kind}() over {t}")
            spec.types[hidden[0]] = pa.int64() if pa.types.is_integer(t) else pa.float64()
            if kind == "sum" and pa.types.is_integer(t):
                lo, hi = f"__p{i}lo", f"__p{i}hi"
                a = agg.aggs[i].args[0]
                for op, h in (("min", lo), ("max", hi)):
                    spec.partial.aggs.append(Expr(op, a).alias(h))
                    spec.merge_ops.append((h, op))
                    spec.types[h] = t
                spec.int_sums[hidden[0]] = (lo, hi)
                need_rows = True
        elif kind in ("min", "max"):
            spec.types[hidden[0]] = zero.field(name).type if zero is not None and name in zero.names else t
        if kind in ("count", "mean"):
            spec.types[hidden[-1]] = pa.int64()
    if need_rows:
        spec.partial.aggs.append(Expr("count", Expr("lit", 1)).alias(spec.rows))
        spec.merge_ops.append((spec.rows, "sum"))
        spec.types[spec.rows] = pa.int64()


@dataclasses.dataclass
class Plan:
    spec: _Spec
    leaf: _Leaf
    batches: List[List[int]]
    batch_target: int
    gpu_cap: Optional[int]
    budget: int
    all_groups: int


def analyse(plan, budget: int, device: str, reproducible: bool):
    """A `Plan` if the plan should and can stream, None if it fits in memory, else the reason it cannot
    (a string)."""
    from .io_parquet import resolve_plan
    resolved = resolve_plan(plan, read=False)
    leaves, rleaves = _leaves(plan), _leaves(resolved)
    feet = []
    for node, read in zip(leaves, rleaves):
        feet.append(_footprint(node, read) if isinstance(node, ParquetScan) else node.table.nbytes)
    if any(f is None for f in feet):
        return None                                           # a dataset directory is among the leaves
    est = lambda f: f if isinstance(f, int) else f.bytes      # noqa: E731
    if sum(est(f) for f in feet) * WORKING_FACTOR <= budget:
        return None                                           # fits: the usual path
    big = max((f for f in feet if isinstance(f, _Leaf)), key=lambda f: f.bytes, default=None)
    try:
        if big is None:
            raise _Decline("no Parquet file among the sources")
        chain = _chain(plan, big.node)
        agg_at = max((i for i, n in enumerate(chain[:-1]) if isinstance(n, Aggregate)), default=None)
        if agg_at is None:
            raise _Decline("no aggregate above the big file (a plan that returns rows is not streamed yet)")
        for n, child in zip(chain[agg_at + 1:-1], chain[agg_at + 2:]):
            if isinstance(n, (Sort, Limit, Distinct, Tail, Aggregate)):
                raise _Decline(f"a {type(n).__name__} below the aggregate")
            if isinstance(n, Project) and _has_op(n.exprs, _WINDOW):
                raise _Decline("a window function below the aggregate")
            if isinstance(n, Join) and child is n.right and n.how != "inner":
                raise _Decline(f"the big file is the build side of a {n.how} join")
        agg = chain[agg_at]
        if sum(1 for l in leaves if isinstance(l, ParquetScan) and os.path.abspath(l.path) == big.path) > 1:
            raise _Decline("the big file is read twice by the plan")
        sub = {id(l) for l in _leaves(agg.input)}
        others = sum(est(f) for l, f in zip(leaves, feet) if id(l) in sub and f is not big)
        if others * RESIDENT_FACTOR > budget // 2:
            raise _Decline(f"the other tables of the aggregate (about {fmt_bytes(others)}) do not fit "
                           f"half of the budget: a join of two big tables needs both sides partitioned")
        spec = _spec(agg)
        _argument_types(spec)
    except _Decline as why:
        return str(why)
    batch = max(int((budget / 2 - others * RESIDENT_FACTOR) / WORKING_FACTOR), 1)
    cap = None
    if device != "cpu" and not reproducible:
        try:
            from . import gpu_stream
            if gpu_stream.cuda_available():
                cap = max(int(gpu_budget.PRECHECK_FRACTION * gpu_budget.free_bytes() / GPU_BATCH_DIVISOR), 1)
                batch = min(batch, cap)
        except Exception:
            cap = None
    batches, cur, size = [], [], 0
    for g, b in zip(big.groups, big.sizes):
        if cur and size + b > batch:
            batches.append(cur)
            cur, size = [], 0
        cur.append(g)
        size += b
    if cur:
        batches.append(cur)
    from . import io_parquet as iop
    return Plan(spec, big, batches, batch, cap, budget, iop._metadata(big.path).num_row_groups)


# --- running the batches and merging ------------------------------------------------------------------------

def _normal(spec: _Spec, t: pa.Table, key_types: Optional[dict]) -> pa.Table:
    """A partial (or merged) table with plain keys and the hidden columns in their fixed types."""
    from .io_parquet import _decode_dict, _is_dict
    cols = []
    for name in t.column_names:
        c = t.column(name)
        if _is_dict(c.type):
            c = _decode_dict(c)
        want = spec.types.get(name) or (key_types or {}).get(name)
        if want is not None and c.type != want:
            c = pc.cast(c, want)
        cols.append(c)
    return pa.table(cols, names=t.column_names)


class _State:
    """The merged partials so far, plus the partials waiting to be merged."""

    def __init__(self, spec: _Spec, budget: int):
        self.spec, self.budget = spec, budget
        self.merged: Optional[pa.Table] = None
        self.pending: List[pa.Table] = []
        self.key_types: Optional[dict] = None
        self.empty: Optional[pa.Table] = None             # a partial with no groups: the answer if all are
        self.merges = 0
        self.peak_state = 0
        self.engine_int64: set = set()                   # integer sums some batch answered as int64 (nulls, big values)

    def add(self, partial: pa.Table) -> None:
        if self.spec.int_sums and partial.num_rows and pc.sum(partial.column(self.spec.rows)).as_py():
            for h in self.spec.int_sums:
                if pa.types.is_integer(partial.schema.field(h).type):
                    self.engine_int64.add(h)
        t = _normal(self.spec, partial, self.key_types)
        if t.num_rows == 0 and self.spec.key_names:
            self.empty = t                               # a keyed aggregate of no rows holds no groups
            return
        if self.key_types is None:                       # the first partial fixes the key types
            self.key_types = {n: t.schema.field(n).type for n in self.spec.key_names}
        self.pending.append(t)
        if sum(p.nbytes for p in self.pending) >= max(self.merged.nbytes if self.merged is not None else 0,
                                                      min(MIN_MERGE_BYTES, self.budget // 16)):
            self.merge()

    def merge(self) -> None:
        parts = ([self.merged] if self.merged is not None else []) + self.pending
        self.pending = []
        if not parts:
            return
        table = parts[0]
        if len(parts) > 1:
            aggs = [Expr(op, col(name)).alias(name) for name, op in self.spec.merge_ops]
            keys = [col(k) for k in self.spec.key_names]
            out = LazyFrame(Aggregate(Scan(pa.concat_tables(parts)), keys, aggs)).compute(
                "cpu", optimize=False, validate=False)
            table = _normal(self.spec, out, self.key_types)
        self.merged = table
        self.merges += 1
        self.peak_state = max(self.peak_state, table.num_rows)
        if table.nbytes > self.budget // 2:
            raise MemoryError(
                f"group state of {table.num_rows:,} groups ({fmt_bytes(table.nbytes)}) is over half of the host "
                f"memory budget ({fmt_bytes(self.budget)}): this query needs hash-partitioned groups "
                f"(F-056 step 3)")

    def _integer_sum_is_exact_in_double(self, t: pa.Table, h: str) -> bool:
        """The engine's rule, on the totals: an integer sum is a double when rows x max |value| stays within
        2^53 and no batch needed Arrow's exact path (nulls in a column it reads, values past 2^53)."""
        if h in self.engine_int64:
            return False
        lo_col, hi_col = self.spec.int_sums[h]
        lo, hi = pc.min(t.column(lo_col)).as_py(), pc.max(t.column(hi_col)).as_py()
        rows = pc.sum(t.column(self.spec.rows)).as_py() or 0
        return lo is None or max(abs(lo), abs(hi)) * max(rows, 1) <= 2 ** 53

    def result(self) -> pa.Table:
        self.merge()
        t = self.merged if self.merged is not None else self.empty
        arrays, names = [t.column(k) for k in self.spec.key_names], list(self.spec.key_names)
        for kind, hidden, name in self.spec.finals:
            if kind == "mean":
                s, c = t.column(hidden[0]), t.column(hidden[1])
                arr = pc.if_else(pc.greater(c, 0), pc.divide(pc.cast(s, pa.float64()), pc.cast(c, pa.float64())),
                                 pa.scalar(None, pa.float64()))
            else:
                arr = t.column(hidden[0])
                if hidden[0] in self.spec.int_sums and self._integer_sum_is_exact_in_double(t, hidden[0]):
                    arr = pc.cast(arr, pa.float64())
            arrays.append(arr)
            names.append(name)
        return pa.table(arrays, names=names)


def describe(plan, device: str = "cpu") -> Optional[str]:
    """One line for `explain()`: whether the plan would stream (and in how many batches) or why it would
    not, decided exactly as `compute()` decides. None when the plan reads no Parquet source."""
    if not any(isinstance(l, ParquetScan) for l in _leaves(plan)):
        return None
    if not enabled():
        return "in memory: streaming is off (MXFRAME_STREAM=0)"
    from . import native_exec
    repro = native_exec.reproducible()
    budget = host_budget(repro)
    if budget is None:
        return "in memory: the host memory size is unknown here"
    got = None if _cheap_fits(plan, budget) else analyse(plan, budget, device, repro)
    if got is None:
        return f"in memory: fits the host budget of {fmt_bytes(budget)}"
    if isinstance(got, str):
        return f"would not stream: {got}"
    return (f"streams in {len(got.batches)} batches of row groups ({len(got.leaf.groups)} of {got.all_groups} row groups "
            f"read, host budget {fmt_bytes(budget)})")


def try_stream(lf, plan, device: str, *, strict: bool, prefer, placement, gpu_fraction, verbose: bool = False):
    """(table, provenance) if `plan` is streamed, None to run it in memory as before. Called by `compute()`
    on the optimized plan, before any Parquet source is read."""
    if _active.get() or not enabled() or not any(isinstance(l, ParquetScan) for l in _leaves(plan)):
        return None
    from . import native_exec
    from .io_parquet import streaming
    repro = native_exec.reproducible()
    budget = host_budget(repro)
    if budget is None or _cheap_fits(plan, budget):
        return None
    got = analyse(plan, budget, device, repro)
    if got is None:
        return None
    if isinstance(got, str):
        warnings.warn(f"mxframe: this query's working set does not fit in memory (budget {fmt_bytes(budget)}); "
                      f"not streamed: {got}. Running it in memory, which may run out of memory.",
                      RuntimeWarning, stacklevel=3)
        return None
    spec, leaf = got.spec, got.leaf
    state = _State(spec, budget)
    devices, pieces, paths = Counter(), Counter(), Counter()
    biggest, size_of = 0, dict(zip(leaf.groups, leaf.sizes))
    token = _active.set(True)
    try:
        with streaming(leaf.path) as st:
            for batch in got.batches:
                st.groups = tuple(batch)
                biggest = max(biggest, sum(size_of[g] for g in batch))
                sub = LazyFrame(spec.partial)
                part = sub.compute(device, optimize=False, validate=False, strict=strict, prefer=prefer,
                                   placement=placement, gpu_fraction=gpu_fraction)
                p = sub.last_compile_provenance or {}
                devices[p.get("auto_device") or p.get("device") or device] += 1
                paths[p.get("path", "")] += 1
                if p.get("gpu_pieces"):
                    pieces[p["gpu_pieces"]] += 1
                state.add(part)
                if verbose:
                    print(f"[mxframe] streamed batch {sum(devices.values())}/{len(got.batches)}: "
                          f"{len(batch)} row groups, {part.num_rows} partial rows")
        merged = state.result()
    finally:
        _active.reset(token)
    above = _replace(plan, spec.agg, Scan(merged))
    result = merged if isinstance(above, Scan) else LazyFrame(above).compute(
        device, optimize=False, validate=False, strict=strict, prefer=prefer, placement=placement,
        gpu_fraction=gpu_fraction)
    prov = {"path": "streamed", "device": device, "rows": result.num_rows,
            "streamed": {"batches": len(got.batches), "row_groups": got.all_groups,
                         "row_groups_read": len(leaf.groups), "host_budget": budget, "gpu_batch_cap": got.gpu_cap,
                         "batch_target_bytes": got.batch_target, "largest_batch_bytes": biggest,
                         "merges": state.merges, "group_state_rows": state.peak_state,
                         "devices": dict(devices), "batch_paths": dict(paths)}}
    if pieces:
        prov["gpu_pieces"] = "; ".join(f"{n} batches: {s}" for s, n in pieces.items())
    return result, prov
