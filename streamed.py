"""F-056 steps 2 and 3a: a query over a Parquet file bigger than memory runs in batches of row groups.

When the working set of a plan's `scan_parquet` source is bigger than the host budget, `try_stream` cuts the
file's row groups into batches that fit the budget and runs the plan once per batch with the streamed scan
reading just that batch (`io_parquet.streaming`). The plan's lowest `Aggregate` is run as a partial aggregate
(sum -> sum, count -> count, min / max -> min / max, mean -> sum + count); the partial tables are merged by
another aggregate and whatever sat above the aggregate runs on the merged table. Each batch is an ordinary
`compute(device)`, so filters, the join to the resident dimension tables, the native executor, the GPU
pipeline and step 1's budget safety net all apply per batch. Memory is one batch + the group state.

Step 3a: group state bigger than the budget spills to hash partitions (`spill.py`, merged one partition at a time);
a plan with no aggregate streams its rows into a `_Sink` (running top-k for ORDER BY + LIMIT, early stop for LIMIT,
else collected and checked against the budget); a row group bigger than a batch is read in slices of rows.

Design, limits and the step 3 list: docs/design/streamed-execution.md.
"""
from __future__ import annotations

import contextlib
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
from .lazy_frame import (Aggregate, Distinct, Filter, Join, LazyFrame, Limit, LogicalPlan, ParquetScan, Project,
                         Scan, Sort, Tail)
from . import spill as _spill

WORKING_FACTOR = 3            # in-memory path: the table, the filtered copy, group ids / gather buffers
RESIDENT_FACTOR = 1.5         # a dimension table stays resident: its columns and the join index built on them
HOST_FRACTION = 0.7           # of MemAvailable
REPRODUCIBLE_FRACTION = 0.5   # of MemTotal: does not move with what other processes hold
GPU_BATCH_DIVISOR = 1.3       # a GPU piece needs about this much per byte of batch (columns + group ids)
MIN_MERGE_BYTES = 8 << 20
MERGE_FACTOR = 4             # the engine's group-by holds about this many times its input (measured: 5 MB in, 20 MB peak)
MIN_SLICE_ROWS = 1024        # a slice of an oversized row group is never smaller

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
    int_sums: set = dataclasses.field(default_factory=set)      # hidden columns of integer sums (typed by _State.result)


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
    the engine's own rule on the merged inputs of every batch (`pipeline_program.SumProbe`), so it does not
    depend on how the rows were batched."""
    agg = spec.agg
    args = [Expr(e.args[0].op, *e.args[0].args, **e.args[0].kwargs).alias(f"__a{i}") for i, e in enumerate(agg.aggs)]
    got = LazyFrame(Project(agg.input, args)).schema
    zero = LazyFrame(agg).schema
    if got is None:
        raise _Decline("the aggregate's arguments cannot be typed")
    for i, (kind, hidden, name) in enumerate(spec.finals):
        t = got.field(f"__a{i}").type
        if pa.types.is_dictionary(t):
            t = t.value_type
        if kind in ("sum", "mean"):
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise _Decline(f"{kind}() over {t}")
            spec.types[hidden[0]] = pa.int64() if pa.types.is_integer(t) else pa.float64()
            if kind == "sum" and pa.types.is_integer(t):
                spec.int_sums.add(hidden[0])
        elif kind in ("min", "max"):
            spec.types[hidden[0]] = zero.field(name).type if zero is not None and name in zero.names else t
        if kind in ("count", "mean"):
            spec.types[hidden[-1]] = pa.int64()


@dataclasses.dataclass
class Plan:
    spec: Optional[_Spec]               # None: the plan returns rows (no aggregate above the big file)
    leaf: _Leaf
    batches: List[List[int]]
    batch_target: int
    gpu_cap: Optional[int]
    budget: int
    all_groups: int
    top: LogicalPlan = None             # the node whose output the batches produce: the aggregate, or the top of the row-local part
    slice_rows: dict = dataclasses.field(default_factory=dict)      # row group -> rows per slice (a group bigger than a batch)
    resident: int = 0                   # bytes of the dimension tables kept for all batches
    prune: bool = False                 # rows: select only the columns the plan reads (a bare Filter over the file)


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
        prune = False
        if agg_at is None:                  # rows: the top of the part that works row by row on the big file
            top_at = len(chain) - 1
            for i in range(len(chain) - 2, -1, -1):
                n = chain[i]
                if isinstance(n, Filter) or (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)) \
                        or (isinstance(n, Join) and (n.how == "inner" or chain[i + 1] is n.left)):
                    top_at = i
                else:
                    break
            top = chain[top_at]
            prune = all(isinstance(n, Filter) for n in chain[top_at:-1])
        else:
            for n, child in zip(chain[agg_at + 1:-1], chain[agg_at + 2:]):
                if isinstance(n, (Sort, Limit, Distinct, Tail, Aggregate)):
                    raise _Decline(f"a {type(n).__name__} below the aggregate")
                if isinstance(n, Project) and _has_op(n.exprs, _WINDOW):
                    raise _Decline("a window function below the aggregate")
                if isinstance(n, Join) and child is n.right and n.how != "inner":
                    raise _Decline(f"the big file is the build side of a {n.how} join")
            top = chain[agg_at]
            for n in reversed(chain[:agg_at]):          # a HAVING filter / projection works group by group
                if isinstance(n, Filter) or (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)):
                    top = n
                else:
                    break
        agg = chain[agg_at] if agg_at is not None else None
        if sum(1 for l in leaves if isinstance(l, ParquetScan) and os.path.abspath(l.path) == big.path) > 1:
            raise _Decline("the big file is read twice by the plan")
        sub = {id(l) for l in _leaves(agg.input if agg is not None else top)}
        others = sum(est(f) for l, f in zip(leaves, feet) if id(l) in sub and f is not big)
        if others * RESIDENT_FACTOR > budget // 2:
            raise _Decline(f"the other tables of the aggregate (about {fmt_bytes(others)}) do not fit "
                           f"half of the budget: a join of two big tables needs both sides partitioned")
        spec = None
        if agg is not None:
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
    batches, cur, size, slices = [], [], 0, {}
    from . import io_parquet as iop
    md = iop._metadata(big.path)
    for g, b in zip(big.groups, big.sizes):
        rows = md.row_group(g).num_rows
        if b > batch and rows > 1:                  # one row group bigger than a batch: read it in slices of rows
            if cur:
                batches.append(cur)
                cur, size = [], 0
            per = max(-(-rows // -(-b // batch)), min(MIN_SLICE_ROWS, rows))
            slices[g] = per
            batches.extend([g] for _ in range(-(-rows // per)))
            continue
        if cur and size + b > batch:
            batches.append(cur)
            cur, size = [], 0
        cur.append(g)
        size += b
    if cur:
        batches.append(cur)
    return Plan(spec, big, batches, batch, cap, budget, md.num_row_groups, top, slices, int(others * RESIDENT_FACTOR), prune)


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


def _read_partition(parts, i: int):
    return parts.read(i)


class _State:
    """The merged partials so far, plus the partials waiting to be merged.

    When the merged state passes half of the budget it is spilled (`spill.Partitioned`): the state so far and every
    later partial go to hash partitions on disk, and `pieces()` merges one partition at a time (splitting a
    partition again, with another salt, if even that one is too big)."""

    def __init__(self, spec: _Spec, budget: int, probe=None, root=None, salt: int = 0, depth: int = 0, held=None):
        self.spec, self.budget, self.probe = spec, budget, probe
        self.held = held or (lambda: 0)          # bytes the query holds elsewhere: resident tables, the sink
        self.root, self.salt, self.depth = root, salt, depth
        self.merged: Optional[pa.Table] = None
        self.pending: List[pa.Table] = []
        self.parts: Optional[_spill.Partitioned] = None
        self.key_types: Optional[dict] = None
        self.empty: Optional[pa.Table] = None             # a partial with no groups: the answer if all are
        self.merges = 0
        self.peak_state = 0

    @property
    def spilled(self) -> bool:
        return self.parts is not None

    def add(self, partial: pa.Table) -> None:
        t = _normal(self.spec, partial, self.key_types)
        if t.num_rows == 0 and self.spec.key_names:
            self.empty = t                               # a keyed aggregate of no rows holds no groups
            return
        if self.key_types is None:                       # the first partial fixes the key types
            self.key_types = {n: t.schema.field(n).type for n in self.spec.key_names}
        if self.parts is not None:
            self.parts.write(t)
            return
        self.pending.append(t)
        if sum(p.nbytes for p in self.pending) >= max(self.merged.nbytes if self.merged is not None else 0,
                                                      min(MIN_MERGE_BYTES, self.budget // 16)):
            self.merge()

    def merge(self) -> None:
        parts = ([self.merged] if self.merged is not None else []) + self.pending
        self.pending = []
        if not parts:
            return
        size = sum(p.nbytes for p in parts)
        if size * MERGE_FACTOR > max(self.budget - self.held(), self.budget // 16):               # merging this much would pass the budget: spill instead
            if self.root is None or self.depth >= _spill.MAX_DEPTH:
                raise MemoryError(
                    f"group state of {sum(p.num_rows for p in parts):,} partial groups ({fmt_bytes(size)}) is too big "
                    f"to merge in the host memory budget ({fmt_bytes(self.budget)}) even after splitting the groups "
                    f"{self.depth} times by hash")
            self.parts = _spill.Partitioned(self.root, self.spec.key_names, self.salt, self.depth + 1)
            for p in parts:                                 # the state so far; every later partial follows it
                self.parts.write(p)
            self.merged = None
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

    def pieces(self):
        """The merged hidden-column tables: one when the state fits, else one per non-empty hash partition (a group
        is in exactly one), in partition order."""
        self.merge()
        if self.parts is None:
            t = self.merged if self.merged is not None else self.empty
            if t is not None:
                yield t
            return
        self.parts.finish()
        for i in range(_spill.PARTITIONS):
            sub = _State(self.spec, self.budget, None, self.root, self.salt + 1, self.depth + 1, self.held)
            sub.key_types = self.key_types
            for t in _read_partition(self.parts, i):
                sub.add(t)
            yield from sub.pieces()

    def result(self) -> pa.Table:
        self.merge()
        return self.format(self.merged if self.merged is not None else self.empty)

    def format(self, t: pa.Table) -> pa.Table:
        """The original aggregate's output from merged hidden columns."""
        arrays, names = [t.column(k) for k in self.spec.key_names], list(self.spec.key_names)
        for kind, hidden, name in self.spec.finals:
            if kind == "mean":
                s, c = t.column(hidden[0]), t.column(hidden[1])
                arr = pc.if_else(pc.greater(c, 0), pc.divide(pc.cast(s, pa.float64()), pc.cast(c, pa.float64())),
                                 pa.scalar(None, pa.float64()))
            else:
                arr = t.column(hidden[0])
                if hidden[0] in self.spec.int_sums and self.probe is not None and not self.probe.inexact():
                    arr = pc.cast(arr, pa.float64())
            arrays.append(arr)
            names.append(name)
        return pa.table(arrays, names=names)


# --- what happens to the rows the streamed part of the plan produces ---------------------------------------------

def _plain(t: pa.Table) -> pa.Table:
    from .io_parquet import _decode_dict, _is_dict
    for i, c in enumerate(t.columns):
        if _is_dict(c.type):
            t = t.set_column(i, t.column_names[i], _decode_dict(c))
    return t


class _Sink:
    """Collects what the streamed part of a plan produces (batches of rows, or the merged groups) up to what is
    above it: `top-k` (Sort then Limit: a running top n + offset, merged as the batches come), `limit` (stops as soon
    as n + offset rows are in) or `collect` (everything, which must fit `limit` bytes: otherwise a MemoryError
    naming the size). What sits above is run on `result()` by the caller, so offset, ties and ordering are the
    engine's own."""

    def __init__(self, kind: str, k: int, sort: Optional[Sort], limit: int, budget: int, above: str = ""):
        self.kind, self.k, self.sort, self.limit, self.budget, self.above = kind, k, sort, limit, budget, above
        self.tables: List[pa.Table] = []
        self.rows = self.bytes = self.pending_rows = 0
        self.empty: Optional[pa.Table] = None

    @property
    def full(self) -> bool:
        return self.kind == "limit" and self.rows >= self.k

    def batch_plan(self, top):
        if self.kind == "top-k":
            return Limit(dataclasses.replace(self.sort, input=top), self.k)
        if self.kind == "limit":
            return Limit(top, max(self.k - self.rows, 0))
        return top

    def add(self, t: pa.Table, progress: Optional[float] = None) -> None:
        t = _plain(t)
        if t.num_rows == 0:
            self.empty = t
            return
        self.tables.append(t)
        self.rows += t.num_rows
        self.pending_rows += t.num_rows
        self.bytes += t.nbytes
        if self.kind == "top-k" and self.pending_rows >= max(self.k, 8192):
            self._reduce()
        if self.bytes > self.limit:
            self._too_big(progress)

    def _reduce(self) -> None:
        table = self._concat()
        out = LazyFrame(Limit(dataclasses.replace(self.sort, input=Scan(table)), self.k)).compute(
            "cpu", optimize=False, validate=False)
        self.tables, self.rows, self.pending_rows, self.bytes = [out], out.num_rows, 0, out.nbytes

    def _concat(self) -> pa.Table:
        if not self.tables:
            return self.empty
        return self.tables[0] if len(self.tables) == 1 else pa.concat_tables(self.tables, promote_options="default")

    def _too_big(self, progress: Optional[float]) -> None:
        if self.kind == "top-k":
            self._reduce()
            if self.bytes <= self.limit:
                return
            what = f"the {self.k:,} rows kept for ORDER BY ... LIMIT"
        else:
            what = f"the {self.rows:,} rows produced so far"
        guess = f", about {fmt_bytes(int(self.bytes / progress))} in all" if progress and progress > 0.01 and self.kind == "collect" else ""
        subject = f"the rows that the {self.above} above the big file needs are" if self.above else "the result of this query is"
        raise MemoryError(
            f"{subject} bigger than memory: {what} are {fmt_bytes(self.bytes)}{guess}, and only "
            f"{fmt_bytes(self.limit)} fits the host memory budget of {fmt_bytes(self.budget)}. Add a LIMIT or a "
            f"tighter filter, aggregate it, or raise MXFRAME_HOST_MEM_BUDGET")

    def result(self) -> pa.Table:
        if self.kind == "top-k" and len(self.tables) > 1:
            self._reduce()
        return self._concat()


def _sink_for(plan, top, limit_direct: int, limit_above: int, budget: int) -> _Sink:
    """The sink for what sits above `top` in `plan` (`_chain` finds it): a Sort then (optionally some row-local
    Projects and) a Limit is a top-k, a Limit alone stops early, anything else collects."""
    above = list(reversed(_chain(plan, top)[:-1]))
    limit = limit_above if above else limit_direct
    name = type(above[0]).__name__ if above else ""
    if above and isinstance(above[0], Limit):
        return _Sink("limit", above[0].n + above[0].offset, None, limit, budget, name)
    if above and isinstance(above[0], Sort):
        for n in above[1:]:
            if isinstance(n, Limit):
                return _Sink("top-k", n.n + n.offset, above[0], limit, budget, name)
            if not (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)):
                break
    return _Sink("collect", 0, None, limit, budget, name)


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
    from .pipeline_program import sum_probe
    devices, pieces, paths = Counter(), Counter(), Counter()
    biggest, size_of = 0, dict(zip(leaf.groups, leaf.sizes))
    total_bytes, done_bytes, sliced = max(sum(leaf.sizes), 1), 0, Counter()
    root = _spill.SpillDir()
    sink, state, batches_run, rows_info = None, None, 0, None
    n_slices = Counter(b[0] for b in got.batches if len(b) == 1 and b[0] in got.slice_rows)
    free = max(budget - got.resident, budget // 8)          # what the dimension tables kept resident leave
    if spec is None:                                        # a batch of input is in flight while rows are collected
        sink = _sink_for(plan, got.top, max(free - got.batch_target, budget // 8), free // WORKING_FACTOR, budget)
    token = _active.set(True)
    try:
        with streaming(leaf.path) as st, (sum_probe(names=spec.int_sums) if spec else contextlib.nullcontext()) as probe:
            state = _State(spec, budget, probe, root, held=lambda: got.resident + (sink.bytes if sink else 0)) if spec else None
            for batch in got.batches:
                st.groups = tuple(batch)
                st.slice_rows = got.slice_rows.get(batch[0]) if len(batch) == 1 else None
                if st.slice_rows:
                    st.slice_no = sliced[batch[0]]
                    sliced[batch[0]] += 1
                share = size_of[batch[0]] / n_slices[batch[0]] if st.slice_rows \
                    else sum(size_of[g] for g in batch)
                biggest = max(biggest, int(share))
                if spec is not None:
                    plan_b = spec.partial
                else:
                    plan_b = sink.batch_plan(_pruned(got) if got.prune else got.top)
                sub = LazyFrame(plan_b)
                if probe:
                    probe.begin()
                part = sub.compute(device, optimize=False, validate=False, strict=strict, prefer=prefer,
                                   placement=placement, gpu_fraction=gpu_fraction)
                p = sub.last_compile_provenance or {}
                devices[p.get("auto_device") or p.get("device") or device] += 1
                paths[p.get("path", "")] += 1
                if p.get("gpu_pieces"):
                    pieces[p["gpu_pieces"]] += 1
                if probe:
                    probe.end()
                batches_run += 1
                done_bytes += share
                if spec is not None:
                    state.add(part)
                else:
                    sink.add(part, done_bytes / total_bytes)
                if verbose:
                    print(f"[mxframe] streamed batch {batches_run}/{len(got.batches)}: "
                          f"{len(batch)} row groups, {part.num_rows} {'partial ' if spec else ''}rows")
                if sink is not None and sink.full:
                    break
            if spec is not None and not state.spilled:
                merged = state.result()                              # the usual case: the groups fit
                above = _replace(plan, spec.agg, Scan(merged))
                table = merged
            else:
                if spec is not None:                                 # spilled groups: one partition at a time
                    sink = _sink_for(plan, got.top, free, free // WORKING_FACTOR, budget)
                    rowwise = got.top is not spec.agg
                    for piece in state.pieces():
                        out = state.format(piece)
                        if rowwise:
                            out = LazyFrame(_replace(got.top, spec.agg, Scan(out))).compute(
                                "cpu", optimize=False, validate=False)
                        sink.add(out)
                        if sink.full:
                            break
                table = sink.result()
                above = _replace(plan, got.top, Scan(table))
                rows_info = {"kind": sink.kind, "batches_run": batches_run}
    finally:
        _active.reset(token)
        root.close()
    result = table if isinstance(above, Scan) else LazyFrame(above).compute(
        device, optimize=False, validate=False, strict=strict, prefer=prefer, placement=placement,
        gpu_fraction=gpu_fraction)
    info = {"batches": len(got.batches), "row_groups": got.all_groups,
            "row_groups_read": len(leaf.groups), "host_budget": budget, "gpu_batch_cap": got.gpu_cap,
            "batch_target_bytes": got.batch_target, "largest_batch_bytes": biggest,
            "devices": dict(devices), "batch_paths": dict(paths)}
    if spec is not None:
        info.update(merges=state.merges, group_state_rows=state.peak_state)
    if root.partitions:
        info["spill"] = {"partitions": root.partitions, "bytes": root.bytes, "depth": root.depth}
    if rows_info is not None and spec is None:
        info["rows"] = rows_info
    if got.slice_rows:
        info["slices"] = {"row_groups": len(got.slice_rows), "rows_per_slice": min(got.slice_rows.values())}
    prov = {"path": "streamed", "device": device, "rows": result.num_rows, "streamed": info}
    if pieces:
        prov["gpu_pieces"] = "; ".join(f"{n} batches: {s}" for s, n in pieces.items())
    return result, prov


def _pruned(got: Plan):
    """The rows plan, selecting only the columns the whole plan reads from the file."""
    return Project(got.top, [col(c) for c in got.leaf.columns])
