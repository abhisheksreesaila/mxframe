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
from .lazy_expr import Expr, col, lit
from .lazy_frame import (Aggregate, Distinct, Filter, Join, LazyFrame, Limit, LogicalPlan, ParquetScan, Project,
                         Scan, Sort, Tail)
from . import np_arrow
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
_out: contextvars.ContextVar = contextvars.ContextVar("mxframe_sink_out", default=None)     # F-113: the file a sink writes
_wide: contextvars.ContextVar = contextvars.ContextVar("mxframe_wide_means", default=False)  # F-126: integer means in two int64 sums
_HALF = 1 << 32
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
    is int64 in every batch and in the merge (F-060), so its type does not depend on how the rows were batched."""
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
        elif kind in ("min", "max"):
            spec.types[hidden[0]] = zero.field(name).type if zero is not None and name in zero.names else t
        if kind in ("count", "mean"):
            spec.types[hidden[-1]] = pa.int64()
        if kind == "mean" and spec.types[hidden[0]] == pa.int64() and _wide.get():
            _widen_mean(spec, i, e=agg.aggs[i], hidden=hidden)


def _widen_mean(spec: _Spec, i: int, e, hidden) -> None:
    """F-126: the mean of an integer whose total passed int64 in a batch or in the merge (the plain run raised and was
    started again with this on). The hidden sum becomes two: the sum of ``x % 2^32`` and the sum of ``(x - x % 2^32) /
    2^32``, each far from int64 (|part| < 2^32 and <= 2^31, so about 2^31 rows add up); the total ``hi * 2^32 + lo``
    is put together as a Python integer in `_State.format`."""
    x = e.args[0]
    lo = x % lit(_HALF)
    hi = ((x - lo) / lit(_HALF)).cast("int64")
    s, c = hidden
    h = f"{s[:-1]}h"
    aggs = spec.partial.aggs
    at = next(j for j, a in enumerate(aggs) if a._alias == s)
    aggs[at] = Expr("sum", lo).alias(s)
    aggs.insert(at + 1, Expr("sum", hi).alias(h))
    spec.merge_ops.insert(at + 1, (h, "sum"))
    spec.types[h] = pa.int64()
    spec.finals[i] = ("imean", (s, h, c), spec.finals[i][2])


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
    grace: Optional[Join] = None        # step 3b: the join of two big sides, run as a Grace hash join (grace.py)
    plan: LogicalPlan = None            # the plan to run: the one given, or its count-distinct rewrite (step 3c)


def _grace_at(chain, agg_at, leaves, feet, big, budget: int) -> Optional[int]:
    """Index in `chain` of the topmost join that needs both its sides partitioned (step 3b), else None: the big
    file is the right side of a left / semi / anti join (a build side cannot stream), the other side of the join
    (together, times the join index) does not fit half of the budget, or the other side reads the same file."""
    size = {id(l): (f if isinstance(f, int) else f.bytes) for l, f in zip(leaves, feet)}
    for i, n in enumerate(chain[:-1]):
        if agg_at is not None and i <= agg_at:
            continue
        if not isinstance(n, Join):
            continue
        child = chain[i + 1]
        other = n.right if child is n.left else n.left
        ls = _leaves(other)
        if (n.how != "inner" and child is n.right) or sum(size[id(l)] for l in ls) * RESIDENT_FACTOR > budget // 2 \
                or any(isinstance(l, ParquetScan) and os.path.abspath(l.path) == big.path for l in ls):
            return i
    return None


def _distinct_to_aggregate(plan):
    """The plan with every `Aggregate(Distinct(X), keys, aggs)` rewritten as `Aggregate(Aggregate(X, every column of X,
    [count(1)]), keys, aggs)`, or None when there is none (F-056 step 3c: count(distinct x) per group).

    Removing duplicate rows is grouping by all their columns, which the engine does with the same rules (one group for
    every NaN, -0.0 with 0.0, one for NULL), so the answer is the same; and a group-by is what streams: its partials
    merge, and when the (keys, x) pairs are more than fit they are hash-partitioned to disk, each partition
    deduplicated on its own."""
    changed = [False]

    def walk(n):
        if isinstance(n, Join):
            return dataclasses.replace(n, left=walk(n.left), right=walk(n.right))
        if getattr(n, "input", None) is None:
            return n
        n = dataclasses.replace(n, input=walk(n.input))
        if isinstance(n, Aggregate) and isinstance(n.input, Distinct) and not n.input.subset:
            names = (LazyFrame(n.input.input).schema or pa.schema([])).names
            if names:
                changed[0] = True
                inner = Aggregate(n.input.input, [col(c) for c in names], [Expr("count", Expr("lit", 1)).alias("__dn")])
                return dataclasses.replace(n, input=inner)
        return n
    out = walk(plan)
    return out if changed[0] else None


def analyse(plan, budget: int, device: str, reproducible: bool):
    """A `Plan` if the plan should and can stream, None if it fits in memory, else the reason it cannot
    (a string). A plan with a Distinct below an Aggregate that declines for it is tried again as the group-by it
    stands for (`_distinct_to_aggregate`); the `Plan` then names the rewritten plan to run."""
    got = _analyse(plan, budget, device, reproducible)
    if isinstance(got, Plan):
        got.plan = plan
    elif isinstance(got, str) and "Distinct" in got:
        alt = _distinct_to_aggregate(plan)
        if alt is not None:
            again = _analyse(alt, budget, device, reproducible)
            if isinstance(again, Plan):
                again.plan = alt
                return again
    return got


def _analyse(plan, budget: int, device: str, reproducible: bool):
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
        gi = _grace_at(chain, agg_at, leaves, feet, big, budget)
        prune = False
        if agg_at is None:                  # rows: the top of the part that works row by row on the big file
            top_at = len(chain) - 1 if gi is None else gi
            for i in range(top_at - 1, -1, -1):
                n = chain[i]
                if isinstance(n, Filter) or (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)) \
                        or (isinstance(n, Join) and (n.how == "inner" or chain[i + 1] is n.left)):
                    top_at = i
                else:
                    break
            top = chain[top_at]
            prune = all(isinstance(n, Filter) for n in chain[top_at:-1])
        else:
            lower = len(chain) - 1 if gi is None else gi              # below a Grace join the sides are streamed on their own
            for n, child in zip(chain[agg_at + 1:lower], chain[agg_at + 2:lower + 1]):
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
        if gi is None and sum(1 for l in leaves if isinstance(l, ParquetScan) and os.path.abspath(l.path) == big.path) > 1:
            raise _Decline("the big file is read twice by the plan")
        sub = {id(l) for l in _leaves(agg.input if agg is not None else top)}
        if gi is not None:                                         # the two sides of the Grace join are not resident
            sub -= {id(l) for l in _leaves(chain[gi])}
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
    if gi is not None:                              # a Grace join: the pairs of partitions are the batches
        return Plan(spec, big, [], batch, cap, budget, md.num_row_groups, top, {}, int(others * RESIDENT_FACTOR), False,
                    chain[gi])
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

    def __init__(self, spec: _Spec, budget: int, root=None, salt: int = 0, depth: int = 0, held=None):
        self.spec, self.budget = spec, budget
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
            sub = _State(self.spec, self.budget, self.root, self.salt + 1, self.depth + 1, self.held)
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
            if kind == "imean":                               # F-126: hi * 2^32 + lo, divided once
                arr = _wide_mean(t, hidden)
            elif kind == "mean" and pa.types.is_integer(t.column(hidden[0]).type):     # an exact int64 total: no cast to a double
                arr = _wide_mean(t, (hidden[0], None, hidden[1]))
            elif kind == "mean":
                s, c = t.column(hidden[0]), t.column(hidden[1])
                arr = pc.if_else(pc.greater(c, 0), pc.divide(pc.cast(s, pa.float64()), pc.cast(c, pa.float64())),
                                 pa.scalar(None, pa.float64()))
            else:
                arr = t.column(hidden[0])
            arrays.append(arr)
            names.append(name)
        return pa.table(arrays, names=names)


def _wide_mean(t: pa.Table, hidden) -> pa.Array:
    """The mean column of merged hidden columns (sum of the low halves, sum of the high halves or None, count): NULL
    for a group with no valid value, the total divided once like DuckDB's HUGEINT."""
    import numpy as np
    from .np_arrow import combine, scalar, to_numpy, array, int_mean_of_totals, int_mean_of_halves
    zero = scalar(0, type=pa.int64())

    def ints(name):
        return to_numpy(pc.fill_null(combine(t.column(name)), zero))
    lo, c = ints(hidden[0]), ints(hidden[2])
    res = int_mean_of_totals(lo, c) if hidden[1] is None else int_mean_of_halves(lo, ints(hidden[1]), c)
    return array(np.ma.array(res, mask=c == 0), type=pa.float64())


# --- F-113: writing a result bigger than memory -------------------------------------------------------------------

class ParquetOut:
    """The Parquet file `LazyFrame.sink_parquet` writes. A streamed query hands it the pieces of its result as they
    are produced (`write`), small ones buffered into row groups of about `flush_bytes`; a query that was not
    streamed hands it its table at the end. The file is written next to its final name and renamed when complete, so
    a failed query leaves neither a partial file nor a damaged old one."""

    def __init__(self, path, compression: str = "zstd"):
        self.path = os.path.abspath(os.fspath(path))
        self.tmp = None                          # a unique file next to `path` (threads may sink to one path), made on open
        self.compression = compression
        self.writer = None
        self.schema: Optional[pa.Schema] = None
        self.empty: Optional[pa.Table] = None
        self.buffer: List[pa.Table] = []
        self.buffered = 0
        self.rows = 0
        self.streamed = False                    # a streamed query wrote into it (its returned table is empty)
        self.flush_bytes = 32 << 20

    def attach(self, budget: int) -> None:
        self.streamed = True
        self.flush_bytes = max(256 << 10, min(32 << 20, budget // 16))

    def write(self, t: pa.Table) -> None:
        t = _plain(t)
        if t.num_rows == 0:
            self.empty = t
            return
        self.buffer.append(t)
        self.buffered += t.nbytes
        if self.buffered >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        t = self.buffer[0] if len(self.buffer) == 1 else pa.concat_tables(self.buffer, promote_options="default")
        self.buffer, self.buffered = [], 0
        self._write(t)

    def _write(self, t: pa.Table) -> None:
        import pyarrow.parquet as pq
        if self.writer is None:
            import tempfile
            fd, self.tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=os.path.basename(self.path) + ".", suffix=".partial")
            os.close(fd)
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(self.tmp, 0o666 & ~umask)
            self.schema = t.schema
            self.writer = pq.ParquetWriter(self.tmp, self.schema, compression=self.compression)
        elif not t.schema.equals(self.schema):
            try:
                t = t.select(self.schema.names).cast(self.schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, KeyError) as e:
                raise TypeError(f"sink_parquet: a later piece of the result has columns {t.schema.names} of types "
                                f"{[str(x) for x in t.schema.types]}, which do not fit the first piece's "
                                f"{[str(x) for x in self.schema.types]}: {e}") from e
        self.writer.write_table(t)
        self.rows += t.num_rows

    def no_rows(self) -> pa.Table:
        sch = self.schema or (self.empty.schema if self.empty is not None else None)
        return np_arrow.empty(sch if sch is not None else pa.schema([]))

    def finish(self, table: Optional[pa.Table] = None) -> int:
        """Close the file (writing `table` first when the query was not streamed) and move it into place."""
        if table is not None and not self.streamed:
            self.write(_plain(table))
            self.empty = self.empty if self.empty is not None else _plain(table)
        self.flush()
        if self.writer is None:                                  # no rows: a file with the columns
            self._write(self.empty if self.empty is not None else np_arrow.empty(pa.schema([])))
        self.writer.close()
        self.writer = None
        os.replace(self.tmp, self.path)
        return self.rows

    def reset(self) -> None:
        """Forget everything written so far (F-126: a streamed run that raised is started again from its first batch):
        the partial file is removed and the next write opens a fresh one."""
        self.abort()
        self.tmp, self.schema, self.empty = None, None, None
        self.buffer, self.buffered, self.rows = [], 0, 0
        self.streamed = False

    def abort(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        self.writer = None
        try:
            if self.tmp:
                os.remove(self.tmp)
        except OSError:
            pass


def sink_parquet(lf, path, device: str = "cpu", *, compression: str = "zstd", **kwargs) -> int:
    """Run `lf` and write its result to the Parquet file `path`; returns the number of rows written. A result bigger
    than the host memory budget is streamed into the file piece by piece (batches of a filter, spilled groups, join
    pairs, sorted ranges) instead of being built as one table."""
    out = ParquetOut(path, compression)
    token = _out.set(out)
    try:
        table = lf.compute(device, **kwargs)
        n = out.finish(table)
    except BaseException:
        out.abort()
        raise
    finally:
        _out.reset(token)
    prov = dict(lf.last_compile_provenance or {})
    prov["sink"] = {"path": out.path, "rows": n, "streamed": out.streamed}
    lf._last_compile_provenance = prov
    return n


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
        self.out: Optional[ParquetOut] = None          # F-113: the rows go straight into this file, none are kept
        self.ext = None                                # kind "sort": the external sort the rows are handed to
        self.sort_out: Optional[ParquetOut] = None     # kind "sort": the file the sorted ranges are written to when the plan allows

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
        if self.out is not None:
            self.out.write(t)
            self.rows += t.num_rows
            return
        if self.ext is not None:
            self.ext.add(t)
            self.rows += t.num_rows
            return
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

    def _too_big(self, progress: Optional[float], size: Optional[int] = None) -> None:
        if size is not None:                                     # a sorted result that does not fit
            raise MemoryError(
                f"the result of this query is bigger than memory: the {self.ext.rows:,} sorted rows are about "
                f"{fmt_bytes(self.ext.nbytes)}, and only {fmt_bytes(self.limit)} fits the host memory budget of {fmt_bytes(self.budget)}. Write it to a file "
                f"with sink_parquet(path), add a LIMIT or a tighter filter, aggregate it, or raise MXFRAME_HOST_MEM_BUDGET")
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
            f"{fmt_bytes(self.limit)} fits the host memory budget of {fmt_bytes(self.budget)}. Write it to a file "
            f"with sink_parquet(path), add a LIMIT or a tighter filter, aggregate it, or raise MXFRAME_HOST_MEM_BUDGET")

    def _post(self, plan, above, r: pa.Table) -> pa.Table:
        """`r` (rows in sorted order) through the row-local nodes above the Sort."""
        return r if not above else LazyFrame(_replace(plan, self.sort, Scan(r))).compute(
            "cpu", optimize=False, validate=False)

    def finish(self, plan, top) -> tuple:
        """(the table the streamed part produced, the plan to run on it): the rest of the original plan with the
        streamed part replaced by that table. A sorted result replaces the Sort node itself."""
        if self.ext is None:
            table = self.result()
            return table, _replace(plan, top, Scan(table))
        above = list(reversed(_chain(plan, self.sort)[:-1]))                 # the nodes over the Sort, nearest first
        local = all(isinstance(n, Filter) or (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)) for n in above)
        if self.sort_out is not None and local:                              # sorted ranges, each finished by the nodes above
            self.sort_out.attach(self.budget)
            wrote = False
            for r in self.ext.ranges():
                wrote = True
                self.sort_out.write(self._post(plan, above, r))
            if not wrote and self.ext.empty is not None:             # no rows: the columns the nodes above give
                self.sort_out.write(self._post(plan, above, self.ext.empty))
            table = self.sort_out.no_rows()
            return table, Scan(table)
        got, size = [], 0
        for r in self.ext.ranges():
            size += r.nbytes
            if size > self.limit:
                self._too_big(None, size)
            got.append(r)
        table = got[0] if len(got) == 1 else pa.concat_tables(got) if got else self.ext.empty
        return table, _replace(plan, self.sort, Scan(table))

    def result(self) -> pa.Table:
        if self.out is not None:
            return self.out.no_rows()
        if self.kind == "top-k" and len(self.tables) > 1:
            self._reduce()
        return self._concat()


def _sink_for(plan, top, limit_direct: int, limit_above: int, budget: int, out: Optional[ParquetOut] = None,
              root=None, held=lambda: 0) -> _Sink:
    """The sink for what sits above `top` in `plan` (`_chain` finds it): a Sort then (optionally some row-local
    Projects and) a Limit is a top-k, a Limit alone stops early, anything else collects."""
    above = list(reversed(_chain(plan, top)[:-1]))
    limit = limit_above if above else limit_direct
    name = type(above[0]).__name__ if above else ""
    if above and isinstance(above[0], Limit) and above[0].n is not None:
        return _Sink("limit", above[0].n + above[0].offset, None, limit, budget, name)
    if above and isinstance(above[0], Sort):
        for n in above[1:]:
            if isinstance(n, Limit):
                if n.n is None:                           # no limit: every row is needed
                    break
                return _Sink("top-k", n.n + n.offset, above[0], limit, budget, name)
            if not (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)):
                break
    if above and isinstance(above[0], Sort) and root is not None:      # no LIMIT: sorted runs on disk, merged by key ranges
        from .extsort import ExternalSort
        sink = _Sink("sort", 0, above[0], limit, budget, name)
        sink.ext, sink.sort_out = ExternalSort(above[0], root, budget, held), out
        return sink
    sink = _Sink("collect", 0, None, limit, budget, name)
    if out is not None and not above:                  # the rows are the result: write them as they come
        sink.out = out
        out.attach(budget)
    return sink


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
    if got.grace is not None:
        return (f"grace join, {_spill.PARTITIONS} partitions: both sides of the {got.grace.how} join on "
                f"{', '.join(got.grace.left_on)} are hash-partitioned to disk and joined pair by pair "
                f"(pair target {fmt_bytes(got.batch_target)}, host budget {fmt_bytes(budget)})")
    return (f"streams in {len(got.batches)} batches of row groups ({len(got.leaf.groups)} of {got.all_groups} row groups "
            f"read, host budget {fmt_bytes(budget)})")


class _Runner:
    """Computes one batch plan on the query's device and counts where it ran (provenance)."""

    def __init__(self, device, strict, prefer, placement, gpu_fraction):
        self.kw = dict(optimize=False, validate=False, strict=strict, prefer=prefer, placement=placement,
                       gpu_fraction=gpu_fraction)
        self.device = device
        self.devices, self.pieces, self.paths = Counter(), Counter(), Counter()
        self.biggest = 0

    def __call__(self, plan_b) -> pa.Table:
        sub = LazyFrame(plan_b)
        part = sub.compute(self.device, **self.kw)
        p = sub.last_compile_provenance or {}
        self.devices[p.get("auto_device") or p.get("device") or self.device] += 1
        self.paths[p.get("path", "")] += 1
        if p.get("gpu_pieces"):
            self.pieces[p["gpu_pieces"]] += 1
        return part


def _stream_parts(got: Plan, st, plan_for, run):
    """Run `plan_for()` once per batch of `got` (row groups, or slices of an oversized one) with the streamed scan
    reading that batch: yields (result, bytes of the file read, description)."""
    size_of = dict(zip(got.leaf.groups, got.leaf.sizes))
    sliced = Counter()
    n_slices = Counter(b[0] for b in got.batches if len(b) == 1 and b[0] in got.slice_rows)
    for batch in got.batches:
        st.groups = tuple(batch)
        st.slice_rows = got.slice_rows.get(batch[0]) if len(batch) == 1 else None
        st.slice_last = False
        if st.slice_rows:
            st.slice_no = sliced[batch[0]]
            sliced[batch[0]] += 1
            st.slice_last = sliced[batch[0]] == n_slices[batch[0]]
        share = size_of[batch[0]] / n_slices[batch[0]] if st.slice_rows else sum(size_of[g] for g in batch)
        run.biggest = max(run.biggest, int(share))
        yield run(plan_for()), share, f"{len(batch)} row groups"


def try_stream(lf, plan, device: str, *, strict: bool, prefer, placement, gpu_fraction, verbose: bool = False):
    """(table, provenance) if `plan` is streamed, None to run it in memory as before. Called by `compute()`
    on the optimized plan, before any Parquet source is read.

    F-126: an integer `avg` never raises on an overflow, but its hidden sum is an int64 like any `sum` (the fast
    kernels). When a batch or the merge overflows, the whole run starts again with every integer mean kept as two
    sums (`_widen_mean`); a real `sum` that overflows raises again there, as F-060 decided."""
    kw = dict(strict=strict, prefer=prefer, placement=placement, gpu_fraction=gpu_fraction, verbose=verbose)
    try:
        return _try_stream(lf, plan, device, **kw)
    except OverflowError:
        if not any(e.op == "mean" for a in _walk_aggregates(plan) for e in a.aggs):
            raise
    if _out.get() is not None:
        _out.get().reset()                       # rows a sink already received would be written again
    token = _wide.set(True)
    try:
        return _try_stream(lf, plan, device, **kw)
    finally:
        _wide.reset(token)


def _walk_aggregates(plan):
    node = plan
    while node is not None:
        if isinstance(node, Aggregate):
            yield node
        if isinstance(node, Join):
            yield from _walk_aggregates(node.left)
            yield from _walk_aggregates(node.right)
            return
        node = getattr(node, "input", None)


def _try_stream(lf, plan, device: str, *, strict: bool, prefer, placement, gpu_fraction, verbose: bool = False):
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
    if isinstance(got, Plan):
        plan = got.plan
    if isinstance(got, str):
        warnings.warn(f"omxcore: this query's working set does not fit in memory (budget {fmt_bytes(budget)}); "
                      f"not streamed: {got}. Running it in memory, which may run out of memory.",
                      RuntimeWarning, stacklevel=3)
        return None
    spec, leaf = got.spec, got.leaf
    root = _spill.SpillDir("join partitions" if got.grace is not None else "group state")
    out = _out.get()
    sink, state, batches_run, rows_info, graced = None, None, 0, None, None
    free = max(budget - got.resident, budget // 8)          # what the dimension tables kept resident leave
    if spec is None:                                        # a batch is in flight (input x working factor) while rows are collected
        sink = _sink_for(plan, got.top, max(free - got.batch_target * WORKING_FACTOR, budget // 8), free // WORKING_FACTOR,
                         budget, out, root, lambda: got.resident)
    token, out_token = _active.set(True), _out.set(None)      # a nested streamed compute (a Grace side) must not write into the sink file
    try:
        with streaming(leaf.path) as st:
            run = _Runner(device, strict, prefer, placement, gpu_fraction)
            state = _State(spec, budget, root, held=lambda: got.resident + (sink.bytes if sink else 0)) if spec else None
            if got.grace is not None:
                from . import grace
                batch_of = (lambda: spec.partial) if spec else (lambda: sink.batch_plan(got.top))
                graced = grace.Grace(plan, got, root, run, batch_of, _Runner("cpu", strict, prefer, placement, gpu_fraction))
                parts = graced.pairs()
            else:
                parts = _stream_parts(got, st, (lambda: spec.partial) if spec else
                                      (lambda: sink.batch_plan(_pruned(got) if got.prune else got.top)), run)
            total_bytes, done_bytes = max(sum(leaf.sizes), 1), 0
            for part, share, n_in in parts:
                batches_run += 1
                done_bytes += share or 0
                if spec is not None:
                    state.add(part)
                else:
                    sink.add(part, done_bytes / total_bytes if share is not None else None)
                if verbose:
                    print(f"[omxcore] streamed batch {batches_run}/{len(got.batches) or '?'}: "
                          f"{n_in}, {part.num_rows} {'partial ' if spec else ''}rows")
                if sink is not None and sink.full:
                    break
            stage_spec, stage_state, stage_top, stages = spec, state, got.top, 1
            while True:
                if stage_spec is not None and not stage_state.spilled:
                    merged = stage_state.result()                # the usual case: the groups fit
                    above = _replace(plan, stage_spec.agg, Scan(merged))
                    table = merged
                    break
                if stage_spec is not None:                       # spilled groups: one partition at a time
                    rowwise = stage_top is not stage_spec.agg
                    nxt = _next_stage(plan, stage_top)
                    if nxt is not None:                          # count(distinct x): another group-by over the groups
                        spec2, top2 = nxt
                        state2 = _State(spec2, budget, root, held=lambda: got.resident)
                        for piece in stage_state.pieces():
                            rows = stage_state.format(piece)
                            if rowwise:
                                rows = LazyFrame(_replace(stage_top, stage_spec.agg, Scan(rows))).compute(
                                    "cpu", optimize=False, validate=False)
                            state2.add(LazyFrame(dataclasses.replace(spec2.partial, input=Scan(rows))).compute(
                                "cpu", optimize=False, validate=False))
                        stage_spec, stage_state, stage_top, stages = spec2, state2, top2, stages + 1
                        continue
                    sink = _sink_for(plan, stage_top, free, free // WORKING_FACTOR, budget, out, root,
                                     lambda: got.resident)
                    for piece in stage_state.pieces():
                        rows = stage_state.format(piece)
                        if rowwise:
                            rows = LazyFrame(_replace(stage_top, stage_spec.agg, Scan(rows))).compute(
                                "cpu", optimize=False, validate=False)
                        sink.add(rows)
                        if sink.full:
                            break
                table, above = sink.finish(plan, stage_top)
                rows_info = {"kind": sink.kind, "batches_run": batches_run}
                if sink.ext is not None:
                    rows_info["sort"] = dict(sink.ext.stats, rows=sink.ext.rows)
                break
    finally:
        _active.reset(token)
        _out.reset(out_token)
        root.close()
    result = table if isinstance(above, Scan) else LazyFrame(above).compute(
        device, optimize=False, validate=False, strict=strict, prefer=prefer, placement=placement,
        gpu_fraction=gpu_fraction)
    info = {"batches": batches_run if got.grace is not None else len(got.batches), "row_groups": got.all_groups,
            "row_groups_read": len(leaf.groups), "host_budget": budget, "gpu_batch_cap": got.gpu_cap,
            "batch_target_bytes": got.batch_target, "largest_batch_bytes": run.biggest,
            "devices": dict(run.devices), "batch_paths": dict(run.paths)}
    if graced is not None:
        info["grace"] = graced.info()
    if spec is not None:
        info.update(merges=state.merges, group_state_rows=state.peak_state)
    if root.partitions:
        info["spill"] = {"partitions": root.partitions, "bytes": root.bytes, "depth": root.depth}
    if rows_info is not None and (spec is None or "sort" in rows_info):
        info["rows"] = rows_info
    if stages > 1:
        info["stages"] = stages
    if got.slice_rows:
        info["slices"] = {"row_groups": len(got.slice_rows), "rows_per_slice": min(got.slice_rows.values())}
    prov = {"path": "streamed", "device": device, "rows": result.num_rows, "streamed": info}
    if run.pieces:
        prov["gpu_pieces"] = "; ".join(f"{n} batches: {s}" for s, n in run.pieces.items())
    return result, prov


def _next_stage(plan, top):
    """(_Spec, top) of a second group-by right above the streamed one (`top` is the topmost row-local node over the
    first Aggregate), else None: the shape of `count(distinct x)` per group, `Aggregate(Aggregate(...))`. The first
    aggregate's groups, one hash partition at a time, are then the batches of the second, and the second's own
    partials merge like any other (it spills too when its groups are many)."""
    above = list(reversed(_chain(plan, top)[:-1]))
    if not above or not isinstance(above[0], Aggregate):
        return None
    try:
        spec = _spec(above[0])
        _argument_types(spec)
    except _Decline:
        return None
    top2 = above[0]
    for n in above[1:]:                                 # a HAVING filter / projection works group by group
        if isinstance(n, Filter) or (isinstance(n, Project) and not _has_op(n.exprs, _WINDOW)):
            top2 = n
        else:
            break
    return spec, top2


def _pruned(got: Plan):
    """The rows plan, selecting only the columns the whole plan reads from the file."""
    return Project(got.top, [col(c) for c in got.leaf.columns])
