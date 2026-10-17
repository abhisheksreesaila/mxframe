"""F-056 step 3b: a Grace hash join for joins where a side does not fit in memory.

`streamed.analyse` finds the topmost join on the path to the big file that cannot stream (the big file is the right
side of a left / semi / anti join, the other side does not fit half of the budget, or both sides read the same
file). `Grace` then:

1. streams each side of that join (row by row, in batches, as step 3a streams a row query) and hash-partitions its
   rows by the join key into `spill.PARTITIONS` files on disk. Both sides use the same hash (`spill.hash_keys`), so
   equal keys under SQL rules land in the same partition number: -0.0 with 0.0, every NaN together, an integer with
   the float of the same value, strings whatever their per-batch dictionaries. A NULL key hashes to a partition of
   its own value and never matches there, so a left join still emits it with NULLs.
2. for each pair of partitions builds a hash table on one side and probes with the other, then runs the rest of the
   plan (filters, joins to resident tables, the aggregate's partial or the row sink of step 3a) on that pair: the
   pair's plan is the batch plan with the join's two sides replaced by the partitions, so `compute(device)` picks
   the GPU when the pair fits the GPU budget and the CPU native executor otherwise. The partials go to step 2's
   merge, the rows to step 3a's sinks.
3. a pair that does not fit is not built whole: the build side stays in memory and the probe side is read in chunks
   (right for left / semi / anti, the smaller side for an inner join: a left row's result depends on the whole right
   side only). When even the build side is too big both partitions are split again with another salt, at most
   `spill.MAX_DEPTH` levels; past that one key is repeated too often and it raises a `MemoryError`.

Everything is in plan order (partition number, then file order), so reproducible mode gets the same bits.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Iterator, Optional

import pyarrow as pa

from . import spill as _spill
from . import streamed as _s
from .gpu_budget import fmt_bytes
from .lazy_expr import Expr, col
from .lazy_frame import Join, LazyFrame, LogicalPlan, Project, Scan
from .io_parquet import _renamed_ref, streaming


def _names_in(value, out: set) -> None:
    """Every column name a plan field refers to (expressions, key lists)."""
    if isinstance(value, Expr):
        if value.op == "col" and value.args and isinstance(value.args[0], str):
            out.add(value.args[0])
        for a in list(value.args) + list(value.kwargs.values()):
            _names_in(a, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                out.add(v)
            else:
                _names_in(v, out)


def _numeric(t) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t)


def _value_type(t):
    return t.value_type if pa.types.is_dictionary(t) else t


def _schema(plan) -> Optional[pa.Schema]:
    try:
        return LazyFrame(plan).schema
    except Exception:
        return None


class Grace:
    def __init__(self, plan, got, root, run, batch_of, side_run):
        self.plan, self.got, self.root, self.run, self.batch_of, self.side_run = plan, got, root, run, batch_of, side_run
        self.join: Join = got.grace
        self.target = max(got.batch_target, 1)             # bytes (in memory) one pair of partitions may hold
        self.budget = got.budget
        self.stats = Counter()
        self.schemas: list = [None, None]
        self.as_float: list = []

    def info(self) -> dict:
        return {"partitions": _spill.PARTITIONS, "partitions_total": self.root.partitions, "depth": self.root.depth,
                "pairs": self.stats["pairs"], "chunked": self.stats["chunked"], "split": self.stats["split"],
                "skipped": self.stats["skipped"], "how": self.join.how, "pair_target_bytes": self.target}

    # --- what each side has to deliver ---------------------------------------------------------------------------

    def _needed(self, side: LogicalPlan, other: LogicalPlan, schema: Optional[pa.Schema], keys) -> Optional[list]:
        """The columns of `side` the rest of the plan reads (a side is partitioned and re-read, so unused columns
        cost disk and memory); None: all of them."""
        chain = _s._chain(self.plan, self.join)
        if schema is None or not any(isinstance(n, (Project, _s.Aggregate)) for n in chain[:-1]):
            return None                                              # nothing above names the columns that are used
        refs: set = set()

        def walk(n):
            for f in dataclasses.fields(n):
                v = getattr(n, f.name)
                if isinstance(v, LogicalPlan):
                    if v is not self.join.left and v is not self.join.right:
                        walk(v)
                else:
                    _names_in(v, refs)
        walk(self.plan)
        _names_in(self.join.left_on, refs)
        _names_in(self.join.right_on, refs)
        other_schema = _schema(other)
        taken = set(other_schema.names) if other_schema is not None else set()
        cols = [n for n in schema.names if n in refs or n in keys or n in taken or _renamed_ref(n, refs)]
        return None if len(cols) == len(schema.names) else cols

    # --- one side to partitions ---------------------------------------------------------------------------------

    def _rows(self, plan) -> Iterator[pa.Table]:
        """The rows of a side as tables of a bounded size: a streamed row query when the side is a big file with
        row-wise work above it (step 3a), a nested Grace join when the side is a join of big tables, else computed
        in memory (which is what a side that fits, or an aggregate over a big file, needs)."""
        from . import native_exec
        got = _s.analyse(plan, self.budget, "cpu", native_exec.reproducible())
        if isinstance(got, _s.Plan) and got.spec is None and got.top is plan:
            if got.grace is not None:
                sub = Grace(plan, got, self.root, self.side_run, lambda: plan, self.side_run)
                with streaming("<grace join>"):               # the nested plan's resident tables are read once
                    for part, _, _ in sub.pairs():
                        yield part
                self.stats.update(sub.stats)
                return
            with streaming(got.leaf.path) as st:
                for part, _, _ in _s._stream_parts(got, st, lambda: got.top, self.side_run):
                    yield part
            return
        # an aggregate over a big file (or a side that fits) is an ordinary compute, which streams by itself: lift the
        # "no nested streaming" flag for it (its result must fit, as any query's does)
        token = _s._active.set(False)
        try:
            out = self.side_run(plan)
        finally:
            _s._active.reset(token)
        yield out

    def _partition(self, side: LogicalPlan, other: LogicalPlan, keys, schema, which: int) -> _spill.Partitioned:
        cols = self._needed(side, other, schema, set(keys))
        plan = side if cols is None else Project(side, [col(c) for c in cols])
        parts = _spill.Partitioned(self.root, keys, 0, 1, self.as_float)
        first = None
        for t in self._rows(plan):
            t = _s._plain(t)
            first = first or t.schema
            parts.write(t)
        parts.finish()
        if first is None and schema is not None:                      # a side with no rows: its declared columns
            first = _s._plain(pa.schema([schema.field(c) for c in (cols or schema.names)]).empty_table()).schema
        self.schemas[which] = first
        return parts

    # --- the pairs ---------------------------------------------------------------------------------------------

    def pairs(self):
        """Yield (result of the rest of the plan on one pair or chunk, None, description) for every pair."""
        j = self.join
        ls, rs = _schema(j.left), _schema(j.right)
        self.as_float = []
        for a, b in zip(j.left_on, j.right_on):
            lt = _value_type(ls.field(a).type) if ls is not None and a in ls.names else None
            rt = _value_type(rs.field(b).type) if rs is not None and b in rs.names else None
            self.as_float.append(bool(lt is not None and rt is not None and _numeric(lt) and _numeric(rt)
                                      and (pa.types.is_floating(lt) or pa.types.is_floating(rt))))
        lp = self._partition(j.left, j.right, j.left_on, ls, 0)
        rp = self._partition(j.right, j.left, j.right_on, rs, 1)
        for i in range(_spill.PARTITIONS):                 # (the resident tables are cached by the caller's stream context)
            yield from self._pair(lp, rp, i, 1, 0)

    def _empty(self, side: int) -> pa.Table:
        sch = self.schemas[side]
        if sch is None:
            raise _spill.SpillError("grace join: a side produced no schema")
        return sch.empty_table()

    def _chunks(self, tables, limit: int):
        cur, size = [], 0
        for t in tables:
            if cur and size + t.nbytes > limit:
                yield pa.concat_tables(cur)
                cur, size = [], 0
            cur.append(t)
            size += t.nbytes
        if cur:
            yield pa.concat_tables(cur)

    def _compute(self, left: pa.Table, right: pa.Table):
        j = dataclasses.replace(self.join, left=Scan(left), right=Scan(right))
        self.stats["pairs"] += 1
        return self.run(_s._replace(self.batch_of(), self.join, j))

    def _pair(self, lp, rp, i: int, depth: int, salt: int):
        how = self.join.how
        lrows, rrows = lp.rows[i], rp.rows[i]
        if lrows == 0 or (rrows == 0 and how in ("inner", "semi")):      # no left row to emit, or nothing can match
            lp.drop(i)
            rp.drop(i)
            self.stats["skipped"] += 1
            return
        lb, rb = lp.nbytes[i], rp.nbytes[i]
        half = self.target // 2
        right_ok, left_ok = rb <= half, how == "inner" and lb <= half
        if right_ok and (not left_ok or rb <= lb):
            build = pa.concat_tables(list(rp.read(i))) if rrows else self._empty(1)
            n = 0
            for chunk in self._chunks(lp.read(i), max(self.target - rb, half)):
                n += 1
                yield self._compute(chunk, build), None, f"pair {i} of {_spill.PARTITIONS}, chunk {n}"
            self.stats["chunked"] += n > 1
        elif left_ok:
            build = pa.concat_tables(list(lp.read(i)))
            n = 0
            for chunk in self._chunks(rp.read(i), max(self.target - lb, half)):
                n += 1
                yield self._compute(build, chunk), None, f"pair {i} of {_spill.PARTITIONS}, chunk {n}"
            self.stats["chunked"] += n > 1
        else:
            if depth >= _spill.MAX_DEPTH:
                raise MemoryError(
                    f"grace join: the rows of one join-key partition ({lrows:,} left rows, {fmt_bytes(lb)}, and "
                    f"{rrows:,} right rows, {fmt_bytes(rb)}) are still bigger than the share of the host memory "
                    f"budget ({fmt_bytes(self.target)}) after splitting the join key by hash {depth} times, and "
                    f"cannot be split any further: one join key is repeated too often to join in memory")
            self.stats["split"] += 1
            nl = _spill.Partitioned(self.root, self.join.left_on, salt + 1, depth + 1, self.as_float)
            for t in lp.read(i):
                nl.write(t)
            nl.finish()
            nr = _spill.Partitioned(self.root, self.join.right_on, salt + 1, depth + 1, self.as_float)
            for t in rp.read(i):
                nr.write(t)
            nr.finish()
            for k in range(_spill.PARTITIONS):
                yield from self._pair(nl, nr, k, depth + 1, salt + 1)
