"""F-056 step 3c: ORDER BY with no LIMIT over more rows than fit in memory (sorted runs, merged by key ranges).

`streamed` hands every batch of rows to `ExternalSort.add`. A batch is sorted on its own (the engine's `Sort` on the CPU,
so NaN is the largest value, NULLs come last, DESC and string keys follow F-070 / F-084 exactly as in memory) and
written to disk as a sorted *run*, with a hidden `__seq` column that numbers its rows (run number x 2^40 + position).
`(sort keys, __seq)` is then a strict total order that every run already follows, and it makes the sort stable by
batch order.

After the last batch the runs are merged by key ranges instead of row by row (a k-way merge with a heap would compare
rows in Python; this one only ever asks the engine to sort):

1. a few rows of every run are kept as samples; the sorted samples give `P - 1` splitter rows, so that every range
   holds about `range_bytes` of the data;
2. each run is loaded alone and cut at the splitters (sorting the splitters together with the run tells how many of
   its rows come before each one); its `P` slices are appended to the range files;
3. range by range, the slices are read back, sorted (a merge of `k` sorted pieces) and yielded without `__seq`.

The ranges come out in order, so their concatenation is the sorted result and can be written to a Parquet file one
range at a time. Memory is one batch while writing runs, one run and the splitters while cutting, and one range while
merging. At most `MAX_RANGES` ranges are open files at a time, so a sort can be about `MAX_RANGES x range_bytes` big
(a MemoryError names the size otherwise).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Iterator, List, Optional

import numpy as np
import pyarrow as pa

from . import np_arrow
from . import spill as _spill
from .gpu_budget import fmt_bytes
from .lazy_expr import col
from .lazy_frame import LazyFrame, Scan, Sort

MAX_RANGES = 256
SAMPLES_PER_RUN = 64
SORT_FACTOR = 6                     # the engine's sort peaks at about 4-5x its input (measured)
SEQ_BITS = 40                       # rows per run: 2^40


def _sort(table: pa.Table, by, desc) -> pa.Table:
    return LazyFrame(Sort(Scan(table), list(by), list(desc))).compute("cpu", optimize=False, validate=False)


class _Files:
    """One Arrow IPC stream per key range under a `SpillDir`; slices are appended in run order."""

    def __init__(self, root: _spill.SpillDir, n: int):
        self.root, self.n = root, n
        self.base = root.name()
        self.writers: dict = {}
        self.sinks: dict = {}
        self.rows = [0] * n
        self.nbytes = [0] * n
        root.partitions += n

    def path(self, i: int) -> str:
        return f"{self.base}_r{i:03d}.arrows"

    def write(self, i: int, t: pa.Table) -> None:
        if t.num_rows == 0:
            return
        if i not in self.writers:
            self.sinks[i] = pa.OSFile(self.path(i), "wb")
            self.writers[i] = pa.ipc.new_stream(self.sinks[i], t.schema)
        self.root.check_space(int(t.nbytes * 1.2))
        try:
            self.writers[i].write_table(t)
        except OSError as e:
            raise _spill.SpillError(f"sorting spilled to {self.root.path} failed after about {self.root.written:,} bytes: {e}") from e
        self.root.written += t.nbytes
        self.rows[i] += t.num_rows
        self.nbytes[i] += t.nbytes

    def finish(self) -> None:
        try:
            for i, w in self.writers.items():
                w.close()
                self.sinks[i].close()
                self.root.bytes += os.path.getsize(self.path(i))
        except OSError as e:
            raise _spill.SpillError(f"sorting spilled to {self.root.path} failed while finishing the range files: {e}") from e
        finally:
            self.writers, self.sinks = {}, {}

    def read(self, i: int) -> pa.Table:
        path = self.path(i)
        if not os.path.exists(path):
            if self.rows[i]:
                raise _spill.SpillError(f"sort range {i} ({self.rows[i]:,} rows) is missing: {path} was removed while the query ran")
            return None
        try:
            with pa.OSFile(path, "rb") as f:
                t = pa.ipc.open_stream(f).read_all()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if t.num_rows != self.rows[i]:
            raise _spill.SpillError(f"sort range {i} of {self.root.path} holds {t.num_rows:,} rows, {self.rows[i]:,} were "
                                    f"written: the file was truncated or changed")
        return t


@dataclasses.dataclass
class _Run:
    path: str
    rows: int
    nbytes: int


class ExternalSort:
    """Sorts the rows handed to `add` (any number of batches, together bigger than memory) by `sort.by` /
    `sort.descending`; `ranges()` yields the sorted result in order, a table at a time."""

    def __init__(self, sort: Sort, root: _spill.SpillDir, budget: int, held=lambda: 0):
        self.by, self.desc = list(sort.by), list(sort.descending)
        self.root, self.budget, self.held = root, budget, held
        self.runs: List[_Run] = []
        self.samples: List[pa.Table] = []
        self.schema: Optional[pa.Schema] = None
        self.empty: Optional[pa.Table] = None
        self.rows = 0
        self.nbytes = 0
        self.stats = {"runs": 0, "ranges": 0, "bytes": 0}

    # --- sorted runs -------------------------------------------------------------------------------------------

    @property
    def run_bytes(self) -> int:
        """The most one run holds: the engine's sort needs about SORT_FACTOR times its input, on top of the batch the
        caller still holds."""
        return max((self.budget - self.held()) // 32, 64 << 10)

    def add(self, t: pa.Table) -> None:
        """Sort a batch of rows into runs of at most `run_bytes` and write them to disk."""
        from .streamed import _plain
        t = _plain(t)
        if t.num_rows == 0:
            self.empty = t
            return
        if self.schema is None:
            self.schema = t.schema
        elif not t.schema.equals(self.schema):
            t = t.select(self.schema.names).cast(self.schema)
        step = max(int(self.run_bytes * t.num_rows / max(t.nbytes, 1)), 1024)
        for at in range(0, t.num_rows, step):
            self._run(t.slice(at, step))

    def _run(self, t: pa.Table) -> None:
        from .streamed import _plain
        run = _sort(t, self.by, self.desc)
        n = run.num_rows
        seq = (len(self.runs) << SEQ_BITS) + np.arange(n, dtype=np.int64)
        run = _plain(run).append_column("__seq", pa.array(seq))
        at = np.unique(np.linspace(0, n - 1, num=min(n, SAMPLES_PER_RUN)).astype(np.int64))
        self.samples.append(np_arrow.take_rows(run, at))
        path = self.root.name() + ".run"
        self.root.check_space(int(run.nbytes * 1.2))
        try:
            with pa.OSFile(path, "wb") as f, pa.ipc.new_stream(f, run.schema) as w:
                w.write_table(run)
        except OSError as e:
            raise _spill.SpillError(f"sorting spilled to {self.root.path} failed after about {self.root.written:,} bytes: {e}") from e
        self.root.written += run.nbytes
        self.root.bytes += os.path.getsize(path)
        self.runs.append(_Run(path, n, run.nbytes))
        self.rows += n
        self.nbytes += run.nbytes
        self.stats["runs"], self.stats["bytes"] = len(self.runs), self.nbytes

    def _load(self, run: _Run) -> pa.Table:
        try:
            with pa.OSFile(run.path, "rb") as f:
                t = pa.ipc.open_stream(f).read_all()
        except OSError as e:
            raise _spill.SpillError(f"sort run {run.path} cannot be read back: {e}") from e
        finally:
            try:
                os.remove(run.path)
            except OSError:
                pass
        if t.num_rows != run.rows:
            raise _spill.SpillError(f"sort run {run.path} holds {t.num_rows:,} rows, {run.rows:,} were written: "
                                    f"the file was truncated or changed")
        return t

    # --- merge by key ranges -----------------------------------------------------------------------------------

    @property
    def range_bytes(self) -> int:
        """What one range should hold: sorting it needs about SORT_FACTOR times that."""
        return max((self.budget - self.held()) // 16, 64 << 10)

    def _splitters(self, p: int) -> Optional[pa.Table]:
        keys = self.by + [col("__seq")]
        s = _sort(pa.concat_tables(self.samples), keys, self.desc + [False])
        pos = np.unique((np.arange(1, p) * s.num_rows) // p)
        return np_arrow.take_rows(s, pos) if len(pos) else None

    def ranges(self) -> Iterator[pa.Table]:
        """The sorted rows, in order, as tables of about `range_bytes` each (the runs are consumed)."""
        if not self.runs:
            return
        hard = max((self.budget - self.held()) // 3, 64 << 10)
        p = min(MAX_RANGES, max(1, -(-self.nbytes // self.range_bytes)))
        if p == 1 and self.nbytes > hard:
            p = 2
        split = self._splitters(p) if p > 1 else None
        if split is None:
            if self.nbytes > hard:
                raise MemoryError(f"sort: the {self.rows:,} rows to sort are {fmt_bytes(self.nbytes)}, more than the "
                                  f"{fmt_bytes(hard)} one sorted range may hold in the host budget of {fmt_bytes(self.budget)}")
            t = pa.concat_tables([self._load(r) for r in self.runs])
            self.stats["ranges"] = 1
            yield self._finish(t)
            return
        p = split.num_rows + 1
        self.stats["ranges"] = p
        files = _Files(self.root, p)
        tags = np.zeros(split.num_rows, dtype=np.int8)
        s_tagged = split.append_column("__tag", pa.array(tags))
        keys = self.by + [col("__seq"), col("__tag")]
        for run in self.runs:
            t = self._load(run)
            both = pa.concat_tables([s_tagged, t.append_column("__tag", pa.array(np.ones(t.num_rows, dtype=np.int8)))])
            tag = np_arrow.to_numpy(_sort(both, keys, self.desc + [False, False]).column("__tag"))
            cuts = np.flatnonzero(tag == 0) - np.arange(split.num_rows)           # run rows before each splitter
            bounds = [0] + cuts.tolist() + [t.num_rows]
            for i in range(p):
                files.write(i, t.slice(bounds[i], bounds[i + 1] - bounds[i]))
            del t, both
        files.finish()
        hard_rows = hard
        for i in range(p):
            if files.nbytes[i] > hard_rows:
                raise MemoryError(f"sort: one key range holds {files.rows[i]:,} rows, {fmt_bytes(files.nbytes[i])}, more than "
                                  f"the {fmt_bytes(hard)} that fits the host budget of {fmt_bytes(self.budget)} while it is "
                                  f"sorted; the {self.rows:,} rows to sort ({fmt_bytes(self.nbytes)}) need more than "
                                  f"{MAX_RANGES} ranges")
            t = files.read(i)
            if t is not None:
                yield self._finish(t)

    def _finish(self, t: pa.Table) -> pa.Table:
        t = _sort(t, self.by + [col("__seq")], self.desc + [False])
        return t.drop_columns(["__seq"])
