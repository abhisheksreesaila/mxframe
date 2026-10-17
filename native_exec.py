"""Native executor front end (F-030 / F-031, milestones M0-M1 of docs/design/mojo-executor.md).

Python only plans: it checks the plan shape, compiles the expressions to bytecode (the existing
`pipeline_program` compiler), hands the input columns over through the Arrow C Data Interface and
makes ONE call, `mx_execute` in ``libmxexec_cpu.so``. Group keys, filtering, aggregation, finishing
means / counts, sorting and building the output columns all run in Mojo; the result buffers are
wrapped zero-copy and freed by the executor when the table is garbage collected.

Covered: ``Limit? Sort? Having? Aggregate(Filter*(Scan))`` on the CPU, with string keys (utf8, or
dictionary-encoded with a dictionary per chunk; merged by hashing in Mojo) and int32 / int64 keys
(any range: direct, compacted or hashed),
null-free numeric value columns, and sum / mean / count / min / max (NaN-aware, as DuckDB). HAVING is
an AND of ``aggregate <op> number``; ORDER BY any mix of group keys and aggregates. Join results are
materialized before the aggregate, so this is the final stage of most queries. Groups come out in key
order on every path (dense, compacted, hashed), as pandas' groupby does by default; ORDER BY ties keep
that order. The old path emits first appearance: SQL leaves both orders undefined. Everything else
returns ``None`` and the existing path runs: the executor never changes an answer.
On by default; ``MXFRAME_NATIVE=0`` turns it off.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time as _time
from typing import List, Optional

import pyarrow as pa

from .lazy_frame import Aggregate, Filter, Limit, Scan, Sort

MAGIC = 0x4D584531
MAX_GROUPS = 1 << 20          # key ranges addressed directly
MAX_COMPACT = 1 << 27         # key ranges compacted through a presence table (bytes + int32 per code)
DISTINCT_MAX_GROUPS = 1 << 17     # SELECT DISTINCT through the group-by: directly addressed key spaces up to this,
                                  # wider ones keep mx_distinct (measured: past ~2e5 groups its hash beats the
                                  # group-by plus the sort by first row)
DISTINCT_FLAG = 1                 # plan word 27: groups directly addressed or E_TOO_MANY_GROUPS
REPRO_FLAG = 2                    # plan word 27: float sums in the defined order (F-064)
HASH_FLAG = 4                     # plan word 27: hashed groups whatever the key ranges (F-069)

import contextvars as _contextvars
_reproducible = _contextvars.ContextVar("mxframe_reproducible", default=None)    # per thread / task, never process-wide


def reproducible() -> bool:
    """Whether the compute() running in this thread asked for reproducible float sums: its explicit argument,
    else MXFRAME_REPRODUCIBLE (read here, never written)."""
    v = _reproducible.get()
    if v is None:
        return os.environ.get("MXFRAME_REPRODUCIBLE", "0") not in ("0", "false", "off", "")
    return bool(v)

_CMP = {"lt": 0, "le": 1, "gt": 2, "ge": 3, "eq": 4, "ne": 5}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
K_DICT, K_I32, K_I64, K_STR, K_EXPR, K_F64, K_PF64, K_F32, K_PF32 = 0, 1, 2, 3, 4, 5, 6, 7, 8
_GROUP_PATHS = {0: "dense", 1: "compact", 2: "hash"}
_OPS = {"sum": 0, "min": 1, "max": 2, "mean": 3, "count": 4, "isum": 5, "imean": 6, "imin": 7, "imax": 8}
_INT_OUT = ("count", "isum", "imin", "imax")      # outputs the executor writes as int64 words
_DT = {pa.int32(): 0, pa.int64(): 1, pa.float32(): 2, pa.float64(): 3}
DT_I64X = 8      # an int64 column only OP_LOADI reads (F-060c): exact past 2^53, so the executor's float64 range check skips it

_lib = None
_lock = threading.Lock()
_last = {"path": None, "decline": None, "py_ms": None, "native_ms": None, "group_path": None}
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def enabled() -> bool:
    """On by default since F-031; MXFRAME_NATIVE=0 turns it off (the existing paths answer)."""
    return os.environ.get("MXFRAME_NATIVE", "1") not in ("0", "false", "off")


def _path():
    from .pipeline_kernels import _find
    return _find("libmxexec_cpu.so", "MXFRAME_EXEC_CPU_LIB")


def available() -> bool:
    return _lib is not None or _path() is not None      # loaded: no filesystem lookup per call


def _load():
    global _lib
    with _lock:
        if _lib is None:
            lib = ctypes.CDLL(str(_path()))
            lib.mx_execute.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_execute.restype = ctypes.c_int64
            lib.mx_exec_free.argtypes = [ctypes.c_int64]
            lib.mx_exec_free.restype = ctypes.c_int64
            lib.mx_filter.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_filter.restype = ctypes.c_int64
            lib.mx_gather.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_gather.restype = ctypes.c_int64
            lib.mx_distinct.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_distinct.restype = ctypes.c_int64
            lib.mx_count_distinct.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_count_distinct.restype = ctypes.c_int64
            lib.mx_str_slice_in.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_str_slice_in.restype = ctypes.c_int64
            lib.mx_str_slice.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_str_slice.restype = ctypes.c_int64
            lib.mx_topk.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_topk.restype = ctypes.c_int64
            lib.mx_filter_topk.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_filter_topk.restype = ctypes.c_int64
            _lib = lib
    return _lib


def last_path():
    return _last["path"]


def last_timing():
    """(Python-side ms, native-call ms) of the last native run."""
    return _last["py_ms"], _last["native_ms"]


def last_decline():
    return _last["decline"]


def _record_limit() -> int:
    """Bytes the hashed group-by may hold in flushed records before it gives up (plan word 28; the call then
    returns E_NO_MEMORY and the caller's path answers). The records are about one per (worker flush, group),
    so up to one per row when the groups are many, and the peak measured at 10M rows, 1M groups was 1.25 x the
    record bytes: a third of MemAvailable leaves room for that, the input and the engine's other buffers.
    MXFRAME_HASH_RECORD_MB overrides (tests); a value that is not a positive number is ignored."""
    try:
        mb = float(os.environ.get("MXFRAME_HASH_RECORD_MB", ""))
        if 0 < mb < float("inf"):
            return int(mb * (1 << 20))
    except ValueError:
        pass                                    # not a number: as if unset
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 // 3
    except (OSError, ValueError):
        pass
    try:                                        # no /proc: a third of the physical memory (0 would mean unlimited)
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 3
    except (ValueError, OSError, AttributeError):
        return 0


def last_group_path():
    """How the last native group-by found its groups: "dense" (directly addressed), "compact" (key range
    compacted through a presence table) or "hash" (F-069: float keys, wide key tuples, partitioned hash
    aggregate); None for a global aggregate or when it declined."""
    return _last["group_path"]


_decline_counts: dict = {}


def strict() -> bool:
    """MXFRAME_STRICT_NATIVE=1: an unexpected exception in a native hook is raised, not turned into a quiet
    decline (the tests set it, so a broken native path cannot hide behind the fallback). Declines the code
    expects (a plan that does not fit) return None and never raise, either way."""
    return os.environ.get("MXFRAME_STRICT_NATIVE", "0") not in ("0", "false", "off", "")


def decline_counts() -> dict:
    """{reason: count} of every decline since the last reset, unexpected exceptions as ``raised: <ExcType>``
    (bench scripts print it: a speed-up or slow-down measured on a quietly declining path is not one)."""
    return dict(_decline_counts)


def reset_decline_counts() -> None:
    _decline_counts.clear()


def _count(key: str) -> None:
    _decline_counts[key] = _decline_counts.get(key, 0) + 1


def _decline(reason: str):
    _last["path"], _last["decline"], _last["group_path"] = None, reason, None
    _count(reason)
    return None


def swallowed(exc: BaseException, where: str):
    """Called from an ``except Exception`` around a native hook: raise ``exc`` under MXFRAME_STRICT_NATIVE,
    else record the decline (``raised: <ExcType>``) and return None so the caller's fallback answers.
    A MemoryError is raised too inside a ``compute(strict=True)`` (``gpu_budget.strict()``)."""
    if strict():
        raise exc
    if isinstance(exc, MemoryError):          # compute(strict=True) promises out-of-memory is never hidden
        from . import gpu_budget
        if gpu_budget.strict():
            raise exc
    _last["path"], _last["decline"] = None, f"{where} raised {type(exc).__name__}: {exc}"
    _count(f"raised: {type(exc).__name__}")
    return None


def _call(lib, plan_addr: int, out_addr: int) -> int:
    """The one native call per query (a separate function so tests can count it)."""
    return lib.mx_execute(plan_addr, out_addr)


class _Arena:
    """Owns the executor's output block; frees it when the last buffer using it is gone."""

    def __init__(self, lib, addr: int):
        self.lib, self.addr = lib, addr

    def __del__(self):
        if self.addr:
            self.lib.mx_exec_free(self.addr)
            self.addr = 0


class _Extras:
    """The chunk table's extra words after a program's columns (one set per chunk): a string column held as
    dictionary codes (dtype 5) has its dictionary's offset in the concatenation of all chunks' dictionaries;
    a plain utf8 column (dtype 6, ``OP_STR``) has three: the chunk's bytes address, its validity bitmap address
    (0: no nulls) and the bit index of the chunk table's row 0 in that bitmap (F-034). A column only tested by
    ``OP_ISNULL`` (dtype 7) has the last two (F-097); its values are never read."""

    def __init__(self, prog):
        self.prog = prog
        self.dcols = [c for c in prog.columns if c in prog.dict_concat]
        self.scols = list(prog.str_cols)
        self.ncols = list(prog.null_cols)
        self.n = len(self.dcols) + 3 * len(self.scols) + 2 * len(self.ncols)
        base = len(prog.columns)
        self.slot = {c: base + i for i, c in enumerate(self.dcols)}
        self.slot.update({c: base + len(self.dcols) + 3 * i for i, c in enumerate(self.scols)})
        self.slot.update({c: base + len(self.dcols) + 3 * len(self.scols) + 2 * i for i, c in enumerate(self.ncols)})

    def dtype(self, name: str) -> Optional[int]:
        """The program column's dtype word if it is one of these string columns, else None."""
        code = 5 if name in self.prog.dict_concat else 7 if name in self.ncols else 6
        return (code | (self.slot[name] << 8)) if name in self.slot else None

    def words(self, piece_of, row0: int) -> list:
        """The extra words of one chunk; ``piece_of(name)`` is the column's slice in it, ``row0`` its first row."""
        from .pipeline_program import _dict_key
        out = [self.prog.dict_concat[n][_dict_key(piece_of(n).dictionary)] for n in self.dcols]
        for n in self.scols:
            a = piece_of(n)
            bufs = a.buffers()
            out += [bufs[2].address if bufs[2] is not None else 0,
                    bufs[0].address if bufs[0] is not None and a.null_count else 0, a.offset - row0]
        for n in self.ncols:
            a = piece_of(n)
            a = a.indices if pa.types.is_dictionary(a.type) else a          # a dictionary column's validity is its indices'
            bitmap = a.buffers()[0]
            out += [bitmap.address if bitmap is not None and a.null_count else 0, a.offset - row0]
        return out

    def patch(self, code):
        """The program's code with the chunk table's width set to include these words."""
        if not self.n:
            return code
        code = code.copy()
        code[7] = len(self.prog.columns) + self.n
        return code


def _key_kind(col: pa.ChunkedArray):
    t = col.type
    if pa.types.is_dictionary(t):
        if t.index_type != pa.int32() or not pa.types.is_string(t.value_type):
            return None
        if any(ch.dictionary.null_count for ch in col.chunks):
            return None
        return K_DICT          # chunks may carry different dictionaries: the executor merges them
    if pa.types.is_string(t):
        return K_STR           # hashed to entry ids in Mojo
    return {pa.int32(): K_I32, pa.int64(): K_I64, pa.float64(): K_F64, pa.float32(): K_F32}.get(t)


def _segments(columns):
    """Cut every column at the union of all columns' chunk boundaries: a list of row segments,
    each a list of one Array per column. Only metadata: slices share the chunks' buffers."""
    cuts = set()
    for c in columns:
        pos = 0
        for ch in c.chunks:
            pos += len(ch)
            cuts.add(pos)
    bounds = sorted(b for b in cuts if b > 0)
    out = []
    per_col = [[(ch, 0) for ch in c.chunks if len(ch)] for c in columns]
    idx = [0] * len(columns)
    start = 0
    for end in bounds:
        seg = []
        for j, chunks in enumerate(per_col):
            while True:
                ch, ch_start = chunks[idx[j]]
                if start < ch_start + len(ch):
                    break
                idx[j] += 1
                if idx[j] < len(chunks):
                    chunks[idx[j]] = (chunks[idx[j]][0], ch_start + len(ch))
            seg.append(ch.slice(start - ch_start, end - start))
        out.append(seg)
        start = end
    return out


def _having_terms(pred, names):
    """AND-ed ``output <op> number`` terms as (output index, cmp code, constant), else None."""
    if pred is None:
        return []
    if pred.op == "and":
        out = []
        for a in pred.args:
            t = _having_terms(a, names)
            if t is None:
                return None
            out += t
        return out
    if pred.op not in _CMP or len(pred.args) != 2:
        return None
    a, b = pred.args
    op = pred.op
    if getattr(a, "op", None) == "lit" and getattr(b, "op", None) == "col":
        a, b, op = b, a, _FLIP[op]
    if getattr(a, "op", None) != "col" or getattr(b, "op", None) != "lit":
        return None
    v = b.args[0]
    if a.args[0] not in names or isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, int) and abs(v) > 2 ** 53:
        return None
    return [(names.index(a.args[0]), _CMP[op], float(v))]


def _split_post_ops(post_ops):
    """([(sort column, descending)], limit or -1) of the Sort? Limit? above an aggregate, or the
    reason it is not that shape (a str)."""
    sort_node, limit = None, -1
    for op in post_ops or []:
        if isinstance(op, Sort) and sort_node is None and limit < 0:
            sort_node = op
        elif isinstance(op, Limit) and limit < 0:
            limit = -1 if op.n is None else int(op.n)         # (no limit: the rows stay as they are)
        else:
            return f"post-op {type(op).__name__} in this position", -1
    sort_by = []                                  # (name, descending)
    if sort_node is not None:
        for e, desc in zip(sort_node.by, sort_node.descending):
            if e.op != "col":
                return "sort by an expression", -1
            sort_by.append((e.args[0], bool(desc)))
    return sort_by, limit


def _sort_spec(sort_by, keys, names):
    """(what, index, descending) per ORDER BY column (0 = group key, 1 = output), or a reason (str)."""
    sorts = []
    for name, desc in sort_by:
        if name in keys:
            sorts.append((0, keys.index(name), int(desc)))
        elif name in names:
            sorts.append((1, names.index(name), int(desc)))
        else:
            return "sort by a column that is not a key or an aggregate"
    return sorts


def try_execute(compiler, plan, having_pred, post_ops, gpu_fallback: bool = False,
                distinct: bool = False, coproc_share: bool = False) -> Optional[pa.Table]:
    """Run the whole plan in one native call, or None (the caller's path runs instead).
    `gpu_fallback`: a GPU session whose own pipeline declined the plan answers it here, on the CPU.
    `distinct`: the plan is a SELECT DISTINCT as a group-by (``distinct_plan``): only directly addressed key
    spaces up to DISTINCT_MAX_GROUPS, no plain-string keys (``mx_distinct`` takes those and wider ones).
    `coproc_share`: this is the CPU share of a CPU+GPU split (``range_partials``): runs inside the policy
    that makes every other native call decline."""
    _last["path"] = None
    t_start = _time.perf_counter()
    if not enabled():
        return _decline("disabled (MXFRAME_NATIVE=0)")
    if getattr(compiler, "_device_arg", None) != "cpu" and not gpu_fallback and not coproc_share:
        return _decline("CPU executor: device='cpu' (or a GPU session's fallback)")
    from . import coproc_fused
    if coproc_fused.current_policy() is not None and not coproc_share:
        return _decline("CPU+GPU co-processing policy active")
    if not available():
        return _decline("libmxexec_cpu.so not built")
    if not isinstance(plan, Aggregate) or not plan.aggs:
        return _decline("plan is not an aggregate")
    preds, node = [], plan.input
    while isinstance(node, Filter):
        preds.append(node.predicate)
        node = node.input
    if not isinstance(node, Scan):
        return _decline("input is not a scan of one table")
    table: pa.Table = node.table
    if table.num_rows == 0:
        return _decline("empty input")

    keys, derived, src_col = [], {}, {}        # derived: key position -> expression; src_col: key position -> its column
    for i, e in enumerate(plan.group_by):
        if e.op == "col" and e._alias in (None, e.args[0]):
            keys.append(e.args[0])
            src_col[i] = e.args[0]
            continue
        if not e._alias:
            return _decline("derived group key without a name")
        keys.append(e._alias)
        if e.op == "col" and e.args[0] in table.column_names:      # a column under another name (SQL `x AS y`)
            src_col[i] = e.args[0]
        else:
            derived[i] = e
    sort_by, limit = _split_post_ops(post_ops)
    if isinstance(sort_by, str):
        return _decline(sort_by)

    from .pipeline_program import compile_pipeline
    from .lazy_expr import Expr
    n_user = len(plan.aggs)
    hidden = [Expr("min", derived[i]).alias(f"__key{i}") for i in sorted(derived)]
    prog = compile_pipeline(preds, list(plan.aggs) + hidden, table, concat_dicts=True)
    if prog is None:
        return _decline("expression not supported by the bytecode compiler")
    key_slot = {}
    for j, i in enumerate(sorted(derived)):
        slot = prog.outputs[n_user + j][2]
        if slot is None or not prog.slot_int[slot]:
            return _decline("derived group key is not an integer expression")
        key_slot[i] = slot
    outputs = prog.outputs[:n_user]
    names = [name for name, _, _ in outputs]
    outputs, imeans = _exact_imeans(outputs)
    ranges, force_hash = {}, False
    if derived:
        # an integer expression key (TPC-H's order year): its range from one native min / max pass,
        # then the executor groups by the program's value of it (directly addressed; a range past
        # MAX_GROUPS is hashed instead, F-069)
        probe = Aggregate(plan.input, [], [a for i in derived
                                           for a in (Expr("min", derived[i]).alias(f"lo{i}"), Expr("max", derived[i]).alias(f"hi{i}"))])
        r = try_execute(compiler, probe, None, None, gpu_fallback, coproc_share=coproc_share)
        if r is None:
            return _decline("derived group key: range pass declined")
        for i in derived:
            lo, hi = r.column(f"lo{i}")[0].as_py(), r.column(f"hi{i}")[0].as_py()
            if lo is None or hi is None or lo != int(lo) or hi != int(hi):
                return _decline("derived group key: not an integer range")
            if hi - lo + 1 > MAX_GROUPS:
                if distinct:
                    return _decline("derived group key: not a small integer range")
                force_hash = True
            ranges[i] = (int(lo), int(hi))
    sorts = _sort_spec(sort_by, keys, names)
    if isinstance(sorts, str):
        return _decline(sorts)
    having = _having_terms(having_pred, names)
    if having is None:
        return _decline("HAVING other than AND-ed aggregate <op> number")
    if imeans and _imean_ordered(names, imeans, sorts, having):
        return _decline("ORDER BY / HAVING over an integer mean")
    if prog.dict_arrays:
        return _decline("string predicates outside concat mode")
    ex = _Extras(prog)                                # string columns in predicates

    cols = [table.column(c) for c in prog.columns]
    dtypes = []
    for name, c in zip(prog.columns, cols):
        if ex.dtype(name) is not None:                # codes + the chunk's offset (dtype 5), or utf8 (dtype 6)
            dtypes.append(ex.dtype(name))
            continue
        if c.type not in _DT or c.null_count:
            return _decline(f"value column type {c.type} or nulls")
        dtypes.append(DT_I64X if name in prog.exact_cols and c.type == pa.int64() else _DT[c.type])
    key_cols, kinds = [], []
    for i, k in enumerate(keys):
        if i in derived:                       # placeholder column: the key comes from the program
            key_cols.append(table.column(0))
            kinds.append(K_EXPR)
            continue
        c = table.column(src_col[i])
        kind = _key_kind(c)
        if kind is None or c.null_count:
            return _decline(f"group key type {c.type} or nulls not in M0")
        if distinct and kind in (K_STR, K_F64, K_F32):
            return _decline("distinct: plain string or float keys")
        key_cols.append(c)
        kinds.append(kind)
    every = cols + key_cols
    if not every:
        if (not keys and not preds and having_pred is None and not sort_by and limit < 0
                and all(op == "count" for _, op, _ in outputs)):
            # COUNT(col) over every row, no filter, no key: the table's row count when the counted
            # columns have no nulls (COUNT skips them); no column is read, so there is no array to hand
            # the executor
            for a in plan.aggs:
                arg = a.args[0] if a.args else None
                if getattr(arg, "op", None) == "lit":
                    continue
                if getattr(arg, "op", None) != "col" or arg.args[0] not in table.column_names \
                        or table.column(arg.args[0]).null_count:
                    return _decline("count of an expression or of a column with nulls")
            import numpy as np
            from .np_arrow import array as _npa
            result = pa.Table.from_arrays([_npa(np.array([table.num_rows], dtype=np.int64)) for _ in names], names=names)
            _last.update(path="native", decline=None, native_ms=0.0, py_ms=(_time.perf_counter() - t_start) * 1000)
            compiler.last_compile_provenance = {
                "device": "cpu", "rows": 1, "grouped": False, "n_groups": 0, "cache_hit": None,
                "compile_ms": 0.0, "path": "native", "gpu_input_path": "arrow_c_data",
            }
            return result
        return _decline("no columns")
    segs = _segments(every)          # every column cut at the same row boundaries (zero-copy slices)
    if not segs:
        return _decline("empty input")

    # ── export (Arrow C Data Interface) and the plan buffer ────────────────────────────────
    width = len(every)
    structs = []
    arrays = (ctypes.c_int64 * (len(segs) * width))()
    try:
        for r, seg in enumerate(segs):
            for j, piece in enumerate(seg):
                st = (ctypes.c_int64 * 10)()
                piece._export_to_c(ctypes.addressof(st))
                structs.append(st)
                arrays[r * width + j] = ctypes.addressof(st)
        n_keys, n_out = len(keys), len(outputs)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_kinds = (ctypes.c_int64 * max(n_keys, 1))(*kinds)
        c_outs = (ctypes.c_int64 * max(3 * n_out, 1))(
            *[v for name, op, slot in outputs
              for v in (_OPS[op], slot if slot is not None else 0, prog.nan_slots.get(name, -1))])
        c_sorts = (ctypes.c_int64 * max(3 * len(sorts), 1))(*[v for t in sorts for v in t])
        c_having = (ctypes.c_int64 * max(3 * len(having), 1))(*[v for j, c, _ in having for v in (j, c, 0)])
        c_hconst = (ctypes.c_double * max(len(having), 1))(*[x for _, _, x in having])
        doffs, row0 = [], 0
        for seg in segs:
            doffs += ex.words(lambda nm: seg[prog.columns.index(nm)], row0)
            row0 += len(seg[0])
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        code = ex.patch(prog.code)                    # the chunk table gets the extra words
        from .pipeline_kernels import default_threads
        c_xkeys = (ctypes.c_int64 * max(3 * n_keys, 1))(
            *[v for i in range(n_keys) for v in ((key_slot[i],) + ranges[i] if i in derived else (0, 0, 0))])
        max_groups = DISTINCT_MAX_GROUPS if distinct else MAX_GROUPS
        P = (ctypes.c_int64 * 29)(
            MAGIC, 2, len(segs), len(cols), n_keys, n_out, len(sorts),
            code.ctypes.data, prog.consts.ctypes.data, prog.set_bytes.ctypes.data,
            prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes), ctypes.addressof(arrays),
            ctypes.addressof(c_kinds), ctypes.addressof(c_outs), ctypes.addressof(c_sorts),
            default_threads(table.num_rows), max_groups, limit, len(having), ctypes.addressof(c_having),
            max_groups if distinct else min(MAX_COMPACT, max(MAX_GROUPS, 8 * table.num_rows)),
            ctypes.addressof(c_hconst), ex.n, ctypes.addressof(c_doffs), ctypes.addressof(c_xkeys),
            0, (DISTINCT_FLAG if distinct else 0) | (REPRO_FLAG if reproducible() else 0) | (HASH_FLAG if force_hash else 0), _record_limit())
        O = (ctypes.c_int64 * (3 + 2 * n_keys + n_out))()
        lib = _load()
        t_call = _time.perf_counter()
        rc = _call(lib, ctypes.addressof(P), ctypes.addressof(O))
        native_ms = (_time.perf_counter() - t_call) * 1000
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return _decline(f"executor declined (code {rc})")

    result = _finish_imeans(_wrap_result(lib, O, keys, kinds, outputs), imeans)
    total_ms = (_time.perf_counter() - t_start) * 1000
    _last.update(path="native", decline=None, native_ms=native_ms, py_ms=total_ms - native_ms)
    compiler.last_compile_provenance = {
        "device": "cpu", "rows": int(result.num_rows), "grouped": bool(keys), "n_groups": int(result.num_rows) if keys else 0,
        "cache_hit": None, "compile_ms": 0.0, "path": "native", "gpu_input_path": "arrow_c_data",
    }
    return result


def _exact_imeans(outputs):
    """F-126: the executor divides an integer mean's int64 total in its own lanes, which rounds a total past 2^53
    twice (DuckDB divides its exact HUGEINT total once). Ask it for the total (`isum`) and a count instead, and
    divide in `_finish_imeans`: (outputs to run, names of the integer means)."""
    run, extra, names = [], [], []
    for name, op, slot in outputs:
        if op == "imean":
            run.append((name, "isum", slot))
            extra.append((f"__imean_n{len(names)}", "count", None))
            names.append(name)
        else:
            run.append((name, op, slot))
    return run + extra, names


def _imean_ordered(names, imeans, sorts, having) -> bool:
    """True when an ORDER BY or HAVING term reads an integer mean (the executor would see its total, not the mean)."""
    idx = {names.index(n) for n in imeans}
    return any(w == 1 and i in idx for w, i, _ in sorts) or any(t[0] in idx for t in having)


def _finish_imeans(result: pa.Table, imeans) -> pa.Table:
    if not imeans:
        return result
    from .np_arrow import int_mean_column
    cols = {n: result.column(n) for n in result.column_names}
    for k, name in enumerate(imeans):
        cols[name] = int_mean_column(cols[name], cols.pop(f"__imean_n{k}"))
    return pa.table(list(cols.values()), names=list(cols))


def _wrap_result(lib, O, keys, kinds, outputs) -> pa.Table:
    """The executor's result words (see mx_execute) as a table: zero-copy views of its arena."""
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    empty_global = bool(O[2] & 1)
    _last["group_path"] = _GROUP_PATHS.get((int(O[2]) >> 1) & 3) if keys else None

    def buf(addr, nbytes):
        return pa.foreign_buffer(int(addr), max(nbytes, 0), base=owner)

    names, out_cols = [], []
    for k, (name, kind) in enumerate(zip(keys, kinds)):
        a, b = int(O[3 + 2 * k]), int(O[4 + 2 * k])
        if kind in (K_DICT, K_STR):
            nbytes = ctypes.c_int32.from_address(a + 4 * n).value
            arr = pa.Array.from_buffers(pa.string(), n, [None, buf(a, 4 * (n + 1)), buf(b, nbytes)])
        elif kind == K_I32:
            arr = pa.Array.from_buffers(pa.int32(), n, [None, buf(a, 4 * n)])
        elif kind in (K_F64, K_PF64):
            arr = pa.Array.from_buffers(pa.float64(), n, [None, buf(a, 8 * n)])
        elif kind in (K_F32, K_PF32):
            arr = pa.Array.from_buffers(pa.float32(), n, [None, buf(a, 4 * n)])
        else:
            arr = pa.Array.from_buffers(pa.int64(), n, [None, buf(a, 8 * n)])
        names.append(name)
        out_cols.append(arr)
    base = 3 + 2 * len(keys)
    for j, (name, op, _) in enumerate(outputs):
        t = pa.int64() if op in _INT_OUT else pa.float64()
        if empty_global and op != "count":
            arr = pa.nulls(1, t)
        else:
            arr = pa.Array.from_buffers(t, n, [None, buf(O[base + j], 8 * n)])
        names.append(name)
        out_cols.append(arr)
    return pa.Table.from_arrays(out_cols, names=names)


_DISTINCT_KIND = {pa.int32(): 0, pa.int64(): 1, pa.string(): 2}
DMAGIC = 0x4D584431


def _call_distinct(lib, plan_addr: int, out_addr: int) -> int:
    """The native distinct call (a separate function so tests can count it)."""
    return lib.mx_distinct(plan_addr, out_addr)


def try_distinct(compiler, table: pa.Table, rows_only: bool = False):
    """The distinct rows of ``table`` in first-appearance order (F-032 M2h): one native call finds
    the first row of every tuple (parallel hashing, 64 partitions built in row order), then the
    kept rows are gathered. Columns int32 / int64 / utf8 without nulls; None: the caller's path.
    ``rows_only``: the kept row numbers (int64 NumPy), or the table itself when every row is kept."""
    if not enabled() or not available():
        return _decline("distinct: native executor off")
    if getattr(compiler, "_device_arg", None) != "cpu":
        return _decline("distinct: device='cpu'")
    if table.num_rows < 2:
        return _decline("distinct: row count")
    import numpy as np
    from ._lazy_pc import pc
    kinds = []
    for c in table.columns:
        if c.type not in _DISTINCT_KIND or c.null_count:
            return _decline(f"distinct: column type {c.type} or nulls")
        kinds.append(_DISTINCT_KIND[c.type])
    flat = [c.combine_chunks() if c.num_chunks != 1 else c.chunk(0) for c in table.columns]
    structs = []
    c_cols = (ctypes.c_int64 * len(flat))()
    try:
        for j, arr in enumerate(flat):
            st = (ctypes.c_int64 * 10)()
            arr._export_to_c(ctypes.addressof(st))
            structs.append(st)
            c_cols[j] = ctypes.addressof(st)
        c_kinds = (ctypes.c_int64 * len(kinds))(*kinds)
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 5)(DMAGIC, len(flat), ctypes.addressof(c_cols), ctypes.addressof(c_kinds),
                                 default_threads(table.num_rows))
        O = (ctypes.c_int64 * 2)()
        lib = _load()
        rc = _call_distinct(lib, ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return _decline(f"distinct: executor declined (code {rc})")
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    if n == table.num_rows:
        return table
    idx = np.frombuffer(pa.foreign_buffer(int(O[1]), 8 * n, base=owner), dtype=np.int64, count=n)
    if rows_only:
        return idx
    out = try_gather(compiler, table, idx)
    if out is None:
        out = pc.take(table, pa.array(idx))
    _last.update(path="native", decline=None)
    return out


CMAGIC = 0x4D584332
E_UNSORTED = -20


def _call_count_distinct(lib, plan_addr: int, out_addr: int) -> int:
    """The native count-distinct call (a separate function so tests can count it)."""
    return lib.mx_count_distinct(plan_addr, out_addr)


CD_PATHS = ("runs", "key partitions", "byte map", "pair partitions")


def _dict_codes(column):
    """A dictionary column as (int32 codes of one dictionary without duplicates, that dictionary):
    chunks with their own dictionaries are remapped to a unified one (F-046h). None with nulls."""
    import numpy as np
    from .np_arrow import array as _arr, to_numpy
    if column.null_count:
        return None
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    if not chunks:
        return None
    unified, maps = _unified_dictionary([c.dictionary for c in chunks])
    if unified.null_count:
        return None
    same = all(len(m) == len(unified) and bool((m == np.arange(len(m), dtype=np.int32)).all()) for m in maps)
    if same and len(chunks) == 1 and chunks[0].indices.type == pa.int32():
        return chunks[0].indices, unified
    parts = [to_numpy(c.indices).astype(np.int32, copy=False) for c in chunks]
    codes = np.concatenate([p if same else m[p] for p, m in zip(parts, maps)]) if parts else np.zeros(0, np.int32)
    return _arr(np.ascontiguousarray(codes, dtype=np.int32)), unified


def try_count_distinct(compiler, table: pa.Table, key: str, val: str, name: str) -> Optional[pa.Table]:
    """``table.distinct().group_by(key).count(val)`` without the distinct table. A non-decreasing
    ``key`` (TPC-H lineitem by order key, and any filter of it) is counted run by run (F-023); any
    other key by a hash aggregate (F-046b): a byte per (key, value) when both ranges are small,
    partitions by the pair's hash for a few keys, by key range otherwise (F-046h). Dictionary
    columns count on their codes. Returns ``key`` (its own type; a dictionary key decoded) and
    ``name`` (int64) in key order; None when the key range is too sparse (past max(2^26, 8 x rows))
    or the columns do not fit (int32 / int64 / dictionary key, int32 / int64 / utf8 / dictionary
    value, no nulls), so the caller's path answers."""
    if not enabled() or not available():
        return _decline("count distinct: native executor off")
    if getattr(compiler, "_device_arg", None) != "cpu":
        return _decline("count distinct: device='cpu'")
    a, b = table.column(key), table.column(val)
    adict = bdict = None
    if pa.types.is_dictionary(a.type):
        got = _dict_codes(a)
        if got is None:
            return _decline("count distinct: column types or nulls")
        a, adict = got
    if pa.types.is_dictionary(b.type):
        got = _dict_codes(b)
        if got is None:
            return _decline("count distinct: column types or nulls")
        b = got[0]
    if a.type not in (pa.int32(), pa.int64()) or b.type not in _DISTINCT_KIND or a.null_count or b.null_count:
        return _decline("count distinct: column types or nulls")
    if table.num_rows < 1:
        return _decline("count distinct: empty input")
    import numpy as np
    flat = [c.combine_chunks() if isinstance(c, pa.ChunkedArray) and c.num_chunks != 1
            else c.chunk(0) if isinstance(c, pa.ChunkedArray) else c for c in (a, b)]
    structs = []
    try:
        for arr in flat:
            st = (ctypes.c_int64 * 10)()
            arr._export_to_c(ctypes.addressof(st))
            structs.append(st)
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 6)(CMAGIC, ctypes.addressof(structs[0]), 0 if flat[0].type == pa.int32() else 1,
                                 ctypes.addressof(structs[1]), _DISTINCT_KIND[b.type],
                                 default_threads(table.num_rows))
        O = (ctypes.c_int64 * 3)()
        lib = _load()
        rc = _call_count_distinct(lib, ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc == E_UNSORTED:
        return _decline("count distinct: key range too sparse")
    if rc < 0:
        return _decline(f"count distinct: executor declined (code {rc})")
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    buf = pa.foreign_buffer(int(O[1]), 16 * n, base=owner)
    keys = pa.Array.from_buffers(pa.int64(), n, [None, buf], offset=0)
    cnts = pa.Array.from_buffers(pa.int64(), n, [None, buf], offset=n)
    from ._lazy_pc import pc
    if adict is not None:                      # codes -> values, in value order (as when decoded first)
        keys = pc.take(adict, keys)
        order = pc.sort_indices(keys)
        keys, cnts = pc.take(keys, order), pc.take(cnts, order)
    elif a.type == pa.int32():
        keys = pc.cast(keys, pa.int32())
    _last.update(path="native", decline=None, cd_path=CD_PATHS[int(O[2])] if 0 <= int(O[2]) < 4 else None)
    return pa.table({key: keys, name: cnts})


SMAGIC = 0x4D585333
SLICE_IN_MAX_VALUES = 32


def try_str_slice_in(arr, offset: int, length, values) -> Optional[pa.Array]:
    """``str_slice(arr, offset, length).isin(values)`` in one native pass (F-043): no sliced string
    array is built. utf8 (int32 offsets) without nulls, string values; None otherwise (the caller
    uses Arrow). Characters are code points, as in Arrow's utf8_slice_codeunits."""
    if not enabled() or not available() or not isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return None
    # each row is compared with each value: past a few dozen values Arrow's hash set wins
    if len(values) > SLICE_IN_MAX_VALUES:
        return None
    if arr.type != pa.string() or arr.null_count or not all(isinstance(v, str) for v in values):
        return None
    if isinstance(arr, pa.ChunkedArray):
        parts = [try_str_slice_in(c, offset, length, values) for c in arr.chunks]
        if any(x is None for x in parts):
            return None
        return pa.chunked_array(parts, type=pa.bool_())
    import numpy as np
    n = len(arr)
    if n == 0:                                   # (pa.array([]) would import pandas)
        return pa.Array.from_buffers(pa.bool_(), 0, [None, pa.py_buffer(b"")])
    enc = [v.encode("utf-8") for v in values]
    data = b"".join(enc) or b"\0"
    voff = np.zeros(len(enc) + 1, dtype=np.int64)
    voff[1:] = np.cumsum([len(e) for e in enc])
    dbuf = ctypes.create_string_buffer(data, len(data))
    out = np.empty(n, dtype=np.uint8)
    st = (ctypes.c_int64 * 10)()
    arr._export_to_c(ctypes.addressof(st))
    try:
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 9)(SMAGIC, ctypes.addressof(st), offset, -1 if length is None else length,
                                 len(enc), ctypes.addressof(dbuf), voff.ctypes.data, out.ctypes.data,
                                 default_threads(n))
        rc = _load().mx_str_slice_in(ctypes.addressof(P), 0)
    finally:
        if st[8]:
            _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return None
    bits = np.packbits(out.view(bool), bitorder="little")
    return pa.Array.from_buffers(pa.bool_(), n, [None, pa.py_buffer(bits)])


LMAGIC = 0x4D58534C


def try_str_slice(arr, offset: int, length) -> Optional[pa.Array]:
    """``str_slice(arr, offset, length)`` as a new utf8 array in one native pass (F-043): code points,
    like Arrow's utf8_slice_codeunits. utf8 (int32 offsets); nulls stay null. None otherwise."""
    if not enabled() or not available() or not isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return None
    if arr.type != pa.string():
        return None
    if isinstance(arr, pa.ChunkedArray):
        parts = [try_str_slice(c, offset, length) for c in arr.chunks]
        if any(x is None for x in parts):
            return None
        return pa.chunked_array(parts, type=pa.string())
    n = len(arr)
    if n == 0:
        return pa.Array.from_buffers(pa.string(), 0, [None, pa.py_buffer(bytes(4)), pa.py_buffer(b"")])
    st = (ctypes.c_int64 * 10)()
    arr._export_to_c(ctypes.addressof(st))
    try:
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 5)(LMAGIC, ctypes.addressof(st), offset, -1 if length is None else length,
                                 default_threads(n))
        O = (ctypes.c_int64 * 2)()
        lib = _load()
        rc = lib.mx_str_slice(ctypes.addressof(P), ctypes.addressof(O))
    finally:
        if st[8]:
            _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return None
    total, base = int(O[0]), int(O[1])
    owner = _Arena(lib, base)
    obytes = ((n + 1) * 4 + 7) // 8 * 8
    offs = pa.foreign_buffer(base, (n + 1) * 4, base=owner)
    data = pa.foreign_buffer(base + obytes, max(total, 1), base=owner)
    out = pa.Array.from_buffers(pa.string(), n, [None, offs, data])
    if not arr.null_count:
        return out
    if arr.offset == 0:                            # the input's validity bitmap, as it is
        return pa.Array.from_buffers(pa.string(), n, [arr.buffers()[0], offs, data], null_count=arr.null_count)
    from ._lazy_pc import pc
    return pc.if_else(pc.is_valid(arr), out, pa.scalar(None, pa.string()))


KMAGIC = 0x4D58544B


def _topk_kind(t: pa.DataType):
    if t in (pa.int32(), pa.date32()):
        return 0
    if t == pa.int64() or pa.types.is_timestamp(t) or t == pa.date64():
        return 1
    if t == pa.float32():
        return 2
    if t == pa.float64():
        return 3
    return None


def try_topk(table: pa.Table, names, descending, k: int):
    """Row numbers of ``table`` sorted by ``names`` (``descending`` per column), the first ``k``
    (F-044): one native pass of per-worker heaps instead of ranking every row. Ties keep row order
    (a stable sort's first k). int32 / int64 / float / date / timestamp columns without nulls; None
    otherwise, or on NaN (the caller's sort decides where NaN goes)."""
    if not enabled() or not available() or k <= 0 or table.num_rows == 0:
        return None
    kinds = []
    for nm in names:
        c = table.column(nm)
        kd = _topk_kind(c.type)
        if kd is None or c.null_count:
            return None
        kinds.append(kd)
    import numpy as np
    flat = [table.column(nm).combine_chunks() if table.column(nm).num_chunks != 1 else table.column(nm).chunk(0)
            for nm in names]
    structs = []
    c_cols = (ctypes.c_int64 * len(flat))()
    try:
        for j, arr in enumerate(flat):
            st = (ctypes.c_int64 * 10)()
            arr._export_to_c(ctypes.addressof(st))
            structs.append(st)
            c_cols[j] = ctypes.addressof(st)
        c_kinds = (ctypes.c_int64 * len(kinds))(*kinds)
        c_desc = (ctypes.c_int64 * len(kinds))(*[1 if d else 0 for d in descending])
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 7)(KMAGIC, len(flat), ctypes.addressof(c_cols), ctypes.addressof(c_kinds),
                                 ctypes.addressof(c_desc), k, default_threads(table.num_rows))
        O = (ctypes.c_int64 * 2)()
        lib = _load()
        rc = lib.mx_topk(ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return None
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    return np.frombuffer(pa.foreign_buffer(int(O[1]), max(8 * n, 8), base=owner), dtype=np.int64, count=n)


TKMAGIC = 0x4D58464B


def try_filter_topk(compiler, table: pa.Table, predicates, names, descending, k: int):
    """Row numbers of the rows of ``table`` passing ``predicates``, sorted by ``names``
    (``descending`` per column), the first ``k`` (F-046e): the predicate runs per block and passing
    rows go straight into per-worker heaps, so no filtered table is built. Ties keep row order.
    Sort columns int32 / int64 / float / date / timestamp without nulls; None otherwise or on NaN."""
    if not enabled() or not available() or getattr(compiler, "_device_arg", None) != "cpu":
        return None
    if k <= 0 or table.num_rows == 0 or not predicates:
        return None
    skinds = []
    for nm in names:
        c = table.column(nm)
        kd = _topk_kind(c.type)
        if kd is None or c.null_count:
            return None
        skinds.append(kd)
    from .pipeline_program import compile_pipeline
    prog = compile_pipeline(list(predicates), [], table, concat_dicts=True)
    if prog is None or prog.dict_arrays:
        return None
    ex = _Extras(prog)
    vcols = [table.column(c) for c in prog.columns]
    dtypes = []
    for name, c in zip(prog.columns, vcols):
        if ex.dtype(name) is not None:
            dtypes.append(ex.dtype(name))
            continue
        if c.type not in _DT or c.null_count:
            return None
        dtypes.append(_DT[c.type])
    scols = [table.column(nm) for nm in names]
    segs = _segments(vcols + scols)
    if not segs:
        return None
    import numpy as np
    width = len(vcols) + len(scols)
    structs = []
    arrays = (ctypes.c_int64 * (len(segs) * width))()
    try:
        for r, seg in enumerate(segs):
            for j, piece in enumerate(seg):
                st = (ctypes.c_int64 * 10)()
                piece._export_to_c(ctypes.addressof(st))
                structs.append(st)
                arrays[r * width + j] = ctypes.addressof(st)
        doffs, row0 = [], 0
        for seg in segs:
            doffs += ex.words(lambda nm: seg[prog.columns.index(nm)], row0)
            row0 += len(seg[0])
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        code = ex.patch(prog.code)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_sk = (ctypes.c_int64 * len(skinds))(*skinds)
        c_desc = (ctypes.c_int64 * len(skinds))(*[1 if d else 0 for d in descending])
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 16)(TKMAGIC, len(segs), len(vcols), code.ctypes.data, prog.consts.ctypes.data,
                                  prog.set_bytes.ctypes.data, prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes),
                                  ctypes.addressof(arrays), len(scols), ctypes.addressof(c_sk),
                                  ctypes.addressof(c_desc), k, default_threads(table.num_rows), ex.n,
                                  ctypes.addressof(c_doffs))
        O = (ctypes.c_int64 * 2)()
        lib = _load()
        rc = lib.mx_filter_topk(ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return None
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    return np.frombuffer(pa.foreign_buffer(int(O[1]), max(8 * n, 8), base=owner), dtype=np.int64, count=n)


# ── M2: native leaf filter (Filter*(Scan) under a join) ─────────────────────────────────────

FMAGIC = 0x4D584631
_FIXED_OK = (pa.types.is_integer, pa.types.is_floating, pa.types.is_date, pa.types.is_timestamp,
             pa.types.is_decimal, pa.types.is_time, pa.types.is_duration)
_filter_last = {"path": None, "decline": None, "maps": []}


def last_filter():
    """(path, decline reason) of the last native leaf-filter attempt."""
    return _filter_last["path"], _filter_last["decline"]


def last_filter_maps():
    """How the last native leaf filter indexed its partners, one dict per lookup: ``kind`` ("direct" array or
    "hash" table), ``multi`` (the partner's keys repeat: rows in CSR layout) and ``bitmap`` (a presence bitmap
    answers a miss first). Tests and the join trace read it."""
    return list(_filter_last["maps"])


def _filter_decline(reason: str):
    _filter_last["path"], _filter_last["decline"] = None, reason
    _count(reason)
    return None


def _call_filter(lib, plan_addr: int, out_addr: int) -> int:
    """The one native call per leaf filter (a separate function so tests can count it)."""
    return lib.mx_filter(plan_addr, out_addr)


def _out_kind(t: pa.DataType):
    """(kind, bytes per value) of an output column, or None."""
    if pa.types.is_dictionary(t):
        if t.index_type == pa.int32() and pa.types.is_string(t.value_type):
            return 2, 0
        return None
    if pa.types.is_string(t):
        return 1, 0
    if any(f(t) for f in _FIXED_OK) and t.bit_width % 8 == 0:
        return 0, t.bit_width // 8
    return None


_KEY_WIDTH = {pa.int32(): 4, pa.int64(): 8, pa.date32(): 4}


def _unified_dictionary(dicts):
    """One dictionary of the distinct values of ``dicts`` (string arrays: per-chunk dictionaries)
    and, per input, an int32 NumPy remap of its codes to the unified codes (F-046d). Dictionaries
    are small (a Parquet row group's distinct values), and shared ones are mapped once."""
    import numpy as np
    from ._lazy_pc import pc
    from .np_arrow import to_numpy
    def ident(d):            # the same dictionary buffers (chunks sharing one dictionary)
        return (d.offset, len(d)) + tuple(b.address if b is not None else 0 for b in d.buffers())

    uniq, seen = [], {}
    for d in dicts:
        if ident(d) not in seen:
            seen[ident(d)] = len(uniq)
            uniq.append(d)
    unified = pc.unique(pa.concat_arrays(uniq))
    maps = [np.ascontiguousarray(to_numpy(pc.index_in(d, value_set=unified)).astype(np.int32)) for d in uniq]
    return unified, [maps[seen[ident(d)]] for d in dicts]


class Lookup:
    """A join partner fused into the leaf filter (F-032): rows of the scanned table whose ``probe``
    columns equal ``build_keys`` of a row of ``build`` keep that row's ``outputs`` (written under
    ``names``). ``build`` keys may repeat (each partner row of a matching key gives one joined row, in build order); 1 or 2
    integer key columns. ``semi``: only whether a partner row exists counts (no outputs)."""
    __slots__ = ("probe", "build", "build_keys", "outputs", "names", "semi", "keep")

    def __init__(self, probe, build: pa.Table, build_keys, outputs, names=None, semi=False, keep=()):
        self.probe, self.build, self.build_keys = list(probe), build, list(build_keys)
        self.outputs = list(outputs)
        self.names = list(names) if names is not None else list(outputs)
        self.semi = bool(semi)
        self.keep = frozenset(keep)        # dictionary outputs written as codes, not decoded (F-046d)


def _lookup_key_kind(t: pa.DataType):
    return "date" if t == pa.date32() else "int"


def _lookup_keys(table: pa.Table, lk: Lookup, pk_names: List[str]):
    """(the partner as one chunk per column, the slots in ``pk_names`` of its probe key columns, the key
    widths word) of a lookup, adding its probe columns to ``pk_names``; or the reason it cannot run (str)."""
    if len(lk.probe) not in (1, 2) or len(lk.probe) != len(lk.build_keys):
        return "lookup: 1 or 2 key columns"
    if lk.semi and lk.outputs:
        return "lookup: a semi join has no outputs"
    b = lk.build.combine_chunks()     # one chunk per column; dictionaries unified, not decoded
    if b.num_rows == 0 or any(b.column(c).num_chunks != 1 for c in b.column_names):
        return "lookup: empty partner"
    pks, widths = [], 0
    for i, (pn, bn) in enumerate(zip(lk.probe, lk.build_keys)):
        pc_, bc_ = table.column(pn), b.column(bn)
        if (pc_.type not in _KEY_WIDTH or bc_.type not in _KEY_WIDTH or pc_.null_count or bc_.null_count
                or _lookup_key_kind(pc_.type) != _lookup_key_kind(bc_.type)):
            return f"lookup key {pn} = {bn}: types {pc_.type} / {bc_.type} or nulls"
        if pn not in pk_names:
            pk_names.append(pn)
        pks.append(pk_names.index(pn))
        widths |= _KEY_WIDTH[pc_.type] << (4 * i) | _KEY_WIDTH[bc_.type] << (8 + 4 * i)
    if lk.semi:
        widths |= 1 << 16
    return b, pks, widths


def try_filter(compiler, table: pa.Table, predicates, needed, lookups=(), keep_dict=()) -> Optional[pa.Table]:
    """The rows of ``table`` that pass every predicate (and find a partner row in every lookup),
    holding the ``needed`` columns (None: all) in table order, then each lookup's outputs, in one
    native call; dictionary string columns come back decoded (as the join path expects). None: the
    caller filters (and joins) as before."""
    _filter_last["path"] = None
    _filter_last["maps"] = []
    if not enabled():
        return _filter_decline("disabled")
    if getattr(compiler, "_device_arg", None) != "cpu":
        return _filter_decline("CPU executor: device='cpu'")
    if not available():
        return _filter_decline("libmxexec_cpu.so not built")
    if table.num_rows == 0 or not (predicates or lookups):
        return _filter_decline("nothing to filter")
    names = [c for c in table.column_names if needed is None or c in needed]
    if not names and not lookups:
        names = table.column_names[:1]

    from .pipeline_program import compile_pipeline
    prog = compile_pipeline(list(predicates), [], table, concat_dicts=True)
    if prog is None:
        return _filter_decline("predicate not supported by the bytecode compiler")
    if prog.dict_arrays:
        return _filter_decline("string predicates outside concat mode")
    ex = _Extras(prog)
    vcols = [table.column(c) for c in prog.columns]
    dtypes = []
    for name, c in zip(prog.columns, vcols):
        if ex.dtype(name) is not None:
            dtypes.append(ex.dtype(name))
            continue
        if c.type not in _DT or c.null_count:
            return _filter_decline(f"predicate column type {c.type} or nulls")
        dtypes.append(_DT[c.type])
    ocols = [table.column(n) for n in names]
    kinds, sizes = [], []
    coded = []                                  # output positions written as dictionary codes
    for n, c in zip(names, ocols):
        k = _out_kind(c.type)
        if k is None or c.null_count:
            return _filter_decline(f"output column {n}: type {c.type} or nulls")
        if k[0] == 2 and any(ch.dictionary.null_count for ch in c.chunks):
            return _filter_decline(f"output column {n}: null in a dictionary")
        if k[0] == 2 and n in keep_dict:
            coded.append(len(kinds))
            k = (3, 0)                           # remap table address filled in below
        kinds.append(k[0])
        sizes.append(k[1])

    # lookups: probe key columns ride along with the scanned table's chunks; partners are one chunk
    pk_names: List[str] = []
    look_words, bout_words, keep_alive = [], [], []
    out_types, out_names = [c.type for c in ocols], list(names)
    for lk in lookups:
        got = _lookup_keys(table, lk, pk_names)
        if isinstance(got, str):
            return _filter_decline(got)
        b, pks, widths = got
        bkeys = [b.column(bn).chunk(0) for bn in lk.build_keys]
        bouts = []
        for on, nm in zip(lk.outputs, lk.names):
            col = b.column(on)
            k = _out_kind(col.type)
            if k is None or col.null_count:
                return _filter_decline(f"lookup output {on}: type {col.type} or nulls")
            if k[0] == 2 and col.chunk(0).dictionary.null_count:
                return _filter_decline(f"lookup output {on}: null in a dictionary")
            arr = col.chunk(0)
            if k[0] == 2 and on in lk.keep:
                k = (3, 0)
            bouts.append((arr, k))
            out_types.append(col.type)
            out_names.append(nm)
        look_words.append((len(pks), pks, bkeys, widths, bouts))
    if not out_names:
        return _filter_decline("no output columns")

    pcols = [table.column(n) for n in pk_names]
    every = vcols + ocols + pcols
    segs = _segments(every)
    if not segs:
        return _filter_decline("empty input")

    width = len(every)
    structs = []
    # dictionary columns kept coded: one unified dictionary each, a code remap per segment
    unified = {}
    for j in coded:
        u, maps = _unified_dictionary([seg[len(vcols) + j].dictionary for seg in segs])
        tab = (ctypes.c_int64 * len(segs))(*[m.ctypes.data for m in maps])
        keep_alive.extend([maps, tab])
        sizes[j] = ctypes.addressof(tab)
        unified[j] = u

    def export(arr):
        st = (ctypes.c_int64 * 10)()
        arr._export_to_c(ctypes.addressof(st))
        structs.append(st)
        return ctypes.addressof(st)

    arrays = (ctypes.c_int64 * (len(segs) * width))()
    try:
        for r, seg in enumerate(segs):
            for j, piece in enumerate(seg):
                arrays[r * width + j] = export(piece)
        L = (ctypes.c_int64 * max(8 * len(look_words), 1))()
        for j, (nk, pks, bkeys, widths, bouts) in enumerate(look_words):
            bo = (ctypes.c_int64 * max(3 * len(bouts), 1))()
            for t, (arr, (kind, size)) in enumerate(bouts):
                if kind == 3:                            # a partner is one chunk: one remap
                    u, (m,) = _unified_dictionary([arr.dictionary])
                    tab = (ctypes.c_int64 * 1)(m.ctypes.data)
                    keep_alive.extend([m, tab])
                    size = ctypes.addressof(tab)
                    unified[("b", j, t)] = u
                bo[3 * t], bo[3 * t + 1], bo[3 * t + 2] = export(arr), kind, size
            keep_alive.append(bo)
            L[8 * j: 8 * j + 8] = [nk, pks[0], pks[1] if nk == 2 else -1, export(bkeys[0]),
                                   export(bkeys[1]) if nk == 2 else 0, widths, len(bouts), ctypes.addressof(bo)]
        doffs, row0 = [], 0
        for seg in segs:
            doffs += ex.words(lambda nm: seg[prog.columns.index(nm)], row0)
            row0 += len(seg[0])
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        code = ex.patch(prog.code)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_kinds = (ctypes.c_int64 * max(len(kinds), 1))(*kinds)
        c_sizes = (ctypes.c_int64 * max(len(sizes), 1))(*sizes)
        from .pipeline_kernels import default_threads
        c_stats = (ctypes.c_int64 * max(4 * len(look_words), 1))()
        P = (ctypes.c_int64 * 20)(
            FMAGIC, 2 if look_words else 1, len(segs), len(vcols), len(ocols), code.ctypes.data, prog.consts.ctypes.data,
            prog.set_bytes.ctypes.data, prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes),
            ctypes.addressof(arrays), ctypes.addressof(c_kinds), ctypes.addressof(c_sizes),
            default_threads(table.num_rows), ex.n, ctypes.addressof(c_doffs),
            len(look_words), ctypes.addressof(L), len(pk_names), ctypes.addressof(c_stats))
        O = (ctypes.c_int64 * (2 + 2 * len(out_names)))()
        lib = _load()
        rc = _call_filter(lib, ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return _filter_decline(f"executor declined (code {rc})")

    n = int(O[0])
    owner = _Arena(lib, int(O[1]))

    def buf(addr, nbytes):
        return pa.foreign_buffer(int(addr), max(nbytes, 0), base=owner)

    coded_out = {j: unified[j] for j in coded}
    pos = len(ocols)
    for jl, (nk, pks, bkeys, widths, bouts) in enumerate(look_words):
        for tb, (arr, (kind, size)) in enumerate(bouts):
            if kind == 3:
                coded_out[pos] = unified[("b", jl, tb)]
            pos += 1
    out = []
    for j, t in enumerate(out_types):
        kind, size = _out_kind(t)
        a, b = int(O[2 + 2 * j]), int(O[3 + 2 * j])
        if j in coded_out:
            idx = pa.Array.from_buffers(pa.int32(), n, [None, buf(a, 4 * n)])
            out.append(pa.DictionaryArray.from_arrays(idx, coded_out[j]))
        elif kind == 0:
            out.append(pa.Array.from_buffers(t, n, [None, buf(a, n * size)]))
        else:
            nbytes = ctypes.c_int32.from_address(a + 4 * n).value
            out.append(pa.Array.from_buffers(pa.string(), n, [None, buf(a, 4 * (n + 1)), buf(b, nbytes)]))
    _filter_last.update(path="native_filter" if not lookups else "native_lookup", decline=None,
                        maps=[{"kind": "hash" if c_stats[4 * j] else "direct", "multi": bool(c_stats[4 * j + 1]),
                               "bitmap": bool(c_stats[4 * j + 2])} for j in range(len(look_words))])
    return pa.Table.from_arrays(out, names=out_names)


# ── F-046a slice 3: join + aggregate fusion (mx_execute plan version 3) ─────────────────────────────────
#
# `Aggregate([Filter] * (star join))` in one native call: per 1024-row block of the fact table, its own filter,
# the partner lookups, the aggregate program's predicate (conditions across partners, filters above the
# join) over the surviving rows with the partner columns gathered into block scratch, then the aggregate.
# Extra plan words (version 3), `F` at P[26]:
#    0 n_fa  fact value columns of the program (the first slots of an `arrays` row; then the n_keys key
#            columns, the pre-filter's columns, the probe key columns: all columns of the fact table)
#    1 colmap*  int64[n_cols]: program column -> its slot, or -(1 + k) for the k-th partner column
#    2 n_pre  3 code*  4 consts*  5 set_bytes*  6 set_meta*  7 dtypes*  8 n_dcols  9 doffs*  (the fact
#    leaf's own predicate program)   10 its first slot in an arrays row
#    11 n_look  12 look*  (as mx_filter: key columns, partner key arrays, widths)  13 first probe key slot
#    14 n_bcols  15 bcols*: int64[n_bcols][4] = (lookup, ArrowArray*, dtype, rank table* or 0)
#    16 (unused)  17 words per arrays row
# Group keys on partner columns (and derived keys) are expression keys: the executor groups by the program's
# value of them. A partner string column's key is its rank in byte order (a dictionary-sized table the
# executor applies while gathering), turned back into strings here.

_DICT_MAX = 1 << 16        # a partner string column is encoded / ranked in Python up to this many distinct values


class FusedJoin:
    """A star join under an aggregate (``join_engine.fused_join_aggregate``): the fact table's scan and
    predicates, the partner ``Lookup``s, where each column of the joined schema lives (``vcols``: name ->
    ("fact", column) or ("partner", lookup index, column of the partner)) and the predicates over the
    joined schema (conditions across partners, filters above the join)."""
    __slots__ = ("scan", "preds", "lookups", "vcols", "post_preds", "cache", "fd")

    def __init__(self, scan, preds, lookups, vcols, post_preds, fd=None):
        self.scan, self.preds, self.lookups, self.vcols, self.post_preds = scan, list(preds), lookups, vcols, post_preds
        self.fd = fd or {}           # group column (a plain partner string) -> the unique key of its table that fixes it
        self.cache = {}              # what a second call over the same join (a key range pass) reuses


class _VTable:
    """The columns of a joined schema, each in the table it lives in (different lengths): just enough of a
    table for ``compile_pipeline`` to read names, types and dictionaries."""

    def __init__(self, cols, num_rows):
        self._cols = cols
        self.num_rows = num_rows         # joined rows <= the fact table's (unique partner keys): bounds an integer sum
        self.column_names = list(cols)
        self.schema = pa.schema([pa.field(n, c.type) for n, c in cols.items()])

    def column(self, name):
        return self._cols[name]


def fused_join_blocked(compiler) -> Optional[str]:
    """Why a fused join + aggregate cannot run here (switched off, not the CPU executor, library missing),
    else None. Cheap: asked before any partner is built."""
    if not enabled():
        return "disabled (MXFRAME_NATIVE=0)"
    if os.environ.get("MXFRAME_FUSED_JOIN_AGG", "1") in ("0", "false", "off"):
        return "disabled (MXFRAME_FUSED_JOIN_AGG=0)"
    if getattr(compiler, "_device_arg", None) != "cpu":
        return "CPU executor: device='cpu'"
    from . import coproc_fused
    if coproc_fused.current_policy() is not None:
        return "CPU+GPU co-processing policy active"
    if not available():
        return "libmxexec_cpu.so not built"
    return None


def try_execute_join(compiler, plan, fj: FusedJoin, having_pred, post_ops, distinct: bool = False,
                     raw_keys: bool = False) -> Optional[pa.Table]:
    """``plan``'s aggregate over the star join ``fj`` in one native call (plan version 3), or None (the
    caller materializes the join and runs the aggregate as before). ``plan.input`` is not read.
    `distinct`: a SELECT DISTINCT as a group-by (see ``try_execute``): key spaces up to DISTINCT_MAX_GROUPS,
    no partner-row grouping."""
    max_groups = DISTINCT_MAX_GROUPS if distinct else MAX_GROUPS
    _last["path"] = None
    t_start = _time.perf_counter()
    why = fused_join_blocked(compiler)
    if why is not None:
        return _decline(why)
    table: pa.Table = fj.scan.table
    if table.num_rows == 0 or not fj.lookups:
        return _decline("empty input")
    import numpy as np
    from ._lazy_pc import pc
    from .lazy_expr import Expr
    from .np_arrow import array as _npa, to_numpy as _np_of
    from .pipeline_kernels import default_threads
    from .pipeline_program import compile_pipeline, _dict_key

    # ── partners and the columns of the joined schema ───────────────────────────────────────────
    mem = fj.cache.get("partners")
    if mem is None:
        pk_names: List[str] = []
        bs, look_words = [], []
        for lk in fj.lookups:
            got = _lookup_keys(table, lk, pk_names)
            if isinstance(got, str):
                return _decline(got)
            b, pks, widths = got
            bs.append(b)
            look_words.append((len(pks), pks, [b.column(bn).chunk(0) for bn in lk.build_keys], widths))
        base = {}
        for name, src in fj.vcols.items():
            if src[0] == "prow":
                continue
            base[name] = table.column(src[1]) if src[0] == "fact" else bs[src[1]].column(src[2])
        mem = fj.cache["partners"] = (pk_names, bs, look_words, base)
    pk_names, bs, look_words, base = mem
    cols = dict(base)
    if plan.group_by and not distinct and not any(src[0] == "prow" for src in fj.vcols.values()):
        got = _try_dependent_keys(compiler, plan, fj, bs, having_pred, post_ops)
        if got is not None:                      # strings fixed by another key: taken by partner row (F-076)
            return got
        got = _try_prow_groups(compiler, plan, fj, bs, having_pred, post_ops, only_if_wide=True)
        if got is not None:                      # keys too many for direct addressing: group by partner row
            return got
    from .join_engine import _cols as _expr_cols
    used = set()
    for e in list(plan.group_by) + list(plan.aggs) + list(fj.post_preds):
        used |= _expr_cols(e)
    for name, src in fj.vcols.items():         # partner strings become dictionaries only now, and only those the
        c = cols.get(name)                     # query reads: a string taken by partner row is never encoded (F-076)
        if name in used and src[0] == "partner" and pa.types.is_string(c.type) and 0 < len(c) <= _DICT_MAX:
            enc = fj.cache.setdefault("encoded", {})
            if name not in enc:
                enc[name] = pa.chunked_array([pc.dictionary_encode(c.chunk(0))])
            cols[name] = enc[name]
    for name, src in fj.vcols.items():
        if src[0] == "prow":             # the partner's row number (group = partner row): an int32 column
            cols[name] = pa.chunked_array([_npa_i32_empty()])

    # ── group keys: fact columns as they are; partner columns and expressions as expression keys ──
    space = 1                           # the partner keys alone already multiply past direct addressing: decline now,
    for e in plan.group_by:             # before a string column is ranked or anything is compiled
        src = fj.vcols.get(e.args[0]) if e.op == "col" else None
        if src is not None and src[0] == "fact":
            r = _probe_range(fj, bs, src[1])
            if r is not None:                   # a join key: only values the partner has reach the aggregate
                space *= max(1, r[1] - r[0] + 1)
            continue
        c = cols.get(e.args[0]) if src is not None and src[0] == "partner" else None
        if c is not None and not c.null_count:
            if c.type in (pa.int32(), pa.int64()):
                a = _np_of(c.chunk(0), zero_copy_only=False)
                space *= int(a.max()) - int(a.min()) + 1
            elif pa.types.is_dictionary(c.type):
                space *= max(1, len(c.chunk(0).dictionary))
    hashed = False                      # F-069: key tuples past direct addressing are hashed (after the partner-row route)
    if space > max_groups:
        got, hashed = _prow_or_hash(compiler, plan, fj, bs, having_pred, post_ops,
                                    "group key space too wide for direct addressing", distinct)
        if not hashed:
            return got
    keys, derived, known, post, fact_key, extra = [], {}, {}, {}, {}, {}
    pf64 = set()                        # float64 partner columns as keys: hashed by their exact values
    fact_range = {}                     # fact join keys bounded by their partners' keys: no scan for their range
    for i, e in enumerate(plan.group_by):
        if e.op == "col":                # (renamed by an alias: the result column takes the alias, F-080)
            name = e.args[0]
            keys.append(e._alias or name)
            src = fj.vcols.get(name)
            if src is None:
                return _decline("group key outside the joined schema")
            if src[0] == "fact":
                fact_key[i] = src[1]
                r = _probe_range(fj, bs, src[1])
                if r is not None:
                    fact_range[i] = r
                continue
            c = cols[name]
            if c.null_count:
                return _decline("partner group key with nulls")
            if src[0] == "prow":
                derived[i], known[i] = e, (0, bs[src[1]].num_rows - 1)
            elif c.type in (pa.int32(), pa.int64()):
                a = _np_of(c.chunk(0), zero_copy_only=False)
                derived[i], known[i], post[i] = e, (int(a.min()), int(a.max())), c.type
            elif (pa.types.is_dictionary(c.type) and c.type.index_type == pa.int32()
                  and pa.types.is_string(c.type.value_type)):
                d = c.chunk(0).dictionary
                if d.null_count or len(d) > _DICT_MAX:
                    return _prow_or_decline(compiler, plan, fj, bs, having_pred, post_ops,
                                            "partner string group key: dictionary too big or with nulls")
                vals = d.to_pylist()
                uniq = sorted(set(vals))
                rank = {v: r for r, v in enumerate(uniq)}
                rk = f"__rank:{name}"
                cols[rk] = pa.chunked_array([c.chunk(0).indices])
                extra[rk] = (src[1], np.array([rank[v] for v in vals], dtype=np.int32))
                derived[i], known[i], post[i] = Expr("col", rk), (0, len(uniq) - 1), _npa(uniq, type=pa.string())
            elif c.type in (pa.float64(), pa.float32()):
                derived[i] = e
                pf64.add(i)
            else:
                return _prow_or_decline(compiler, plan, fj, bs, having_pred, post_ops,
                                        f"partner group key type {c.type}")
        else:
            if not e._alias:
                return _decline("derived group key without a name")
            keys.append(e._alias)
            derived[i] = e
    sort_by, limit = _split_post_ops(post_ops)
    if isinstance(sort_by, str):
        return _decline(sort_by)

    n_user = len(plan.aggs)
    hidden = [Expr("min", derived[i]).alias(f"__key{i}") for i in sorted(derived)]
    prog = compile_pipeline(list(fj.post_preds), list(plan.aggs) + hidden, _VTable(cols, table.num_rows), concat_dicts=True,
                            int_extremes=False)
    if prog is None:
        return _decline("expression not supported by the bytecode compiler")
    if prog.dict_arrays:
        return _decline("string predicates outside concat mode")
    if prog.str_cols or prog.null_cols:
        return _decline("plain string predicate or IS NULL over joined columns")
    key_slot = {}
    for j, i in enumerate(sorted(derived)):
        slot = prog.outputs[n_user + j][2]
        if i in pf64:                    # read from the program's column table: which column
            key_slot[i] = prog.columns.index(derived[i].args[0])
            continue
        if slot is None or not prog.slot_int[slot]:
            return _decline("derived group key is not an integer expression")
        key_slot[i] = slot
    outputs = prog.outputs[:n_user]
    names = [name for name, _, _ in outputs]
    outputs, imeans = _exact_imeans(outputs)
    ranges = dict(known)
    for i in pf64:
        ranges[i] = (0, 0)
    for i in [i for i in sorted(derived) if i not in known and i not in pf64]:
        bnd = _bounds(derived[i], lambda n: _partner_range(fj, cols, n))
        if bnd is not None:                  # an integer expression of partner columns: bounded without a pass
            known[i] = ranges[i] = bnd
    if not hashed and all(i in fact_range for i in fact_key) and all(i in known for i in derived if i not in pf64):
        d = 1                       # every key's range is known here: a key space too big hashes before any pass
        for lo, hi in list(known.values()) + list(fact_range.values()):
            d *= hi - lo + 1
        if d > max_groups:
            got, hashed = _prow_or_hash(compiler, plan, fj, bs, having_pred, post_ops,
                                        "group key space too wide for direct addressing", distinct)
            if not hashed:
                return got
    probe_idx = [i for i in sorted(derived) if i not in known and i not in pf64]
    if hashed:                         # the hashed groups need no range: a placeholder reaches the executor
        for i in probe_idx:
            ranges[i] = (0, 0)
        probe_idx = []
    if probe_idx:                      # an expression key's range: one fused min / max pass over the same join
        probe = Aggregate(None, [], [a for i in probe_idx
                                     for a in (Expr("min", derived[i]).alias(f"lo{i}"), Expr("max", derived[i]).alias(f"hi{i}"))])
        r = try_execute_join(compiler, probe, fj, None, None)
        if r is None:
            return _decline("derived group key: range pass declined")
        for i in probe_idx:
            lo, hi = r.column(f"lo{i}")[0].as_py(), r.column(f"hi{i}")[0].as_py()
            if lo is None or hi is None or lo != int(lo) or hi != int(hi):
                return _decline("derived group key: not an integer range")
            ranges[i] = (int(lo), int(hi))
    if not hashed and any(hi - lo + 1 > max_groups for lo, hi in list(ranges.values()) + list(fact_range.values())):
        if distinct:
            return _decline("group key: range too wide for direct addressing")
        if not any(src[0] == "prow" for src in fj.vcols.values()):
            got = _try_prow_groups(compiler, plan, fj, bs, having_pred, post_ops)
            if got is not None:
                return got
        hashed = True
    sorts = _sort_spec(sort_by, keys, names)
    if isinstance(sorts, str):
        return _decline(sorts)
    having = _having_terms(having_pred, names)
    if having is None:
        return _decline("HAVING other than AND-ed aggregate <op> number")
    if imeans and _imean_ordered(names, imeans, sorts, having):
        return _decline("ORDER BY / HAVING over an integer mean")

    # ── the program's columns: fact columns (read from the chunks), partner columns (gathered per block) ──
    dcols = [c for c in prog.columns if c in prog.dict_concat]
    fcols, fpos, colmap, dtypes, bspec = [], {}, [], [], []
    for name in prog.columns:
        c = cols[name]
        if c.null_count:
            return _decline(f"column {name} has nulls")
        if name in prog.dict_concat:
            dt = 5 | ((len(prog.columns) + dcols.index(name)) << 8)
        elif c.type in _DT:
            dt = _DT[c.type]
        else:
            return _decline(f"column {name}: type {c.type}")
        dtypes.append(dt)
        src = fj.vcols.get(name)
        if src is not None and src[0] == "fact":
            colmap.append(len(fcols))
            fpos[name] = len(fcols)
            fcols.append(c)
        else:
            j = src[1] if src is not None else extra[name][0]
            colmap.append(-(len(bspec) + 1))
            bspec.append((j, name, 7 if src is not None and src[0] == "prow"
                          else 0 if pa.types.is_dictionary(c.type) else _DT[c.type]))
    key_cols, kinds = [], []
    for i, k in enumerate(keys):
        if i in derived:
            key_cols.append(table.column(0))                 # placeholder: the key comes from the program
            kinds.append((K_PF64 if cols[derived[i].args[0]].type == pa.float64() else K_PF32) if i in pf64 else K_EXPR)
            continue
        c = table.column(fact_key[i])
        kind = _key_kind(c)
        if kind is None or c.null_count:
            return _decline(f"group key type {c.type} or nulls")
        key_cols.append(c)
        kinds.append(kind)

    # ── the fact leaf's own predicate (its columns ride along as more columns of the fact table) ──────
    pre = compile_pipeline(list(fj.preds), [], table, concat_dicts=True)
    if pre is None:
        return _decline("fact predicate not supported by the bytecode compiler")
    if pre.dict_arrays:
        return _decline("string predicates outside concat mode")
    pex = _Extras(pre)
    pre_cols = [table.column(c) for c in pre.columns]
    pdtypes = []
    for name, c in zip(pre.columns, pre_cols):
        if pex.dtype(name) is not None:
            pdtypes.append(pex.dtype(name))
        elif c.type in _DT and not c.null_count:
            pdtypes.append(_DT[c.type])
        else:
            return _decline(f"predicate column type {c.type} or nulls")
    probe_cols = [table.column(n) for n in pk_names]
    every = fcols + key_cols + pre_cols + probe_cols
    segs = _segments(every)
    if not segs:
        return _decline("empty input")

    # ── export (Arrow C Data Interface) and the plan buffer ──────────────────────────────────────
    n_fa, n_keys, n_out = len(fcols), len(keys), len(outputs)
    pre_off = n_fa + n_keys
    probe_off = pre_off + len(pre_cols)
    width = len(every)
    structs, keep = [], []

    def export(arr):
        st = (ctypes.c_int64 * 10)()
        arr._export_to_c(ctypes.addressof(st))
        structs.append(st)
        return ctypes.addressof(st)

    arrays = (ctypes.c_int64 * (len(segs) * width))()
    try:
        for r, seg in enumerate(segs):
            for j, piece in enumerate(seg):
                arrays[r * width + j] = export(piece)
        L = (ctypes.c_int64 * max(8 * len(look_words), 1))()
        for j, (nk, pks, bkeys, widths) in enumerate(look_words):
            L[8 * j: 8 * j + 8] = [nk, pks[0], pks[1] if nk == 2 else -1, export(bkeys[0]),
                                   export(bkeys[1]) if nk == 2 else 0, widths, 0, 0]
        bwords = []
        for j, name, dcode in bspec:
            rm = 0
            if name in extra:
                m = extra[name][1]
                keep.append(m)
                rm = m.ctypes.data
            bwords += [j, export(look_words[j][2][0] if dcode == 7 else cols[name].chunk(0)), dcode, rm]
        c_bcols = (ctypes.c_int64 * max(len(bwords), 1))(*bwords)
        doffs = []
        for seg in segs:
            for name in dcols:
                d = seg[fpos[name]].dictionary if name in fpos else cols[name].chunk(0).dictionary
                doffs.append(prog.dict_concat[name][_dict_key(d)])
        pdoffs, row0 = [], 0
        for seg in segs:
            pdoffs += pex.words(lambda nm: seg[pre_off + pre.columns.index(nm)], row0)
            row0 += len(seg[0])
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        c_pdoffs = (ctypes.c_int64 * max(len(pdoffs), 1))(*pdoffs)
        code, pcode = prog.code, pre.code
        if dcols:
            code = code.copy()
            code[7] = len(prog.columns) + len(dcols)
        pcode = pex.patch(pcode)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_pdtypes = (ctypes.c_int32 * max(len(pdtypes), 1))(*pdtypes)
        c_colmap = (ctypes.c_int64 * max(len(colmap), 1))(*colmap)
        c_kinds = (ctypes.c_int64 * max(n_keys, 1))(*kinds)
        c_outs = (ctypes.c_int64 * max(3 * n_out, 1))(
            *[v for name, op, slot in outputs
              for v in (_OPS[op], slot if slot is not None else 0, prog.nan_slots.get(name, -1))])
        c_sorts = (ctypes.c_int64 * max(3 * len(sorts), 1))(*[v for t in sorts for v in t])
        c_having = (ctypes.c_int64 * max(3 * len(having), 1))(*[v for j, c, _ in having for v in (j, c, 0)])
        c_hconst = (ctypes.c_double * max(len(having), 1))(*[x for _, _, x in having])
        c_xkeys = (ctypes.c_int64 * max(3 * n_keys, 1))(
            *[v for i in range(n_keys) for v in ((key_slot[i],) + ranges[i] if i in derived
                                                 else (-1,) + fact_range[i] if i in fact_range else (0, 0, 0))])
        F = (ctypes.c_int64 * 18)(
            n_fa, ctypes.addressof(c_colmap), len(pre_cols), pcode.ctypes.data, pre.consts.ctypes.data,
            pre.set_bytes.ctypes.data, pre.set_meta.ctypes.data, ctypes.addressof(c_pdtypes), pex.n,
            ctypes.addressof(c_pdoffs), pre_off, len(look_words), ctypes.addressof(L), probe_off,
            len(bspec), ctypes.addressof(c_bcols), 0, width)
        P = (ctypes.c_int64 * 29)(
            MAGIC, 3, len(segs), len(prog.columns), n_keys, n_out, len(sorts),
            code.ctypes.data, prog.consts.ctypes.data, prog.set_bytes.ctypes.data,
            prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes), ctypes.addressof(arrays),
            ctypes.addressof(c_kinds), ctypes.addressof(c_outs), ctypes.addressof(c_sorts),
            default_threads(table.num_rows), max_groups, limit, len(having), ctypes.addressof(c_having),
            max_groups, ctypes.addressof(c_hconst),
            len(dcols), ctypes.addressof(c_doffs), ctypes.addressof(c_xkeys), ctypes.addressof(F),
            (DISTINCT_FLAG if distinct else 0) | (REPRO_FLAG if reproducible() else 0) | (HASH_FLAG if hashed else 0), _record_limit())
        O = (ctypes.c_int64 * (3 + 2 * n_keys + n_out))()
        lib = _load()
        t_call = _time.perf_counter()
        rc = _call(lib, ctypes.addressof(P), ctypes.addressof(O))
        native_ms = (_time.perf_counter() - t_call) * 1000
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc == -11 and not any(src[0] == "prow" for src in fj.vcols.values()):
        got = _try_prow_groups(compiler, plan, fj, bs, having_pred, post_ops)   # E_BAD_KEY: keys past direct addressing
        if got is not None:
            return got
    if rc < 0:
        return _decline(f"executor declined (code {rc})")

    result = _finish_imeans(_wrap_result(lib, O, keys, kinds, outputs), imeans)
    for i, how in post.items() if not raw_keys else ():      # expression keys back to the column's own values
        c = result.column(i)
        result = result.set_column(i, keys[i], pc.take(how, c) if isinstance(how, pa.Array) else pc.cast(c, how))
    total_ms = (_time.perf_counter() - t_start) * 1000
    _last.update(path="native", decline=None, native_ms=native_ms, py_ms=total_ms - native_ms)
    compiler.last_compile_provenance = {
        "device": "cpu", "rows": int(result.num_rows), "grouped": bool(keys), "n_groups": int(result.num_rows) if keys else 0,
        "cache_hit": None, "compile_ms": 0.0, "path": "native", "gpu_input_path": "arrow_c_data", "fused_join": True,
    }
    return result


def _probe_range(fj: FusedJoin, bs, name):
    """(min, max) a fact column can take among the rows that reach the aggregate when it is a join key: only
    rows whose key finds a partner row count, so the partners' own key ranges bound it (the narrowest of them).
    None: not a join key, or not a plain integer key."""
    from .np_arrow import to_numpy as _np_of
    out = None
    for lk, b in zip(fj.lookups, bs):
        for pn, bn in zip(lk.probe, lk.build_keys):
            c = b.column(bn)
            if pn != name or c.null_count or c.type not in (pa.int32(), pa.int64()):
                continue
            a = _np_of(c.chunk(0), zero_copy_only=False)
            r = (int(a.min()), int(a.max()))
            out = r if out is None else (max(out[0], r[0]), min(out[1], r[1]))
    return out


def _partner_range(fj: FusedJoin, cols, name):
    """(min, max) of an integer partner column of the joined schema, else None (a fact column: no pass over it)."""
    import numpy as np
    from .np_arrow import to_numpy as _np_of
    src = fj.vcols.get(name)
    c = cols.get(name)
    if src is None or src[0] != "partner" or c is None or c.null_count or c.type not in (pa.int32(), pa.int64()):
        return None
    a = _np_of(c.chunk(0), zero_copy_only=False)
    return int(a.min()), int(a.max())


def _bounds(e, rng):
    """A (lo, hi) interval every value of integer expression ``e`` lies in, from the ranges ``rng(name)`` of its
    columns, or None. Conservative: a CASE WHEN is the union of its branches. Group ids outside the values
    that occur just stay empty."""
    op = e.op
    if op == "lit":
        v = e.args[0]
        return (int(v), int(v)) if isinstance(v, int) and not isinstance(v, bool) else None
    if op == "col":
        return rng(e.args[0])
    if op == "case_when":
        a, b = _bounds(e.args[1], rng), _bounds(e.args[2], rng)
        return None if a is None or b is None else (min(a[0], b[0]), max(a[1], b[1]))
    if op == "mod" and len(e.args) == 2 and e.args[1].op == "lit" and isinstance(e.args[1].args[0], int) and e.args[1].args[0]:
        a = _bounds(e.args[0], rng)                      # the sign follows the dividend, |x % k| < |k|
        if a is None:
            return None
        m = abs(e.args[1].args[0]) - 1
        return (max(a[0], -m) if a[0] < 0 else 0, min(a[1], m) if a[1] > 0 else 0)
    if op in ("add", "sub", "mul"):
        a, b = _bounds(e.args[0], rng), _bounds(e.args[1], rng)
        if a is None or b is None:
            return None
        if op == "add":
            return a[0] + b[0], a[1] + b[1]
        if op == "sub":
            return a[0] - b[1], a[1] - b[0]
        p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
        return min(p), max(p)
    return None


def _prow_or_decline(compiler, plan, fj, bs, having_pred, post_ops, reason, distinct=False):
    """A group key the expression-key route cannot take (a string column with too many distinct values, a
    float): group by the partner's row when that shape fits, else decline with ``reason``
    (never for a SELECT DISTINCT: its wide key spaces are ``mx_distinct``'s)."""
    if not distinct and not any(src[0] == "prow" for src in fj.vcols.values()):
        got = _try_prow_groups(compiler, plan, fj, bs, having_pred, post_ops)
        if got is not None:
            return got
    return _decline(reason)


def _prow_or_hash(compiler, plan, fj, bs, having_pred, post_ops, reason, distinct=False):
    """Keys past direct addressing: group by the partner's row when that shape fits (the result is returned),
    else (F-069) hash the key tuples: (None, True). A SELECT DISTINCT declines instead (its wide key spaces
    are ``mx_distinct``'s): (None, False)."""
    if distinct:
        return _decline(reason), False
    if not any(src[0] == "prow" for src in fj.vcols.values()):
        got = _try_prow_groups(compiler, plan, fj, bs, having_pred, post_ops)
        if got is not None:
            return got, False
    return None, True


def _npa_i32_empty():
    import numpy as np
    from .np_arrow import array as _npa
    return _npa(np.empty(0, dtype=np.int32))


def _try_prow_groups(compiler, plan, fj: FusedJoin, bs, having_pred, post_ops, only_if_wide=False) -> Optional[pa.Table]:
    """Group keys past direct addressing (TPC-H Q3's per-order keys, Q10's per-customer keys): when every key is
    a column of ONE partner (or the fact column equal to its join key), group by the partner's ROW (a dense id
    below its row count: `K_PROW`, the executor gathers the row number like any partner column), then take
    key columns from the partner at those rows. Only when the keys include the partner's unique join key:
    otherwise several partner rows share a key tuple (Q10's per-customer keys through orders) and their groups
    would have to be merged, which is left to a native merge (declined here). HAVING and ORDER BY / LIMIT run
    on that small result, the way the old path runs them. None: not this shape (or the partner has more rows
    than a direct id space)."""
    import numpy as np
    from ._lazy_pc import pc
    from .lazy_expr import Expr
    from .np_arrow import array as _npa, scalar as _npscalar, to_numpy as _np_of, combine as _combine
    srcs = []
    for e in plan.group_by:
        if not (e.op == "col" and e._alias in (None, e.args[0])) or fj.vcols.get(e.args[0]) is None:
            return None
        srcs.append((e.args[0], fj.vcols[e.args[0]]))
    if not srcs:
        return None
    owners = {src[1] for _, src in srcs if src[0] == "partner"}
    if len(owners) > 1:
        return None
    if owners:
        j = next(iter(owners))
    else:
        j = next((jj for jj, lk in enumerate(fj.lookups) if all(src[1] in lk.probe for _, src in srcs)), None)
        if j is None:
            return None
    lk, b = fj.lookups[j], bs[j]
    take_from = []                      # per key: its column of the partner
    for name, src in srcs:
        if src[0] == "partner":
            take_from.append(src[2])
        elif src[1] in lk.probe:
            take_from.append(lk.build_keys[lk.probe.index(src[1])])
        else:
            return None
    if b.num_rows > MAX_GROUPS:
        return None
    if only_if_wide:        # direct addressing would do when the keys' ranges multiply to few enough groups
        from .np_arrow import to_numpy as _np
        est = 1
        for bn in take_from:
            c = b.column(bn)
            if c.null_count == 0 and c.type in (pa.int32(), pa.int64()):
                a = _np(c.chunk(0), zero_copy_only=False)
                est *= int(a.max()) - int(a.min()) + 1
            elif pa.types.is_dictionary(c.type):
                est *= max(1, len(c.chunk(0).dictionary))
            else:
                est *= max(1, b.num_rows)
            if est > MAX_GROUPS:
                break
        if est <= MAX_GROUPS:
            return None
    if set(lk.build_keys) - set(take_from):          # several partner rows per key tuple: needs a merge
        return None
    vcols = dict(fj.vcols)
    vcols["__prow"] = ("prow", j)
    fj2 = FusedJoin(fj.scan, fj.preds, fj.lookups, vcols, fj.post_preds)
    fj2.cache = fj.cache
    R = try_execute_join(compiler, Aggregate(None, [Expr("col", "__prow")], list(plan.aggs)), fj2, None, None)
    if R is None:
        return None
    out_names = [a._alias or f"agg_{i}" for i, a in enumerate(plan.aggs)]
    gp = np.asarray(_np_of(_combine(R.column("__prow")), zero_copy_only=False)).astype(np.int64)    # partner row per group
    return _keys_by_row(compiler, plan, fj, b, srcs, take_from, gp, [R.column(n) for n in out_names], out_names,
                        having_pred, post_ops)


def _try_dependent_keys(compiler, plan, fj: FusedJoin, bs, having_pred, post_ops) -> Optional[pa.Table]:
    """Group keys that are plain partner strings determined by another key of the same source table (Q10:
    ``GROUP BY c_custkey, c_name, c_address, ...``; ``fj.fd`` maps each such column to the unique key that fixes
    it, F-076): group by the other keys and one representative partner row per group (``min`` of the row
    number), then take every key column from the partner at those rows after HAVING / ORDER BY / LIMIT, as
    ``_try_prow_groups`` does. No string is hashed or encoded. None: not this shape."""
    import numpy as np
    from .lazy_expr import Expr
    from .np_arrow import to_numpy as _np_of, combine as _combine
    fd = fj.fd
    names = [e.args[0] for e in plan.group_by if e.op == "col"]
    if not fd or len(names) != len(plan.group_by) or not any(n in fd and fd[n] in names for n in names):
        return None
    if any(src[0] == "prow" for src in fj.vcols.values()):
        return None
    dep = {n for n in names if n in fd and fd[n] in names}
    if any(fj.vcols.get(n) is None for n in names):
        return None
    srcs = [(e._alias or e.args[0], fj.vcols[e.args[0]]) for e in plan.group_by]
    j = fj.vcols[next(iter(dep))][1]
    if any(fj.vcols[n][0] == "partner" and fj.vcols[n][1] != j for n in names):      # every partner key is j's
        return None
    b = bs[j]
    if b.num_rows > MAX_GROUPS * 64:
        return None
    take_from = []
    for (nm, src), e in zip(srcs, plan.group_by):
        if src[0] == "partner":
            take_from.append(src[2])
        else:                                           # a fact key: the partner's own copy of its join key
            lk = fj.lookups[j]
            if src[1] not in lk.probe:
                return None
            take_from.append(lk.build_keys[lk.probe.index(src[1])])
    vcols = dict(fj.vcols)
    vcols["__prow"] = ("prow", j)
    fj2 = FusedJoin(fj.scan, fj.preds, fj.lookups, vcols, fj.post_preds)
    fj2.cache = fj.cache
    keep = [e for e in plan.group_by if e.args[0] not in dep]
    R = try_execute_join(compiler, Aggregate(None, keep, list(plan.aggs) + [Expr("min", Expr("col", "__prow")).alias("__rep")]),
                         fj2, None, None, raw_keys=True)         # (the keys come from the partner rows below)
    if R is None:
        return None
    out_names = [a._alias or f"agg_{i}" for i, a in enumerate(plan.aggs)]
    gp = np.asarray(_np_of(_combine(R.column("__rep")), zero_copy_only=False)).astype(np.int64)
    return _keys_by_row(compiler, plan, fj, b, srcs, take_from, gp, [R.column(n) for n in out_names], out_names,
                        having_pred, post_ops)


def _keys_by_row(compiler, plan, fj: FusedJoin, b, srcs, take_from, gp, agg_cols, out_names, having_pred, post_ops):
    """The result of a grouped aggregate whose groups are known by one partner row each: HAVING and the top-k
    cut on the aggregates first, then the key columns taken from the partner at the surviving rows."""
    import numpy as np
    from ._lazy_pc import pc
    from .np_arrow import array as _npa, scalar as _npscalar, to_numpy as _np_of, combine as _combine

    def subset(idx):
        nonlocal gp, agg_cols
        gp = gp[idx]
        agg_cols = [pc.take(c, _npa(idx.astype(np.int64))) for c in agg_cols]

    post = compiler._key_tiebreak(plan, list(post_ops or []))
    names = [n for n, _ in srcs] + out_names
    if having_pred is not None:
        # HAVING reads aggregates: evaluate it before any key is materialized
        from .optimizer import _expr_columns
        hc = _expr_columns(having_pred)
        if hc is not None and set(hc) <= set(out_names):
            mask = compiler._eval_predicate(having_pred, pa.Table.from_arrays(agg_cols, names=out_names))
            subset(np.flatnonzero(np.asarray(_np_of(pc.fill_null(mask, _npscalar(False, type=pa.bool_())),
                                                    zero_copy_only=False))))
            having_pred = None
    sort = post_ops[0] if post_ops and isinstance(post_ops[0], Sort) else None
    if (sort is not None and len(post_ops) == 2 and isinstance(post_ops[1], Limit) and post_ops[1].n is not None
            and len(gp) > 8 * max(post_ops[1].n, 1)
            and sort.by[0].op == "col" and sort.by[0].args[0] in out_names):
        # ORDER BY an aggregate LIMIT k: only the rows that can reach the top k need their keys (and the
        # key tie-break): those at least as good as the k-th by the first sort column, ties included
        desc = sort.descending[0] if isinstance(sort.descending, (list, tuple)) else bool(sort.descending)
        v = np.asarray(_np_of(_combine(agg_cols[out_names.index(sort.by[0].args[0])]), zero_copy_only=False)).astype(np.float64)
        k = max(post_ops[1].n, 1)
        if not np.isnan(v).any():
            w = -v if desc else v
            thr = np.partition(w, k - 1)[k - 1]
            subset(np.flatnonzero(w <= thr))
    key_cols = []
    for (name, src), bn in zip(srcs, take_from):
        t = pc.take(b.column(bn), _npa(gp))
        if pa.types.is_dictionary(t.type):
            t = pc.cast(t, pa.string())
        if src[0] == "fact" and t.type != fj.scan.table.column(src[1]).type:
            t = pc.cast(t, fj.scan.table.column(src[1]).type)
        key_cols.append(t)
    t = pa.Table.from_arrays(key_cols + agg_cols, names=names)
    if having_pred is not None:
        mask = compiler._eval_predicate(having_pred, t)
        t = pc.filter(t, pc.fill_null(mask, _npscalar(False, type=pa.bool_())))
    if sort is None and t.num_rows > 1:
        t = pc.take(t, pc.sort_indices(t, sort_keys=[(n, "ascending") for n, _ in srcs]))        # key order
    if post:
        t = compiler._apply_post_ops_custom(t, post)
    return t


# ── M2: native gather (a join's output rows) ───────────────────────────────────────────────

GMAGIC = 0x4D584731


def _call_gather(lib, plan_addr: int, out_addr: int) -> int:
    """The one native call per gather (a separate function so tests can count it)."""
    return lib.mx_gather(plan_addr, out_addr)


def try_gather(compiler, table: pa.Table, idx, decode_dicts: bool = False) -> Optional[pa.Table]:
    """``table.take(idx)`` in one native call (rows in ``idx`` order; ``idx`` an int32 / int64 NumPy
    array of row numbers). Dictionary string columns come back decoded, so only when
    ``decode_dicts`` (the caller decodes them anyway); else None. None: the caller takes as before."""
    import numpy as np
    if not enabled() or getattr(compiler, "_device_arg", None) != "cpu" or not available():
        return None
    if table.num_columns == 0 or table.num_rows == 0:
        return None
    idx = np.asarray(idx)
    if idx.dtype not in (np.int32, np.int64) or idx.ndim != 1 or len(idx) == 0:
        return None
    idx = np.ascontiguousarray(idx)
    cols = [table.column(i) for i in range(table.num_columns)]
    kinds, sizes = [], []
    for c in cols:
        k = _out_kind(c.type)
        if k is None or c.null_count or (k[0] == 2 and not decode_dicts):
            return None
        if k[0] == 2 and any(ch.dictionary.null_count for ch in c.chunks):
            return None
        kinds.append(k[0])
        sizes.append(k[1])
    segs = _segments(cols)
    if not segs:
        return None
    width = len(cols)
    structs = []
    arrays = (ctypes.c_int64 * (len(segs) * width))()
    try:
        for r, seg in enumerate(segs):
            for j, piece in enumerate(seg):
                st = (ctypes.c_int64 * 10)()
                piece._export_to_c(ctypes.addressof(st))
                structs.append(st)
                arrays[r * width + j] = ctypes.addressof(st)
        c_kinds = (ctypes.c_int64 * width)(*kinds)
        c_sizes = (ctypes.c_int64 * width)(*sizes)
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 11)(GMAGIC, 1, len(segs), width, ctypes.addressof(arrays),
                                  ctypes.addressof(c_kinds), ctypes.addressof(c_sizes),
                                  idx.ctypes.data, len(idx), idx.dtype.itemsize, default_threads(len(idx)))
        O = (ctypes.c_int64 * (2 + 2 * width))()
        lib = _load()
        rc = _call_gather(lib, ctypes.addressof(P), ctypes.addressof(O))
    finally:
        for st in structs:
            if st[8]:
                _RELEASE(st[8])(ctypes.addressof(st))
    if rc < 0:
        return None
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))

    def buf(addr, nbytes):
        return pa.foreign_buffer(int(addr), max(nbytes, 0), base=owner)

    out = []
    for j, (c, kind, size) in enumerate(zip(cols, kinds, sizes)):
        a, b = int(O[2 + 2 * j]), int(O[3 + 2 * j])
        if kind == 0:
            out.append(pa.Array.from_buffers(c.type, n, [None, buf(a, n * size)]))
        else:
            nbytes = ctypes.c_int32.from_address(a + 4 * n).value
            out.append(pa.Array.from_buffers(pa.string(), n, [None, buf(a, 4 * (n + 1)), buf(b, nbytes)]))
    return pa.Table.from_arrays(out, names=table.column_names)


# ── the CPU share of a CPU+GPU split (F-057) ───────────────────────────────────────────────────────────────

_GID = "__mx_gid__"
_CNT = "__mx_cnt__"


def range_partials(compiler, prog, table: pa.Table, predicates, agg_node, gids, n_groups: int,
                   start: int, end: int):
    """The native executor over rows ``[start, end)`` of ``table``, as the ``Partials`` the fused pipeline's
    CPU kernel would return for that range (so they merge with the GPU's), or None when it declines
    (the caller runs the pipeline kernel over the range instead).

    The slice is a zero-copy ``table.slice`` and the rows are grouped by their group id (a column of
    ``gids``), so the result's rows land at their group's position without comparing keys. Each of the
    program's slots is computed by one aggregate of the same expression: a mean's slot is its sum
    (``finalize`` divides by the count), and a float min / max (DuckDB: NaN above every number) is turned
    into the pipeline's slot pair, the NaN-free extreme plus a NaN count (see ``pipeline_program``)."""
    import numpy as np
    from .lazy_expr import Expr, col, lit
    from .np_arrow import array as _npa, to_numpy as _np_of
    from .pipeline_program import Partials

    vals = np.repeat(prog.identity[:, None], n_groups, axis=1)
    cnt = np.zeros(n_groups, dtype=np.int64)
    if end <= start:
        return Partials(vals=vals, cnt=cnt)
    grouped = gids is not None
    sliced = table.slice(start, end - start)
    if grouped:
        sliced = sliced.append_column(_GID, pa.chunked_array([_npa(np.ascontiguousarray(gids[start:end]))]))
    slot_agg = {}                                    # slot -> (aggregate name, op)
    aggs = [Expr("count", lit(1)).alias(_CNT)]
    for e, (name, op, slot) in zip(agg_node.aggs, prog.outputs):
        if slot is None or slot in slot_agg:
            continue
        kind = {"mean": "sum", "isum": "sum", "imean": "sum", "imin": "min", "imax": "max"}.get(op, op)
        slot_agg[slot] = f"__mx_s{slot}"
        aggs.append(Expr(kind, e.args[0]).alias(slot_agg[slot]))
    nan_of = {}                                      # hidden NaN-count slot -> [(aggregate name, op)]
    out_of = {n: (o, sl) for n, o, sl in prog.outputs}
    for name, nan_slot in prog.nan_slots.items():
        op, slot = out_of[name]
        nan_of.setdefault(nan_slot, []).append((slot_agg[slot], op))
    node = Scan(sliced)
    for p in reversed(predicates):
        node = Filter(node, p)
    plan = Aggregate(node, [col(_GID)] if grouped else [], aggs)
    res = try_execute(compiler, plan, None, None, coproc_share=True)
    if res is None:
        return None
    idx = _np_of(res.column(_GID), zero_copy_only=False).astype(np.int64) if grouped else np.zeros(1, np.int64)

    def column(name):
        return np.asarray(_np_of(res.column(name), zero_copy_only=False), dtype=np.float64)

    def ints(name):                                  # an integer sum: int64 (the executor adds it exactly), not a float
        return np.asarray(_np_of(res.column(name), zero_copy_only=False), dtype=np.int64)

    c = column(_CNT).astype(np.int64)
    if not grouped and c[0] == 0:                    # a global aggregate over no surviving rows
        return Partials(vals=vals, cnt=cnt)
    cnt[idx] = c
    for slot, name in slot_agg.items():
        if prog.slot_kinds[slot] in ("isum", "imin", "imax"):
            vals[slot].view(np.int64)[idx] = ints(name)
            continue
        v = column(name)
        if prog.slot_kinds[slot] in ("min", "max"):      # the pipeline's slot never holds a NaN
            v = np.where(np.isnan(v), prog.identity[slot], v)
        vals[slot, idx] = v
    for slot, users in nan_of.items():
        n_nan = np.zeros(len(idx))
        for name, op in users:
            hit = np.isnan(column(name))
            n_nan = np.maximum(n_nan, np.where(hit, c if op == "min" else 1.0, 0.0))
        vals[slot, idx] = n_nan
    return Partials(vals=vals, cnt=cnt)
