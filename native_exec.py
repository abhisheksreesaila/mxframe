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
_CMP = {"lt": 0, "le": 1, "gt": 2, "ge": 3, "eq": 4, "ne": 5}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
K_DICT, K_I32, K_I64, K_STR, K_EXPR = 0, 1, 2, 3, 4
_OPS = {"sum": 0, "min": 1, "max": 2, "mean": 3, "count": 4}
_DT = {pa.int32(): 0, pa.int64(): 1, pa.float32(): 2, pa.float64(): 3}

_lib = None
_lock = threading.Lock()
_last = {"path": None, "decline": None, "py_ms": None, "native_ms": None}
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
            _lib = lib
    return _lib


def last_path():
    return _last["path"]


def last_timing():
    """(Python-side ms, native-call ms) of the last native run."""
    return _last["py_ms"], _last["native_ms"]


def last_decline():
    return _last["decline"]


def _decline(reason: str):
    _last["path"], _last["decline"] = None, reason
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
    return {pa.int32(): K_I32, pa.int64(): K_I64}.get(t)


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


def try_execute(compiler, plan, having_pred, post_ops, gpu_fallback: bool = False) -> Optional[pa.Table]:
    """Run the whole plan in one native call, or None (the caller's path runs instead).
    `gpu_fallback`: a GPU session whose own pipeline declined the plan answers it here, on the CPU."""
    _last["path"] = None
    t_start = _time.perf_counter()
    if not enabled():
        return _decline("disabled (MXFRAME_NATIVE=0)")
    if getattr(compiler, "_device_arg", None) != "cpu" and not gpu_fallback:
        return _decline("CPU executor: device='cpu' (or a GPU session's fallback)")
    from . import coproc_fused
    if coproc_fused.current_policy() is not None:
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

    keys, derived = [], {}                     # derived: key position -> integer expression
    for i, e in enumerate(plan.group_by):
        if e.op == "col" and e._alias in (None, e.args[0]):
            keys.append(e.args[0])
            continue
        if not e._alias:
            return _decline("derived group key without a name")
        keys.append(e._alias)
        derived[i] = e
    sort_node, limit = None, -1
    for op in post_ops or []:
        if isinstance(op, Sort) and sort_node is None and limit < 0:
            sort_node = op
        elif isinstance(op, Limit) and limit < 0:
            limit = int(op.n)
        else:
            return _decline(f"post-op {type(op).__name__} in this position")
    sort_by = []                                  # (name, descending)
    if sort_node is not None:
        for e, desc in zip(sort_node.by, sort_node.descending):
            if e.op != "col":
                return _decline("sort by an expression")
            sort_by.append((e.args[0], bool(desc)))

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
    ranges = {}
    if derived:
        # an integer expression key (TPC-H's order year): its range from one native min / max pass,
        # then the executor groups by the program's value of it (directly addressed)
        probe = Aggregate(plan.input, [], [a for i, e in derived.items()
                                           for a in (Expr("min", e).alias(f"lo{i}"), Expr("max", e).alias(f"hi{i}"))])
        r = try_execute(compiler, probe, None, None, gpu_fallback)
        if r is None:
            return _decline("derived group key: range pass declined")
        for i in derived:
            lo, hi = r.column(f"lo{i}")[0].as_py(), r.column(f"hi{i}")[0].as_py()
            if lo is None or hi is None or lo != int(lo) or hi != int(hi) or hi - lo + 1 > MAX_GROUPS:
                return _decline("derived group key: not a small integer range")
            ranges[i] = (int(lo), int(hi))
    sorts = []
    for name, desc in sort_by:
        if name in keys:
            sorts.append((0, keys.index(name), int(desc)))
        elif name in names:
            sorts.append((1, names.index(name), int(desc)))
        else:
            return _decline("sort by a column that is not a key or an aggregate")
    having = _having_terms(having_pred, names)
    if having is None:
        return _decline("HAVING other than AND-ed aggregate <op> number")
    if prog.dict_arrays:
        return _decline("string predicates outside concat mode")
    dcols = [c for c in prog.columns if c in prog.dict_concat]      # string columns in predicates

    cols = [table.column(c) for c in prog.columns]
    dtypes = []
    for name, c in zip(prog.columns, cols):
        if name in prog.dict_concat:                  # codes + the chunk's offset (dtype 5)
            dtypes.append(5 | ((len(prog.columns) + dcols.index(name)) << 8))
            continue
        if c.type not in _DT or c.null_count:
            return _decline(f"value column type {c.type} or nulls")
        dtypes.append(_DT[c.type])
    key_cols, kinds = [], []
    for i, k in enumerate(keys):
        if i in derived:                       # placeholder column: the key comes from the program
            key_cols.append(table.column(0))
            kinds.append(K_EXPR)
            continue
        c = table.column(k)
        kind = _key_kind(c)
        if kind is None or c.null_count:
            return _decline(f"group key type {c.type} or nulls not in M0")
        key_cols.append(c)
        kinds.append(kind)
    every = cols + key_cols
    if not every:
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
        from .pipeline_program import _dict_key
        doffs = []
        for seg in segs:
            for name in dcols:
                doffs.append(prog.dict_concat[name][_dict_key(seg[prog.columns.index(name)].dictionary)])
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        code = prog.code
        if dcols:                                     # the chunk table gets one offset slot per column
            code = code.copy()
            code[7] = len(prog.columns) + len(dcols)
        from .pipeline_kernels import default_threads
        c_xkeys = (ctypes.c_int64 * max(3 * n_keys, 1))(
            *[v for i in range(n_keys) for v in ((key_slot[i],) + ranges[i] if i in derived else (0, 0, 0))])
        P = (ctypes.c_int64 * 26)(
            MAGIC, 2, len(segs), len(cols), n_keys, n_out, len(sorts),
            code.ctypes.data, prog.consts.ctypes.data, prog.set_bytes.ctypes.data,
            prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes), ctypes.addressof(arrays),
            ctypes.addressof(c_kinds), ctypes.addressof(c_outs), ctypes.addressof(c_sorts),
            default_threads(table.num_rows), MAX_GROUPS, limit, len(having), ctypes.addressof(c_having),
            min(MAX_COMPACT, max(MAX_GROUPS, 8 * table.num_rows)), ctypes.addressof(c_hconst),
            len(dcols), ctypes.addressof(c_doffs), ctypes.addressof(c_xkeys))
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

    # ── wrap the result: zero-copy views of the executor's arena ─────────────────────────────
    n = int(O[0])
    owner = _Arena(lib, int(O[1]))
    empty_global = bool(O[2] & 1)

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
        else:
            arr = pa.Array.from_buffers(pa.int64(), n, [None, buf(a, 8 * n)])
        names.append(name)
        out_cols.append(arr)
    base = 3 + 2 * len(keys)
    for j, (name, op, _) in enumerate(outputs):
        t = pa.int64() if op == "count" else pa.float64()
        if empty_global and op != "count":
            arr = pa.nulls(1, t)
        else:
            arr = pa.Array.from_buffers(t, n, [None, buf(O[base + j], 8 * n)])
        names.append(name)
        out_cols.append(arr)
    result = pa.Table.from_arrays(out_cols, names=names)
    total_ms = (_time.perf_counter() - t_start) * 1000
    _last.update(path="native", decline=None, native_ms=native_ms, py_ms=total_ms - native_ms)
    compiler.last_compile_provenance = {
        "device": "cpu", "rows": int(n), "grouped": bool(keys), "n_groups": int(n) if keys else 0,
        "cache_hit": None, "compile_ms": 0.0, "path": "native", "gpu_input_path": "arrow_c_data",
    }
    return result


_DISTINCT_KIND = {pa.int32(): 0, pa.int64(): 1, pa.string(): 2}
DMAGIC = 0x4D584431


def _call_distinct(lib, plan_addr: int, out_addr: int) -> int:
    """The native distinct call (a separate function so tests can count it)."""
    return lib.mx_distinct(plan_addr, out_addr)


def try_distinct(compiler, table: pa.Table) -> Optional[pa.Table]:
    """The distinct rows of ``table`` in first-appearance order (F-032 M2h): one native call finds
    the first row of every tuple (parallel hashing, 64 partitions built in row order), then the
    kept rows are gathered. Columns int32 / int64 / utf8 without nulls; None: the caller's path."""
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
    out = try_gather(compiler, table, idx)
    if out is None:
        out = pc.take(table, pa.array(idx))
    _last.update(path="native", decline=None)
    return out


# ── M2: native leaf filter (Filter*(Scan) under a join) ─────────────────────────────────────

FMAGIC = 0x4D584631
_FIXED_OK = (pa.types.is_integer, pa.types.is_floating, pa.types.is_date, pa.types.is_timestamp,
             pa.types.is_decimal, pa.types.is_time, pa.types.is_duration)
_filter_last = {"path": None, "decline": None}


def last_filter():
    """(path, decline reason) of the last native leaf-filter attempt."""
    return _filter_last["path"], _filter_last["decline"]


def _filter_decline(reason: str):
    _filter_last["path"], _filter_last["decline"] = None, reason
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


class Lookup:
    """A join partner fused into the leaf filter (F-032): rows of the scanned table whose ``probe``
    columns equal ``build_keys`` of a row of ``build`` keep that row's ``outputs`` (written under
    ``names``). ``build`` keys must be unique (a duplicate declines); 1 or 2 integer key columns.
    ``semi``: only whether a partner row exists counts (keys may repeat, no outputs)."""
    __slots__ = ("probe", "build", "build_keys", "outputs", "names", "semi")

    def __init__(self, probe, build: pa.Table, build_keys, outputs, names=None, semi=False):
        self.probe, self.build, self.build_keys = list(probe), build, list(build_keys)
        self.outputs = list(outputs)
        self.names = list(names) if names is not None else list(outputs)
        self.semi = bool(semi)


def _lookup_key_kind(t: pa.DataType):
    return "date" if t == pa.date32() else "int"


def try_filter(compiler, table: pa.Table, predicates, needed, lookups=()) -> Optional[pa.Table]:
    """The rows of ``table`` that pass every predicate (and find a partner row in every lookup),
    holding the ``needed`` columns (None: all) in table order, then each lookup's outputs, in one
    native call; dictionary string columns come back decoded (as the join path expects). None: the
    caller filters (and joins) as before."""
    _filter_last["path"] = None
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

    from .pipeline_program import compile_pipeline, _dict_key
    prog = compile_pipeline(list(predicates), [], table, concat_dicts=True)
    if prog is None:
        return _filter_decline("predicate not supported by the bytecode compiler")
    if prog.dict_arrays:
        return _filter_decline("string predicates outside concat mode")
    dcols = [c for c in prog.columns if c in prog.dict_concat]
    vcols = [table.column(c) for c in prog.columns]
    dtypes = []
    for name, c in zip(prog.columns, vcols):
        if name in prog.dict_concat:
            dtypes.append(5 | ((len(prog.columns) + dcols.index(name)) << 8))
            continue
        if c.type not in _DT or c.null_count:
            return _filter_decline(f"predicate column type {c.type} or nulls")
        dtypes.append(_DT[c.type])
    ocols = [table.column(n) for n in names]
    kinds, sizes = [], []
    for n, c in zip(names, ocols):
        k = _out_kind(c.type)
        if k is None or c.null_count:
            return _filter_decline(f"output column {n}: type {c.type} or nulls")
        if k[0] == 2 and any(ch.dictionary.null_count for ch in c.chunks):
            return _filter_decline(f"output column {n}: null in a dictionary")
        kinds.append(k[0])
        sizes.append(k[1])

    # lookups: probe key columns ride along with the scanned table's chunks; partners are one chunk
    pk_names: List[str] = []
    look_words, bout_words, keep_alive = [], [], []
    out_types, out_names = [c.type for c in ocols], list(names)
    for lk in lookups:
        if len(lk.probe) not in (1, 2) or len(lk.probe) != len(lk.build_keys):
            return _filter_decline("lookup: 1 or 2 key columns")
        if lk.semi and lk.outputs:
            return _filter_decline("lookup: a semi join has no outputs")
        b = lk.build.combine_chunks()     # one chunk per column; dictionaries unified, not decoded
        if b.num_rows == 0 or any(b.column(c).num_chunks != 1 for c in b.column_names):
            return _filter_decline("lookup: empty partner")
        pks, widths = [], 0
        for i, (pn, bn) in enumerate(zip(lk.probe, lk.build_keys)):
            pc_, bc_ = table.column(pn), b.column(bn)
            if (pc_.type not in _KEY_WIDTH or bc_.type not in _KEY_WIDTH or pc_.null_count or bc_.null_count
                    or _lookup_key_kind(pc_.type) != _lookup_key_kind(bc_.type)):
                return _filter_decline(f"lookup key {pn} = {bn}: types {pc_.type} / {bc_.type} or nulls")
            if pn not in pk_names:
                pk_names.append(pn)
            pks.append(pk_names.index(pn))
            widths |= _KEY_WIDTH[pc_.type] << (4 * i) | _KEY_WIDTH[bc_.type] << (8 + 4 * i)
        if lk.semi:
            widths |= 1 << 16
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
                bo[3 * t], bo[3 * t + 1], bo[3 * t + 2] = export(arr), kind, size
            keep_alive.append(bo)
            L[8 * j: 8 * j + 8] = [nk, pks[0], pks[1] if nk == 2 else -1, export(bkeys[0]),
                                   export(bkeys[1]) if nk == 2 else 0, widths, len(bouts), ctypes.addressof(bo)]
        doffs = [prog.dict_concat[n][_dict_key(seg[prog.columns.index(n)].dictionary)]
                 for seg in segs for n in dcols]
        c_doffs = (ctypes.c_int64 * max(len(doffs), 1))(*doffs)
        code = prog.code
        if dcols:
            code = code.copy()
            code[7] = len(prog.columns) + len(dcols)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_kinds = (ctypes.c_int64 * max(len(kinds), 1))(*kinds)
        c_sizes = (ctypes.c_int64 * max(len(sizes), 1))(*sizes)
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 19)(
            FMAGIC, 2 if look_words else 1, len(segs), len(vcols), len(ocols), code.ctypes.data, prog.consts.ctypes.data,
            prog.set_bytes.ctypes.data, prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes),
            ctypes.addressof(arrays), ctypes.addressof(c_kinds), ctypes.addressof(c_sizes),
            default_threads(table.num_rows), len(dcols), ctypes.addressof(c_doffs),
            len(look_words), ctypes.addressof(L), len(pk_names))
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

    out = []
    for j, t in enumerate(out_types):
        kind, size = _out_kind(t)
        a, b = int(O[2 + 2 * j]), int(O[3 + 2 * j])
        if kind == 0:
            out.append(pa.Array.from_buffers(t, n, [None, buf(a, n * size)]))
        else:
            nbytes = ctypes.c_int32.from_address(a + 4 * n).value
            out.append(pa.Array.from_buffers(pa.string(), n, [None, buf(a, 4 * (n + 1)), buf(b, nbytes)]))
    _filter_last.update(path="native_filter" if not lookups else "native_lookup", decline=None)
    return pa.Table.from_arrays(out, names=out_names)


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
