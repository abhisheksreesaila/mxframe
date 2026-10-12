"""Native executor front end (F-030, milestone M0 of docs/design/mojo-executor.md).

Python only plans: it checks the plan shape, compiles the expressions to bytecode (the existing
`pipeline_program` compiler), hands the input columns over through the Arrow C Data Interface and
makes ONE call, `mx_execute` in ``libmxexec_cpu.so``. Group keys, filtering, aggregation, finishing
means / counts, sorting and building the output columns all run in Mojo; the result buffers are
wrapped zero-copy and freed by the executor when the table is garbage collected.

M0 covers ``Sort(all group keys)? Aggregate(Filter*(Scan))`` on the CPU, with dictionary-encoded
string keys (each chunk may have its own dictionary; up to 4096 distinct strings) and int32 / int64 keys, null-free numeric value columns, and sum / mean / count (min /
max without NaN handling). Everything else returns ``None`` and the existing path runs: the
executor never changes an answer. Opt-in during the spike: ``MXFRAME_NATIVE=1``.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time as _time
from typing import Optional

import pyarrow as pa

from .lazy_frame import Aggregate, Filter, Scan, Sort

MAGIC = 0x4D584531
MAX_GROUPS = 1 << 20
K_DICT, K_I32, K_I64 = 0, 1, 2
_OPS = {"sum": 0, "min": 1, "max": 2, "mean": 3, "count": 4}
_DT = {pa.int32(): 0, pa.int64(): 1, pa.float32(): 2, pa.float64(): 3}

_lib = None
_lock = threading.Lock()
_last = {"path": None, "decline": None, "py_ms": None, "native_ms": None}
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def enabled() -> bool:
    return os.environ.get("MXFRAME_NATIVE", "0") in ("1", "true", "on")


def _path():
    from .pipeline_kernels import _find
    return _find("libmxexec_cpu.so", "MXFRAME_EXEC_CPU_LIB")


def available() -> bool:
    return _path() is not None


def _load():
    global _lib
    with _lock:
        if _lib is None:
            lib = ctypes.CDLL(str(_path()))
            lib.mx_execute.argtypes = [ctypes.c_int64, ctypes.c_int64]
            lib.mx_execute.restype = ctypes.c_int64
            lib.mx_exec_free.argtypes = [ctypes.c_int64]
            lib.mx_exec_free.restype = ctypes.c_int64
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


def try_execute(compiler, plan, having_pred, post_ops) -> Optional[pa.Table]:
    """Run the whole plan in one native call, or None (the caller's path runs instead)."""
    _last["path"] = None
    t_start = _time.perf_counter()
    if not enabled():
        return _decline("disabled (MXFRAME_NATIVE != 1)")
    if getattr(compiler, "_device_arg", None) != "cpu":
        return _decline("M0 runs device='cpu' only")
    if not available():
        return _decline("libmxexec_cpu.so not built")
    if having_pred is not None:
        return _decline("HAVING not in M0")
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

    keys = []
    for e in plan.group_by:
        if e.op != "col" or e._alias:
            return _decline("derived group key")
        keys.append(e.args[0])
    sorts = []
    for op in post_ops or []:
        if not isinstance(op, Sort) or sorts:
            return _decline(f"post-op {type(op).__name__} not in M0")
        for e, desc in zip(op.by, op.descending):
            if e.op != "col" or e.args[0] not in keys:
                return _decline("sort by something other than the group keys")
            sorts.append((keys.index(e.args[0]), bool(desc)))
    if keys and sorted(k for k, _ in sorts) != list(range(len(keys))):
        return _decline("grouped result without a sort on every group key")

    from .pipeline_program import compile_pipeline
    prog = compile_pipeline(preds, list(plan.aggs), table)
    if prog is None:
        return _decline("expression not supported by the bytecode compiler")
    if prog.dict_arrays or prog.nan_slots:
        return _decline("string predicates / NaN-aware min-max not in M0")

    cols = [table.column(c) for c in prog.columns]
    dtypes = []
    for c in cols:
        if c.type not in _DT or c.null_count:
            return _decline(f"value column type {c.type} or nulls not in M0")
        dtypes.append(_DT[c.type])
    key_cols = [table.column(k) for k in keys]
    kinds = []
    for c in key_cols:
        kind = _key_kind(c)
        if kind is None or c.null_count:
            return _decline(f"group key type {c.type} or nulls not in M0")
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
        n_keys, n_out = len(keys), len(prog.outputs)
        c_dtypes = (ctypes.c_int32 * max(len(dtypes), 1))(*dtypes)
        c_kinds = (ctypes.c_int64 * max(n_keys, 1))(*kinds)
        c_outs = (ctypes.c_int64 * max(2 * n_out, 1))(
            *[v for _, op, slot in prog.outputs for v in (_OPS[op], slot if slot is not None else 0)])
        c_sorts = (ctypes.c_int64 * max(2 * len(sorts), 1))(*[v for k, d in sorts for v in (k, int(d))])
        from .pipeline_kernels import default_threads
        P = (ctypes.c_int64 * 18)(
            MAGIC, 1, len(segs), len(cols), n_keys, n_out, len(sorts),
            prog.code.ctypes.data, prog.consts.ctypes.data, prog.set_bytes.ctypes.data,
            prog.set_meta.ctypes.data, ctypes.addressof(c_dtypes), ctypes.addressof(arrays),
            ctypes.addressof(c_kinds), ctypes.addressof(c_outs), ctypes.addressof(c_sorts),
            default_threads(), MAX_GROUPS)
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
        if kind == K_DICT:
            nbytes = ctypes.c_int32.from_address(a + 4 * n).value
            arr = pa.Array.from_buffers(pa.string(), n, [None, buf(a, 4 * (n + 1)), buf(b, nbytes)])
        elif kind == K_I32:
            arr = pa.Array.from_buffers(pa.int32(), n, [None, buf(a, 4 * n)])
        else:
            arr = pa.Array.from_buffers(pa.int64(), n, [None, buf(a, 8 * n)])
        names.append(name)
        out_cols.append(arr)
    base = 3 + 2 * len(keys)
    for j, (name, op, _) in enumerate(prog.outputs):
        t = pa.int64() if op == "count" else pa.float64()
        if empty_global and op != "count":
            arr = pa.nulls(1, t)
        else:
            arr = pa.Array.from_buffers(t, n, [None, buf(O[base + j], 8 * n)])
        names.append(name)
        out_cols.append(arr)
    result = pa.Table.from_arrays(out_cols, names=names)
    total_ms = (_time.perf_counter() - t_start) * 1000
    _last.update(path="native_m0", decline=None, native_ms=native_ms, py_ms=total_ms - native_ms)
    compiler.last_compile_provenance = {
        "device": "cpu", "rows": int(n), "grouped": bool(keys), "n_groups": int(n) if keys else 0,
        "cache_hit": None, "compile_ms": 0.0, "path": "native_m0", "gpu_input_path": "arrow_c_data",
    }
    return result
