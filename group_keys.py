"""Group ids for dictionary-encoded keys, straight from the Arrow code buffers (F-015).

``build(table, keys)`` returns ``(dense_ids, n_groups, unique_key_cols)`` exactly as
``CustomOpsCompiler._build_group_ids`` does, or ``None`` when the shape is not handled
(the caller then uses the existing path): a key that is not dictionary typed, nulls, an
unsupported index type or a composite space too large for a dense presence table.

Cost model: one parallel pass over the code buffers (any of int8/int16/int32/uint8/uint16,
read in place, no widening copy) writes int32 composite ids and a presence table; a second
parallel pass rewrites them to dense ids when some combinations never occur. Chunks that
carry different dictionaries (one per Parquet row group) are unified on the tiny
dictionaries only, never on the rows, and chunks that share a dictionary skip that too.
Only combinations that occur become groups, so unused dictionary entries need no check.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
from ._lazy_pc import pc
from .np_arrow import array as _npa, to_numpy as _np_of

__all__ = ["available", "build", "MAX_SPACE", "encode_int"]

MAX_SPACE = 1 << 18            # dense presence/remap table limit (composite id space)
_MAX_DICT_TOTAL = 1 << 20      # total chunk-dictionary entries unified in Python
_HERE = Path(__file__).parent
_lock = threading.Lock()
_lib = None
_I64 = ctypes.c_int64

_IK_ARGS = {"mx_ik_range": [_I64] * 5, "mx_ik_first": [_I64] * 7, "mx_ik_rank": [_I64] * 5,
            "mx_ik_finish": [_I64] * 8, "mx_ik_apply": [_I64] * 7,
            "mx_ik_composite": [_I64] * 7, "mx_ik_present": [_I64] * 7, "mx_ik_prefix": [_I64] * 4,
            "mx_ik_uniq": [_I64] * 6, "mx_ik_rank_sorted": [_I64] * 6,
            "mx_ik_rank_bucketed": [_I64] * 9}
_IK_WIDTH = {np.dtype("int8"): 0, np.dtype("int16"): 1, np.dtype("int32"): 2, np.dtype("int64"): 3,
             np.dtype("uint8"): 4, np.dtype("uint16"): 5, np.dtype("uint32"): 6}
IK_MAX_RANGE_FACTOR = 4       # direct-address table entries per row, at most
IK_MIN_ROWS = 50_000           # below this NumPy/Arrow are as fast and skip the thread hand-off
_WIDTH = {pa.int8(): (0, 1), pa.int16(): (1, 2), pa.int32(): (2, 4), pa.uint8(): (3, 1), pa.uint16(): (4, 2)}


def _find() -> Optional[Path]:
    env = os.environ.get("MXFRAME_GROUPKEYS_LIB")
    if env and Path(env).exists():
        return Path(env)
    for cand in (_HERE / "kernels_aot" / "libmxgroupkeys_cpu.so", _HERE / "libmxgroupkeys_cpu.so"):
        if cand.exists():
            return cand
    return None


def available() -> bool:
    return _find() is not None


def _load():
    global _lib
    with _lock:
        if _lib is None:
            path = _find()
            if path is None:
                raise FileNotFoundError("libmxgroupkeys_cpu.so not found")
            lib = ctypes.CDLL(str(path))
            lib.mx_groupkeys_composite.restype = _I64
            lib.mx_groupkeys_composite.argtypes = [_I64] * 9
            lib.mx_groupkeys_remap.restype = _I64
            lib.mx_groupkeys_remap.argtypes = [_I64] * 4
            lib.mx_gk_init_runtime.restype = _I64
            lib.mx_gk_touch.restype = _I64
            lib.mx_gk_touch.argtypes = [_I64] * 3
            for name in _IK_ARGS:
                fn = getattr(lib, name, None)
                if fn is not None:
                    fn.restype = _I64
                    fn.argtypes = _IK_ARGS[name]
            lib.mx_gk_init_runtime()
            _lib = lib
    return _lib


def _threads() -> int:
    from .pipeline_kernels import default_threads
    return default_threads()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _key_chunks(col, n_rows):
    """The non-empty dictionary chunks of ``col`` or None if the column is not handled."""
    if isinstance(col, pa.ChunkedArray):
        chunks = [c for c in col.chunks if len(c)]
    else:
        chunks = [col] if len(col) else []
    if not chunks or not pa.types.is_dictionary(chunks[0].type):
        return None
    if sum(len(c) for c in chunks) != n_rows:
        return None
    for c in chunks:
        if c.null_count or c.dictionary.null_count or c.type.index_type not in _WIDTH:
            return None
    return chunks


def _unify(chunks):
    """(unified dictionary, per-chunk int32 remap arrays or None when all chunks share one dictionary)."""
    d0 = chunks[0].dictionary
    if all(c.dictionary is d0 or c.dictionary.equals(d0) for c in chunks[1:]):
        return d0, None
    dicts = [c.dictionary for c in chunks]
    if sum(len(d) for d in dicts) > _MAX_DICT_TOTAL:
        return None
    enc = pc.dictionary_encode(pa.concat_arrays(dicts))
    ids = _np_of(enc.indices, zero_copy_only=False).astype(np.int32, copy=False)
    remaps, off = [], 0
    for d in dicts:
        remaps.append(np.ascontiguousarray(ids[off:off + len(d)]))
        off += len(d)
    return enc.dictionary, remaps


def build(table, keys):
    if not available():
        return None
    n_rows = table.num_rows
    if n_rows == 0 or not keys or len(keys) > 8:
        return None
    dicts, sizes, rows, keep = [], [], [], []
    for k in keys:
        chunks = _key_chunks(table.column(k), n_rows)
        if chunks is None or any(len(c.dictionary) == 0 for c in chunks):
            return None      # non-empty chunk with an empty dictionary: invalid codes, old path decides
        u = _unify(chunks)
        if u is None:
            return None
        d, remaps = u
        dicts.append(d)
        sizes.append(len(d))
        keep.append((chunks, remaps))
    strides = []
    for i in range(len(sizes)):
        s = 1
        for j in range(i + 1, len(sizes)):
            s *= sizes[j]
        strides.append(s)
    space = strides[0] * sizes[0]
    if space > MAX_SPACE or space < 1:
        return None

    tbl_rows, key_off = [], [0]
    for chunks, remaps in keep:
        start = 0
        for j, c in enumerate(chunks):
            idx = c.indices
            wcode, isz = _WIDTH[idx.type]
            base = idx.buffers()[1].address + idx.offset * isz - start * isz
            tbl_rows.append((start, start + len(c), base, _addr(remaps[j]) if remaps else 0, wcode,
                             len(remaps[j]) if remaps else len(c.dictionary)))
            start += len(c)
        key_off.append(len(tbl_rows))
    tbl = np.array(tbl_rows, dtype=np.int64)
    key_off = np.array(key_off, dtype=np.int64)
    strides_a = np.array(strides, dtype=np.int64)
    ids = np.empty(n_rows, dtype=np.int32)
    present = np.empty(space, dtype=np.uint8)

    lib = _load()
    th = _threads()
    lib.mx_gk_touch(_addr(ids), ids.nbytes, th)     # fault the pages in with all cores first
    rc = lib.mx_groupkeys_composite(_addr(tbl), _addr(key_off), len(keys), _addr(strides_a), n_rows,
                                    _addr(ids), _addr(present), space, th)
    if rc != 0:
        return None
    pres = np.flatnonzero(present)
    n_groups = len(pres)
    if n_groups == 0:
        return None
    if n_groups != space:
        table_a = np.zeros(space, dtype=np.int32)
        table_a[pres] = np.arange(n_groups, dtype=np.int32)
        if lib.mx_groupkeys_remap(_addr(ids), n_rows, _addr(table_a), th) != 0:
            return None
    unique_key_cols = {}
    for i, k in enumerate(keys):
        key_idx = ((pres // strides[i]) % sizes[i]).astype(np.int64)
        unique_key_cols[k] = pc.take(dicts[i], _npa(key_idx))
    return ids, n_groups, unique_key_cols


def encode_int(values: np.ndarray):
    """Dense group ids for an integer column, numbered in order of first appearance (F-020).

    Returns ``(ids int32[n], uniq int64[g], first int32[g])`` (``uniq[g]`` is group g's value,
    ``first[g]`` its first row, ascending) exactly as ``Array.dictionary_encode`` would number the
    groups, or ``None`` when the kernel does not apply: too few rows, a dtype it does not read, or a
    value range too wide to address directly (the caller then hashes as before).
    """
    n = len(values)
    if n < IK_MIN_ROWS or n >= (1 << 31) - 1 or not available():
        return None
    w = _IK_WIDTH.get(values.dtype)
    if w is None or values.ndim != 1:
        return None
    lib = _load()
    if not hasattr(lib, "mx_ik_range"):
        return None
    values = np.ascontiguousarray(values)
    th = _threads()
    mm = np.empty(2, dtype=np.int64)
    if lib.mx_ik_range(_addr(values), w, n, _addr(mm), th) != 0:
        return None
    lo, hi = int(mm[0]), int(mm[1])
    r = hi - lo + 1
    if r > IK_MAX_RANGE_FACTOR * n + 65536 or r >= (1 << 31) - 1:
        return None
    tab = np.empty(r, dtype=np.int32)
    lib.mx_gk_touch(_addr(tab), tab.nbytes, th)
    if lib.mx_ik_first(_addr(values), w, n, lo, r, _addr(tab), th) != 0:
        return None
    ids = np.empty(n, dtype=np.int32)             # scratch for the ranks, then the result
    lib.mx_gk_touch(_addr(ids), ids.nbytes, th)
    g = lib.mx_ik_rank(_addr(tab), r, n, _addr(ids), th)
    if g <= 0:
        return None
    uniq = np.empty(g, dtype=np.int64)
    first = np.empty(g, dtype=np.int32)
    lib.mx_ik_finish(_addr(tab), r, n, lo, _addr(ids), _addr(uniq), _addr(first), th)
    lib.mx_ik_apply(_addr(values), w, n, lo, _addr(tab), _addr(ids), th)
    return ids, uniq, first


def composite(codes, strides):
    """int64 ``sum(code_k * stride_k)`` of per-key code arrays (any integer dtype), or None."""
    n = len(codes[0])
    if n < IK_MIN_ROWS or not available() or not 1 <= len(codes) <= 8:
        return None
    lib = _load()
    if not hasattr(lib, "mx_ik_composite"):
        return None
    codes = [np.ascontiguousarray(c) for c in codes]
    widths = [_IK_WIDTH.get(c.dtype) for c in codes]
    if None in widths or any(len(c) != n for c in codes):
        return None
    addrs = np.array([_addr(c) for c in codes], dtype=np.int64)
    ws = np.array(widths, dtype=np.int64)
    st = np.array(strides, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    if lib.mx_ik_composite(len(codes), _addr(addrs), _addr(ws), _addr(st), n, _addr(out), _threads()) != 0:
        return None
    return out


def dense_sorted(comp: np.ndarray, lead=None):
    """``np.unique(comp, return_inverse=True)[::-1]`` for non-negative int64 ``comp``, or None.

    Returns ``(ids int32[n], unique int64[g], first int32[g] or None)``: dense ids numbered in
    ascending value order.  Small value ranges use a presence table, wide ones a parallel radix sort,
    or, when ``lead = (stride, n_codes)`` says ``comp // stride`` is a code in [0, n_codes) shared
    by only a few rows each (an order key leading a composite), a counting sort on that code.
    """
    n = len(comp)
    if n < IK_MIN_ROWS or n >= (1 << 31) - 1 or comp.dtype != np.int64 or not available():
        return None
    lib = _load()
    if not hasattr(lib, "mx_ik_present"):
        return None
    comp = np.ascontiguousarray(comp)
    th = _threads()
    mm = np.empty(2, dtype=np.int64)
    if lib.mx_ik_range(_addr(comp), 3, n, _addr(mm), th) != 0:
        return None
    lo, hi = int(mm[0]), int(mm[1])
    r = hi - lo + 1
    if r <= IK_MAX_RANGE_FACTOR * n + 65536 and r < (1 << 31) - 1:
        pres = np.zeros(r, dtype=np.uint8)
        if lib.mx_ik_present(_addr(comp), 3, n, lo, r, _addr(pres), th) != 0:
            return None
        tab = np.empty(r, dtype=np.int32)
        g = lib.mx_ik_prefix(_addr(pres), r, _addr(tab), th)
        uniq = np.empty(g, dtype=np.int64)
        lib.mx_ik_uniq(_addr(pres), _addr(tab), r, lo, _addr(uniq), th)
        ids = np.empty(n, dtype=np.int32)
        lib.mx_ik_apply(_addr(comp), 3, n, lo, _addr(tab), _addr(ids), th)
        return ids, uniq, None
    if lead is not None:
        ranked = _rank_bucketed(lib, comp, *lead, th)
        if ranked is not None:
            return ranked
    try:
        from .aot_kernels import AOTKernels
        from .sort_check import checked_sort_indices
        perm = checked_sort_indices(AOTKernels(), comp)
    except Exception:
        return None
    ids = np.empty(n, dtype=np.int32)
    first = np.empty(n, dtype=np.int32)
    g = lib.mx_ik_rank_sorted(_addr(comp), _addr(perm), n, _addr(ids), _addr(first), th)
    if g <= 0:
        return None
    first = first[:g]
    return ids, comp[first], first


def _rank_bucketed(lib, comp, stride, n_codes, th):
    """``dense_sorted``'s sorted-path result through buckets of ``comp // stride``, or None."""
    n = len(comp)
    if not hasattr(lib, "mx_ik_rank_bucketed") or stride <= 0 or not 0 < n_codes < (1 << 31) - 1:
        return None
    if n_codes * 16 < n:            # long buckets: the per-bucket insertion sort would lose
        return None
    ids = np.empty(n, dtype=np.int32)
    first = np.empty(n, dtype=np.int32)
    cnt = np.empty(n_codes + 1, dtype=np.int32)
    rows = np.empty(n, dtype=np.int32)
    g = lib.mx_ik_rank_bucketed(_addr(comp), n, int(stride), int(n_codes), _addr(ids), _addr(first),
                                _addr(cnt), _addr(rows), th)
    if g <= 0:
        return None
    first = first[:g]
    return ids, comp[first], first


def first_rows(ids: np.ndarray, n_groups: int):
    """Ascending row numbers of the first row of every group (``ids`` int32 in [0, n_groups)), or None."""
    n = len(ids)
    if n < IK_MIN_ROWS or n >= (1 << 31) - 1 or ids.dtype != np.int32 or not available():
        return None
    lib = _load()
    if not hasattr(lib, "mx_ik_first"):
        return None
    ids = np.ascontiguousarray(ids)
    tab = np.empty(n_groups, dtype=np.int32)
    if lib.mx_ik_first(_addr(ids), 2, n, 0, n_groups, _addr(tab), _threads()) != 0:
        return None
    mark = np.zeros(n, dtype=np.bool_)
    mark[tab[tab < n]] = True          # n = group without rows
    return np.flatnonzero(mark)
